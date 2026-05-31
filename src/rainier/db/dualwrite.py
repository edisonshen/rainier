"""Safe-rollout glue for Phase 2 dual-write (task plan §4).

The dual-write posture is *additive and non-fatal*: parquet is the load-bearing
output; the Postgres write is a mirror. Two ways PG can be unusable, both must
leave the parquet pipeline whole (exit 0):

  1. DATABASE_URL unset (local dev, CI without a DB) -> skip + warn (quiet).
  2. DATABASE_URL set but PG is unreachable / unmigrated / missing migration
     0002 -> the connect or upsert raises a SQLAlchemyError. We catch it, emit a
     LOUD, greppable diagnostic (PG-MIRROR-FAILURE), and continue. A flaky mirror
     DB must NOT abort `run-daily` (which would otherwise crash after writing
     labels parquet but before rendering).

``mirror_guard`` centralizes BOTH so all call sites share identical behavior:

    with mirror_guard("backfill_thematic_universe") as eng:
        if eng is None:
            ...            # parquet already written; PG skipped/failed + warned
        else:
            market_upsert(eng, ...)
    # engine disposed + SQLAlchemyError swallowed on the way out (rainier owns
    # the lifecycle of what it opens; a mirror failure is never fatal).

Visibility design (docs/DESIGN-mirror-guard-loud-on-failure.md): a set-but-failing
mirror emits a distinct ``PG-MIRROR-FAILURE`` block to stderr — host-redacted,
credential-scrubbed, greppable in cron logs — while the benign DATABASE_URL-unset
(or empty) skip stays quiet. The failure surfaces BEFORE *or* DURING the body:

    ┌─ mirror_guard ────────────────────────────────────────────────┐
    │ try:                                                           │
    │   try: eng = pg_engine_or_skip()   # may raise BEFORE yield    │
    │   except SQLAlchemyError: _emit(...); eng = None  ──┐          │
    │   yield eng   # ALWAYS reached (None on failure) <──┘          │
    │ except SQLAlchemyError: _emit(...)  # body (upsert) failure    │
    │ finally: if eng: eng.dispose()      # cleanup, both paths      │
    └────────────────────────────────────────────────────────────────┘

Always reaching the ``yield`` fixes a latent fatal bug: a before-``yield``
SQLAlchemyError used to escape as ``RuntimeError: generator didn't yield`` and
abort the caller, contradicting the non-fatal contract.
"""

from __future__ import annotations

import contextlib
import os
import re
import sys
from collections.abc import Iterator

from sqlalchemy.engine import Engine, make_url
from sqlalchemy.exc import SQLAlchemyError

_DESIGN_DOC = "docs/DESIGN-rainier-postgres-canonical-store.md"

# Drop a `<scheme>://<userinfo>@` segment, with OR without a `:password` part, so
# neither the password NOR the username survives. Scheme allows `+driver` forms
# (e.g. postgresql+psycopg). userinfo runs up to the LAST '@' before the host —
# greedy `[^/\s]*@` so an unescaped '@' INSIDE the password (e.g.
# `postgresql://u:pa@ss@db/prod`) is consumed too (the userinfo ends at the final
# '@' before the path/host); `[^/\s]` keeps us from crossing into the path.
_URL_USERINFO_RE = re.compile(
    r"(?P<scheme>[a-zA-Z][a-zA-Z0-9+.\-]*://)(?P<userinfo>[^/\s]*@)"
)
# password=<v> / pwd=<v>, optionally quoted (key/value connect-arg + JSON-ish).
# Two alternatives for the value so a QUOTED secret with embedded whitespace /
# separators (e.g. password='foo bar') is consumed WHOLE — an unquoted class that
# stops at whitespace would leave the tail of the secret in the scrubbed text.
_PASSWORD_KV_RE = re.compile(
    r"""(?P<key>\b(?:password|pwd)\b)   # password / pwd, word-bounded
        (?P<sep>['"]?\s*[:=]\s*)        # optional key-closing quote + : or = sep
        (?:
            (?P<sq>')(?:\\.|[^'\\])*(?P=sq)  # single-quoted: consume \-escapes so
                                              #   an escaped \' inside stays part of
                                              #   the secret; stop at the real close
          | (?P<dq>")(?:\\.|[^"\\])*(?P=dq)  # double-quoted: same, for \" (JSON)
          | [^'"\s,;]*                       # unquoted value: stop at ws/sep
        )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _scrub_credentials(text: str) -> str:
    """Strip credentials from arbitrary error text. Bounded scope (design §3.2):

      1. URL-shaped creds: ``<scheme>://<userinfo>@`` -> ``<scheme>://@`` (with or
         without a ``:password`` — the username never survives either way).
      2. key/value forms: ``password=<v>`` / ``pwd=<v>`` (quoted or not,
         case-insensitive) -> ``password=***``.

    Anything outside these two forms is left untouched. Never raises.
    """
    def _kv_repl(m: re.Match[str]) -> str:
        # Re-wrap with the captured quote when the value was quoted, so the
        # output stays well-formed (password='***'); bare *** when unquoted.
        quote = m.group("sq") or m.group("dq") or ""
        return f"{m.group('key')}{m.group('sep')}{quote}***{quote}"

    try:
        out = _URL_USERINFO_RE.sub(r"\g<scheme>@", text)
        out = _PASSWORD_KV_RE.sub(_kv_repl, out)
        return out
    except Exception:  # pragma: no cover - defensive; regex on str cannot raise
        return text


def _redact_host(database_url: str) -> str:
    """Render a credential-free host identifier from a DATABASE_URL.

    Parses with SQLAlchemy's ``make_url`` and renders host/port/database only —
    never username or password. Handles driver-prefixed URLs, IPv6 hosts,
    percent-encoded passwords, and query-string options. For a host-less /
    Unix-socket form, renders a MEANINGFUL identifier (the socket path from the
    ``host`` query param or the URL host field, plus the db name) rather than a
    bare ``/db``. On ANY parse failure, returns the literal ``"<unparseable>"``
    (never echoes the raw URL — it carries creds). Never raises.
    """
    try:
        url = make_url(database_url)
    except Exception:
        return "<unparseable>"

    try:
        db = url.database or ""
        host = url.host or ""
        # An unescaped '@' in the password (e.g. postgresql://u:pa@ss@db/prod)
        # makes make_url parse the host as `ss@db` — a password fragment bled into
        # the host field. Any '@' in the host means the parse was ambiguous and
        # creds may be embedded; refuse to render it (never leak to cron logs).
        if "@" in host:
            return "<unparseable>"
        # A "socket" host is a filesystem path (leading '/'), or it lives in the
        # `host` query param (libpq Unix-socket convention SQLAlchemy passes
        # through). Prefer the query-param socket dir when present.
        query = url.query or {}
        socket_q = query.get("host")
        if isinstance(socket_q, (list, tuple)):
            socket_q = socket_q[0] if socket_q else None

        socket_path = None
        if socket_q and str(socket_q).startswith("/"):
            socket_path = str(socket_q)
        elif host.startswith("/"):
            socket_path = host

        if socket_path:
            # Meaningful socket identifier: socket path + db name when available.
            return f"{socket_path}/{db}" if db else socket_path

        if not host:
            # No TCP host and no socket path — degenerate. Render whatever db we
            # have rather than an empty string, but never an empty/misleading host.
            return db or "<unparseable>"

        # TCP host. IPv6 hosts already come back bracket-free from make_url; wrap
        # them so the rendered form is unambiguous.
        rendered_host = f"[{host}]" if ":" in host else host
        if url.port:
            rendered_host = f"{rendered_host}:{url.port}"
        return f"{rendered_host}/{db}" if db else rendered_host
    except Exception:  # pragma: no cover - defensive
        return "<unparseable>"


def _emit_mirror_failure(writer_name: str, exc: BaseException) -> None:
    """Print the LOUD, greppable PG-MIRROR-FAILURE diagnostic to stderr.

    Non-fatal: the caller swallows the error after this. Host is redacted via
    ``_redact_host`` and the error message is credential-scrubbed via
    ``_scrub_credentials`` before printing.
    """
    host = _redact_host(os.environ.get("DATABASE_URL", ""))
    message = _scrub_credentials(str(exc))
    print(
        f"PG-MIRROR-FAILURE: {writer_name} dual-write FAILED — NOT written to {host}\n"
        f"  error: {type(exc).__name__}: {message}\n"
        f"  impact: NON-FATAL (parquet output is load-bearing and unaffected; exit 0)\n"
        f"  fix: ensure the mirror DB is reachable and migrated to head\n"
        f"       (see {_DESIGN_DOC})",
        file=sys.stderr,
    )


def pg_engine_or_skip(writer_name: str) -> Engine | None:
    """Return a fresh Engine, or None (with a warning) when PG is unconfigured.

    None means "DATABASE_URL is unset (or empty) — skip the PG mirror". The
    caller must have already done its parquet write so the pipeline stays whole.

    NOTE: this only handles the *unset/empty* case. A set-but-broken DATABASE_URL
    is handled by ``mirror_guard``, which catches the SQLAlchemyError the engine
    creation or upsert raises and emits the loud diagnostic. Prefer
    ``mirror_guard`` at call sites so broken-PG is non-fatal.
    """
    if not os.environ.get("DATABASE_URL"):
        print(
            f"warning: DATABASE_URL not set — {writer_name} skipping the "
            f"Postgres dual-write (parquet output unaffected). Set DATABASE_URL "
            f"to mirror into market.* (see {_DESIGN_DOC}).",
            file=sys.stderr,
        )
        return None

    # Import here so modules that never dual-write don't pay the engine import.
    from rainier.db.engine import get_engine

    return get_engine()


@contextlib.contextmanager
def mirror_guard(writer_name: str) -> Iterator[Engine | None]:
    """Context manager for an additive, non-fatal PG mirror write.

    Yields a fresh Engine (or None when DATABASE_URL is unset/empty). The caller
    runs its ``market_upsert`` calls in the ``with`` body. On exit the engine is
    disposed, and ANY ``SQLAlchemyError`` raised — either BEFORE the yield (engine
    creation: malformed/unreachable DATABASE_URL) or DURING the body (PG down,
    schema unmigrated, missing migration 0002, etc.) — is caught, emitted as a
    LOUD ``PG-MIRROR-FAILURE`` diagnostic to stderr, and swallowed, so a broken
    mirror DB never aborts the load-bearing parquet pipeline. Non-SQLAlchemy
    errors (programmer bugs) propagate.

    The generator ALWAYS reaches ``yield``: an engine-creation failure emits the
    diagnostic and yields ``None`` (caller takes its existing skip path), instead
    of escaping as ``RuntimeError: generator didn't yield`` and aborting the
    caller (the latent fatal bug this fixes).
    """
    eng: Engine | None = None
    try:
        try:
            eng = pg_engine_or_skip(writer_name)  # may raise BEFORE the yield
        except (SQLAlchemyError, ValueError) as exc:
            # SQLAlchemyError: unreachable/unmigrated PG. ValueError: a malformed
            # DATABASE_URL whose make_url/create_engine parse fails (e.g. a
            # non-numeric port → `int('notaport')`) raises a bare ValueError, not
            # SQLAlchemyError. Both are mirror-config failures on a SET url, so
            # both are non-fatal here (design §3.1). The BODY handler below stays
            # SQLAlchemyError-only so a programmer-bug ValueError still propagates.
            _emit_mirror_failure(writer_name, exc)
            eng = None  # …and fall through so we still yield
        yield eng  # ALWAYS yields (None on engine-creation failure)
    except SQLAlchemyError as exc:  # body (upsert) failure
        _emit_mirror_failure(writer_name, exc)
    finally:
        if eng is not None:
            eng.dispose()
