"""Write helpers shared by the Streamlit dashboard and the CLI.

These wrappers exist so the dashboard does not shell out to `rainier`
(slow, brittle) — both UI surfaces hit the same Python paths.

Every function returns a small dict the caller can render as a status
line. Errors raise standard exceptions so the Streamlit page can show a
red banner without being coupled to Click's exception types.

  ┌──────────────┐         ┌──────────────────────────┐
  │   app.py     │ ──────▶ │       actions.py         │
  │ (Streamlit)  │         │  toggle / accept / reject │
  └──────────────┘         └────────┬─────────────────┘
                                    │
                              ┌─────▼─────────────────────────┐
                              │  llm_thesis.research          │
                              │   (ACTION_EXECUTORS for accept │
                              │    + ruamel.yaml round-trip)   │
                              └────────────────────────────────┘
"""

from __future__ import annotations

import logging
from datetime import date as date_cls
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rainier.core.config import load_settings_fresh
from rainier.core.database import get_session
from rainier.core.models import ResearchInsight
from rainier.core.types import StockCandidate
from rainier.llm_thesis.research import apply_action
from rainier.llm_thesis.signals import REGISTRY
from rainier.llm_thesis.signals.base import SignalContext

log = logging.getLogger(__name__)


_DEFAULT_SETTINGS_PATH = "config/settings.yaml"


# ---------------------------------------------------------------------------
# Internal: ruamel YAML helper (mirrors research._load_yaml_round_trip but
# duplicated here so the dashboard's signal toggle uses the comment-
# preserving loader, matching the auto-research accept handler. The CLI's
# pre-PR3 toggle used PyYAML; we standardize on ruamel for PR4+.)
# ---------------------------------------------------------------------------


def _load_round_trip(path: Path):
    """Round-trip YAML loader — preserves comments + key order on dump."""
    from ruamel.yaml import YAML

    yaml = YAML(typ="rt")
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)
    if path.exists():
        with path.open("r") as f:
            data = yaml.load(f) or {}
    else:
        data = {}
    return yaml, data


def _atomic_dump(yaml, data, path: Path) -> None:
    """Atomic write via temp-file + os.replace. Keeps settings.yaml from
    being half-written if the process crashes mid-dump.
    """
    import os
    import tempfile

    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=str(parent))
    try:
        with os.fdopen(fd, "w") as f:
            yaml.dump(data, f)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# toggle_signal — Tab 1
# ---------------------------------------------------------------------------


def toggle_signal(
    signal_name: str,
    enabled: bool,
    *,
    settings_path: str | Path = _DEFAULT_SETTINGS_PATH,
) -> dict[str, Any]:
    """Flip `llm_thesis.signals.<name>.enabled` in settings.yaml.

    Mirrors the CLI's `rainier thesis signals enable/disable`
    semantics but uses ruamel.yaml so comments + key order survive
    the round-trip (matching the auto-research executors).

    Raises `ValueError` for unknown signals — the dashboard surfaces
    this as an error toast.
    """
    if signal_name not in REGISTRY:
        raise ValueError(
            f"Unknown signal {signal_name!r}. Known: {sorted(REGISTRY)}"
        )
    path = Path(settings_path)
    yaml, data = _load_round_trip(path)
    section = data.setdefault("llm_thesis", {}).setdefault("signals", {})
    entry = section.get(signal_name)
    if entry is None:
        section[signal_name] = {
            "enabled": bool(enabled),
            "params": {},
            "weight": 1.0,
        }
    else:
        entry["enabled"] = bool(enabled)
    _atomic_dump(yaml, data, path)
    log.info(
        "dashboard_toggle_signal name=%s enabled=%s path=%s",
        signal_name,
        enabled,
        path,
    )
    return {
        "signal": signal_name,
        "enabled": bool(enabled),
        "settings_path": str(path),
    }


# ---------------------------------------------------------------------------
# accept_insight / reject_insight — Tab 4
# ---------------------------------------------------------------------------


def accept_insight(
    insight_id: int,
    *,
    settings_path: str | Path = _DEFAULT_SETTINGS_PATH,
    decided_by: str = "dashboard",
) -> dict[str, Any]:
    """Apply the insight's structured action and mark it accepted.

    Mirrors the CLI accept handler exactly (validate-then-apply order
    so a YAML failure does not leak a half-accepted DB row). Raises
    `LookupError` for unknown id, `ValueError` for non-pending status
    or unknown action kind.
    """
    path = Path(settings_path)

    with get_session() as session:
        row = session.get(ResearchInsight, int(insight_id))
        if row is None:
            raise LookupError(f"No ResearchInsight with id={insight_id}")
        if row.status != "pending":
            raise ValueError(
                f"Insight {insight_id} has status={row.status!r}, not pending."
            )
        action = row.action or {}
        # apply_action validates the kind first (no side-effects) before
        # touching the YAML. A ValueError here leaves both DB + YAML
        # untouched.
        diff = apply_action(action, path)
        row.status = "accepted"
        row.decided_at = datetime.now(timezone.utc)
        row.decided_by = decided_by[:200]
        row.applied_change = diff
        session.flush()
    log.info(
        "dashboard_accept_insight id=%s kind=%s target=%s",
        insight_id,
        action.get("kind"),
        action.get("target"),
    )
    return {
        "insight_id": int(insight_id),
        "status": "accepted",
        "action_kind": action.get("kind"),
        "applied_change": diff,
    }


def reject_insight(
    insight_id: int,
    reason: str,
    *,
    decided_by: str = "dashboard",
) -> dict[str, Any]:
    """Mark a pending insight as rejected.

    The reason is stored in `decided_by` (200-char limit, matching the
    CLI). settings.yaml is not touched.
    """
    if not reason or not reason.strip():
        raise ValueError("Reject requires a non-empty reason.")
    decided = f"{decided_by}: {reason.strip()}" if decided_by else reason.strip()

    with get_session() as session:
        row = session.get(ResearchInsight, int(insight_id))
        if row is None:
            raise LookupError(f"No ResearchInsight with id={insight_id}")
        if row.status != "pending":
            raise ValueError(
                f"Insight {insight_id} has status={row.status!r}, not pending."
            )
        row.status = "rejected"
        row.decided_at = datetime.now(timezone.utc)
        row.decided_by = decided[:200]
        session.flush()
    log.info("dashboard_reject_insight id=%s reason=%s", insight_id, reason)
    return {
        "insight_id": int(insight_id),
        "status": "rejected",
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# test_signal — Tab 1 expander
# ---------------------------------------------------------------------------


def test_signal(
    signal_name: str,
    symbol: str,
    *,
    settings_path: str | Path = _DEFAULT_SETTINGS_PATH,
) -> dict[str, Any]:
    """Dry-run a single signal against the latest QU100 snapshot.

    Mirrors `rainier thesis signals test NAME --symbol X`. Returns a dict
    suitable for rendering as JSON in the Streamlit expander:

      {"signal": ..., "symbol": ..., "value": <SignalValue>,
       "render_for_prompt": <str>}

    Heavy imports (the screener — pandas, yfinance) live inside this
    function so the dashboard's import time stays reasonable when the
    operator only browses Tab 4.
    """
    if signal_name not in REGISTRY:
        raise ValueError(
            f"Unknown signal {signal_name!r}. Known: {sorted(REGISTRY)}"
        )

    # Local import: screener is heavy and only needed when the operator
    # actually runs the Test button.
    from rainier.analysis.stock_screener import screen_stocks

    settings = load_settings_fresh(str(settings_path))
    candidates, ohlcv = screen_stocks(settings)
    target: StockCandidate | None = next(
        (c for c in candidates if c.symbol.upper() == symbol.upper()),
        None,
    )
    if target is None:
        # Build a stub candidate so the operator can still test the
        # signal against a symbol the screener didn't pick up — same
        # shape the CLI builds.
        target = StockCandidate(
            symbol=symbol.upper(),
            rank=0,
            rank_change=0,
            long_short="Long in",
            capital_flow_direction="N",
            sector="Unknown",
            signal_strength=0.0,
        )

    cfg = settings.llm_thesis.signals.get(signal_name)
    params = dict(cfg.params) if cfg else {}
    ctx = SignalContext(
        symbol=target.symbol,
        scan_date=date_cls.today(),
        session_name="dashboard-test",
        candidate=target,
        ohlcv_df=ohlcv.get(target.symbol),
        params=params,
    )
    sig = REGISTRY[signal_name]()
    value = sig.compute(ctx)
    rendered = sig.render_for_prompt(value) if value is not None else ""
    return {
        "signal": signal_name,
        "symbol": target.symbol,
        "value": value,
        "render_for_prompt": rendered,
    }


__all__ = [
    "accept_insight",
    "reject_insight",
    "test_signal",
    "toggle_signal",
]
