"""Experiment-spec interpreter for the A/B substrate (design §10.4).

An experiment is a YAML file, not engine code: champion + challengers +
the one contended ``layer`` + reward keys + window. Specs live in
``config/experiments/*.yaml``; each carries ``status: active | retired``
and :func:`load_active_specs` is the single discovery entry point (cron
shadow arm, evaluator CLI).

    experiment.yaml ──► interpreter ──► validate override keys ──► deep-merge
                            │             (model_fields +            (existing
                            │              pattern names;             merge_stock_
                            │              reject unknown)            screener_config)
                            └──► mutual exclusion across ACTIVE specs:
                                 layer label unique AND override-key sets disjoint

The override-key validation is load-bearing and CANNOT be delegated
downstream: ``merge_stock_screener_config`` performs no key validation,
and ``StockScreenerConfig(**merged)`` silently drops unknown kwargs
(``core/champion.py:86-88``) — an unvalidated typo'd key would yield a
challenger identical to the champion, a silent no-op A/B that "finds" no
difference. The interpreter never invents config paths: a knob must exist
as a ``StockScreenerConfig`` field before it can be experimented on.

Reward keys (``primary`` / ``guardrails``) are opaque strings here,
resolved at run time by the reward registry — no import of the rewards
module (keeps this contract independent of the registry task).

``champion:`` names the champion model file the experiment was authored
against (``champion.yaml`` — a file reference, not a version pin; §10.4
has no version field). Champion-drift attribution across promotions is
NOT this layer's job: the evaluator replays champion + challengers from
ONE :func:`materialize_challengers` call over one corpus (so within any
evaluation they share the exact base), and every result row is stamped
with the champion ``version`` + ``corpus_hash`` in the results registry
(§10.5) — a promotion mid-experiment changes future rows' recorded
baseline, never silently mixes baselines within one comparison. The
operator-gated promote flow is a separate task and does not exist yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:  # runtime imports stay local (config load cost, cycle safety)
    from rainier.core.config import StockScreenerConfig

# Spec home (pinned in the task plan; consumed by the shadow arm + evaluator).
DEFAULT_EXPERIMENTS_DIR = Path("config/experiments")

# Default purge/embargo gap for walk-forward validation = the max
# forward-return horizon H (20 trading days), per §10.4.
DEFAULT_EMBARGO_DAYS = 20

_VALID_STATUSES = ("active", "retired")

# Design §10.4 pins the exact spec shape — unknown keys are rejected loudly.
# Without this, a typo'd `guardrail:` (singular) would parse cleanly into a
# guardrail-free experiment, and `embargo_dayz:` would silently keep the
# default embargo — the same silent-no-op class the override-key check kills.
_ALLOWED_SPEC_KEYS = frozenset(
    {"id", "status", "champion", "layer", "challengers", "primary", "guardrails", "window"}
)
_ALLOWED_WINDOW_KEYS = frozenset({"train", "holdout", "embargo_days"})
_ALLOWED_CHALLENGER_KEYS = frozenset({"id", "override"})


class ExperimentSpecError(ValueError):
    """Raised when an experiment spec fails parse or validation."""


class _NoDuplicateKeysLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate mapping keys.

    Plain ``yaml.safe_load`` silently last-write-wins on duplicate keys —
    a spec with two ``buy_threshold:`` lines (or two ``window:`` blocks)
    would load the second and drop the first without a sound, defeating
    the interpreter's declare-it-once stance.
    """

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict:
        seen: list[Any] = []
        for key_node, _value_node in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge":
                # `<<:` merge keys are flattened by SafeLoader with documented
                # override semantics (explicit keys win) — not author duplicates.
                continue
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key {key!r} — last-write-wins would be "
                    f"silent; declare it once",
                    key_node.start_mark,
                )
            seen.append(key)
        return super().construct_mapping(node, deep)


@dataclass(frozen=True)
class ExperimentWindow:
    """Train/holdout ranges (``start..end`` strings) + embargo gap."""

    train: str
    holdout: str
    embargo_days: int = DEFAULT_EMBARGO_DAYS


@dataclass(frozen=True)
class Challenger:
    """One challenger arm: id + validated override (dotted paths translated)."""

    id: str
    override: dict[str, Any]


@dataclass(frozen=True)
class ExperimentSpec:
    """A parsed, validated experiment spec (design §10.4)."""

    id: str
    status: str
    champion: str
    layer: str
    challengers: tuple[Challenger, ...]
    primary: str
    guardrails: tuple[str, ...]
    window: ExperimentWindow
    source: str = "<memory>"

    @property
    def is_active(self) -> bool:
        return self.status == "active"

    @property
    def override_keys(self) -> frozenset[str]:
        """Granular knob set this experiment contends, across all challengers.

        ``pattern_weights`` is expanded to per-pattern ``pattern_weights.<p>``
        keys so two experiments touching DIFFERENT patterns don't conflict.
        Used by :func:`load_active_specs` for cross-spec mutual exclusion.
        """
        keys: set[str] = set()
        for challenger in self.challengers:
            for key, value in challenger.override.items():
                if key == "pattern_weights" and isinstance(value, dict):
                    keys.update(f"pattern_weights.{p}" for p in value)
                else:
                    keys.add(key)
        return frozenset(keys)


def _validate_override(
    override: Any, *, spec_id: str, challenger_id: str, source: str
) -> dict[str, Any]:
    """Validate every override key; translate ``pattern_weights.<p>`` dotted paths.

    Reuses the check pattern from ``load_champion_overrides``
    (``core/champion.py:91-110``): flat keys against
    ``StockScreenerConfig.model_fields``, nested/dotted pattern keys against
    the known pattern names. Rejects the spec on ANY unknown key — see the
    module docstring for why this is interpreter-side.
    """
    from rainier.core.config import StockScreenerConfig

    ctx = f"{source}: experiment {spec_id!r} challenger {challenger_id!r}"
    if not isinstance(override, dict) or not override:
        raise ExperimentSpecError(
            f"{ctx}: override must be a non-empty mapping — an empty override "
            f"yields a challenger identical to the champion (silent no-op A/B)"
        )

    known_fields = set(StockScreenerConfig.model_fields)
    known_patterns = set(StockScreenerConfig().pattern_weights)
    out: dict[str, Any] = {}
    for key, value in override.items():
        if not isinstance(key, str):
            # YAML 1.1 quirks (`on:` → True, bare ints) must surface as the
            # spec-contract exception, not a downstream TypeError.
            raise ExperimentSpecError(
                f"{ctx}: override keys must be str, got {key!r} "
                f"({type(key).__name__})"
            )
        if key.startswith("pattern_weights."):
            pattern = key.split(".", 1)[1]
            if pattern not in known_patterns:
                raise ExperimentSpecError(
                    f"{ctx}: unknown pattern_weights key {pattern!r} "
                    f"(typo? known: {sorted(known_patterns)})"
                )
            pw = out.setdefault("pattern_weights", {})
            if pattern in pw:
                raise ExperimentSpecError(
                    f"{ctx}: pattern {pattern!r} assigned twice (dotted "
                    f"`pattern_weights.{pattern}` AND nested `pattern_weights:`) "
                    f"— last-write-wins would be silent; declare it once"
                )
            pw[pattern] = value
        elif key == "pattern_weights":
            if not isinstance(value, dict) or not value:
                # An EMPTY mapping would translate to an empty override —
                # a challenger identical to the champion (silent no-op A/B)
                # that would also escape cross-spec mutual exclusion.
                raise ExperimentSpecError(
                    f"{ctx}: pattern_weights override must be a non-empty "
                    f"mapping, got {value!r}"
                )
            non_str_pw = [p for p in value if not isinstance(p, str)]
            if non_str_pw:
                raise ExperimentSpecError(
                    f"{ctx}: pattern_weights keys must be str, got {non_str_pw!r}"
                )
            unknown_pw = sorted(set(value) - known_patterns)
            if unknown_pw:
                raise ExperimentSpecError(
                    f"{ctx}: unknown pattern_weights key(s): "
                    f"{', '.join(unknown_pw)} (typo? known: {sorted(known_patterns)})"
                )
            pw = out.setdefault("pattern_weights", {})
            dup = sorted(set(value) & set(pw))
            if dup:
                raise ExperimentSpecError(
                    f"{ctx}: pattern(s) {', '.join(dup)} assigned twice (dotted "
                    f"`pattern_weights.<p>` AND nested `pattern_weights:`) "
                    f"— last-write-wins would be silent; declare each once"
                )
            pw.update(value)
        elif key in known_fields:
            out[key] = value
        else:
            raise ExperimentSpecError(
                f"{ctx}: unknown override key {key!r} — override keys must be "
                f"real StockScreenerConfig fields (or pattern_weights.<pattern> "
                f"dotted paths); the interpreter never invents config paths"
            )

    # Value smoke-check at parse time: a bad value (e.g. a non-numeric weight)
    # must fail at spec-authoring/load time as ExperimentSpecError, not at
    # 8am inside the cron shadow arm as a raw pydantic ValidationError.
    # STRICT validation: lax construction would coerce `buy_threshold: true`
    # to 1.0 (a plausible-looking losing challenger — silent corruption of
    # the A/B result) and accept numeric strings like "0.35". Strict mode
    # still accepts int where a float is declared, so `weight: 1` stays valid.
    from pydantic import ValidationError

    from rainier.core.champion import merge_stock_screener_config

    try:
        StockScreenerConfig.model_validate(
            merge_stock_screener_config(None, out), strict=True
        )
    except ValidationError as exc:
        raise ExperimentSpecError(
            f"{ctx}: override value(s) rejected by StockScreenerConfig — {exc}"
        ) from exc

    # Strict mode still admits `.nan` / `.inf` YAML floats: a NaN threshold
    # makes every score comparison False and a NaN weight poisons composite
    # scores — the same silent-corruption class, so reject non-finite here.
    _require_finite_values(out, ctx)
    return out


def _require_finite_values(overrides: dict[str, Any], ctx: str) -> None:
    """Reject nan/inf floats in an override layer (incl. nested pattern_weights).

    pydantic strict mode admits ``.nan`` / ``.inf`` floats; non-finite
    thresholds/weights poison score comparisons silently instead of failing.
    Used for spec-authored overrides at parse time AND caller-supplied
    ``base_overrides`` at materialization time.
    """
    import math

    def _check(label: str, value: Any) -> None:
        if isinstance(value, float) and not math.isfinite(value):
            raise ExperimentSpecError(
                f"{ctx}: {label!r} is {value!r} — non-finite values "
                f"(nan/inf) poison score comparisons silently"
            )

    for key, value in overrides.items():
        if key == "pattern_weights" and isinstance(value, dict):
            for pattern, weight in value.items():
                _check(f"pattern_weights.{pattern}", weight)
        else:
            _check(key, value)


def _require_str(raw: dict[str, Any], key: str, source: str) -> str:
    val = raw.get(key)
    if not isinstance(val, str) or not val:
        raise ExperimentSpecError(
            f"{source}: spec field {key!r} must be a non-empty str, got {val!r}"
        )
    return val


_WINDOW_RANGE_HINT = "YYYY-MM-DD..YYYY-MM-DD"


def _parse_date_range(value: Any, *, field: str, ctx: str) -> tuple[date, date]:
    """Parse one ``start..end`` window string into (start, end) dates.

    A garbage/empty/reversed range would otherwise parse cleanly here and
    detonate (or worse, silently mis-slice) downstream in the evaluator at
    cron time — the exact failure class the parse-time value smoke-check
    exists to prevent for override values.
    """
    if not isinstance(value, str) or not value:
        raise ExperimentSpecError(
            f"{ctx}: window.{field} is a required non-empty string "
            f"({_WINDOW_RANGE_HINT}), got {value!r}"
        )
    parts = value.split("..")
    if len(parts) != 2:
        raise ExperimentSpecError(
            f"{ctx}: window.{field} must be {_WINDOW_RANGE_HINT!r} "
            f"(exactly one `..` separator), got {value!r}"
        )
    try:
        start, end = date.fromisoformat(parts[0]), date.fromisoformat(parts[1])
    except ValueError as exc:
        raise ExperimentSpecError(
            f"{ctx}: window.{field} must be {_WINDOW_RANGE_HINT!r}, "
            f"got {value!r} — {exc}"
        ) from exc
    if start > end:
        raise ExperimentSpecError(
            f"{ctx}: window.{field} start {parts[0]} is after end {parts[1]} "
            f"— reversed range"
        )
    return start, end


def parse_spec(raw: Any, source: str = "<memory>") -> ExperimentSpec:
    """Parse + validate one raw spec mapping into an :class:`ExperimentSpec`."""
    if not isinstance(raw, dict):
        raise ExperimentSpecError(
            f"{source}: spec must be a YAML mapping, got {type(raw).__name__}"
        )
    non_str = [k for k in raw if not isinstance(k, str)]
    if non_str:
        raise ExperimentSpecError(
            f"{source}: spec keys must be str, got {non_str!r} "
            f"(YAML 1.1 parses bare `on`/`yes` as booleans — quote them)"
        )
    unknown_keys = sorted(set(raw) - _ALLOWED_SPEC_KEYS)
    if unknown_keys:
        raise ExperimentSpecError(
            f"{source}: unknown spec key(s): {', '.join(unknown_keys)} — the "
            f"§10.4 spec shape is exact (allowed: {sorted(_ALLOWED_SPEC_KEYS)}); "
            f"a typo'd key (e.g. `guardrail:`) would otherwise be silently ignored"
        )
    spec_id = _require_str(raw, "id", source)
    status = _require_str(raw, "status", source)
    if status not in _VALID_STATUSES:
        raise ExperimentSpecError(
            f"{source}: experiment {spec_id!r} has invalid status {status!r} "
            f"(must be one of {list(_VALID_STATUSES)})"
        )
    champion = _require_str(raw, "champion", source)
    layer = _require_str(raw, "layer", source)
    primary = _require_str(raw, "primary", source)

    guardrails_raw = raw.get("guardrails")
    if guardrails_raw is None:
        guardrails_raw = []
    if not isinstance(guardrails_raw, list) or not all(
        isinstance(g, str) and g for g in guardrails_raw
    ):
        raise ExperimentSpecError(
            f"{source}: experiment {spec_id!r}: guardrails must be a list of "
            f"non-empty strings"
        )
    dup_guardrails = sorted({g for g in guardrails_raw if guardrails_raw.count(g) > 1})
    if dup_guardrails:
        raise ExperimentSpecError(
            f"{source}: experiment {spec_id!r}: duplicate guardrail(s): "
            f"{', '.join(dup_guardrails)}"
        )
    if primary in guardrails_raw:
        raise ExperimentSpecError(
            f"{source}: experiment {spec_id!r}: primary reward {primary!r} is "
            f"also listed in guardrails — a reward key is either the "
            f"optimization target or a guardrail, not both"
        )

    window_raw = raw.get("window")
    if not isinstance(window_raw, dict):
        raise ExperimentSpecError(
            f"{source}: experiment {spec_id!r}: window must be a mapping with "
            f"train/holdout (got {window_raw!r})"
        )
    non_str_w = [k for k in window_raw if not isinstance(k, str)]
    if non_str_w:
        raise ExperimentSpecError(
            f"{source}: experiment {spec_id!r}: window keys must be str, "
            f"got {non_str_w!r}"
        )
    unknown_window = sorted(set(window_raw) - _ALLOWED_WINDOW_KEYS)
    if unknown_window:
        raise ExperimentSpecError(
            f"{source}: experiment {spec_id!r}: unknown window key(s): "
            f"{', '.join(unknown_window)} (allowed: {sorted(_ALLOWED_WINDOW_KEYS)}; "
            f"a typo'd `embargo_dayz:` would otherwise silently keep the default)"
        )
    ctx = f"{source}: experiment {spec_id!r}"
    train = window_raw.get("train")
    holdout = window_raw.get("holdout")
    _, train_end = _parse_date_range(train, field="train", ctx=ctx)
    holdout_start, _ = _parse_date_range(holdout, field="holdout", ctx=ctx)
    if holdout_start <= train_end:
        # Overlapping (or reversed) train/holdout leaks training data into
        # the holdout. The embargo gap in TRADING days is enforced by the
        # evaluator, which owns the trading calendar — here we can only pin
        # the calendar-level ordering invariant.
        raise ExperimentSpecError(
            f"{ctx}: window.holdout ({holdout}) must start after window.train "
            f"ends ({train}) — overlapping windows leak train data into the "
            f"holdout"
        )
    embargo_days = window_raw.get("embargo_days", DEFAULT_EMBARGO_DAYS)
    if not isinstance(embargo_days, int) or isinstance(embargo_days, bool) or embargo_days < 0:
        raise ExperimentSpecError(
            f"{source}: experiment {spec_id!r}: embargo_days must be a "
            f"non-negative int, got {embargo_days!r}"
        )

    challengers_raw = raw.get("challengers")
    if not isinstance(challengers_raw, list) or not challengers_raw:
        raise ExperimentSpecError(
            f"{source}: experiment {spec_id!r}: challengers must be a non-empty list"
        )
    challengers: list[Challenger] = []
    seen_ids: set[str] = set()
    for entry in challengers_raw:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            raise ExperimentSpecError(
                f"{source}: experiment {spec_id!r}: each challenger needs an "
                f"`id` and an `override` mapping (got {entry!r})"
            )
        non_str_c = [k for k in entry if not isinstance(k, str)]
        if non_str_c:
            raise ExperimentSpecError(
                f"{source}: experiment {spec_id!r}: challenger keys must be "
                f"str, got {non_str_c!r}"
            )
        unknown_c = sorted(set(entry) - _ALLOWED_CHALLENGER_KEYS)
        if unknown_c:
            raise ExperimentSpecError(
                f"{source}: experiment {spec_id!r}: unknown challenger key(s): "
                f"{', '.join(unknown_c)} (allowed: id, override)"
            )
        cid = entry["id"]
        if not cid:
            raise ExperimentSpecError(
                f"{source}: experiment {spec_id!r}: challenger id must be a "
                f"non-empty str"
            )
        if cid in seen_ids:
            raise ExperimentSpecError(
                f"{source}: experiment {spec_id!r}: duplicate challenger id {cid!r}"
            )
        seen_ids.add(cid)
        override = _validate_override(
            entry.get("override"), spec_id=spec_id, challenger_id=cid, source=source
        )
        challengers.append(Challenger(id=cid, override=override))

    return ExperimentSpec(
        id=spec_id,
        status=status,
        champion=champion,
        layer=layer,
        challengers=tuple(challengers),
        primary=primary,
        guardrails=tuple(guardrails_raw),
        window=ExperimentWindow(train=train, holdout=holdout, embargo_days=embargo_days),
        source=source,
    )


def load_spec(path: str | Path) -> ExperimentSpec:
    """Load + validate one experiment spec YAML file.

    Every failure — including a YAML syntax error — surfaces as
    :class:`ExperimentSpecError`, so consumers catching the spec-contract
    exception can't miss a broken file.
    """
    p = Path(path)
    try:
        with p.open("r") as fh:
            raw = yaml.load(fh, Loader=_NoDuplicateKeysLoader)  # noqa: S506 — SafeLoader subclass
    except yaml.YAMLError as exc:
        raise ExperimentSpecError(f"{p}: invalid YAML — {exc}") from exc
    except (OSError, UnicodeDecodeError) as exc:
        # PermissionError, IsADirectoryError, non-UTF8 bytes, … — same
        # contract exception.
        raise ExperimentSpecError(f"{p}: unreadable — {exc}") from exc
    return parse_spec(raw, source=str(p))


def load_active_specs(directory: str | Path | None = None) -> list[ExperimentSpec]:
    """Discover specs in ``config/experiments/*.yaml``; return ACTIVE ones only.

    The single discovery entry point for every consumer. Every spec in the
    directory must parse (a malformed spec — bad status, unknown override
    key — fails the whole load loudly; retired-but-broken specs don't get to
    rot silently). Mutual exclusion is evaluated across the ACTIVE set:
    two active specs claiming the same ``layer`` label is an error, duplicate
    active experiment ids are an error, and — because a layer label is just a
    name — two active specs overriding the SAME config knob under different
    labels is ALSO an error (entangled experiments would contaminate
    attribution silently). Retired specs never conflict.

    Discovery itself fails loudly too: a MISSING directory raises (a wrong
    CWD or a deploy that lost ``config/experiments`` would otherwise be
    indistinguishable from a legitimately empty queue — the same silent-no-op
    class the interpreter exists to kill), and stray ``*.yml`` files are
    rejected rather than silently skipped.
    """
    base = Path(directory) if directory is not None else DEFAULT_EXPERIMENTS_DIR
    if not base.is_dir():
        raise ExperimentSpecError(
            f"experiments directory {base} does not exist — a missing "
            f"directory would silently disable every experiment "
            f"(indistinguishable from an empty queue); create it or pass "
            f"the right path"
        )
    # Any yaml-ish file that `*.yaml` won't match (.yml, .YAML, .YML, …) is
    # rejected loudly instead of silently skipped. Dotfiles (editor lockfiles
    # like `.#spec.yaml`, backups) are not specs and are ignored entirely —
    # an operator merely having a spec open in an editor at cron time must
    # not fail the run.
    stray = sorted(
        p.name
        for p in base.iterdir()
        if not p.name.startswith(".")
        and p.suffix.lower() in {".yml", ".yaml"}
        and p.suffix != ".yaml"
    )
    if stray:
        raise ExperimentSpecError(
            f"stray spec file(s) in {base}: {', '.join(stray)} — discovery "
            f"only reads *.yaml (lowercase); rename so the spec isn't "
            f"silently ignored"
        )
    active: list[ExperimentSpec] = []
    layer_owner: dict[str, str] = {}
    key_owner: dict[str, str] = {}
    challenger_owner: dict[str, str] = {}
    seen_ids: dict[str, str] = {}
    for path in sorted(base.glob("*.yaml")):
        if path.name.startswith("."):
            continue  # editor lockfiles / backups, not specs
        spec = load_spec(path)
        if not spec.is_active:
            continue
        if spec.id in seen_ids:
            raise ExperimentSpecError(
                f"duplicate active experiment id {spec.id!r}: "
                f"{seen_ids[spec.id]} and {path}"
            )
        seen_ids[spec.id] = str(path)
        if spec.layer in layer_owner:
            raise ExperimentSpecError(
                f"mutual exclusion: layer {spec.layer!r} claimed by two active "
                f"experiments: {layer_owner[spec.layer]!r} and {spec.id!r} — "
                f"retire one before activating the other"
            )
        layer_owner[spec.layer] = spec.id
        overlap = sorted(k for k in spec.override_keys if k in key_owner)
        if overlap:
            raise ExperimentSpecError(
                f"mutual exclusion: override key(s) {', '.join(overlap)} "
                f"contended by two active experiments: "
                f"{key_owner[overlap[0]]!r} and {spec.id!r} — a knob belongs "
                f"to at most one active experiment (layer labels don't "
                f"substitute for disjoint key sets)"
            )
        for k in spec.override_keys:
            key_owner[k] = spec.id
        for challenger in spec.challengers:
            if challenger.id in challenger_owner:
                # Scorecards key on candidate_id (base_scorecard carries no
                # experiment_id), so two live arms sharing an id would emit
                # indistinguishable scorecards — misattribution, silently.
                raise ExperimentSpecError(
                    f"challenger id {challenger.id!r} used by two active "
                    f"experiments: {challenger_owner[challenger.id]!r} and "
                    f"{spec.id!r} — arm ids must be unique across the active set"
                )
            challenger_owner[challenger.id] = spec.id
        active.append(spec)
    return active


def materialize_challengers(
    spec: ExperimentSpec,
    base_overrides: dict[str, Any] | None = None,
) -> tuple["StockScreenerConfig", dict[str, "StockScreenerConfig"]]:
    """Materialize (champion_config, {challenger_id: config}) for a spec.

    ``base_overrides`` is the champion-effective override layer (e.g.
    ``settings.yaml:stock_screener`` already merged with ``champion.yaml``
    via the existing loaders). Each challenger = that layer deep-merged with
    its validated override via the existing ``merge_stock_screener_config``
    — so a challenger differs from the champion ONLY in the overridden keys,
    and a partial ``pattern_weights`` override keeps every other pattern's
    weight. Deep-merge happens only AFTER override-key validation (done at
    parse time); this function never sees unvalidated keys.

    ``base_overrides`` come from the CALLER (settings/champion layer), not
    the spec, so they are validated here — a bad value must still surface as
    the contract exception, not a raw pydantic ``ValidationError`` inside the
    unattended cron shadow arm. Construction stays lax (parity with the live
    config loaders); spec-authored values were already strict-checked at
    parse time.

    CONSUMER CONTRACT (champion-drift attribution, §10.5): every evaluation
    must take champion AND challengers from ONE call to this function over
    one corpus — never mix configs materialized before and after a champion
    promotion — and must stamp the champion ``version`` + ``corpus_hash``
    into the results registry row. ``spec.champion`` is a file reference,
    not a version pin (§10.4 defines no version field); baseline identity is
    recorded per-result by the registry, not frozen at parse time.
    """
    from pydantic import ValidationError

    from rainier.core.champion import merge_stock_screener_config
    from rainier.core.config import StockScreenerConfig

    if base_overrides:
        # Key-check the caller's layer too: StockScreenerConfig silently
        # drops unknown kwargs, so a typo'd base key would materialize a
        # "champion" that doesn't match the live config — the silent-no-op
        # class this module never delegates downstream.
        known = set(StockScreenerConfig.model_fields)
        unknown = sorted(
            repr(k) for k in base_overrides if not isinstance(k, str) or k not in known
        )
        if unknown:
            raise ExperimentSpecError(
                f"experiment {spec.id!r}: unknown base_overrides key(s): "
                f"{', '.join(unknown)} — StockScreenerConfig would silently "
                f"drop them (materialized champion would not match the live config)"
            )
        base_pw = base_overrides.get("pattern_weights")
        if isinstance(base_pw, dict):
            known_patterns = set(StockScreenerConfig().pattern_weights)
            unknown_pw = sorted(repr(p) for p in set(base_pw) - known_patterns)
            if unknown_pw:
                raise ExperimentSpecError(
                    f"experiment {spec.id!r}: unknown base_overrides "
                    f"pattern_weights key(s): {', '.join(unknown_pw)}"
                )
        # A nan/inf in the base layer would materialize a champion (and every
        # challenger inheriting the field) with non-finite thresholds/weights
        # — the same silent-poisoning _validate_override guards against for
        # spec-authored values.
        _require_finite_values(
            base_overrides, f"experiment {spec.id!r}: base_overrides"
        )

    try:
        merged_champion = merge_stock_screener_config(base_overrides, None)
        # Value smoke-check the base layer STRICTLY, mirroring the parse-time
        # check on spec-authored overrides: lax construction would silently
        # coerce `buy_threshold: true` → 1.0 or "0.35" → 0.35 in the champion
        # (and every challenger inheriting the field). A malformed base
        # config fails loudly here instead of skewing the A/B comparison.
        StockScreenerConfig.model_validate(merged_champion, strict=True)
        champion_cfg = StockScreenerConfig(**merged_champion)
    except ValidationError as exc:
        raise ExperimentSpecError(
            f"experiment {spec.id!r}: champion base overrides rejected by "
            f"StockScreenerConfig — {exc}"
        ) from exc
    challenger_cfgs: dict[str, StockScreenerConfig] = {}
    for challenger in spec.challengers:
        merged = merge_stock_screener_config(base_overrides, challenger.override)
        try:
            challenger_cfgs[challenger.id] = StockScreenerConfig(**merged)
        except ValidationError as exc:
            raise ExperimentSpecError(
                f"experiment {spec.id!r} challenger {challenger.id!r}: merged "
                f"config rejected by StockScreenerConfig — {exc}"
            ) from exc
    return champion_cfg, challenger_cfgs
