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
"""

from __future__ import annotations

from dataclasses import dataclass
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
            if not isinstance(value, dict):
                raise ExperimentSpecError(
                    f"{ctx}: pattern_weights override must be a mapping, "
                    f"got {type(value).__name__}"
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
    from pydantic import ValidationError

    from rainier.core.champion import merge_stock_screener_config

    try:
        StockScreenerConfig(**merge_stock_screener_config(None, out))
    except ValidationError as exc:
        raise ExperimentSpecError(
            f"{ctx}: override value(s) rejected by StockScreenerConfig — {exc}"
        ) from exc
    return out


def _require_str(raw: dict[str, Any], key: str, source: str) -> str:
    val = raw.get(key)
    if not isinstance(val, str) or not val:
        raise ExperimentSpecError(
            f"{source}: spec field {key!r} must be a non-empty str, got {val!r}"
        )
    return val


def parse_spec(raw: Any, source: str = "<memory>") -> ExperimentSpec:
    """Parse + validate one raw spec mapping into an :class:`ExperimentSpec`."""
    if not isinstance(raw, dict):
        raise ExperimentSpecError(
            f"{source}: spec must be a YAML mapping, got {type(raw).__name__}"
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

    guardrails_raw = raw.get("guardrails") or []
    if not isinstance(guardrails_raw, list) or not all(
        isinstance(g, str) for g in guardrails_raw
    ):
        raise ExperimentSpecError(
            f"{source}: experiment {spec_id!r}: guardrails must be a list of strings"
        )

    window_raw = raw.get("window")
    if not isinstance(window_raw, dict):
        raise ExperimentSpecError(
            f"{source}: experiment {spec_id!r}: window must be a mapping with "
            f"train/holdout (got {window_raw!r})"
        )
    unknown_window = sorted(set(window_raw) - _ALLOWED_WINDOW_KEYS)
    if unknown_window:
        raise ExperimentSpecError(
            f"{source}: experiment {spec_id!r}: unknown window key(s): "
            f"{', '.join(unknown_window)} (allowed: {sorted(_ALLOWED_WINDOW_KEYS)}; "
            f"a typo'd `embargo_dayz:` would otherwise silently keep the default)"
        )
    train = window_raw.get("train")
    holdout = window_raw.get("holdout")
    if not isinstance(train, str) or not isinstance(holdout, str):
        raise ExperimentSpecError(
            f"{source}: experiment {spec_id!r}: window.train and window.holdout "
            f"are required strings (start..end)"
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
    with p.open("r") as fh:
        try:
            raw = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            raise ExperimentSpecError(f"{p}: invalid YAML — {exc}") from exc
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
    """
    base = Path(directory) if directory is not None else DEFAULT_EXPERIMENTS_DIR
    active: list[ExperimentSpec] = []
    layer_owner: dict[str, str] = {}
    key_owner: dict[str, str] = {}
    seen_ids: dict[str, str] = {}
    for path in sorted(base.glob("*.yaml")):
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
    """
    from rainier.core.champion import merge_stock_screener_config
    from rainier.core.config import StockScreenerConfig

    champion_cfg = StockScreenerConfig(
        **merge_stock_screener_config(base_overrides, None)
    )
    challenger_cfgs: dict[str, StockScreenerConfig] = {}
    for challenger in spec.challengers:
        merged = merge_stock_screener_config(base_overrides, challenger.override)
        challenger_cfgs[challenger.id] = StockScreenerConfig(**merged)
    return champion_cfg, challenger_cfgs
