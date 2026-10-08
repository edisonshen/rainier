"""Validator + formatter for the D-031 stdout summary blocks.

Subagents grep one line out of stdout; humans eyeball the block; CI
asserts every summary parses against this module. The schema YAML lives
at ``src/rainier/research/output_schema.yaml`` — extend that file when a
new verb adds a block (not by inventing fields ad-hoc in CLI code).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SCHEMA_PATH = Path(__file__).parent / "output_schema.yaml"


class SchemaError(ValueError):
    """Raised when a summary block fails validation or parse."""


_TYPE_TO_PY = {
    "int": int,
    "float": (int, float),  # YAML ints are valid floats; permissive.
    "str": str,
    "bool": bool,
    "list": list,
    "dict": dict,
}


@dataclass(frozen=True)
class _FieldSpec:
    name: str
    py_type: tuple[type, ...] | type
    enum: tuple[str, ...] | None
    nullable: bool


def load(path: str | Path | None = None) -> dict[str, Any]:
    """Read the schema YAML; returned dict is the raw shape (schemas: {...})."""
    p = Path(path) if path is not None else SCHEMA_PATH
    try:
        with p.open("r") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        # Consumers catch SchemaError (the module contract) — a raw
        # ParserError from a broken schema file must not bypass them.
        raise SchemaError(f"{p}: invalid YAML — {exc}") from exc
    if not isinstance(data, dict) or "schemas" not in data:
        raise SchemaError(f"{p} has no top-level `schemas:` key")
    return data


def _coerce_fields(schema: dict[str, Any]) -> list[_FieldSpec]:
    out: list[_FieldSpec] = []
    raw = schema.get("fields") or {}
    for name, body in raw.items():
        t = body.get("type")
        if t not in _TYPE_TO_PY:
            raise SchemaError(f"field {name!r} has unknown type {t!r}")
        py_type = _TYPE_TO_PY[t]
        enum = body.get("enum")
        out.append(
            _FieldSpec(
                name=name,
                py_type=py_type,
                enum=tuple(enum) if enum else None,
                nullable=bool(body.get("nullable", False)),
            )
        )
    return out


def _check_payload(specs: list[_FieldSpec], payload: dict[str, Any], strict: bool) -> None:
    """Shared field checks: presence, strict extras, types, enums."""
    declared = {spec.name for spec in specs}

    missing = declared - payload.keys()
    if missing:
        raise SchemaError(f"missing required field(s): {sorted(missing)}")

    if strict:
        extra = payload.keys() - declared
        if extra:
            raise SchemaError(f"extra field(s) not in schema: {sorted(extra)}")

    for spec in specs:
        val = payload[spec.name]
        if val is None and spec.nullable:
            continue
        allowed = spec.py_type if isinstance(spec.py_type, tuple) else (spec.py_type,)
        if isinstance(val, bool) and bool not in allowed:
            # bool is a subclass of int: `n_selection_days: true` would
            # otherwise pass the int/float check and land as 1/1.0 in
            # promotion artifacts — silent type corruption.
            raise SchemaError(
                f"field {spec.name!r}: expected {spec.py_type}, got bool ({val!r})"
            )
        if not isinstance(val, spec.py_type):  # type: ignore[arg-type]
            raise SchemaError(
                f"field {spec.name!r}: expected {spec.py_type}, got "
                f"{type(val).__name__} ({val!r})"
            )
        if spec.enum is not None and val not in spec.enum:
            raise SchemaError(
                f"field {spec.name!r}: value {val!r} not in enum {spec.enum}"
            )


def validate(name: str, payload: dict[str, Any], strict: bool = True) -> None:
    """Validate ``payload`` against the schema named ``name``.

    Strict mode (default) rejects any extra field not declared in the schema.
    Non-strict mode tolerates forward-compatible extras — used at runtime so
    a new field added in a future commit doesn't blow up an older CLI.
    """
    data = load()
    schemas = data["schemas"]
    if name not in schemas:
        raise SchemaError(f"unknown schema {name!r}; known: {sorted(schemas)}")
    _check_payload(_coerce_fields(schemas[name]), payload, strict)


def validate_composed(
    base: str,
    payload: dict[str, Any],
    extensions: Sequence[str] = (),
    strict: bool = True,
) -> None:
    """Validate ``payload`` against a base schema plus optional extensions.

    A/B-substrate scorecards (design §10.5) compose a candidate-agnostic
    ``base_scorecard`` with optional per-candidate-type extensions (e.g.
    ``llm_extension``). The composed field set is the union of the named
    schemas: every field of the base AND every named extension is required;
    extras beyond the union fail strict mode. An extension payload alone
    fails on the missing base fields — the extension is never standalone.
    """
    if isinstance(extensions, str):
        # str is itself a Sequence[str]; a bare 'llm_extension' would iterate
        # per-character and fail with a baffling "unknown schema 'l'".
        raise SchemaError(
            f"extensions must be a sequence of schema names, "
            f"got bare string {extensions!r} — wrap it in a list"
        )
    data = load()
    schemas = data["schemas"]
    specs: list[_FieldSpec] = []
    seen: set[str] = set()
    for name in (base, *extensions):
        if name not in schemas:
            raise SchemaError(f"unknown schema {name!r}; known: {sorted(schemas)}")
        for spec in _coerce_fields(schemas[name]):
            if spec.name in seen:
                raise SchemaError(
                    f"field {spec.name!r} declared by more than one of "
                    f"{[base, *extensions]}"
                )
            seen.add(spec.name)
            specs.append(spec)
    _check_payload(specs, payload, strict)


def format_block(name: str, payload: dict[str, Any]) -> str:
    """Render ``payload`` as a fenced YAML block matching schema ``name``.

    The output is exactly the D-031 wire shape:
        ---
        key: value
        ...
        ---

    Field order follows the schema declaration order so byte-identical
    inputs produce byte-identical output (subagent grep stability).
    """
    validate(name, payload, strict=True)
    data = load()
    specs = _coerce_fields(data["schemas"][name])
    lines = ["---"]
    for spec in specs:
        val = payload[spec.name]
        # Use yaml.safe_dump for the value so floats / lists round-trip.
        dumped = yaml.safe_dump(val, default_flow_style=True).strip()
        # Strip the trailing `\n...\n` document terminator yaml adds.
        if dumped.endswith("\n..."):
            dumped = dumped[: -len("\n...")]
        lines.append(f"{spec.name}: {dumped}")
    lines.append("---")
    lines.append("")  # trailing newline
    return "\n".join(lines)


def format_block_composed(
    base: str,
    payload: dict[str, Any],
    extensions: Sequence[str] = (),
) -> str:
    """Render a composed (base + extensions) payload as a fenced YAML block.

    Mirrors :func:`format_block` for the composed-scorecard case (§10.5) —
    without this, an evaluator emitting ``base_scorecard + llm_extension``
    would have no serializer (plain ``format_block`` rejects the extension
    fields as extras). Field order is base-schema declaration order, then
    each extension's declaration order, so byte-identical inputs produce
    byte-identical output. ``parse_block`` is the shared inverse.
    """
    validate_composed(base, payload, extensions=extensions, strict=True)
    data = load()
    lines = ["---"]
    for name in (base, *extensions):
        for spec in _coerce_fields(data["schemas"][name]):
            val = payload[spec.name]
            dumped = yaml.safe_dump(val, default_flow_style=True).strip()
            if dumped.endswith("\n..."):
                dumped = dumped[: -len("\n...")]
            lines.append(f"{spec.name}: {dumped}")
    lines.append("---")
    lines.append("")  # trailing newline
    return "\n".join(lines)


def parse_block(text: str) -> dict[str, Any]:
    """Inverse of ``format_block`` — extract a single fenced YAML block.

    Rejects payloads without the `---` fences (per D-031). Whitespace
    before/after the fences is tolerated.
    """
    stripped = text.strip()
    if not (stripped.startswith("---") and stripped.endswith("---")):
        raise SchemaError("block is not `---` fenced")
    # Drop the fences.
    inner = stripped[3:-3].strip()
    try:
        parsed = yaml.safe_load(inner)
    except yaml.YAMLError as exc:
        # A truncated block (process killed mid-write) must surface as the
        # module contract exception, not a raw yaml ParserError.
        raise SchemaError(f"block is not valid YAML — {exc}") from exc
    if not isinstance(parsed, dict):
        raise SchemaError(f"block must parse to a dict; got {type(parsed).__name__}")
    return parsed
