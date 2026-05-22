"""Output-schema validator tests — D-031 stdout summary-block contract.

The schema lives at src/rainier/research/output_schema.yaml. Every
`llm-research <verb>` subcommand that prints a summary block must
produce a fenced (`---` … `---`) YAML payload that round-trips through
the schema validator. Tests here pin the validator surface — they
don't care about CLI plumbing, just the loader / validator / strict-mode.
"""

from __future__ import annotations

import pytest

from rainier.research import output_schema


def _valid_cost_pilot_block() -> dict:
    """A canonical example of the cost-pilot summary block."""
    return {
        "slice": 0,
        "skill_id": "cost_pilot_v0",
        "calls_total": 500,
        "calls_completed": 500,
        "mean_cost_usd": 0.0123,
        "p95_cost_usd": 0.0456,
        "p99_cost_usd": 0.0789,
        "schema_retry_rate": 0.02,
        "cache_hit_rate": 0.18,
        "total_spend_usd": 6.15,
        "verdict": "PASS",
        "report_html": "docs/slice0-cost-report.html",
        "ledger_sha": "0" * 40,
    }


def test_loader_returns_dict_with_schemas_key():
    loaded = output_schema.load()
    assert "schemas" in loaded
    assert "cost_pilot" in loaded["schemas"]


def test_validate_accepts_canonical_block():
    output_schema.validate("cost_pilot", _valid_cost_pilot_block())


def test_validate_rejects_missing_field():
    block = _valid_cost_pilot_block()
    del block["verdict"]
    with pytest.raises(output_schema.SchemaError) as exc:
        output_schema.validate("cost_pilot", block)
    assert "verdict" in str(exc.value)


def test_validate_rejects_extra_field_strict_mode():
    block = _valid_cost_pilot_block()
    block["never_declared_field"] = "boom"
    with pytest.raises(output_schema.SchemaError) as exc:
        output_schema.validate("cost_pilot", block, strict=True)
    assert "never_declared_field" in str(exc.value)


def test_validate_accepts_extra_field_non_strict_mode():
    block = _valid_cost_pilot_block()
    block["forward_compat"] = "ok"
    # Non-strict mode tolerates extra fields (forward-compat shim).
    output_schema.validate("cost_pilot", block, strict=False)


def test_format_block_round_trips():
    """`format_block(name, payload)` → str → parse_block → payload."""
    block = _valid_cost_pilot_block()
    text = output_schema.format_block("cost_pilot", block)
    assert text.startswith("---\n")
    assert text.endswith("---\n")
    parsed = output_schema.parse_block(text)
    # YAML floats round-trip safely; ints round-trip safely.
    assert parsed == block


def test_parse_block_rejects_unfenced_payload():
    with pytest.raises(output_schema.SchemaError):
        output_schema.parse_block("just: a flat yaml document\n")


def test_verdict_enum_is_pinned():
    schema = output_schema.load()["schemas"]["cost_pilot"]
    # PASS / CONDITIONAL / FAIL are the three values Slice 0 ships with.
    assert set(schema["fields"]["verdict"]["enum"]) == {"PASS", "CONDITIONAL", "FAIL"}
