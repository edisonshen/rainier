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


# ---------------------------------------------------------------------------
# A/B substrate scorecards (design §10.5) — candidate-agnostic base +
# optional LLM extension, composed via validate_composed(). Additive: the
# LLM-shaped `backtest` entry stays untouched for its existing consumer.
# ---------------------------------------------------------------------------


def _valid_base_scorecard() -> dict:
    return {
        "candidate_id": "mf35",
        "candidate_type": "screener",
        "window": "2025-05-27..2026-03-31",
        "n_selection_days": 210,
        "corpus_hash": "a" * 40,
        "rewards": {
            "primary": {"sharpe": 1.12},
            "guardrails": {"max_drawdown": -0.14, "turnover": 0.32},
        },
        "regime_scores": {"risk_on": {"sharpe": 1.30}, "risk_off": {"sharpe": 0.41}},
        # null until the §11-task-6 DSR spec lands — never a naive number.
        "deflated_sharpe": None,
        "evaluator_sha": "b" * 40,
    }


def _valid_llm_extension() -> dict:
    return {
        "valid_thesis_rate": 0.92,
        "cost_usd": 4.31,
        "filled_R": 3.4,
        "filled_rate": 0.71,
        "tqqq_bh_R": 2.1,
        "skill_yaml_sha": "c" * 40,
    }


def test_base_scorecard_validates_without_llm_fields():
    output_schema.validate("base_scorecard", _valid_base_scorecard())


def test_base_scorecard_deflated_sharpe_null_validates():
    block = _valid_base_scorecard()
    assert block["deflated_sharpe"] is None
    output_schema.validate("base_scorecard", block)


def test_base_scorecard_deflated_sharpe_float_also_validates():
    block = _valid_base_scorecard()
    block["deflated_sharpe"] = 0.87
    output_schema.validate("base_scorecard", block)


def test_composed_base_plus_llm_extension_validates():
    block = {**_valid_base_scorecard(), **_valid_llm_extension()}
    output_schema.validate_composed("base_scorecard", block, extensions=["llm_extension"])


def test_llm_extension_alone_is_rejected():
    # The extension is not a standalone scorecard — base fields are required.
    with pytest.raises(output_schema.SchemaError) as exc:
        output_schema.validate_composed(
            "base_scorecard", _valid_llm_extension(), extensions=["llm_extension"]
        )
    assert "candidate_id" in str(exc.value)


def test_composed_rejects_extension_fields_when_extension_not_named():
    block = {**_valid_base_scorecard(), **_valid_llm_extension()}
    with pytest.raises(output_schema.SchemaError):
        output_schema.validate_composed("base_scorecard", block)


def test_composed_rejects_undeclared_extra_field():
    block = {**_valid_base_scorecard(), **_valid_llm_extension()}
    block["never_declared"] = 1
    with pytest.raises(output_schema.SchemaError) as exc:
        output_schema.validate_composed("base_scorecard", block, extensions=["llm_extension"])
    assert "never_declared" in str(exc.value)


def test_composed_unknown_extension_name_rejected():
    with pytest.raises(output_schema.SchemaError):
        output_schema.validate_composed(
            "base_scorecard", _valid_base_scorecard(), extensions=["no_such_extension"]
        )


def test_existing_schemas_unchanged_regression():
    """The pre-existing entries still validate their canonical payloads."""
    output_schema.validate("cost_pilot", _valid_cost_pilot_block())
    output_schema.validate(
        "survivorship",
        {
            "from_date": "2025-05-27",
            "to_date": "2026-06-25",
            "verdict": "PASS",
            "delisted_tickers": [],
            "missing_days": [],
            "next_step": None,
        },
    )
    output_schema.validate(
        "backtest",
        {
            "skill_id": "qu100_v1",
            "reward_fn": "filled_R",
            "filled_R": 3.4,
            "all_call_R": 2.9,
            "calls": 120,
            "valid_thesis_rate": 0.92,
            "filled_rate": 0.71,
            "cost_usd": 4.31,
            "tqqq_bh_R": 2.1,
            "delta_vs_tqqq": 1.3,
            "deflated_sharpe": 0.87,
            "evaluator_sha": "b" * 40,
            "skill_yaml_sha": "c" * 40,
            "ledger_sha": "0" * 40,
        },
    )


def test_composed_duplicate_field_across_schemas_rejected():
    # base_scorecard and the pre-existing backtest schema both declare
    # deflated_sharpe/evaluator_sha — composing them must fail loudly.
    with pytest.raises(output_schema.SchemaError, match="declared by more than one"):
        output_schema.validate_composed("base_scorecard", {}, extensions=["backtest"])


def test_composed_non_strict_tolerates_extras():
    block = {**_valid_base_scorecard(), **_valid_llm_extension()}
    block["future_field"] = 1
    output_schema.validate_composed(
        "base_scorecard", block, extensions=["llm_extension"], strict=False
    )


def test_composed_bare_string_extensions_rejected():
    # str is a Sequence[str]; without the guard this iterates per-character.
    with pytest.raises(output_schema.SchemaError, match="bare string"):
        output_schema.validate_composed(
            "base_scorecard", _valid_base_scorecard(), extensions="llm_extension"
        )


def test_bool_rejected_for_int_field():
    # bool is a subclass of int: `n_selection_days: true` would otherwise
    # pass the int check and land as 1 in promotion artifacts.
    block = _valid_base_scorecard()
    block["n_selection_days"] = True
    with pytest.raises(output_schema.SchemaError, match="n_selection_days"):
        output_schema.validate("base_scorecard", block)


def test_bool_rejected_for_float_field():
    block = _valid_base_scorecard()
    block["deflated_sharpe"] = True
    with pytest.raises(output_schema.SchemaError, match="deflated_sharpe"):
        output_schema.validate("base_scorecard", block)


def test_base_alone_rejected_when_extension_named():
    # Converse of the extension-alone case: naming an extension makes ALL of
    # its fields required, so a base-only payload must fail on them.
    with pytest.raises(output_schema.SchemaError, match="valid_thesis_rate"):
        output_schema.validate_composed(
            "base_scorecard", _valid_base_scorecard(), extensions=["llm_extension"]
        )


def test_parse_block_wraps_invalid_yaml_as_schema_error():
    # A truncated block (process killed mid-write) must surface as the module
    # contract exception, not a raw yaml.parser.ParserError.
    with pytest.raises(output_schema.SchemaError, match="not valid YAML"):
        output_schema.parse_block("---\nkey: [unclosed\n---")


def test_load_wraps_invalid_yaml_as_schema_error(tmp_path):
    broken = tmp_path / "broken_schema.yaml"
    broken.write_text("schemas: [unclosed\n")
    with pytest.raises(output_schema.SchemaError, match="invalid YAML"):
        output_schema.load(broken)
