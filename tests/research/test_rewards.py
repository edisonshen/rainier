"""Tests for the generalized reward registry + screener basket rewards.

Design: docs/DESIGN-qu100-ab-testing.md §4.2, §10.3, §11.2
Task plan: docs/TASK-PLAN-ab-reward-registry-b3b8.md

Fixture baskets are constructed by hand (no evaluator dependency — that is
ab-replay-evaluator-edb5's scope) and every shipped reward is pinned to a
hand-computed expected value.
"""

from __future__ import annotations

import math
from datetime import date

import pytest

from rainier.research import rewards
from rainier.research.rewards import (
    REGISTRY,
    RewardSpec,
    get,
    guardrail_breached,
    make_ratio,
    names,
    register,
    score,
)
from rainier.research.rewards.basket import BasketDay, BasketOutcomes

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _registry_guard():
    """Snapshot/restore REGISTRY so test-registered rewards never leak."""
    snapshot = dict(REGISTRY)
    yield
    REGISTRY.clear()
    REGISTRY.update(snapshot)


@pytest.fixture
def basket() -> BasketOutcomes:
    """Hand-computed three-day fixture.

    Per-day equal-weight basket returns at H=5 (selected, non-None only):
      d1: (0.10 + 0.02) / 2 = 0.06
      d2: (0.04 - 0.12) / 2 = -0.04
      d3: BBB is None -> only DDD  = 0.06

    Hand-computed expectations:
      total_return = 0.06 - 0.04 + 0.06                  = 0.08
      sharpe       = mean/stdev(ddof=1) = 0.0266667/0.0577350 = 0.4618802
      hit_rate     = 2 positive days / 3                 = 2/3
      max_drawdown = 1.06 -> 1.0176 peak-to-trough       = 0.04
      turnover     = mean(1/2, 2/2)                      = 0.75
      dodged_loss  = |−0.08| (CCC d1) + |−0.05| (DDD d2) = 0.13
    """
    return BasketOutcomes(
        days=[
            BasketDay(
                date=date(2026, 1, 5),
                symbols=["AAA", "BBB"],
                fwd_return={5: {"AAA": 0.10, "BBB": 0.02, "CCC": -0.08}},
                regime="bull",
            ),
            BasketDay(
                date=date(2026, 1, 6),
                symbols=["AAA", "CCC"],
                fwd_return={5: {"AAA": 0.04, "CCC": -0.12, "DDD": -0.05}},
                regime="bull",
            ),
            BasketDay(
                date=date(2026, 1, 7),
                symbols=["BBB", "DDD"],
                fwd_return={5: {"BBB": None, "DDD": 0.06, "EEE": 0.03}},
                regime="bear",
            ),
        ]
    )


@pytest.fixture
def empty_basket() -> BasketOutcomes:
    return BasketOutcomes(days=[])


def _uniform_basket(day_returns: list[float], horizon: int = 5) -> BasketOutcomes:
    """One-symbol basket with the given per-day forward returns."""
    return BasketOutcomes(
        days=[
            BasketDay(
                date=date(2026, 1, 5 + i),
                symbols=["AAA"],
                fwd_return={horizon: {"AAA": r}},
                regime="bull",
            )
            for i, r in enumerate(day_returns)
        ]
    )


# ---------------------------------------------------------------------------
# Registration + metadata
# ---------------------------------------------------------------------------


class TestRegister:
    def test_register_new_basket_reward_selectable_with_metadata(self):
        @register("my_reward", input_type="basket", role="secondary", direction="increase")
        def my_reward(outcomes: BasketOutcomes) -> float:
            return 1.0

        assert "my_reward" in names()
        spec = get("my_reward")
        assert isinstance(spec, RewardSpec)
        assert spec.input_type == "basket"
        assert spec.role == "secondary"
        assert spec.direction == "increase"
        assert spec.threshold is None
        assert spec.fn is my_reward  # decorator returns the fn unwrapped

    def test_register_guardrail_carries_threshold(self):
        @register(
            "my_guardrail",
            input_type="basket",
            role="guardrail",
            direction="decrease",
            threshold=0.3,
        )
        def my_guardrail(outcomes: BasketOutcomes) -> float:
            return 0.0

        assert get("my_guardrail").threshold == 0.3

    def test_duplicate_name_rejected(self):
        @register("dup", input_type="basket", role="secondary", direction="increase")
        def first(outcomes):
            return 0.0

        with pytest.raises(ValueError, match="dup"):

            @register("dup", input_type="basket", role="secondary", direction="increase")
            def second(outcomes):
                return 0.0

    def test_duplicate_name_rejected_at_decoration_time(self):
        # Regression: two factories obtained BEFORE either decorator is
        # applied must not silently clobber — the dup check must also run
        # at insertion time, not only in the factory.
        deco_a = register("dup2", input_type="basket", role="secondary", direction="increase")
        deco_b = register("dup2", input_type="basket", role="secondary", direction="increase")

        def first(outcomes):
            return 1.0

        def second(outcomes):
            return 2.0

        deco_a(first)
        with pytest.raises(ValueError, match="dup2"):
            deco_b(second)
        assert get("dup2").fn is first  # first registration survives intact

    def test_invalid_input_type_rejected(self):
        with pytest.raises(ValueError, match="input_type"):
            register("bad", input_type="portfolio", role="primary", direction="increase")

    def test_invalid_role_rejected(self):
        with pytest.raises(ValueError, match="role"):
            register("bad", input_type="basket", role="tertiary", direction="increase")

    def test_invalid_direction_rejected(self):
        with pytest.raises(ValueError, match="direction"):
            register("bad", input_type="basket", role="primary", direction="up")

    def test_guardrail_without_threshold_rejected(self):
        with pytest.raises(ValueError, match="threshold"):
            register("bad", input_type="basket", role="guardrail", direction="decrease")

    def test_guardrail_non_finite_threshold_rejected(self):
        # Regression: a NaN threshold fails every >/< comparison, so the
        # guardrail would silently never breach.
        for bad in (math.nan, math.inf, -math.inf):
            with pytest.raises(ValueError, match="finite"):
                register(
                    "bad",
                    input_type="basket",
                    role="guardrail",
                    direction="decrease",
                    threshold=bad,
                )

    def test_non_guardrail_with_threshold_rejected(self):
        with pytest.raises(ValueError, match="threshold"):
            register(
                "bad", input_type="basket", role="primary", direction="increase", threshold=0.1
            )

    def test_get_unknown_name_raises_with_known_names(self):
        with pytest.raises(ValueError, match="no_such_reward"):
            get("no_such_reward")


# ---------------------------------------------------------------------------
# Input-type refusal (§4.2: no silent mis-scoring)
# ---------------------------------------------------------------------------


class TestInputTypeRefusal:
    def test_trades_reward_refuses_basket_candidate(self, basket):
        @register("trades_only", input_type="trades", role="primary", direction="increase")
        def trades_only(records) -> float:
            return 1.0

        with pytest.raises(ValueError, match="input_type"):
            score("basket", "trades_only", basket)

    def test_basket_reward_refuses_trades_candidate(self):
        with pytest.raises(ValueError, match="input_type"):
            score("trades", "sharpe", [])

    def test_matching_input_type_scores(self, basket):
        assert score("basket", "total_return", basket) == pytest.approx(0.08)

    def test_unknown_candidate_type_rejected(self, basket):
        # A typo'd candidate_type must be flagged as such, not blamed on
        # the reward's input_type.
        with pytest.raises(ValueError, match="candidate_type"):
            score("baskets", "total_return", basket)


# ---------------------------------------------------------------------------
# Shipped basket rewards — hand-computed fixtures
# ---------------------------------------------------------------------------


class TestShippedRewards:
    def test_shipped_set_registered(self):
        expected = {
            "total_return",
            "sharpe",
            "hit_rate",
            "max_drawdown",
            "turnover",
            "dodged_loss",
        }
        assert expected <= set(names())
        for name in expected:
            assert get(name).input_type == "basket"

    def test_roles_and_directions(self):
        assert get("sharpe").role == "primary"
        assert get("max_drawdown").role == "guardrail"
        assert get("max_drawdown").direction == "decrease"
        assert get("max_drawdown").threshold == 0.20  # design-pinned promote-gate bound
        assert get("turnover").role == "guardrail"
        assert get("turnover").direction == "decrease"
        assert get("turnover").threshold == 0.50  # design-pinned promote-gate bound
        assert get("total_return").direction == "increase"
        assert get("dodged_loss").direction == "increase"

    def test_total_return(self, basket):
        assert score("basket", "total_return", basket) == pytest.approx(0.08)

    def test_sharpe(self, basket):
        assert score("basket", "sharpe", basket) == pytest.approx(0.4618802, abs=1e-6)

    def test_hit_rate(self, basket):
        assert score("basket", "hit_rate", basket) == pytest.approx(2 / 3)

    def test_max_drawdown(self, basket):
        assert score("basket", "max_drawdown", basket) == pytest.approx(0.04)

    def test_turnover(self, basket):
        assert score("basket", "turnover", basket) == pytest.approx(0.75)

    def test_dodged_loss(self, basket):
        assert score("basket", "dodged_loss", basket) == pytest.approx(0.13)

    def test_empty_basket_all_rewards_zero(self, empty_basket):
        for name in ("total_return", "sharpe", "hit_rate", "max_drawdown", "turnover"):
            assert score("basket", name, empty_basket) == 0.0
        assert score("basket", "dodged_loss", empty_basket) == 0.0

    def test_sharpe_single_day_zero(self):
        outcomes = _uniform_basket([0.10])
        assert score("basket", "sharpe", outcomes) == 0.0

    def test_sharpe_zero_stdev_zero(self):
        # Identical day returns -> stdev 0 -> 0.0, not ZeroDivisionError.
        outcomes = _uniform_basket([0.05, 0.05, 0.05])
        assert score("basket", "sharpe", outcomes) == 0.0

    def test_turnover_skips_empty_symbol_days(self):
        # [A,B], [], [A,B]: the empty day is dropped, so day1->day3 is the
        # consecutive pair — nothing entered — turnover 0.0 (pinned skip
        # semantics; the empty day never enters the denominator).
        outcomes = BasketOutcomes(
            days=[
                BasketDay(
                    date=date(2026, 1, 5),
                    symbols=["AAA", "BBB"],
                    fwd_return={5: {"AAA": 0.01, "BBB": 0.01}},
                    regime="bull",
                ),
                BasketDay(date=date(2026, 1, 6), symbols=[], fwd_return={5: {}}, regime="bull"),
                BasketDay(
                    date=date(2026, 1, 7),
                    symbols=["AAA", "BBB"],
                    fwd_return={5: {"AAA": 0.01, "BBB": 0.01}},
                    regime="bull",
                ),
            ]
        )
        assert score("basket", "turnover", outcomes) == 0.0


# ---------------------------------------------------------------------------
# None handling (§10.3: None near window end, never 0)
# ---------------------------------------------------------------------------


class TestNoneHandling:
    def test_none_excluded_not_zeroed(self):
        # Two days: d1 all-None (excluded entirely), d2 = 0.10.
        # If None were treated as 0, sharpe's mean/stdev and hit_rate's
        # denominator would both change.
        outcomes = BasketOutcomes(
            days=[
                BasketDay(
                    date=date(2026, 1, 5),
                    symbols=["AAA"],
                    fwd_return={5: {"AAA": None}},
                    regime="bull",
                ),
                BasketDay(
                    date=date(2026, 1, 6),
                    symbols=["AAA"],
                    fwd_return={5: {"AAA": 0.10}},
                    regime="bull",
                ),
            ]
        )
        assert score("basket", "total_return", outcomes) == pytest.approx(0.10)
        assert score("basket", "hit_rate", outcomes) == pytest.approx(1.0)  # 1/1, not 1/2

    def test_partial_none_within_day_excluded(self, basket):
        # d3 has BBB=None; the day return must be 0.06 (DDD only), not 0.03.
        # total_return pins this: 0.06 - 0.04 + 0.06 = 0.08.
        assert score("basket", "total_return", basket) == pytest.approx(0.08)


# ---------------------------------------------------------------------------
# Non-finite input rejection (§4.2: loud, never silent)
# ---------------------------------------------------------------------------


class TestNonFiniteRejection:
    def test_nan_return_raises_not_disarms_drawdown(self):
        # Regression: NaN on day 1 then two -30% days used to yield
        # max_drawdown == 0.0 (equity *= nan; max(worst, nan) keeps worst)
        # — a clean-looking guardrail on corrupt data.
        outcomes = _uniform_basket([math.nan, -0.30, -0.30])
        with pytest.raises(ValueError, match="non-finite"):
            score("basket", "max_drawdown", outcomes)

    def test_nan_return_raises_for_aggregations(self):
        outcomes = _uniform_basket([0.10, math.nan])
        for name in ("total_return", "sharpe", "hit_rate"):
            with pytest.raises(ValueError, match="non-finite"):
                score("basket", name, outcomes)

    def test_inf_return_raises(self):
        outcomes = _uniform_basket([math.inf])
        with pytest.raises(ValueError, match="non-finite"):
            score("basket", "total_return", outcomes)

    def test_nan_in_dodged_loss_pool_raises(self):
        outcomes = BasketOutcomes(
            days=[
                BasketDay(
                    date=date(2026, 1, 5),
                    symbols=["AAA"],
                    fwd_return={5: {"AAA": 0.01, "CCC": math.nan}},
                    regime="bull",
                ),
            ]
        )
        with pytest.raises(ValueError, match="non-finite"):
            score("basket", "dodged_loss", outcomes)


# ---------------------------------------------------------------------------
# Horizon handling (kwargs forwarding + missing-horizon loudness)
# ---------------------------------------------------------------------------


class TestHorizon:
    def test_kwargs_horizon_forwarded(self):
        # Two horizons with different returns: score() must forward horizon=
        # to the reward and aggregate the requested one.
        outcomes = BasketOutcomes(
            days=[
                BasketDay(
                    date=date(2026, 1, 5),
                    symbols=["AAA"],
                    fwd_return={5: {"AAA": 0.10}, 10: {"AAA": 0.25}},
                    regime="bull",
                ),
            ]
        )
        assert score("basket", "total_return", outcomes) == pytest.approx(0.10)
        assert score("basket", "total_return", outcomes, horizon=10) == pytest.approx(0.25)

    def test_horizon_absent_from_every_day_raises(self, basket):
        # A typo'd/unconfigured horizon must not silently score 0.0 for
        # every candidate (valid-looking noise).
        with pytest.raises(ValueError, match="horizon 10"):
            score("basket", "total_return", basket, horizon=10)
        with pytest.raises(ValueError, match="horizon 10"):
            score("basket", "dodged_loss", basket, horizon=10)

    def test_horizon_missing_on_some_days_skips_them(self):
        # Partially missing horizon: days without the key are skipped
        # (like None returns), not fatal.
        outcomes = BasketOutcomes(
            days=[
                BasketDay(
                    date=date(2026, 1, 5),
                    symbols=["AAA"],
                    fwd_return={5: {"AAA": 0.10}},
                    regime="bull",
                ),
                BasketDay(
                    date=date(2026, 1, 6),
                    symbols=["AAA"],
                    fwd_return={5: {"AAA": 0.02}, 10: {"AAA": 0.30}},
                    regime="bull",
                ),
            ]
        )
        assert score("basket", "total_return", outcomes, horizon=10) == pytest.approx(0.30)

    def test_empty_basket_any_horizon_zero(self, empty_basket):
        # No days at all -> nothing to validate against; stays 0.0.
        assert score("basket", "total_return", empty_basket, horizon=10) == 0.0


# ---------------------------------------------------------------------------
# Guardrail evaluation helper
# ---------------------------------------------------------------------------


class TestGuardrail:
    def test_decrease_direction_breach_when_above_threshold(self):
        @register(
            "gd_dec", input_type="basket", role="guardrail", direction="decrease", threshold=0.20
        )
        def gd_dec(outcomes):
            return 0.0

        assert guardrail_breached("gd_dec", 0.25) is True
        assert guardrail_breached("gd_dec", 0.04) is False
        assert guardrail_breached("gd_dec", 0.20) is False  # at threshold = clean

    def test_increase_direction_breach_when_below_threshold(self):
        @register(
            "gd_inc", input_type="basket", role="guardrail", direction="increase", threshold=0.5
        )
        def gd_inc(outcomes):
            return 0.0

        assert guardrail_breached("gd_inc", 0.4) is True
        assert guardrail_breached("gd_inc", 0.6) is False
        assert guardrail_breached("gd_inc", 0.5) is False  # at threshold = clean

    def test_non_guardrail_rejected(self):
        with pytest.raises(ValueError, match="guardrail"):
            guardrail_breached("sharpe", 1.0)

    def test_non_finite_value_rejected(self):
        # Regression: NaN fails every >/< comparison, so it used to read as
        # "clean" — silently disarming the guardrail on corrupt input.
        with pytest.raises(ValueError, match="non-finite"):
            guardrail_breached("max_drawdown", math.nan)
        with pytest.raises(ValueError, match="non-finite"):
            guardrail_breached("max_drawdown", math.inf)


# ---------------------------------------------------------------------------
# Ratio composition (§4.2: ratio rewards compose two aggregations)
# ---------------------------------------------------------------------------


class TestRatio:
    def test_ratio_composes_two_registered_rewards(self, basket):
        fn = make_ratio("total_return", "max_drawdown")
        assert fn(basket) == pytest.approx(0.08 / 0.04)

    def test_ratio_zero_over_zero_is_zero(self, empty_basket):
        # Empty window: numerator and denominator both 0 -> 0.0 (no data).
        fn = make_ratio("total_return", "max_drawdown")
        assert fn(empty_basket) == 0.0

    def test_ratio_positive_over_zero_is_inf(self):
        # Regression: a flawless candidate (positive return, zero drawdown)
        # used to score 0.0 — ranking BELOW every mediocre candidate.
        fn = make_ratio("total_return", "max_drawdown")
        outcomes = _uniform_basket([0.05, 0.10])
        assert fn(outcomes) == math.inf

    def test_ratio_negative_over_zero_is_neg_inf(self):
        @register("neg_const", input_type="basket", role="secondary", direction="increase")
        def neg_const(outcomes, **kwargs):
            return -1.0

        @register("zero_const", input_type="basket", role="secondary", direction="increase")
        def zero_const(outcomes, **kwargs):
            return 0.0

        fn = make_ratio("neg_const", "zero_const")
        assert fn(BasketOutcomes(days=[])) == -math.inf

    def test_ratio_closure_has_descriptive_name(self):
        fn = make_ratio("total_return", "max_drawdown")
        assert fn.__name__ == "ratio_total_return_over_max_drawdown"

    def test_ratio_requires_matching_input_types(self):
        @register("trades_num", input_type="trades", role="secondary", direction="increase")
        def trades_num(records):
            return 1.0

        with pytest.raises(ValueError, match="input_type"):
            make_ratio("trades_num", "max_drawdown")

    def test_ratio_result_registrable(self, basket):
        register(
            "return_over_drawdown",
            input_type="basket",
            role="secondary",
            direction="increase",
        )(make_ratio("total_return", "max_drawdown"))
        assert score("basket", "return_over_drawdown", basket) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Determinism + purity
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_outcomes_identical_value_across_calls(self, basket):
        for name in ("total_return", "sharpe", "hit_rate", "max_drawdown", "turnover",
                     "dodged_loss"):
            assert score("basket", name, basket) == score("basket", name, basket)

    def test_scoring_does_not_mutate_outcomes(self, basket):
        before = [(d.date, list(d.symbols), {h: dict(m) for h, m in d.fwd_return.items()})
                  for d in basket.days]
        score("basket", "sharpe", basket)
        score("basket", "turnover", basket)
        after = [(d.date, list(d.symbols), {h: dict(m) for h, m in d.fwd_return.items()})
                 for d in basket.days]
        assert before == after


# ---------------------------------------------------------------------------
# BasketOutcomes shape (design §10.3, pinned)
# ---------------------------------------------------------------------------


class TestBasketOutcomesShape:
    def test_frozen(self, basket):
        with pytest.raises(AttributeError):
            basket.days = []
        with pytest.raises(AttributeError):
            basket.days[0].regime = "bear"


class TestModuleSurface:
    def test_names_sorted(self):
        assert names() == sorted(names())

    def test_registry_module_reexports(self):
        # The package surface other tasks import against.
        assert rewards.REGISTRY is REGISTRY
        assert callable(rewards.register)
        assert callable(rewards.score)
