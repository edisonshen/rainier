"""Tests for the analysis orchestrator: analyze, analyze_multi_tf, merge helpers."""

from rainier.analysis.analyzer import (
    _dedup_levels,
    _merge_multi_tf_levels,
    _merge_sr_levels,
    analyze,
    analyze_multi_tf,
)
from rainier.core.types import SRLevel, SRRole, SRType, Timeframe


def _level(price, strength=0.5, touches=1, source_tf=None, sr_type=SRType.HORIZONTAL):
    return SRLevel(
        price=price, sr_type=sr_type, role=SRRole.SUPPORT,
        strength=strength, touches=touches, source_tf=source_tf,
    )


class TestAnalyze:
    def test_short_dataframe_returns_empty_result(self, flat_candles):
        result = analyze(flat_candles.iloc[:1], "NQ", Timeframe.H1)
        assert result.symbol == "NQ"
        assert result.timeframe == Timeframe.H1
        assert result.sr_levels == []
        assert result.pin_bars == []
        assert result.pivots == []
        assert result.bias is None

    def test_populates_result_fields(self, pin_bar_candles):
        result = analyze(pin_bar_candles, "NQ", Timeframe.H1, min_touches=1)
        assert result.symbol == "NQ"
        assert result.timeframe == Timeframe.H1
        assert isinstance(result.sr_levels, list)
        assert isinstance(result.pin_bars, list)
        assert isinstance(result.inside_bars, list)

    def test_skip_own_levels_uses_only_higher_tf(self, pin_bar_candles):
        htf = _level(100.0, strength=0.9, touches=5, source_tf=Timeframe.D1)
        result = analyze(
            pin_bar_candles, "NQ", Timeframe.M5,
            higher_tf_levels=[htf], min_touches=1, skip_own_levels=True,
        )
        assert result.sr_levels == [htf]

    def test_levels_outside_price_range_are_clipped(self, pin_bar_candles):
        # Price range of fixture is ~99.5-107.5; a level at 200 is beyond +10% margin
        far = _level(200.0, strength=0.9, touches=5, source_tf=Timeframe.D1)
        result = analyze(
            pin_bar_candles, "NQ", Timeframe.M5,
            higher_tf_levels=[far], min_touches=1, skip_own_levels=True,
        )
        assert result.sr_levels == []

    def test_matched_pin_bars_reference_final_levels(self, pin_bar_candles):
        htf = _level(99.5, strength=0.9, touches=5, source_tf=Timeframe.D1)
        result = analyze(
            pin_bar_candles, "NQ", Timeframe.M5,
            higher_tf_levels=[htf], min_touches=1, skip_own_levels=True,
        )
        assert len(result.pin_bars) >= 1
        for pb in result.pin_bars:
            assert pb.nearest_sr is htf


class TestAnalyzeMultiTf:
    def test_result_is_for_trading_timeframe(self, pin_bar_candles, flat_candles):
        data = {Timeframe.H1: pin_bar_candles, Timeframe.M5: flat_candles}
        result = analyze_multi_tf(data, "NQ", Timeframe.M5, min_touches=1)
        assert result.timeframe == Timeframe.M5
        assert result.symbol == "NQ"

    def test_higher_tf_levels_are_tagged_with_source_tf(self, pin_bar_candles):
        # Trading TF shares the same data so higher-TF levels stay in price range
        data = {Timeframe.H1: pin_bar_candles, Timeframe.M5: pin_bar_candles}
        result = analyze_multi_tf(data, "NQ", Timeframe.M5, min_touches=1)
        horizontal = [l for l in result.sr_levels if l.sr_type == SRType.HORIZONTAL]
        assert len(horizontal) >= 1
        for level in horizontal:
            assert level.source_tf == Timeframe.H1

    def test_only_trading_tf_in_data(self, pin_bar_candles):
        data = {Timeframe.M5: pin_bar_candles}
        result = analyze_multi_tf(data, "NQ", Timeframe.M5, min_touches=1)
        # No higher TFs → no levels at all (own levels skipped)
        assert result.sr_levels == []

    def test_higher_tf_with_too_few_bars_is_skipped(self, pin_bar_candles):
        data = {
            Timeframe.D1: pin_bar_candles.iloc[:1],
            Timeframe.M5: pin_bar_candles,
        }
        result = analyze_multi_tf(data, "NQ", Timeframe.M5, min_touches=1)
        assert result.sr_levels == []


class TestMergeMultiTfLevels:
    TF_ORDER = [Timeframe.W1, Timeframe.D1, Timeframe.H4, Timeframe.H1,
                Timeframe.M30, Timeframe.M15, Timeframe.M5, Timeframe.M1]

    def test_lower_tf_absorbed_into_higher_tf(self):
        d1 = _level(100.0, strength=0.5, touches=2, source_tf=Timeframe.D1)
        h1 = _level(100.3, strength=0.9, touches=3, source_tf=Timeframe.H1)
        merged = _merge_multi_tf_levels([h1, d1], self.TF_ORDER, merge_dist=0.5)
        assert len(merged) == 1
        keep = merged[0]
        assert keep.source_tf == Timeframe.D1  # higher TF label kept
        assert keep.price == 100.3  # refined to lower-TF price
        assert keep.strength == 0.6  # 0.5 + 0.1 confluence boost
        assert keep.touches == 5  # 2 + 3

    def test_strength_boost_capped_at_1(self):
        d1 = _level(100.0, strength=0.95, touches=1, source_tf=Timeframe.D1)
        h1 = _level(100.1, strength=0.5, touches=1, source_tf=Timeframe.H1)
        merged = _merge_multi_tf_levels([d1, h1], self.TF_ORDER, merge_dist=0.5)
        assert len(merged) == 1
        assert merged[0].strength == 1.0

    def test_distant_levels_not_merged(self):
        d1 = _level(100.0, strength=0.5, touches=2, source_tf=Timeframe.D1)
        h1 = _level(105.0, strength=0.9, touches=3, source_tf=Timeframe.H1)
        merged = _merge_multi_tf_levels([d1, h1], self.TF_ORDER, merge_dist=0.5)
        assert len(merged) == 2

    def test_boundary_distance_exactly_merge_dist_merges(self):
        d1 = _level(100.0, strength=0.5, touches=1, source_tf=Timeframe.D1)
        h1 = _level(100.5, strength=0.5, touches=1, source_tf=Timeframe.H1)
        merged = _merge_multi_tf_levels([d1, h1], self.TF_ORDER, merge_dist=0.5)
        assert len(merged) == 1

    def test_same_tf_does_not_refine_price(self):
        a = _level(100.0, strength=0.9, touches=1, source_tf=Timeframe.D1)
        b = _level(100.2, strength=0.5, touches=1, source_tf=Timeframe.D1)
        merged = _merge_multi_tf_levels([a, b], self.TF_ORDER, merge_dist=0.5)
        assert len(merged) == 1
        assert merged[0].price == 100.0  # same rank → no refinement


class TestMergeSrLevels:
    def test_secondary_dup_dropped(self):
        pri = _level(100.0)
        sec = _level(100.2)
        merged = _merge_sr_levels([pri], [sec], dedup_dist=0.3)
        assert merged == [pri]

    def test_secondary_kept_when_far(self):
        pri = _level(100.0)
        sec = _level(101.0)
        merged = _merge_sr_levels([pri], [sec], dedup_dist=0.3)
        assert merged == [pri, sec]

    def test_boundary_distance_exactly_dedup_dist_is_dup(self):
        pri = _level(100.0)
        sec = _level(100.3)
        merged = _merge_sr_levels([pri], [sec], dedup_dist=0.3)
        assert merged == [pri]


class TestDedupLevels:
    def test_keeps_first_of_near_duplicates(self):
        a = _level(100.0, strength=0.9)
        b = _level(100.1, strength=0.5)
        c = _level(102.0, strength=0.4)
        kept = _dedup_levels([a, b, c], dedup_dist=0.3)
        assert kept == [a, c]

    def test_empty_input(self):
        assert _dedup_levels([], dedup_dist=0.3) == []
