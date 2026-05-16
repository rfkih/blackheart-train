"""Unit tests for the M5g.9 regime sub-cut module.

Pure-function tests over synthetic per-trade PnL arrays + regime
flags. Each test is sub-second.
"""
from __future__ import annotations

import numpy as np
import pytest

from blackheart_train.regime_subcuts import (
    MAX_FAILING_REGIMES_DEFAULT,
    MIN_TRADES_PER_REGIME_DEFAULT,
    P_THRESHOLD_DEFAULT,
    classify_trend_regimes_from_train_quantiles,
    classify_vol_regimes_from_train_quantiles,
    regime_subcut_metrics,
)


# ── Pinned constants ──────────────────────────────────────────────────────


def test_blueprint_thresholds_pinned():
    """Blueprint § 7.4 numerical thresholds — pin so accidental edits
    surface during audit."""
    assert MIN_TRADES_PER_REGIME_DEFAULT == 10
    assert P_THRESHOLD_DEFAULT == 0.05
    assert MAX_FAILING_REGIMES_DEFAULT == 1


# ── regime_subcut_metrics ─────────────────────────────────────────────────


def test_all_positive_regimes_pass_gate_6():
    """Both regimes solidly profitable → gate 6 passes (zero failing).
    With both regimes fully populated (>min_trades) the caveat is 0."""
    # 40 trades, all +50 bps gross. Cost = 20 bps. Net = +30 bps.
    n = 40
    gross = np.full(n, 50.0)
    flags = {
        "high_vol": np.array([True] * 20 + [False] * 20),
        "low_vol": np.array([False] * 20 + [True] * 20),
    }
    m = regime_subcut_metrics(
        gross_bps_per_trade=gross,
        cost_per_trade_bps=20.0,
        regime_flags=flags,
    )
    assert m["regime_high_vol_n_trades"] == 20
    assert m["regime_low_vol_n_trades"] == 20
    assert m["regime_high_vol_mean_net_pnl_bps"] == pytest.approx(30.0)
    assert m["regime_low_vol_mean_net_pnl_bps"] == pytest.approx(30.0)
    # MR1: degenerate-variance branch (all rows identical). Code now
    # decides explicitly: mean > 0 → p=1.0, not-failing.
    assert m["regime_high_vol_p_value"] == 1.0
    assert m["regime_high_vol_failing"] == 0.0
    assert m["regime_low_vol_failing"] == 0.0
    assert m["regime_n_failing"] == 0.0
    assert m["regime_gate_6_pass"] == 1.0
    # MR2: both regimes had n=20 >= min_trades → no caveat.
    assert m["regime_gate_6_caveat"] == 0.0


def test_one_failing_regime_still_passes_gate_6():
    """High_vol significantly negative, low_vol positive → one failure
    allowed under max_failing=1 (blueprint § 7.4 Fix 9 default)."""
    rng = np.random.default_rng(42)
    # 30 high_vol trades, gross ~ -10 bps (so net = -30 after 20 cost)
    high_gross = rng.normal(loc=-10.0, scale=5.0, size=30)
    # 30 low_vol trades, gross ~ +40 bps (so net = +20)
    low_gross = rng.normal(loc=40.0, scale=5.0, size=30)
    gross = np.concatenate([high_gross, low_gross])
    flags = {
        "high_vol": np.array([True] * 30 + [False] * 30),
        "low_vol": np.array([False] * 30 + [True] * 30),
    }
    m = regime_subcut_metrics(
        gross_bps_per_trade=gross,
        cost_per_trade_bps=20.0,
        regime_flags=flags,
    )
    assert m["regime_high_vol_failing"] == 1.0
    assert m["regime_low_vol_failing"] == 0.0
    assert m["regime_n_failing"] == 1.0
    # Default max_failing=1 → still passes
    assert m["regime_gate_6_pass"] == 1.0


def test_two_failing_regimes_fails_gate_6():
    """Both regimes significantly negative → gate fails."""
    rng = np.random.default_rng(7)
    high_gross = rng.normal(loc=-10.0, scale=5.0, size=30)
    low_gross = rng.normal(loc=-10.0, scale=5.0, size=30)
    gross = np.concatenate([high_gross, low_gross])
    flags = {
        "high_vol": np.array([True] * 30 + [False] * 30),
        "low_vol": np.array([False] * 30 + [True] * 30),
    }
    m = regime_subcut_metrics(
        gross_bps_per_trade=gross,
        cost_per_trade_bps=20.0,
        regime_flags=flags,
    )
    assert m["regime_high_vol_failing"] == 1.0
    assert m["regime_low_vol_failing"] == 1.0
    assert m["regime_n_failing"] == 2.0
    assert m["regime_gate_6_pass"] == 0.0


def test_skip_regime_with_too_few_trades():
    """Regime with n < min_trades_per_regime → t-test skipped, NOT
    counted as failing. Blueprint § 7.4: 'skip with caveat'.
    MR2: caveat=1.0 since at least one regime was under-populated."""
    n_high = 5    # below default min_trades=10
    n_low = 20
    gross = np.concatenate([
        np.full(n_high, -100.0),    # would fail if checked
        np.full(n_low, +100.0),
    ])
    flags = {
        "high_vol": np.array([True] * n_high + [False] * n_low),
        "low_vol": np.array([False] * n_high + [True] * n_low),
    }
    m = regime_subcut_metrics(
        gross_bps_per_trade=gross,
        cost_per_trade_bps=20.0,
        regime_flags=flags,
    )
    # high_vol mean still computed
    assert m["regime_high_vol_n_trades"] == 5
    assert m["regime_high_vol_mean_net_pnl_bps"] == pytest.approx(-120.0)
    # ...but p_value is NaN and failing=0
    assert np.isnan(m["regime_high_vol_p_value"])
    assert m["regime_high_vol_failing"] == 0.0
    assert m["regime_n_regimes_checked"] == 1.0   # only low_vol checked
    assert m["regime_gate_6_pass"] == 1.0
    # MR2: caveat fires because high_vol had n < min_trades.
    assert m["regime_gate_6_caveat"] == 1.0


def test_empty_regime_handled_without_crashing():
    """Regime with zero trades → NaN mean/p, not failing, no crash.
    MR2: caveat=1.0 because high_vol was empty (incomplete evidence)."""
    n = 30
    gross = np.full(n, 50.0)
    flags = {
        "high_vol": np.zeros(n, dtype=bool),     # zero in regime
        "low_vol": np.ones(n, dtype=bool),       # all in this regime
    }
    m = regime_subcut_metrics(
        gross_bps_per_trade=gross,
        cost_per_trade_bps=20.0,
        regime_flags=flags,
    )
    assert m["regime_high_vol_n_trades"] == 0
    assert np.isnan(m["regime_high_vol_mean_net_pnl_bps"])
    assert np.isnan(m["regime_high_vol_p_value"])
    assert m["regime_high_vol_failing"] == 0.0
    assert m["regime_low_vol_failing"] == 0.0
    assert m["regime_gate_6_pass"] == 1.0
    # MR2: empty regime is an evidence hole → caveat fires.
    assert m["regime_gate_6_caveat"] == 1.0


def test_misaligned_regime_flags_raise():
    """regime_flags arrays must match gross_bps_per_trade shape."""
    gross = np.full(10, 50.0)
    flags = {
        "high_vol": np.array([True] * 8),    # wrong length
    }
    with pytest.raises(ValueError, match="shape"):
        regime_subcut_metrics(
            gross_bps_per_trade=gross,
            cost_per_trade_bps=20.0,
            regime_flags=flags,
        )


def test_non_overlapping_regimes_not_required():
    """A trade can be in BOTH high_vol AND bull simultaneously — the
    module does NOT enforce mutual exclusion. Each regime evaluated
    independently."""
    n = 30
    gross = np.full(n, +50.0)
    flags = {
        "high_vol": np.ones(n, dtype=bool),    # every trade in BOTH
        "bull": np.ones(n, dtype=bool),
    }
    m = regime_subcut_metrics(
        gross_bps_per_trade=gross,
        cost_per_trade_bps=20.0,
        regime_flags=flags,
    )
    assert m["regime_high_vol_n_trades"] == n
    assert m["regime_bull_n_trades"] == n
    assert m["regime_gate_6_pass"] == 1.0


def test_max_failing_override_lets_strict_gate_pass_only_when_all_positive():
    """Setting max_failing=0 makes the gate strict: any failing regime
    fails the gate. Useful for stress-testing."""
    rng = np.random.default_rng(11)
    high_gross = rng.normal(loc=-10.0, scale=5.0, size=30)
    low_gross = rng.normal(loc=+30.0, scale=5.0, size=30)
    gross = np.concatenate([high_gross, low_gross])
    flags = {
        "high_vol": np.array([True] * 30 + [False] * 30),
        "low_vol": np.array([False] * 30 + [True] * 30),
    }
    m = regime_subcut_metrics(
        gross_bps_per_trade=gross,
        cost_per_trade_bps=20.0,
        regime_flags=flags,
        max_failing=0,
    )
    assert m["regime_n_failing"] == 1.0
    # With strict max_failing=0, one failure fails the gate.
    assert m["regime_gate_6_pass"] == 0.0


# ── classify_vol_regimes_from_train_quantiles ────────────────────────────


def test_vol_classifier_basic_split():
    """Train vol = [0..99]; eval contains values below/above q25/q75.
    Verify split is correct."""
    vol_train = np.arange(100, dtype=float)
    # q25 ≈ 24.75, q75 ≈ 74.25
    vol_eval = np.array([10.0, 50.0, 80.0])
    flags = classify_vol_regimes_from_train_quantiles(vol_train, vol_eval)
    assert flags["low_vol"].tolist() == [True, False, False]
    assert flags["high_vol"].tolist() == [False, False, True]


def test_vol_classifier_handles_nan_in_train():
    """NaN in vol_train shouldn't propagate through np.quantile."""
    vol_train = np.array([np.nan, 1.0, 2.0, 3.0, 4.0, 5.0, np.nan])
    vol_eval = np.array([0.5, 4.5])
    flags = classify_vol_regimes_from_train_quantiles(vol_train, vol_eval)
    # 5 finite values, q25=2.0, q75=4.0
    # 0.5 < 2.0 → low_vol; 4.5 > 4.0 → high_vol
    assert flags["low_vol"].tolist() == [True, False]
    assert flags["high_vol"].tolist() == [False, True]


def test_vol_classifier_raises_when_train_all_nan():
    """Cannot compute quantiles if all train vol is NaN."""
    vol_train = np.array([np.nan, np.nan, np.nan])
    vol_eval = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="no finite values"):
        classify_vol_regimes_from_train_quantiles(vol_train, vol_eval)


def test_vol_classifier_custom_quantiles():
    """Non-default low_q / high_q knobs should shift the boundaries."""
    vol_train = np.arange(100, dtype=float)
    vol_eval = np.array([5.0, 50.0, 95.0])
    # 10/90 splits: q10≈9.9, q90≈89.1
    flags = classify_vol_regimes_from_train_quantiles(
        vol_train, vol_eval, low_q=0.10, high_q=0.90,
    )
    assert flags["low_vol"].tolist() == [True, False, False]
    assert flags["high_vol"].tolist() == [False, False, True]


def test_vol_classifier_returns_arrays_of_correct_length():
    """Output flags align with vol_eval, NOT vol_train."""
    vol_train = np.arange(100, dtype=float)
    vol_eval = np.arange(7, dtype=float)
    flags = classify_vol_regimes_from_train_quantiles(vol_train, vol_eval)
    assert flags["low_vol"].shape == (7,)
    assert flags["high_vol"].shape == (7,)


def test_vol_classifier_no_overlap_between_high_and_low():
    """A given vol_eval row must be at most one of {low_vol, high_vol}.
    Mid_vol rows are NEITHER (excluded by quantile-band design)."""
    rng = np.random.default_rng(31)
    vol_train = rng.normal(size=200)
    vol_eval = rng.normal(size=100)
    flags = classify_vol_regimes_from_train_quantiles(vol_train, vol_eval)
    # No row should be both high and low
    assert not np.any(flags["low_vol"] & flags["high_vol"])
    # A meaningful fraction should be neither (the [q25, q75] middle band)
    neither = ~(flags["low_vol"] | flags["high_vol"])
    assert neither.any()


# ── MR1: degenerate variance ──────────────────────────────────────────────


def test_mr1_degenerate_variance_negative_mean_is_failing():
    """MR1: every trade in a regime has identical net PnL = -20 bps
    (e.g., every trade exited at horizon, paying costs without earning
    gross). std=0 → scipy returns p=NaN, the OLD code would silently
    mark not-failing. NEW code: mean<0 → p=0.0, failing=1.0."""
    n = 30
    # gross_bps_per_trade = 0 for every trade, cost = 20 bps → net = -20
    gross = np.zeros(n, dtype="float64")
    flags = {"high_vol": np.ones(n, dtype=bool)}
    m = regime_subcut_metrics(
        gross_bps_per_trade=gross,
        cost_per_trade_bps=20.0,
        regime_flags=flags,
    )
    assert m["regime_high_vol_mean_net_pnl_bps"] == pytest.approx(-20.0)
    assert m["regime_high_vol_p_value"] == 0.0
    assert m["regime_high_vol_failing"] == 1.0
    assert m["regime_n_failing"] == 1.0


def test_mr1_degenerate_variance_positive_mean_is_not_failing():
    """MR1 mirror: every trade has identical net = +30 bps (gross=50,
    cost=20). std=0, mean>0 → p=1.0, NOT failing."""
    n = 30
    gross = np.full(n, 50.0)
    flags = {"high_vol": np.ones(n, dtype=bool)}
    m = regime_subcut_metrics(
        gross_bps_per_trade=gross,
        cost_per_trade_bps=20.0,
        regime_flags=flags,
    )
    assert m["regime_high_vol_mean_net_pnl_bps"] == pytest.approx(30.0)
    assert m["regime_high_vol_p_value"] == 1.0
    assert m["regime_high_vol_failing"] == 0.0


def test_mr1_degenerate_variance_zero_mean_is_not_failing():
    """MR1 boundary: identical net = 0 (gross = cost). std=0, mean=0
    → not failing (gate's H0 is 'mean ≥ 0', and zero satisfies)."""
    n = 15
    gross = np.full(n, 20.0)   # gross == cost → net = 0
    flags = {"high_vol": np.ones(n, dtype=bool)}
    m = regime_subcut_metrics(
        gross_bps_per_trade=gross,
        cost_per_trade_bps=20.0,
        regime_flags=flags,
    )
    assert m["regime_high_vol_mean_net_pnl_bps"] == pytest.approx(0.0)
    assert m["regime_high_vol_p_value"] == 1.0
    assert m["regime_high_vol_failing"] == 0.0


# ── MR2: regime_gate_6_caveat ─────────────────────────────────────────────


def test_mr2_caveat_zero_when_all_regimes_fully_populated():
    """All regimes have n >= min_trades → caveat=0 (binding evidence)."""
    rng = np.random.default_rng(101)
    n = 60
    gross = rng.normal(loc=50.0, scale=10.0, size=n)
    flags = {
        "high_vol": np.array([True] * 30 + [False] * 30),
        "low_vol": np.array([False] * 30 + [True] * 30),
    }
    m = regime_subcut_metrics(
        gross_bps_per_trade=gross,
        cost_per_trade_bps=20.0,
        regime_flags=flags,
    )
    assert m["regime_gate_6_caveat"] == 0.0


def test_mr2_caveat_fires_on_empty_regime():
    """One empty regime → caveat=1 even though the gate mechanically
    passes via max_failing=1."""
    rng = np.random.default_rng(102)
    n = 30
    gross = rng.normal(loc=50.0, scale=5.0, size=n)
    flags = {
        "high_vol": np.ones(n, dtype=bool),
        "low_vol": np.zeros(n, dtype=bool),   # empty
    }
    m = regime_subcut_metrics(
        gross_bps_per_trade=gross,
        cost_per_trade_bps=20.0,
        regime_flags=flags,
    )
    assert m["regime_low_vol_n_trades"] == 0
    assert m["regime_gate_6_pass"] == 1.0
    assert m["regime_gate_6_caveat"] == 1.0


def test_mr2_caveat_fires_on_under_populated_regime():
    """A regime with 1 ≤ n < min_trades fires the caveat. This is the
    exact failure mode the M5g.9 walk-forward smoke surfaced: low_vol
    typically has 0-3 trades while high_vol has 100s."""
    rng = np.random.default_rng(103)
    # 25 high_vol trades, 5 low_vol trades (< min_trades=10)
    n_hi, n_lo = 25, 5
    gross = np.concatenate([
        rng.normal(loc=50.0, scale=5.0, size=n_hi),
        rng.normal(loc=50.0, scale=5.0, size=n_lo),
    ])
    flags = {
        "high_vol": np.array([True] * n_hi + [False] * n_lo),
        "low_vol": np.array([False] * n_hi + [True] * n_lo),
    }
    m = regime_subcut_metrics(
        gross_bps_per_trade=gross,
        cost_per_trade_bps=20.0,
        regime_flags=flags,
    )
    assert m["regime_low_vol_n_trades"] == n_lo
    assert m["regime_gate_6_caveat"] == 1.0


def test_mr2_caveat_does_not_change_gate_pass_decision():
    """Caveat is informational only — it doesn't gate the gate. A
    caveated pass is still a pass; a caveated fail is still a fail.
    The gauntlet aggregator decides what to do with caveat=1."""
    rng = np.random.default_rng(104)
    # Caveat on low_vol (empty) + high_vol significantly negative → fails
    high_gross = rng.normal(loc=-30.0, scale=5.0, size=30)
    n = 30
    flags = {
        "high_vol": np.ones(n, dtype=bool),
        "low_vol": np.zeros(n, dtype=bool),
    }
    m = regime_subcut_metrics(
        gross_bps_per_trade=high_gross,
        cost_per_trade_bps=20.0,
        regime_flags=flags,
    )
    assert m["regime_gate_6_caveat"] == 1.0
    # max_failing=1 default absorbs the one high_vol failure → passes
    # mechanically. The caveat is the operator's signal that the pass
    # was on incomplete evidence.
    assert m["regime_high_vol_failing"] == 1.0
    assert m["regime_n_failing"] == 1.0
    assert m["regime_gate_6_pass"] == 1.0


# ── classify_trend_regimes_from_train_quantiles (Phase 2) ────────────────


def test_trend_classifier_partitions_into_three_buckets():
    """Tertile defaults — bull/chop/bear are each ~1/3 of train. Bucket
    sizes should be similar (not exact due to discrete sampling)."""
    rng = np.random.default_rng(200)
    trend_train = rng.standard_normal(3000)
    trend_eval = rng.standard_normal(900)
    flags = classify_trend_regimes_from_train_quantiles(trend_train, trend_eval)
    assert set(flags.keys()) == {"bear", "chop", "bull"}
    n_bear, n_chop, n_bull = flags["bear"].sum(), flags["chop"].sum(), flags["bull"].sum()
    # Each bucket should be within ±10pp of 1/3 by sample-size law
    # of large numbers.
    n_eval = len(trend_eval)
    for name, n in [("bear", n_bear), ("chop", n_chop), ("bull", n_bull)]:
        assert abs(n / n_eval - 1.0 / 3.0) < 0.10, (
            f"{name} bucket {n}/{n_eval} too far from 1/3"
        )


def test_trend_buckets_are_mutually_exclusive_and_exhaustive():
    """Every finite eval row lands in exactly one of bear/chop/bull."""
    rng = np.random.default_rng(201)
    trend_train = rng.standard_normal(1000)
    trend_eval = rng.standard_normal(500)
    flags = classify_trend_regimes_from_train_quantiles(trend_train, trend_eval)
    # Pairwise disjoint
    assert not (flags["bear"] & flags["chop"]).any()
    assert not (flags["chop"] & flags["bull"]).any()
    assert not (flags["bear"] & flags["bull"]).any()
    # Exhaustive over finite values
    finite_mask = np.isfinite(trend_eval)
    assert int(
        (flags["bear"] | flags["chop"] | flags["bull"])[finite_mask].sum()
    ) == int(finite_mask.sum())


def test_trend_classifier_pit_safe_thresholds_from_train_only():
    """The eval rows MUST NOT influence the cut points. Verify by
    constructing a pathological eval whose distribution would shift the
    quantiles if it were included — the classifier should still produce
    the train-side bucketing."""
    train = np.linspace(-1.0, 1.0, 300)  # uniform in [-1, +1]
    # Train tertile cuts at ~-0.33 and ~+0.33.
    flags = classify_trend_regimes_from_train_quantiles(
        train, np.array([0.0, 0.5, -0.5, 0.4]),
    )
    # 0.0 → chop, 0.5 → bull (>0.33), -0.5 → bear (<-0.33), 0.4 → bull
    assert list(flags["bear"]) == [False, False, True, False]
    assert list(flags["chop"]) == [True, False, False, False]
    assert list(flags["bull"]) == [False, True, False, True]


def test_trend_classifier_rejects_invalid_quantiles():
    train = np.linspace(-1.0, 1.0, 100)
    eval_ = np.linspace(-1.0, 1.0, 50)
    with pytest.raises(ValueError, match="quantiles must satisfy"):
        classify_trend_regimes_from_train_quantiles(
            train, eval_, bear_q=0.7, bull_q=0.3,
        )
    with pytest.raises(ValueError, match="quantiles must satisfy"):
        classify_trend_regimes_from_train_quantiles(
            train, eval_, bear_q=0.5, bull_q=0.5,
        )


def test_trend_classifier_empty_finite_train_raises():
    """All-NaN train can't produce thresholds — surface as clear
    ValueError rather than silently emitting all-False masks."""
    train = np.array([np.nan, np.nan, np.nan])
    eval_ = np.array([0.0, 0.1, -0.1])
    with pytest.raises(ValueError, match="no finite values"):
        classify_trend_regimes_from_train_quantiles(train, eval_)


def test_trend_classifier_at_threshold_lands_in_bull_not_chop():
    """At exactly the q_bull threshold, a row should land in 'bull'
    (inclusive >= per the chop's half-open [q_bear, q_bull) contract).
    The chop bucket is exclusive on the bull side."""
    train = np.linspace(0.0, 1.0, 100)
    # q_bear ≈ 0.333, q_bull ≈ 0.667.
    flags = classify_trend_regimes_from_train_quantiles(
        train, np.array([0.667]),
    )
    assert bool(flags["bull"][0]) is True
    assert bool(flags["chop"][0]) is False


def test_trend_classifier_combined_with_vol_yields_five_regimes():
    """Combined vol + trend flags create 5 regime buckets. The
    regime_subcut_metrics function should process all 5 without
    threshold change — blueprint's max_failing=1 was designed for 5."""
    rng = np.random.default_rng(202)
    n = 100
    trend_train = rng.standard_normal(n)
    vol_train = np.abs(rng.standard_normal(n))
    trend_eval = rng.standard_normal(n)
    vol_eval = np.abs(rng.standard_normal(n))

    trend_flags = classify_trend_regimes_from_train_quantiles(trend_train, trend_eval)
    vol_flags = classify_vol_regimes_from_train_quantiles(vol_train, vol_eval)

    combined = {**vol_flags, **trend_flags}
    assert set(combined.keys()) == {"low_vol", "high_vol", "bear", "chop", "bull"}

    # Run regime_subcut_metrics with all five — synthetic uniform-positive
    # PnL so the gate passes cleanly.
    gross = np.full(n, 30.0)
    m = regime_subcut_metrics(
        gross_bps_per_trade=gross,
        cost_per_trade_bps=10.0,
        regime_flags=combined,
    )
    # All 5 regimes report n_trades + mean + p_value keys.
    for r in combined:
        assert f"regime_{r}_n_trades" in m
        assert f"regime_{r}_mean_net_pnl_bps" in m
        assert f"regime_{r}_p_value" in m
        assert f"regime_{r}_failing" in m
    assert m["regime_n_failing"] == 0.0
    assert m["regime_gate_6_pass"] == 1.0
