"""Unit tests for the M5g.8 cost model.

Pure-function tests over synthetic (y_true, predicted, take_trade)
arrays. Each test is sub-second.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from blackheart_train.cost_model import (
    ATR_BPS_BY_INTERVAL,
    FEE_BPS_ROUND_TRIP,
    FUNDING_BPS_ROUND_TRIP,
    K_SL,
    K_TP,
    SLIPPAGE_BPS_BY_REGIME,
    _per_trade_atr_units,
    simulate_cost_regime_metrics,
)


# ── Pinned constants ──────────────────────────────────────────────────────


def test_cost_constants_match_blueprint():
    """Operator-controlled cost-model knobs — pin so accidental edits
    surface during audit. Blueprint § 7.2."""
    assert FEE_BPS_ROUND_TRIP == 8.0           # Binance taker × 2
    assert FUNDING_BPS_ROUND_TRIP == 2.0       # ~1-2 funding intervals
    assert SLIPPAGE_BPS_BY_REGIME == {
        "realistic": 10.0,
        "conservative": 35.0,
        "adversarial": 60.0,
    }
    assert K_TP == 1.5 and K_SL == 1.0
    assert ATR_BPS_BY_INTERVAL["1h"] == 60.0


# ── _per_trade_atr_units ──────────────────────────────────────────────────


def test_long_trade_wins_K_TP_atr_units_on_TP_hit():
    """predicted=2 (long), actual=2 (TP_long hit) → +K_TP ATR units."""
    assert _per_trade_atr_units(2, 2) == +K_TP


def test_long_trade_loses_K_SL_atr_units_on_SL_hit():
    """predicted=2 (long), actual=0 (SL_long hit) → -K_SL ATR units."""
    assert _per_trade_atr_units(2, 0) == -K_SL


def test_long_trade_zero_on_horizon_end():
    """predicted=2 (long), actual=1 (horizon) → 0 (no movement)."""
    assert _per_trade_atr_units(2, 1) == 0.0


def test_short_trade_wins_K_SL_atr_units_on_SL_long_hit():
    """predicted=0 (short), actual=0 (price down to SL_long) →
    +K_SL ATR units (short wins, but only K_SL because triple-barrier
    is long-biased)."""
    assert _per_trade_atr_units(0, 0) == +K_SL


def test_short_trade_loses_K_SL_atr_units_on_TP_long_hit():
    """predicted=0 (short), actual=2 (price rose to TP_long at +K_TP ATR) →
    -K_SL ATR units.  The short's SL sits at +K_SL ATR < K_TP ATR, so the
    short is stopped out at K_SL before the LONG TP is hit.  MC1 fix:
    original formula incorrectly used -K_TP here."""
    assert _per_trade_atr_units(0, 2) == -K_SL


def test_no_trade_when_predicted_horizon_end():
    """predicted=1 (horizon) → 0 regardless of actual."""
    for actual in (0, 1, 2):
        assert _per_trade_atr_units(1, actual) == 0.0


# ── simulate_cost_regime_metrics ─────────────────────────────────────────


def test_zero_trades_when_take_trade_all_false():
    """Meta-label rejected every row → no trades simulated."""
    y_true = np.array([0, 1, 2, 0, 2], dtype=int)
    pred = np.array([2, 1, 2, 0, 2], dtype=int)
    take = np.zeros(5, dtype=bool)
    m = simulate_cost_regime_metrics(y_true, pred, take, interval="1h")
    assert m["cost_n_trades"] == 0
    # No per-regime keys when nothing traded.
    assert "cost_realistic_net_pnl_bps_per_trade" not in m


def test_all_correct_long_trades_profitable_under_realistic():
    """Perfect prediction (every TP correctly predicted as long),
    every trade taken → strongly positive PnL under realistic costs."""
    n = 50
    y_true = np.full(n, 2, dtype=int)   # all TP_long
    pred = np.full(n, 2, dtype=int)     # all predicted long
    take = np.ones(n, dtype=bool)
    m = simulate_cost_regime_metrics(y_true, pred, take, interval="1h")
    assert m["cost_n_trades"] == n
    # Each trade: gross = K_TP * 60 = 90 bps
    # Realistic cost = 8 + 10 + 2 = 20 bps
    # Net per trade = 90 - 20 = 70 bps
    assert m["cost_gross_pnl_bps_per_trade"] == pytest.approx(K_TP * 60.0)
    assert m["cost_realistic_net_pnl_bps_per_trade"] == pytest.approx(K_TP * 60.0 - 20.0)
    assert m["cost_realistic_profitable"] == 1.0
    # Hit rate is 1.0 (every trade a winner).
    assert m["cost_hit_rate"] == 1.0


def test_perfect_wrong_long_trades_lose_under_realistic():
    """Always predict long, always actual SL → every trade loses 1.0 ATR
    + costs. Strongly negative net PnL."""
    n = 30
    y_true = np.full(n, 0, dtype=int)   # all SL hit
    pred = np.full(n, 2, dtype=int)     # all predicted long
    take = np.ones(n, dtype=bool)
    m = simulate_cost_regime_metrics(y_true, pred, take, interval="1h")
    # Gross per trade: -K_SL * 60 = -60 bps. Realistic cost = 20 bps.
    # Net per trade = -60 - 20 = -80 bps.
    assert m["cost_gross_pnl_bps_per_trade"] == pytest.approx(-K_SL * 60.0)
    assert m["cost_realistic_net_pnl_bps_per_trade"] == pytest.approx(-60.0 - 20.0)
    assert m["cost_realistic_profitable"] == 0.0


def test_adversarial_regime_strictly_more_expensive_than_realistic():
    """For the same gross PnL, adversarial regime costs more per trade
    than conservative which costs more than realistic. Verifies the
    pinned slippage ordering."""
    n = 20
    y_true = np.full(n, 2, dtype=int)
    pred = np.full(n, 2, dtype=int)
    take = np.ones(n, dtype=bool)
    m = simulate_cost_regime_metrics(y_true, pred, take, interval="1h")
    assert (m["cost_realistic_cost_per_trade_bps"]
            < m["cost_conservative_cost_per_trade_bps"]
            < m["cost_adversarial_cost_per_trade_bps"])
    assert (m["cost_realistic_net_pnl_bps_per_trade"]
            > m["cost_conservative_net_pnl_bps_per_trade"]
            > m["cost_adversarial_net_pnl_bps_per_trade"])


def test_horizon_predictions_dont_trade_even_when_take_is_true():
    """Even with meta-label saying 'take_trade=True', a class-1 prediction
    means no trade. The model's directional call decides, not just the
    gate."""
    n = 20
    y_true = np.array([2] * n, dtype=int)
    pred = np.full(n, 1, dtype=int)   # ALL predicted horizon-end
    take = np.ones(n, dtype=bool)     # meta-label says trade them all
    m = simulate_cost_regime_metrics(y_true, pred, take, interval="1h")
    assert m["cost_n_trades"] == 0


def test_unknown_interval_raises():
    with pytest.raises(KeyError, match="ATR estimate"):
        simulate_cost_regime_metrics(
            np.array([0, 1, 2]),
            np.array([2, 1, 0]),
            np.ones(3, dtype=bool),
            interval="3h",   # not in ATR_BPS_BY_INTERVAL
        )


def test_misaligned_inputs_raise():
    with pytest.raises(ValueError, match="aligned"):
        simulate_cost_regime_metrics(
            np.array([0, 1, 2]),
            np.array([2, 1]),
            np.ones(3, dtype=bool),
        )


def test_empty_inputs_raise_with_clear_message():
    """MC2 fix: empty inputs raise a clear ValueError naming the
    cause, not the misleading 'aligned' message that the old code
    produced."""
    with pytest.raises(ValueError, match="non-empty inputs"):
        simulate_cost_regime_metrics(
            np.array([], dtype=int),
            np.array([], dtype=int),
            np.array([], dtype=bool),
        )


# ── Hit rate ──────────────────────────────────────────────────────────────


def test_hit_rate_reflects_actual_win_fraction():
    """Mixed outcomes — half wins (long+TP), half losses (long+SL).
    Hit rate should be 0.5."""
    y_true = np.array([2, 0, 2, 0, 2, 0], dtype=int)
    pred = np.array([2, 2, 2, 2, 2, 2], dtype=int)   # all long
    take = np.ones(6, dtype=bool)
    m = simulate_cost_regime_metrics(y_true, pred, take, interval="1h")
    assert m["cost_n_trades"] == 6
    assert m["cost_hit_rate"] == pytest.approx(0.5)


# ── MC7: horizon-end exit counts as trade-taken with 0 PnL ────────────────


def test_long_trade_at_horizon_end_counts_as_trade_with_zero_gross():
    """MC7: predicted=long (2), actual=horizon (1) — we entered long
    and exited at expiry. Counts as a TRADE (paid fees + funding) but
    gross PnL is 0."""
    y_true = np.array([1], dtype=int)   # actual horizon
    pred = np.array([2], dtype=int)     # predicted long
    take = np.ones(1, dtype=bool)
    m = simulate_cost_regime_metrics(y_true, pred, take, interval="1h")
    assert m["cost_n_trades"] == 1
    assert m["cost_gross_pnl_bps_per_trade"] == 0.0
    # Net PnL is just the negative of cost per trade — we paid costs
    # without earning anything gross.
    assert m["cost_realistic_net_pnl_bps_per_trade"] == -20.0  # 8 + 10 + 2


def test_short_trade_at_horizon_end_counts_as_trade_with_zero_gross():
    """MC7: predicted=short (0), actual=horizon (1) — same pattern as
    LONG version, with SHORT entry."""
    y_true = np.array([1], dtype=int)
    pred = np.array([0], dtype=int)
    take = np.ones(1, dtype=bool)
    m = simulate_cost_regime_metrics(y_true, pred, take, interval="1h")
    assert m["cost_n_trades"] == 1
    assert m["cost_gross_pnl_bps_per_trade"] == 0.0


def test_horizon_end_hit_rate_classified_as_not_hit():
    """MC6 + MC7: in a 50/50 win/horizon mix, hit_rate is 0.5 (the
    horizon trades count as not-hits, NOT as ignored)."""
    # 2 winning long trades (predicted=2, actual=2) and 2 horizon-end
    # trades (predicted=2, actual=1). 4 trades total.
    y_true = np.array([2, 2, 1, 1], dtype=int)
    pred = np.array([2, 2, 2, 2], dtype=int)
    take = np.ones(4, dtype=bool)
    m = simulate_cost_regime_metrics(y_true, pred, take, interval="1h")
    assert m["cost_n_trades"] == 4
    # 2 wins of K_TP*60=90 bps, 2 ties of 0 bps → gross mean = 45
    assert m["cost_gross_pnl_bps_per_trade"] == pytest.approx(0.5 * K_TP * 60.0)
    # Hit rate: 2 wins / 4 trades total = 0.5
    assert m["cost_hit_rate"] == pytest.approx(0.5)


# ── MC3 vectorization equivalence to MC3-removed for-loop ────────────────


def test_vectorized_matches_per_trade_table_on_random_inputs():
    """MC3 fix: the vectorized inner-loop replacement must produce the
    same per-trade PnL as the unit-tested ``_per_trade_atr_units``
    table for every (predicted, actual) combination."""
    rng = np.random.default_rng(123)
    n = 500
    y_true = rng.integers(0, 3, size=n)
    pred = rng.integers(0, 3, size=n)
    take = np.ones(n, dtype=bool)
    m = simulate_cost_regime_metrics(y_true, pred, take, interval="1h")
    # Recompute expected via the table.
    expected_gross = 0.0
    n_traded = 0
    for i in range(n):
        if pred[i] == 1:
            continue
        n_traded += 1
        expected_gross += _per_trade_atr_units(int(pred[i]), int(y_true[i])) * 60.0
    expected_mean = expected_gross / max(1, n_traded)
    assert m["cost_n_trades"] == n_traded
    assert m["cost_gross_pnl_bps_per_trade"] == pytest.approx(expected_mean)
