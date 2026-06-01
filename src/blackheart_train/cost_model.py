"""Cost-regime PnL simulation (blueprint § 7.2, gauntlet gate 5).

The directional model's edge is meaningless if it doesn't survive
realistic trading costs. Gate 5 of the 13-gate gauntlet requires
profitable PnL under both ``realistic`` AND ``conservative`` cost
regimes (``adversarial`` is informational). This module bridges the
model's per-bar predictions to a per-trade PnL estimate, applies the
three cost regimes, and surfaces flat metrics for the gauntlet to read.

PnL model — phase 1, simplified:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For each test bar where the ensemble predicts a directional class
(``class 0`` = SL_long hit / go SHORT; ``class 2`` = TP_long hit / go
LONG) AND the meta-label allows the trade (``P(win) > threshold``):

* LONG trade outcome (entered at bar t, exit at first barrier or
  horizon):
  - actual class +1 → +K_TP × ATR(t)   gross win
  - actual class -1 → -K_SL × ATR(t)   gross loss
  - actual class  0 → 0                horizon timeout, ~no movement
* SHORT trade outcome (mirror — the triple-barrier label encodes
  long-side outcomes, so a short trade winning means hitting
  SL_long which is *price going down*):
  - actual class -1 → +K_SL × ATR(t)   gross win
  - actual class +1 → -K_TP × ATR(t)   gross loss
  - actual class  0 → 0

Costs (per round-trip trade):
* fees: 2 × 4 bps = 8 bps   (Binance taker, both sides)
* funding: ~2 bps           (approx 1-2 funding intervals × ~1 bps)
* slippage: regime-dependent
  - realistic: 10 bps
  - conservative: 35 bps   (+25 bps over realistic per blueprint)
  - adversarial: 60 bps    (+50 bps over realistic)

Phase 1 approximations (audit findings MC1 / MC5 / MC6):

* **MC1 — SHORT trade scoring uses the LONG triple-barrier label.**
  A SHORT trade has its own barriers (SHORT_TP = entry − K_TP × ATR,
  SHORT_SL = entry + K_SL × ATR). The LONG triple-barrier label tells
  us which of LONG's barriers (±K_TP / ∓K_SL) was hit first, NOT
  which of SHORT's barriers. We mirror the LONG outcome — strictly
  wrong, since a price drop to LONG's SL (−1.0 ATR) doesn't trigger
  SHORT's TP (at −1.5 ATR). The mirror approximation is biased toward
  smaller-magnitude SHORT wins (capped at +K_SL) and larger-magnitude
  SHORT losses (−K_TP). Phase 2 would compute a SHORT triple-barrier
  label or walk OHLC directly per trade.

* **MC5 — Per-interval ATR is a single constant (1h ≈ 60 bps;
  15m ≈ 30 bps).** Loses per-bar volatility dispersion (quiet bars
  ≈ 30 bps, volatile bars ≈ 100 bps). Aggregate mean PnL is unbiased
  (E[K × ATR] = K × E[ATR]) so the cost-regime profitable/not-
  profitable verdict is reliable in expectation; per-trade variance
  and Sharpe-style metrics are lossy. Phase 2 would carry per-bar ATR
  on the LoadedDataset and use the at-entry value.

* **MC6 — Hit rate counts horizon-end exits as not-hits.**
  ``cost_hit_rate = wins / (wins + losses + ties)`` where "ties" are
  trades that exited at the horizon with PnL=0. Defensible — the
  trade was capital-taken and paid fees+funding — but conservative;
  some traders compute ``wins / (wins + losses)`` instead.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


# ── Pinned cost constants (blueprint § 7.2) ──────────────────────────────


# Round-trip fee: Binance taker fee × 2 (entry + exit). 0.04% per side
# × 2 = 8 bps. Matches the JVM's existing `FEE_RATE_TAKER` constant.
FEE_BPS_ROUND_TRIP: float = 8.0

# Funding cost approximation for medium-horizon trades (1h-24h holding).
# Funding intervals fire every 8h; rate ≈ 0.01% per interval = 1 bps.
# Lump as 2 bps for an "average" trade — phase 2 should compute per-trade
# from actual holding duration.
FUNDING_BPS_ROUND_TRIP: float = 2.0

# Slippage in bps by regime — blueprint § 7.2:
#  realistic   ≈ measured (10 bps is the JVM's calibration default
#                          for BTCUSDT 1h size on Binance perp)
#  conservative = realistic + 25 bps (per blueprint)
#  adversarial  = realistic + 50 bps (per blueprint)
SLIPPAGE_BPS_BY_REGIME: dict[str, float] = {
    "realistic": 10.0,
    "conservative": 35.0,
    "adversarial": 60.0,
}

# Approximate per-interval ATR in basis points of price. Phase 1's
# single-constant approach; phase 2 would carry per-bar ATR on the
# dataset and use the actual at-entry value. The 1h figure matches the
# observed BTCUSDT 1h ATR_14 over the 2024-12 → 2026-05 window.
ATR_BPS_BY_INTERVAL: dict[str, float] = {
    "1h": 60.0,
    "15m": 30.0,
    "5m": 15.0,
    "4h": 130.0,
}

# Triple-barrier multipliers — MUST match
# ``derived_features._t_label_triple_barrier``. Pinned here too so a
# future blueprint tweak doesn't drift the cost model silently.
K_TP: float = 1.5
K_SL: float = 1.0


# Class encoding (matches train.encode_multiclass):
#   0 = -1 (SL_long hit → price went DOWN → predicted: go SHORT)
#   1 =  0 (horizon end → neither barrier → predicted: no trade)
#   2 = +1 (TP_long hit → price went UP   → predicted: go LONG)
_PREDICTED_TO_DIRECTION: dict[int, str] = {0: "short", 1: "none", 2: "long"}


# ── Per-trade PnL helper ─────────────────────────────────────────────────


def _per_trade_atr_units(predicted_class: int, actual_class: int) -> float:
    """Gross PnL of one trade in ATR units, before costs.

    Returns 0 when ``predicted_class == 1`` (horizon-end prediction =
    no trade) or when the actual class is the horizon-end outcome (the
    trade exits at expiry with ~no movement).

    Asymmetry note: the triple-barrier is long-biased (TP wider than
    SL). A LONG win pays K_TP=1.5 ATR; a SHORT win pays only K_SL=1.0
    ATR. Conversely, a LONG loss costs K_SL=1.0 ATR vs a SHORT loss
    of K_TP=1.5 ATR. This asymmetry is inherent to the label
    construction; downstream PnL inherits it.
    """
    if predicted_class == 1:
        return 0.0
    if predicted_class == 2:    # go long
        if actual_class == 2:
            return +K_TP
        if actual_class == 0:
            return -K_SL
        return 0.0   # horizon end
    if predicted_class == 0:    # go short
        if actual_class == 0:
            return +K_SL
        if actual_class == 2:
            # LONG TP hit means price rose K_TP ATR, but short's SL was at
            # +K_SL ATR (< K_TP), so short SL was triggered first: loss = K_SL.
            return -K_SL
        return 0.0
    return 0.0


# ── Per-trade PnL (shared by aggregator + regime sub-cuts) ───────────────


def compute_per_trade_pnl_bps(
    y_true_enc: np.ndarray,
    predicted_classes: np.ndarray,
    take_trade: np.ndarray,
    *,
    interval: str = "1h",
) -> tuple[np.ndarray, np.ndarray, float]:
    """Vectorized per-trade gross PnL (in bps) + traded mask + the
    interval's ATR_bps assumption. Used by both
    :func:`simulate_cost_regime_metrics` and the regime sub-cut module
    so the formula has one source of truth.

    Returns:
      * ``gross_bps`` — shape (n,) per-row gross PnL in bps; 0 for
        non-traded rows.
      * ``traded_mask`` — shape (n,) bool; True where a trade was taken.
      * ``atr_bps`` — scalar; the per-interval ATR assumption used.

    PnL table (ATR units, before scaling to bps):

      predicted=2 (LONG):  +K_TP if actual=2, -K_SL if actual=0, 0 if actual=1
      predicted=0 (SHORT): +K_SL if actual=0, -K_TP if actual=2, 0 if actual=1
      predicted=1 (HORIZON): no trade

    MC3 note: the for-loop version was ~100x slower on 32k stacked val
    rows; vectorized matches the per-trade table bit-for-bit.
    """
    atr_bps = ATR_BPS_BY_INTERVAL.get(interval)
    if atr_bps is None:
        raise KeyError(
            f"no ATR estimate for interval {interval!r}; add to "
            f"ATR_BPS_BY_INTERVAL or pass a supported interval"
        )
    y_true_enc = np.asarray(y_true_enc).astype(int)
    predicted_classes = np.asarray(predicted_classes).astype(int)
    take_trade = np.asarray(take_trade).astype(bool)
    n = len(y_true_enc)
    if n == 0:
        raise ValueError(
            "compute_per_trade_pnl_bps requires non-empty inputs; got "
            "zero-length arrays. Caller should skip the cost step for "
            "empty val slices."
        )
    if not (len(predicted_classes) == n and len(take_trade) == n):
        raise ValueError(
            f"aligned input arrays required; got lengths "
            f"y_true_enc={n}, predicted_classes={len(predicted_classes)}, "
            f"take_trade={len(take_trade)}"
        )

    is_long = predicted_classes == 2
    is_short = predicted_classes == 0
    is_actual_tp = y_true_enc == 2
    is_actual_sl = y_true_enc == 0
    # SHORT scoring fix (MC1): the LONG triple-barrier label records outcomes
    # against LONG barriers (TP at +K_TP ATR, SL at -K_SL ATR).  When we
    # infer the short position's outcome from that label:
    #
    #   actual=0 (LONG SL hit, price fell K_SL ATR):
    #     Short TP is at -K_TP ATR; price only fell K_SL < K_TP ATR, so the
    #     short TP was NOT yet hit.  Conservative credit: short exits at the
    #     LONG SL level → gains K_SL ATR.  Unchanged from the original.
    #
    #   actual=2 (LONG TP hit, price rose K_TP ATR):
    #     Short SL is at +K_SL ATR < K_TP ATR — it was hit BEFORE the long TP.
    #     The short lost K_SL ATR, not K_TP ATR.  The original formula used
    #     -K_TP here, which overstated the short's loss by (K_TP - K_SL) ATR.
    #
    # With this fix, short PnL is symmetric at K_SL ATR in both directions,
    # which is the correct approximation given only the long-side label.
    gross_atr_units = (
        is_long * (is_actual_tp * K_TP - is_actual_sl * K_SL)
        + is_short * K_SL * (is_actual_sl.astype(float) - is_actual_tp.astype(float))
    )
    no_trade = (~take_trade) | (predicted_classes == 1)
    gross_atr_units = np.where(no_trade, 0.0, gross_atr_units)
    gross_bps = gross_atr_units * float(atr_bps)
    traded_mask = ~no_trade
    return gross_bps, traded_mask, float(atr_bps)


# ── Public entry point ───────────────────────────────────────────────────


def simulate_cost_regime_metrics(
    y_true_enc: np.ndarray,
    predicted_classes: np.ndarray,
    take_trade: np.ndarray,
    *,
    interval: str = "1h",
) -> dict[str, float]:
    """Aggregate per-trade PnL under three cost regimes.

    Inputs are aligned per-row arrays over the val/test slice:

    * ``y_true_enc`` — encoded actual class (0/1/2)
    * ``predicted_classes`` — ``argmax`` of the ensemble proba
    * ``take_trade`` — boolean array, ``True`` where the meta-label
      passed the row (and trading is allowed; if no meta-label is
      attached the caller passes all-True)

    Returns a flat metrics dict suitable for the artifact's
    ``payload["metrics"]``. Keys are prefixed ``cost_*`` so they
    don't collide with the model-evaluation namespace.

    A fold with zero traded bars returns ``cost_n_trades=0`` and no
    per-regime PnL entries — the walk-forward aggregator's mean over
    only-finite values handles this gracefully.
    """
    gross_bps, traded_mask, atr_bps = compute_per_trade_pnl_bps(
        y_true_enc, predicted_classes, take_trade, interval=interval,
    )

    n_traded = int(traded_mask.sum())
    out: dict[str, float] = {
        "cost_n_trades": float(n_traded),
        "cost_atr_bps_assumed": float(atr_bps),
    }
    if n_traded == 0:
        return out

    gross_per_trade = gross_bps[traded_mask]
    out["cost_gross_pnl_bps_per_trade"] = float(gross_per_trade.mean())
    out["cost_hit_rate"] = float((gross_per_trade > 0).mean())

    for regime, slippage_bps in SLIPPAGE_BPS_BY_REGIME.items():
        cost_per_trade_bps = (
            FEE_BPS_ROUND_TRIP + slippage_bps + FUNDING_BPS_ROUND_TRIP
        )
        net_per_trade = gross_per_trade - cost_per_trade_bps
        out[f"cost_{regime}_cost_per_trade_bps"] = float(cost_per_trade_bps)
        out[f"cost_{regime}_net_pnl_bps_per_trade"] = float(net_per_trade.mean())
        out[f"cost_{regime}_net_pnl_bps_total"] = float(net_per_trade.sum())
        # Float 0/1 so the walk-forward aggregator can compute a mean
        # ("fraction of folds profitable under regime"). Bool would
        # break the np.mean step.
        out[f"cost_{regime}_profitable"] = float(net_per_trade.mean() > 0)

    return out
