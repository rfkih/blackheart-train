"""Per-regime PnL sub-cuts and gate 6 (blueprint § 7.4 Fix 9).

A model whose overall PnL looks positive might still be losing
catastrophically in one regime (e.g., +200 bps in low-vol but -150 bps
in high-vol). Gate 6 of the 13-gate gauntlet catches this: split per-
trade net PnL by market regime, run a one-sided t-test for each, and
fail if too many regimes are significantly negative.

Per blueprint § 7.4:

* If ``n_trades < 10`` in a regime → skip the t-test with caveat
* Otherwise: one-sided t-test, H0 mean ≥ 0, alternative=less, α=0.05
* Significant negative = regime fails
* **Gate 6 passes if at most 1 regime fails.** Two+ failures = reject.

Phase-1 scope: vol-only regimes (``high_vol`` / ``low_vol``). The
training spec's ``btc_realized_vol_30d`` feature gives us per-bar
volatility; train-set quantiles set the regime thresholds (PIT-safe).
Trend regimes (bull/bear/chop) are blueprint § 7.4 but not yet wired —
would need a per-bar trend feature (e.g., 30d rolling log-return) in
the dataset. Phase 2 adds them.

Why "max 1 failure" not "all must pass":

Five regimes × any one of them being unlucky in a small fold = false
positive rate ≈ 22% under H0 (1 − 0.95⁵). Allowing 1 regime to fail
caps the per-spec false-positive rate. Blueprint § 7.4 Fix 9 softened
the gate from "all must be positive" to "max 1 with caveat" precisely
for this multiplicity reason.

Audit findings (M5g.9 audit, 2026-05-15):

* **MR1 (fixed) — degenerate variance.** If every trade in a regime
  produces identical net PnL (e.g., every trade exited at horizon-end
  with gross=0 → net = -cost_per_trade_bps for every row), ``std==0``
  and ``ttest_1samp`` returns ``pvalue=nan``. The old code's
  ``p < threshold`` then evaluated False → silently marked as
  not-failing, missing the most obvious failure mode (paying costs,
  earning zero). Now: detect ``std==0`` first; ``failing=1`` when
  ``mean < 0``, else ``failing=0`` with ``p_value=1.0`` to signal "no
  variance" rather than NaN.

* **MR2 (fixed) — gate_6_caveat.** Gate 6 can pass mechanically when a
  regime is empty (or has < min_trades): the empty regime is counted
  as not-failing, and ``max_failing=1`` then absorbs any other
  regime's significance failure. Observed in walk-forward: 80/20 and
  every fold had ``low_vol n_trades = 0`` (eval window above train
  q25), so the gate "passed" while ``high_vol`` was failing with
  p≈1e-6. New ``regime_gate_6_caveat`` flag is 1.0 when any regime
  had ``n_trades < min_trades_per_regime`` (including 0). Gauntlet
  aggregator (M5h) can require ``caveat == 0`` for a binding pass.

* **MR3 (documented) — t-test misspecification.** Triple-barrier PnL is
  bimodal-heavy-tailed ({+K_TP·ATR, -K_SL·ATR, 0} atoms); the t-test
  assumes normality of the sample mean. At n=10-30, the p-value can
  be miscalibrated by a multiple. Blueprint § 7.4 specifies t-test so
  we keep it; phase-2 replacement is bootstrap-based or Wilcoxon
  signed-rank.

* **MN3 (documented) — failing=0.0 is ambiguous.** A regime can record
  ``failing=0.0`` for any of: (a) t-test passed, (b) ``n < min_trades``
  skip, (c) empty regime, (d) MR1's degenerate-variance branch when
  mean≥0. The unambiguous read is the pair ``failing`` +
  ``p_value``: p_value=NaN → skipped/empty, p_value=1.0 → degenerate
  variance non-negative mean, p_value<threshold → t-test failed.
"""
from __future__ import annotations

import logging

import numpy as np
from scipy.stats import ttest_1samp

logger = logging.getLogger(__name__)


# Blueprint § 7.4 thresholds.
MIN_TRADES_PER_REGIME_DEFAULT: int = 10
P_THRESHOLD_DEFAULT: float = 0.05
MAX_FAILING_REGIMES_DEFAULT: int = 1


def regime_subcut_metrics(
    *,
    gross_bps_per_trade: np.ndarray,
    cost_per_trade_bps: float,
    regime_flags: dict[str, np.ndarray],
    min_trades_per_regime: int = MIN_TRADES_PER_REGIME_DEFAULT,
    p_threshold: float = P_THRESHOLD_DEFAULT,
    max_failing: int = MAX_FAILING_REGIMES_DEFAULT,
) -> dict[str, float]:
    """One-sided t-test per regime against zero mean net PnL.

    Inputs:

    * ``gross_bps_per_trade`` — shape (n_traded,) gross PnL in bps for
      each TRADED row (caller has already filtered out non-traded
      bars). Net PnL = gross − cost_per_trade_bps.
    * ``cost_per_trade_bps`` — scalar (typically the ``realistic``
      regime's fee + slippage + funding total). MN2: we test against
      realistic costs only — the binding-regime choice per blueprint.
      Conservative/adversarial would surface MORE failures.
    * ``regime_flags`` — dict ``{regime_name: boolean array shape
      (n_traded,)}`` indicating which trades are in each regime.
      Regimes are NOT mutually exclusive — a trade can be in both
      ``high_vol`` and ``bull`` simultaneously.
    * ``min_trades_per_regime`` — skip the t-test (regime counted as
      not-failing) below this threshold.
    * ``p_threshold`` — one-sided t-test significance level.
    * ``max_failing`` — gate 6 passes if ≤ this many regimes fail.

    Returns flat metrics dict with per-regime stats + ``gate_6_pass`` +
    ``gate_6_caveat``. Keys are prefixed ``regime_*``.

    ``regime_gate_6_caveat`` (MR2) is 1.0 when any regime had fewer
    than ``min_trades_per_regime`` trades — a "passed but on
    incomplete evidence" flag the gauntlet aggregator should require
    to be 0 for a binding pass.
    """
    net_pnl = np.asarray(gross_bps_per_trade, dtype="float64") - float(cost_per_trade_bps)
    out: dict[str, float] = {}
    n_failing = 0
    n_regimes_checked = 0
    n_regimes_caveated = 0   # regime had n < min_trades (incl. zero)
    for regime_name, flag in regime_flags.items():
        flag = np.asarray(flag, dtype=bool)
        if flag.shape != net_pnl.shape:
            raise ValueError(
                f"regime_flags[{regime_name!r}] shape {flag.shape} != "
                f"gross_bps_per_trade shape {net_pnl.shape}"
            )
        in_regime = net_pnl[flag]
        n = int(len(in_regime))
        out[f"regime_{regime_name}_n_trades"] = float(n)
        if n == 0:
            out[f"regime_{regime_name}_mean_net_pnl_bps"] = float("nan")
            out[f"regime_{regime_name}_p_value"] = float("nan")
            out[f"regime_{regime_name}_failing"] = 0.0
            n_regimes_caveated += 1   # empty regime — incomplete evidence
            continue
        mean_net = float(in_regime.mean())
        out[f"regime_{regime_name}_mean_net_pnl_bps"] = mean_net
        if n < min_trades_per_regime:
            # Skip t-test — not enough trades for a reliable check.
            # Blueprint § 7.4 calls this "skip with caveat"; we don't
            # count it as failing.
            out[f"regime_{regime_name}_p_value"] = float("nan")
            out[f"regime_{regime_name}_failing"] = 0.0
            n_regimes_caveated += 1
            continue
        # MR1 fix: degenerate-variance guard. ttest_1samp with std==0
        # returns pvalue=NaN (with a RuntimeWarning); NaN < threshold
        # is False, which would silently mark this regime as
        # not-failing — wrong for "every trade exited at horizon and
        # paid costs" (mean=-cost, std=0). Decide explicitly: mean<0
        # → failing with p=0.0; mean>=0 → not-failing with p=1.0.
        # Both branches bypass scipy entirely.
        sample_std = float(in_regime.std(ddof=1)) if n > 1 else 0.0
        if sample_std == 0.0:
            if mean_net < 0.0:
                p = 0.0
                failing = True
            else:
                p = 1.0
                failing = False
        else:
            # One-sided t-test: H0 mean ≥ 0, H1 mean < 0.
            result = ttest_1samp(in_regime, popmean=0.0, alternative="less")
            p = float(result.pvalue)
            failing = p < p_threshold
        out[f"regime_{regime_name}_p_value"] = p
        n_regimes_checked += 1
        out[f"regime_{regime_name}_failing"] = 1.0 if failing else 0.0
        if failing:
            n_failing += 1
            logger.warning(
                "regime sub-cut FAIL | regime=%s n_trades=%d mean=%.2f bps p=%.4f",
                regime_name, n, mean_net, p,
            )

    out["regime_n_regimes_checked"] = float(n_regimes_checked)
    out["regime_n_failing"] = float(n_failing)
    out["regime_gate_6_pass"] = 1.0 if n_failing <= max_failing else 0.0
    # MR2: caveat surfaces "passed only because some regime was
    # untestable" — caller (gauntlet aggregator) decides whether
    # caveated passes are binding.
    out["regime_gate_6_caveat"] = 1.0 if n_regimes_caveated > 0 else 0.0
    return out


def classify_trend_regimes_from_train_quantiles(
    trend_train: np.ndarray,
    trend_eval: np.ndarray,
    *,
    bear_q: float = 1.0 / 3.0,
    bull_q: float = 2.0 / 3.0,
) -> dict[str, np.ndarray]:
    """Phase 2: classify ``trend_eval`` rows into ``bear`` / ``chop`` /
    ``bull`` using quantile thresholds learned from ``trend_train``.

    PIT-safe: thresholds come from training data only. The three
    buckets are mutually exclusive AND collectively exhaustive over
    finite values — every bar lands in exactly one regime.

    The "trend" feature is caller-chosen — typically a short-window
    log return (``btc_log_return_24h``) or a longer-window momentum
    feature (``btc_momentum_30d`` if registered). The classifier
    doesn't care about the semantics; it just partitions on quantiles.

    Default quantiles are tertiles (1/3, 2/3) so the three buckets have
    comparable sample size by construction. Vol's classifier uses
    25/75 because vol distributions are right-skewed and the tails
    carry more informational value than the median — trend distributions
    on log returns are closer to symmetric so tertiles are the natural
    cut.

    Returns a dict with three boolean arrays the same length as
    ``trend_eval``:

    * ``bear``: ``trend_eval < q_bear``
    * ``chop``: ``q_bear <= trend_eval < q_bull``
    * ``bull``: ``trend_eval >= q_bull``

    See ``classify_vol_regimes_from_train_quantiles`` for the
    methodological caveats around NaN handling (same pattern).
    """
    if not (0.0 < bear_q < bull_q < 1.0):
        raise ValueError(
            f"quantiles must satisfy 0 < bear_q < bull_q < 1; "
            f"got bear_q={bear_q!r} bull_q={bull_q!r}"
        )
    trend_train = np.asarray(trend_train, dtype="float64")
    trend_eval = np.asarray(trend_eval, dtype="float64")
    train_finite = trend_train[np.isfinite(trend_train)]
    if len(train_finite) == 0:
        raise ValueError(
            "trend_train has no finite values; cannot compute thresholds"
        )
    q_bear = float(np.quantile(train_finite, bear_q))
    q_bull = float(np.quantile(train_finite, bull_q))
    # Strict-vs-inclusive: bear is exclusive lower, chop is
    # [q_bear, q_bull) half-open, bull is inclusive at q_bull. This
    # makes the three buckets exhaustive over finite values without
    # double-counting any bar at exactly the quantile.
    bear = trend_eval < q_bear
    bull = trend_eval >= q_bull
    chop = (~bear) & (~bull) & np.isfinite(trend_eval)
    return {
        "bear": bear,
        "chop": chop,
        "bull": bull,
    }


def classify_vol_regimes_from_train_quantiles(
    vol_train: np.ndarray,
    vol_eval: np.ndarray,
    *,
    low_q: float = 0.25,
    high_q: float = 0.75,
) -> dict[str, np.ndarray]:
    """Classify ``vol_eval`` rows into ``high_vol`` / ``low_vol`` using
    quantile thresholds learned from ``vol_train``.

    PIT-safe: thresholds come from training data only, so the eval
    classification doesn't look at future quantiles. Returns a dict
    with two boolean arrays the same length as ``vol_eval``.

    Bars in the middle (``[low_q, high_q]`` quantile range) are NEITHER
    high_vol nor low_vol — they're "mid_vol" by exclusion. We don't
    expose mid_vol explicitly because the t-test is most informative
    in the tail buckets.

    MN1 note: NaN values in ``vol_eval`` silently fall into NEITHER
    bucket (numpy's ``nan < x`` and ``nan > x`` both evaluate False).
    Upstream ``loader.py`` strips rows with NaN features before fit,
    so this is theoretical for the directional pipeline, but a future
    caller that feeds NaN-containing eval volume would see those rows
    excluded from regime sub-cuts without warning.
    """
    vol_train = np.asarray(vol_train, dtype="float64")
    vol_eval = np.asarray(vol_eval, dtype="float64")
    # Drop NaN from train when computing quantiles — quantile would
    # propagate NaN otherwise and break the threshold.
    train_finite = vol_train[np.isfinite(vol_train)]
    if len(train_finite) == 0:
        raise ValueError("vol_train has no finite values; cannot compute thresholds")
    q_low = float(np.quantile(train_finite, low_q))
    q_high = float(np.quantile(train_finite, high_q))
    return {
        "low_vol": vol_eval < q_low,
        "high_vol": vol_eval > q_high,
    }
