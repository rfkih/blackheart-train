"""Deflated Sharpe Ratio — gate 7 of the 13-gate gauntlet (blueprint § 7.5).

A strategy that explored 100 hyperparam variants and reports the best one
has an inflated Sharpe: under the null hypothesis of zero true alpha,
the maximum of N noisy Sharpe estimates is *large* by chance alone.
The Deflated Sharpe Ratio (DSR; Bailey & López de Prado 2014) is the
trial-count discount: it asks whether the OBSERVED Sharpe exceeds the
EXPECTED maximum under the null, accounting for the higher moments of
the return distribution.

Why this matters for crypto-perp directional models specifically:

* Triple-barrier returns are heavy-tailed and asymmetric (K_TP≠K_SL),
  so the simple Sharpe ratio's normality assumption is wrong. PSR
  (Probabilistic Sharpe Ratio; same paper) corrects for skew and
  excess kurtosis of the return series.
* Phase 3's design space — choice of K_TP, K_SL, horizon, feature
  set, ensemble weights, meta-label gate, regime classifier — covers
  hundreds of implicit trials. Without a DSR-style discount, every
  binding gate decision is mining-vulnerable.

Definitions (Bailey & López de Prado 2014, eq. 13-14;
Mertens 2002 / Lo 2002 for the variance formula):

  SR_obs = mean(returns) / std(returns)                      # sample Sharpe
  γ_3    = skew(returns)                                      # 3rd standardized moment
  γ_4    = kurt(returns) — PEARSON form, normal = 3           # 4th standardized moment

  σ²(SR) = (1 − γ_3·SR_obs + ((γ_4 − 1)/4)·SR_obs²) / (T − 1)
  PSR(SR*) = Φ((SR_obs − SR*) / σ(SR))

  E[max{SR_n}] ≈ √V · ((1 − γ_em)·Φ⁻¹(1 − 1/N)
                       + γ_em·Φ⁻¹(1 − 1/(N·e)))             # Cramér 1946 approx
  DSR = PSR(SR* = E[max{SR_n}])

For a Normal distribution: γ_3=0, γ_4=3, so the variance term reduces
to ``(1 + 0.5·SR²)/(T − 1)`` — the textbook Mertens formula. The PSR
denominator is what tests for excess skew/kurtosis penalties vs.
the Normal benchmark.

Note on the kurtosis convention: scipy ``kurtosis(..., fisher=True)``
returns the EXCESS form (``γ_4 − 3``, normal = 0). The PSR formula
above uses Pearson ``γ_4``, so we add 3 internally — equivalent to
``((excess_kurt + 2)/4)·SR²`` in the denominator (since
``(γ_4 − 1)/4 = (excess_kurt + 3 − 1)/4 = (excess_kurt + 2)/4``).

where:

* V = variance of Sharpe estimates across the N trials. If unknown,
  V=1 is a defensible normalized default (the SR_n distribution under
  the null with unit variance).
* γ_em ≈ 0.5772156649 (Euler-Mascheroni constant)
* e = 2.7182818 (Euler's number)
* Φ, Φ⁻¹ = standard normal CDF and its inverse

Gate 7 (blueprint § 7.5): **DSR > 0.95** to pass. That's "95%
confidence the strategy's Sharpe exceeds the deflated null benchmark
after accounting for both trial multiplicity and return non-normality."

Caveats baked in here:

* **MR-A.** If ``n_trials=1`` and ``sr_variance=0``, SR* = 0 and DSR
  collapses to plain PSR(SR*=0) = "probability the strategy's true
  Sharpe exceeds zero." Useful as a sanity check but does NO trial
  discounting — caller must pass realistic n_trials and sr_variance
  for the gate to have teeth. The ``dsr_trial_discount_active`` flag
  in the output dict signals whether real deflation happened.
* **MR-B.** Degenerate variance (every return identical) → std=0 →
  SR is +∞ or -∞ or undefined. We detect and short-circuit:
  std=0 ∧ mean>0 → DSR=1.0 (trivially passes); std=0 ∧ mean≤0 →
  DSR=0.0 (trivially fails). Both are "perfect-evidence" outcomes;
  the gauntlet's regime sub-cut already catches the "every trade at
  horizon" case. Note the sentinel ``±inf`` value of
  ``dsr_sr_observed`` JSON-serializes as ``null`` via the CLI's
  ``_sanitize_for_json``; a Python consumer reading the dict
  directly gets ``float('inf')``.
* **MR-C.** PSR's denominator can in principle go non-positive for
  extreme moments + large SR (e.g., very negative skew with large
  positive SR). We clamp to a small positive epsilon
  (``_DENOMINATOR_FLOOR``) and log a warning rather than return NaN.
* **MR-DSR1 (fixed 2026-05-15).** Earlier audit caught the wrong
  kurtosis convention in the PSR denominator: the original code used
  ``excess_kurtosis / 4`` but Bailey-López de Prado 2014 / Mertens
  2002 use ``(γ_4 − 1) / 4`` with γ_4 = Pearson kurtosis. The
  corrected formula is ``(excess_kurtosis + 2) / 4`` — for Normal
  returns (excess=0) the term is 0.5, matching the Mertens variance
  ``1 + 0.5·SR²``. Test
  ``test_psr_matches_ldp_paper_closed_form`` pins this.

Methodological notes (phase-2 follow-ups, not currently bugs):

* **PSR assumes IID returns.** Per-trade returns from a walk-forward
  fold are roughly IID across folds but within a single fold the
  trades share market regime, so adjacent returns are weakly
  correlated. A HAC-corrected variance or block-bootstrap of the
  Sharpe distribution would be the proper fix; we accept the IID
  approximation for phase 1.
* **Per-trade Sharpe, NOT annualized.** ``dsr_sr_observed`` is the
  Sharpe over the trade series, not scaled to annual. A reader
  comparing to "industry annualized Sharpe ≥ 1" benchmarks should
  multiply by ``√(n_trades_per_year)`` before that comparison. The
  gate threshold (0.95) is on the PSR (a probability, unit-free), so
  annualization doesn't matter for the gate itself.
"""
from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
from scipy.stats import kurtosis, norm, skew

logger = logging.getLogger(__name__)


# Blueprint § 7.5 threshold.
DSR_GATE_THRESHOLD_DEFAULT: float = 0.95

# Euler-Mascheroni constant for the SR* approximation (Cramér 1946).
EULER_MASCHERONI: float = 0.5772156649015329

# Denominator floor for PSR formula — see MR-C in the module docstring.
_DENOMINATOR_FLOOR: float = 1e-6


def expected_max_sharpe_under_null(
    n_trials: int,
    sr_variance: float = 1.0,
) -> float:
    """SR* — Cramér 1946 approximation to the expected maximum of N
    i.i.d. Normal(0, V) Sharpe estimates.

    Inputs:

    * ``n_trials`` — N. Number of independent trials (model variants
      evaluated). ``n_trials=1`` → SR* = 0 (no trial discount).
    * ``sr_variance`` — V. Variance of Sharpe estimates across the N
      trials. V=1.0 is the normalized default (LdP convention).

    Returns SR* as a non-negative float. Caller passes this to
    :func:`probabilistic_sharpe_ratio` to compute the DSR.
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1; got {n_trials}")
    if sr_variance < 0.0:
        raise ValueError(f"sr_variance must be non-negative; got {sr_variance}")
    if n_trials == 1:
        return 0.0
    if sr_variance == 0.0:
        return 0.0
    # Cramér's approximation (López de Prado 2014, eq. 4).
    inv_phi_a = norm.ppf(1.0 - 1.0 / n_trials)
    inv_phi_b = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    sr_star = math.sqrt(sr_variance) * (
        (1.0 - EULER_MASCHERONI) * inv_phi_a + EULER_MASCHERONI * inv_phi_b
    )
    # The approximation is for non-negative N≥2; defensively floor.
    return max(0.0, float(sr_star))


def probabilistic_sharpe_ratio(
    sr_observed: float,
    sr_star: float,
    *,
    n_returns: int,
    skewness: float,
    excess_kurtosis: float,
) -> float:
    """PSR(SR*) — probability that the strategy's TRUE Sharpe exceeds
    the benchmark SR*, given the observed Sharpe, return moments,
    and sample size.

    The PSR formula (Bailey & López de Prado 2012, 2014):

      PSR(SR*) = Φ(((SR_obs − SR*) · √(T − 1))
                   / √(1 − γ_skew·SR_obs + (γ_kurt/4)·SR_obs²))

    where Φ is the standard normal CDF. Returns a float in [0, 1].
    """
    if n_returns < 2:
        raise ValueError(
            f"PSR requires at least 2 returns to estimate Sharpe; got "
            f"n_returns={n_returns}. Caller should skip when too few trades."
        )
    numerator = (sr_observed - sr_star) * math.sqrt(n_returns - 1)
    # MR-DSR1 fix: the PSR denominator uses ``(γ_4 − 1) / 4`` where
    # γ_4 is the PEARSON kurtosis (normal = 3), not Fisher excess
    # kurtosis (normal = 0). With ``excess_kurtosis = γ_4 − 3`` the
    # term is ``(excess_kurtosis + 2) / 4``. For Normal returns this
    # yields the textbook Mertens variance ``1 + 0.5·SR²``.
    # MR-C: the denominator can go non-positive for extreme moments
    # combined with very large |SR_obs|. Clamp + warn.
    denom_sq = (
        1.0
        - skewness * sr_observed
        + ((excess_kurtosis + 2.0) / 4.0) * (sr_observed ** 2)
    )
    if denom_sq <= _DENOMINATOR_FLOOR:
        logger.warning(
            "PSR denominator non-positive | sr_obs=%.4f skew=%.4f exc_kurt=%.4f "
            "denom_sq=%.4f → clamping to %.0e",
            sr_observed, skewness, excess_kurtosis, denom_sq, _DENOMINATOR_FLOOR,
        )
        denom_sq = _DENOMINATOR_FLOOR
    z = numerator / math.sqrt(denom_sq)
    return float(norm.cdf(z))


def compute_dsr(
    returns_per_trade: np.ndarray,
    *,
    n_trials: int = 1,
    sr_variance_across_trials: float = 1.0,
    gate_threshold: float = DSR_GATE_THRESHOLD_DEFAULT,
) -> dict[str, float]:
    """Deflated Sharpe Ratio — gauntlet gate 7.

    Inputs:

    * ``returns_per_trade`` — shape (n_trades,) per-trade NET returns
      (in any consistent unit; bps is fine). Caller has already paid
      costs into these numbers.
    * ``n_trials`` — number of independent trial runs that contributed
      to the strategy's design space. Larger → more deflation.
    * ``sr_variance_across_trials`` — V in the SR* formula. V=1.0 is
      the normalized default. Caller with an actual trial registry
      should compute the empirical variance of SR estimates and pass
      that here.
    * ``gate_threshold`` — DSR cutoff for gate 7. Default 0.95 per
      blueprint § 7.5.

    Returns flat metrics dict suitable for the artifact's
    ``payload["metrics"]``. Keys are prefixed ``dsr_*`` so they don't
    collide with cost or regime metrics.

    A fold with <2 trades returns ``dsr_n_returns`` and bails (DSR
    requires sample-Sharpe which needs std). The walk-forward
    aggregator handles missing keys gracefully via union-and-mean.
    """
    returns = np.asarray(returns_per_trade, dtype="float64")
    n = int(returns.shape[0])
    # MR-DSR3: surface whether trial discount actually deflates SR*.
    # ``n_trials=1`` OR ``sr_variance=0`` ⇒ SR* = 0 ⇒ DSR collapses
    # to PSR(SR*=0). Useful as a sanity check but not a binding
    # multiple-testing gate. The gauntlet aggregator (M5h) should
    # demote a caveated pass when ``trial_discount_active == 0``.
    trial_discount_active = float(n_trials > 1 and sr_variance_across_trials > 0.0)
    out: dict[str, Any] = {
        "dsr_n_returns": float(n),
        "dsr_n_trials": float(n_trials),
        "dsr_sr_variance_across_trials": float(sr_variance_across_trials),
        "dsr_trial_discount_active": trial_discount_active,
    }
    if n < 2:
        # Single-trade or empty samples can't yield a Sharpe.
        return out

    mean_r = float(returns.mean())
    std_r = float(returns.std(ddof=1))
    # MR-B: degenerate-variance short-circuit. std=0 means every return
    # is identical; Sharpe is ±∞ or undefined.
    if std_r == 0.0:
        out["dsr_sr_observed"] = float("inf") if mean_r > 0 else float("-inf") if mean_r < 0 else 0.0
        out["dsr_sr_star"] = 0.0
        out["dsr_value"] = 1.0 if mean_r > 0 else 0.0
        out["dsr_skewness"] = 0.0
        out["dsr_excess_kurtosis"] = 0.0
        out["dsr_gate_7_pass"] = 1.0 if out["dsr_value"] > gate_threshold else 0.0
        return out

    sr_obs = mean_r / std_r
    # scipy.stats.kurtosis uses Fisher's definition (excess kurtosis,
    # i.e. K − 3) by default. Matches the LdP DSR paper's γ_kurt term.
    sk = float(skew(returns, bias=False))
    ek = float(kurtosis(returns, fisher=True, bias=False))
    sr_star = expected_max_sharpe_under_null(n_trials, sr_variance_across_trials)
    dsr = probabilistic_sharpe_ratio(
        sr_obs, sr_star, n_returns=n, skewness=sk, excess_kurtosis=ek,
    )
    out["dsr_sr_observed"] = float(sr_obs)
    out["dsr_sr_star"] = float(sr_star)
    out["dsr_skewness"] = sk
    out["dsr_excess_kurtosis"] = ek
    out["dsr_value"] = float(dsr)
    out["dsr_gate_threshold"] = float(gate_threshold)
    out["dsr_gate_7_pass"] = 1.0 if dsr > gate_threshold else 0.0
    if dsr <= gate_threshold:
        logger.warning(
            "DSR FAIL | sr_obs=%.4f sr_star=%.4f n=%d skew=%.3f exc_kurt=%.3f → DSR=%.4f (threshold=%.2f)",
            sr_obs, sr_star, n, sk, ek, dsr, gate_threshold,
        )
    return out
