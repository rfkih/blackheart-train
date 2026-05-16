"""Unit tests for the M5g.10 Deflated Sharpe Ratio module.

Tests cover: SR* monotonicity in N, PSR's null behavior, sample-size
sensitivity, degenerate-variance branch, denominator clamp, gate-7
pass/fail edges.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from blackheart_train.dsr import (
    DSR_GATE_THRESHOLD_DEFAULT,
    EULER_MASCHERONI,
    compute_dsr,
    expected_max_sharpe_under_null,
    probabilistic_sharpe_ratio,
)


# ── Pinned constants ──────────────────────────────────────────────────────


def test_blueprint_threshold_and_constants_pinned():
    """Operator-controlled knobs — pin so accidental edits surface."""
    assert DSR_GATE_THRESHOLD_DEFAULT == 0.95
    assert EULER_MASCHERONI == pytest.approx(0.5772156649, abs=1e-9)


# ── expected_max_sharpe_under_null (SR*) ──────────────────────────────────


def test_sr_star_zero_when_n_trials_is_one():
    """Single trial → no trial-discount → SR* = 0."""
    assert expected_max_sharpe_under_null(1) == 0.0


def test_sr_star_zero_when_variance_is_zero():
    """V=0 means Sharpe estimates don't vary at all under the null →
    nothing to maximize over → SR* = 0."""
    assert expected_max_sharpe_under_null(100, sr_variance=0.0) == 0.0


def test_sr_star_monotonic_in_n_trials():
    """More trials → higher expected max under the null → larger SR*.
    Strictly increasing for N >= 2."""
    sr_stars = [expected_max_sharpe_under_null(n) for n in (2, 5, 10, 100, 1000)]
    for prev, nxt in zip(sr_stars, sr_stars[1:]):
        assert nxt > prev


def test_sr_star_scales_with_sqrt_variance():
    """SR* is proportional to √V."""
    base = expected_max_sharpe_under_null(50, sr_variance=1.0)
    quadruple = expected_max_sharpe_under_null(50, sr_variance=4.0)
    # √4 = 2 → quadruple should be 2× base (within floating-point tolerance)
    assert quadruple == pytest.approx(2.0 * base, rel=1e-9)


def test_sr_star_rejects_negative_n_or_variance():
    with pytest.raises(ValueError, match="n_trials"):
        expected_max_sharpe_under_null(0)
    with pytest.raises(ValueError, match="n_trials"):
        expected_max_sharpe_under_null(-1)
    with pytest.raises(ValueError, match="sr_variance"):
        expected_max_sharpe_under_null(10, sr_variance=-0.1)


# ── probabilistic_sharpe_ratio (PSR) ──────────────────────────────────────


def test_psr_equals_half_when_observed_equals_star():
    """If observed Sharpe exactly matches SR*, the strategy has 50%
    probability of being better than the benchmark (normal CDF at 0)."""
    psr = probabilistic_sharpe_ratio(
        sr_observed=0.5, sr_star=0.5,
        n_returns=100, skewness=0.0, excess_kurtosis=0.0,
    )
    assert psr == pytest.approx(0.5, abs=1e-9)


def test_psr_increases_with_observed_sharpe():
    """Holding SR* fixed, higher observed SR → higher PSR."""
    common = dict(sr_star=0.0, n_returns=200, skewness=0.0, excess_kurtosis=0.0)
    p1 = probabilistic_sharpe_ratio(sr_observed=0.1, **common)
    p2 = probabilistic_sharpe_ratio(sr_observed=0.5, **common)
    p3 = probabilistic_sharpe_ratio(sr_observed=1.0, **common)
    assert p1 < p2 < p3


def test_psr_decreases_with_sr_star():
    """Holding observed SR fixed, higher benchmark SR* → lower PSR
    (harder to beat a higher hurdle)."""
    common = dict(sr_observed=0.5, n_returns=200, skewness=0.0, excess_kurtosis=0.0)
    p_low_star = probabilistic_sharpe_ratio(sr_star=0.0, **common)
    p_high_star = probabilistic_sharpe_ratio(sr_star=0.4, **common)
    assert p_low_star > p_high_star


def test_psr_more_certain_with_more_returns():
    """For SR_obs > SR*, more samples → higher PSR (tighter sample-
    Sharpe distribution)."""
    common = dict(sr_observed=0.3, sr_star=0.0, skewness=0.0, excess_kurtosis=0.0)
    p_short = probabilistic_sharpe_ratio(n_returns=50, **common)
    p_long = probabilistic_sharpe_ratio(n_returns=500, **common)
    assert p_long > p_short


def test_psr_negative_skew_penalizes_high_sharpe():
    """Negative skew INCREASES the PSR denominator (term:
    1 − γ_skew·SR_obs with negative γ_skew and positive SR_obs → 1+).
    Larger denominator → smaller z → lower PSR. Skewed-left tail risk
    properly punishes apparent Sharpe."""
    common = dict(sr_observed=0.5, sr_star=0.0, n_returns=200, excess_kurtosis=0.0)
    p_symmetric = probabilistic_sharpe_ratio(skewness=0.0, **common)
    p_neg_skew = probabilistic_sharpe_ratio(skewness=-1.0, **common)
    assert p_neg_skew < p_symmetric


def test_psr_high_kurtosis_penalizes():
    """Fat tails → larger denominator term ((γ_kurt/4)·SR²) → lower PSR."""
    common = dict(sr_observed=0.5, sr_star=0.0, n_returns=200, skewness=0.0)
    p_normal = probabilistic_sharpe_ratio(excess_kurtosis=0.0, **common)
    p_fat = probabilistic_sharpe_ratio(excess_kurtosis=5.0, **common)
    assert p_fat < p_normal


def test_psr_clamps_extreme_denominator_without_returning_nan():
    """MR-C: combination of large |SR_obs| and large skew can drive
    the denominator negative. We clamp + warn rather than return NaN."""
    # γ_skew=10, SR_obs=2 → 1 - 10*2 + 0 = -19 (very negative)
    psr = probabilistic_sharpe_ratio(
        sr_observed=2.0, sr_star=0.0,
        n_returns=100, skewness=10.0, excess_kurtosis=0.0,
    )
    assert math.isfinite(psr)
    assert 0.0 <= psr <= 1.0


def test_psr_rejects_too_few_returns():
    with pytest.raises(ValueError, match="at least 2 returns"):
        probabilistic_sharpe_ratio(
            sr_observed=0.5, sr_star=0.0,
            n_returns=1, skewness=0.0, excess_kurtosis=0.0,
        )


# ── MR-DSR1: PSR formula closed-form pinning ──────────────────────────────


def test_psr_matches_mertens_closed_form_for_normal_returns():
    """MR-DSR1: pin the PSR variance term against Mertens 2002 /
    Lo 2002. For Normal returns (skew=0, γ_4=3 → excess_kurt=0), the
    sample-Sharpe variance is (1 + 0.5·SR²) / (T − 1) — the textbook
    formula. PSR should match Φ((SR − SR*)·√(T-1) / √(1 + 0.5·SR²)).

    The pre-fix code used ``excess_kurtosis / 4`` (= 0 for Normal),
    yielding denom = 1 instead of 1.125 at SR=0.5 — overstating PSR.
    """
    import math
    from scipy.stats import norm
    sr_obs = 0.5
    sr_star = 0.0
    T = 100
    psr = probabilistic_sharpe_ratio(
        sr_observed=sr_obs, sr_star=sr_star,
        n_returns=T, skewness=0.0, excess_kurtosis=0.0,
    )
    # Closed form for Normal returns (Mertens variance).
    expected_var = 1.0 + 0.5 * sr_obs ** 2
    expected_psr = norm.cdf((sr_obs - sr_star) * math.sqrt(T - 1) / math.sqrt(expected_var))
    assert psr == pytest.approx(expected_psr, abs=1e-12)


def test_psr_pearson_kurtosis_convention_pinned():
    """MR-DSR1 belt-and-suspenders: pin the formula coefficient on
    excess_kurtosis to ``(excess_kurt + 2)/4`` (NOT ``excess_kurt/4``).
    Computes PSR with non-trivial skew + excess_kurt and matches
    against the hand-derived expected value."""
    import math
    from scipy.stats import norm
    sr_obs = 0.83
    sr_star = 0.85
    T = 24
    sk = -2.448
    ek = 7.164    # Pearson γ_4 = 10.164, excess = 7.164
    psr = probabilistic_sharpe_ratio(
        sr_observed=sr_obs, sr_star=sr_star,
        n_returns=T, skewness=sk, excess_kurtosis=ek,
    )
    # Hand-computed using the correct (LdP 2014 eq. 14) formula.
    expected_denom_sq = 1.0 - sk * sr_obs + ((ek + 2.0) / 4.0) * sr_obs ** 2
    expected_z = (sr_obs - sr_star) * math.sqrt(T - 1) / math.sqrt(expected_denom_sq)
    expected_psr = norm.cdf(expected_z)
    assert psr == pytest.approx(expected_psr, abs=1e-12)
    # The OLD (buggy) formula would have used ek/4 not (ek+2)/4 →
    # denom_sq = 4.266 instead of 4.610. PSR ≈ 0.4815 vs 0.4822.
    # Verify the corrected branch gives ≥0.482, not the buggy 0.4815.
    assert psr > 0.4818


# ── compute_dsr (top-level) ───────────────────────────────────────────────


def test_compute_dsr_strong_signal_passes_gate_7():
    """A return series with clear positive Sharpe (mean=+5, std=1,
    T=500) passes gate 7 even with realistic trial discount (N=100)."""
    rng = np.random.default_rng(0)
    returns = rng.normal(loc=5.0, scale=1.0, size=500)
    m = compute_dsr(returns, n_trials=100, sr_variance_across_trials=1.0)
    assert m["dsr_n_returns"] == 500
    assert m["dsr_sr_observed"] > 4.0    # ~5/1 ≈ 5
    assert m["dsr_value"] > 0.95
    assert m["dsr_gate_7_pass"] == 1.0


def test_compute_dsr_pure_noise_fails_gate_7():
    """A zero-mean Gaussian series (no edge) → DSR ≤ 0.95 → gate fails.
    Use the deflated benchmark to amplify: N=100 trials, V=1 → SR*≈2."""
    rng = np.random.default_rng(42)
    returns = rng.normal(loc=0.0, scale=1.0, size=200)
    m = compute_dsr(returns, n_trials=100, sr_variance_across_trials=1.0)
    assert m["dsr_value"] < 0.95
    assert m["dsr_gate_7_pass"] == 0.0


def test_compute_dsr_more_trials_reduces_dsr_at_fixed_sample():
    """Same return series, more trials → larger SR* → lower DSR."""
    rng = np.random.default_rng(7)
    returns = rng.normal(loc=0.3, scale=1.0, size=300)
    m_few = compute_dsr(returns, n_trials=2, sr_variance_across_trials=1.0)
    m_many = compute_dsr(returns, n_trials=1000, sr_variance_across_trials=1.0)
    assert m_few["dsr_sr_star"] < m_many["dsr_sr_star"]
    assert m_few["dsr_value"] > m_many["dsr_value"]


def test_compute_dsr_n_trials_one_collapses_to_psr_zero():
    """N=1 trials → SR* = 0 → DSR = P(true SR > 0). For a strongly
    positive sample with N=1 trial, this is essentially 1.0."""
    rng = np.random.default_rng(11)
    returns = rng.normal(loc=2.0, scale=1.0, size=300)
    m = compute_dsr(returns, n_trials=1, sr_variance_across_trials=1.0)
    assert m["dsr_sr_star"] == 0.0
    assert m["dsr_value"] > 0.99


def test_compute_dsr_empty_returns_bails_without_crashing():
    """Empty or single-trade samples can't yield a Sharpe — return
    minimal dict, no crash."""
    for n in (0, 1):
        m = compute_dsr(np.zeros(n), n_trials=10)
        assert m["dsr_n_returns"] == n
        assert "dsr_value" not in m
        assert "dsr_gate_7_pass" not in m


def test_compute_dsr_degenerate_variance_positive_mean_perfect_pass():
    """MR-B: every return identical and positive → DSR=1.0 (no
    variance, no doubt). Edge case mostly impossible in practice but
    we don't want to return NaN."""
    returns = np.full(50, 3.0)
    m = compute_dsr(returns, n_trials=100)
    assert m["dsr_value"] == 1.0
    assert m["dsr_gate_7_pass"] == 1.0


def test_compute_dsr_degenerate_variance_negative_mean_perfect_fail():
    """MR-B mirror: every return identical and negative → DSR=0.0."""
    returns = np.full(50, -2.0)
    m = compute_dsr(returns, n_trials=100)
    assert m["dsr_value"] == 0.0
    assert m["dsr_gate_7_pass"] == 0.0


def test_compute_dsr_degenerate_variance_zero_mean_fails():
    """MR-B boundary: identical zero returns → mean=0, no edge → fail."""
    returns = np.zeros(50)
    m = compute_dsr(returns, n_trials=100)
    assert m["dsr_value"] == 0.0
    assert m["dsr_gate_7_pass"] == 0.0


def test_compute_dsr_propagates_skew_and_kurtosis_to_payload():
    """Sample stats are observable in the output so the audit log can
    explain a fail."""
    rng = np.random.default_rng(13)
    # Strongly negatively-skewed + heavy-tailed sample (clip + scale)
    base = rng.standard_t(df=3, size=400)
    returns = np.where(base > 1.0, 1.0, base)   # asymmetric clip
    m = compute_dsr(returns, n_trials=10)
    assert "dsr_skewness" in m
    assert "dsr_excess_kurtosis" in m
    assert math.isfinite(m["dsr_skewness"])
    assert math.isfinite(m["dsr_excess_kurtosis"])


def test_compute_dsr_gate_threshold_override():
    """Caller can tighten or loosen the gate. A 0.5 threshold should
    pass a near-noise series that would fail at 0.95."""
    rng = np.random.default_rng(2)
    returns = rng.normal(loc=0.3, scale=1.0, size=200)
    strict = compute_dsr(returns, n_trials=20, gate_threshold=0.95)
    lax = compute_dsr(returns, n_trials=20, gate_threshold=0.30)
    # Same DSR value, different threshold decision.
    assert strict["dsr_value"] == lax["dsr_value"]
    if strict["dsr_value"] < 0.30:
        assert lax["dsr_gate_7_pass"] == 0.0
    elif strict["dsr_value"] > 0.95:
        assert strict["dsr_gate_7_pass"] == 1.0
    else:
        # Most-common path: DSR in (0.30, 0.95) → strict fails, lax passes.
        assert strict["dsr_gate_7_pass"] == 0.0
        assert lax["dsr_gate_7_pass"] == 1.0


# ── MR-DSR3: trial-discount-active flag ───────────────────────────────────


def test_dsr_trial_discount_inactive_when_n_trials_is_one():
    """MR-DSR3: n_trials=1 means no multiple-testing penalty. The
    flag must surface that the gauntlet can't treat this DSR as a
    binding trial-discount pass."""
    returns = np.full(50, 1.0)
    m = compute_dsr(returns, n_trials=1, sr_variance_across_trials=1.0)
    assert m["dsr_trial_discount_active"] == 0.0


def test_dsr_trial_discount_inactive_when_variance_is_zero():
    """MR-DSR3: V=0 → SR*=0 regardless of N. Flag must be 0."""
    returns = np.full(50, 1.0)
    m = compute_dsr(returns, n_trials=100, sr_variance_across_trials=0.0)
    assert m["dsr_trial_discount_active"] == 0.0


def test_dsr_trial_discount_active_when_both_set_meaningfully():
    """MR-DSR3: N≥2 and V>0 → real deflation. Flag fires."""
    rng = np.random.default_rng(31)
    returns = rng.normal(loc=0.3, scale=1.0, size=200)
    m = compute_dsr(returns, n_trials=50, sr_variance_across_trials=1.0)
    assert m["dsr_trial_discount_active"] == 1.0
    assert m["dsr_sr_star"] > 0.0   # actual deflation applied


def test_dsr_trial_discount_flag_propagates_through_empty_samples():
    """Empty / single-trade returns still report the trial config so
    the gauntlet can see the intent even when DSR wasn't computed."""
    m = compute_dsr(np.zeros(0), n_trials=50, sr_variance_across_trials=1.0)
    assert m["dsr_trial_discount_active"] == 1.0
    assert "dsr_value" not in m   # didn't actually compute
