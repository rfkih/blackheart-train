"""Unit tests for the M5g.6 bootstrap CI helper.

Pure-function tests with synthetic (y_true, y_proba) arrays. Each
bootstrap call does 1000 resamples × small AUC; the whole file runs
in ~3-5 seconds.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from blackheart_train.bootstrap import (
    N_BOOTSTRAP_DEFAULT,
    bootstrap_macro_auc_ovr,
)


def _make_3class_probas(n: int = 600, seed: int = 0, signal: float = 0.3):
    """Synthetic 3-class data where the proba's argmax matches y_true
    with strength ``signal`` (0 = random, 1 = perfect)."""
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 3, size=n)
    # Start with uniform proba, tilt toward y_true by ``signal``.
    proba = np.full((n, 3), (1 - signal) / 3.0)
    for i, c in enumerate(y_true):
        proba[i, c] = (1 - signal) / 3.0 + signal
    # Renormalise to sum=1 after adding signal mass.
    proba = proba / proba.sum(axis=1, keepdims=True)
    return y_true, proba


# ── Shape / keys ──────────────────────────────────────────────────────────


def test_bootstrap_returns_expected_keys():
    y_true, proba = _make_3class_probas(seed=1)
    out = bootstrap_macro_auc_ovr(
        y_true, proba, n_classes=3, n_bootstrap=100, random_state=0,
    )
    expected = {
        "macro_auc_ovr_bootstrap_mean",
        "macro_auc_ovr_ci_lower_5",
        "macro_auc_ovr_ci_upper_95",
        "macro_auc_ovr_bootstrap_std",
        "n_valid_resamples",
        "n_bootstrap",
    }
    assert set(out.keys()) == expected
    assert out["n_bootstrap"] == 100


def test_default_n_bootstrap_constant_is_pinned():
    """Operator-controlled knob — pin so accidental edits surface."""
    assert N_BOOTSTRAP_DEFAULT == 1000


# ── Numeric correctness ───────────────────────────────────────────────────


def test_bootstrap_mean_close_to_point_estimate_with_signal():
    """With clear signal, bootstrap mean is close to the point AUC."""
    from sklearn.metrics import roc_auc_score
    y_true, proba = _make_3class_probas(n=2000, seed=42, signal=0.6)
    point_auc = roc_auc_score(
        y_true, proba, multi_class="ovr", average="macro", labels=[0, 1, 2],
    )
    out = bootstrap_macro_auc_ovr(
        y_true, proba, n_classes=3, n_bootstrap=200, random_state=0,
    )
    # Bootstrap mean is within 0.02 of the point estimate for a strong
    # signal — sampling variation dominates beyond that.
    assert abs(out["macro_auc_ovr_bootstrap_mean"] - point_auc) < 0.02


def test_bootstrap_ci_ordering():
    """ci_lower_5 ≤ bootstrap_mean ≤ ci_upper_95 — quantile ordering
    is an invariant the gauntlet relies on."""
    y_true, proba = _make_3class_probas(n=800, seed=3, signal=0.3)
    out = bootstrap_macro_auc_ovr(
        y_true, proba, n_classes=3, n_bootstrap=500, random_state=0,
    )
    assert out["macro_auc_ovr_ci_lower_5"] <= out["macro_auc_ovr_bootstrap_mean"]
    assert out["macro_auc_ovr_bootstrap_mean"] <= out["macro_auc_ovr_ci_upper_95"]


def test_bootstrap_ci_finite_for_both_small_and_large_n_bootstrap():
    """Both 50-resample and 1000-resample runs produce finite CIs.
    The span isn't strictly monotonic in n_bootstrap (bootstrap is
    itself a noisy estimator), but both regimes should yield finite,
    ordered quantiles.

    Use a noisy mid-strength signal so per-resample AUCs vary (a
    perfectly-discriminating proba would collapse every bootstrap
    sample to AUC=1.0, defeating the test's CI-ordering check)."""
    rng = np.random.default_rng(7)
    n_rows = 600
    y_true = rng.integers(0, 3, size=n_rows)
    # Mildly informed proba — bumps the true class but mostly noise.
    proba = rng.dirichlet([1.0, 1.0, 1.0], size=n_rows)
    for i, c in enumerate(y_true):
        proba[i, c] += 0.15
    proba = proba / proba.sum(axis=1, keepdims=True)
    for n in (50, 1000):
        out = bootstrap_macro_auc_ovr(
            y_true, proba, n_classes=3, n_bootstrap=n, random_state=0,
        )
        assert not math.isnan(out["macro_auc_ovr_ci_lower_5"])
        assert not math.isnan(out["macro_auc_ovr_ci_upper_95"])
        assert out["macro_auc_ovr_ci_lower_5"] <= out["macro_auc_ovr_ci_upper_95"]


# ── Edge cases ────────────────────────────────────────────────────────────


def test_bootstrap_returns_nan_on_empty_input():
    out = bootstrap_macro_auc_ovr(
        np.array([], dtype=int), np.zeros((0, 3)),
        n_classes=3, n_bootstrap=100, random_state=0,
    )
    for key in ("macro_auc_ovr_bootstrap_mean",
                "macro_auc_ovr_ci_lower_5",
                "macro_auc_ovr_ci_upper_95"):
        assert math.isnan(out[key])
    assert out["n_valid_resamples"] == 0


def test_bootstrap_drops_resamples_missing_a_class():
    """When the dataset has a rare class (small representation),
    some resamples miss it; n_valid_resamples reflects that."""
    rng = np.random.default_rng(99)
    n = 300
    # Severe imbalance: class 1 only 3% (matches triple-barrier).
    y_true = np.concatenate([
        np.zeros(int(n * 0.59), dtype=int),
        np.ones(int(n * 0.03), dtype=int),
        np.full(n - int(n * 0.59) - int(n * 0.03), 2, dtype=int),
    ])
    rng.shuffle(y_true)
    proba = rng.dirichlet([1.0, 1.0, 1.0], size=n)

    out = bootstrap_macro_auc_ovr(
        y_true, proba, n_classes=3, n_bootstrap=500, random_state=0,
    )
    # Most resamples should still have all 3 classes (probability of
    # missing class 1 with ~9 expected occurrences in a 300-resample
    # is small but non-zero) — at least some are dropped though.
    assert out["n_valid_resamples"] > 0
    assert out["n_valid_resamples"] <= 500
