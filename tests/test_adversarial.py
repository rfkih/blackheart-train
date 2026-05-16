"""Unit tests for the M5g.6 adversarial-validation helper.

Pure-function tests on synthetic feature matrices. Each call fits a
small LightGBM 5-fold CV; the whole file runs in ~5-10 seconds.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from blackheart_train.adversarial import adversarial_auc


# ── Discrimination semantics ──────────────────────────────────────────────


def test_adversarial_auc_near_random_when_train_test_same_distribution():
    """If X_train and X_test come from the same distribution, the
    adversarial classifier can't distinguish them — AUC near 0.5."""
    rng = np.random.default_rng(42)
    n = 800
    X = pd.DataFrame({
        "a": rng.standard_normal(n),
        "b": rng.standard_normal(n),
        "c": rng.standard_normal(n),
    })
    X_train = X.iloc[: n // 2].copy()
    X_test = X.iloc[n // 2 :].copy()
    auc = adversarial_auc(X_train, X_test, random_state=0)
    # Random ≈ 0.5; allow a generous band for sampling noise across
    # 5 inner folds.
    assert 0.40 < auc < 0.60


def test_adversarial_auc_high_when_train_test_distributions_differ():
    """If train and test have shifted means, the classifier picks up
    the difference — AUC well above 0.6 (the gauntlet's gate-4 threshold)."""
    rng = np.random.default_rng(7)
    n = 400
    X_train = pd.DataFrame({
        "a": rng.normal(0.0, 1.0, n),
        "b": rng.normal(0.0, 1.0, n),
    })
    X_test = pd.DataFrame({
        "a": rng.normal(3.0, 1.0, n),    # mean-shifted
        "b": rng.normal(3.0, 1.0, n),
    })
    auc = adversarial_auc(X_train, X_test, random_state=0)
    assert auc > 0.9


def test_adversarial_auc_deterministic_under_same_seed():
    """Same inputs + same random_state → same AUC. Two runs back-to-back."""
    rng = np.random.default_rng(11)
    X_train = pd.DataFrame(rng.standard_normal((300, 4)),
                            columns=["a", "b", "c", "d"])
    X_test = pd.DataFrame(rng.standard_normal((300, 4)),
                           columns=["a", "b", "c", "d"])
    a = adversarial_auc(X_train, X_test, random_state=42)
    b = adversarial_auc(X_train, X_test, random_state=42)
    assert a == b


# ── interval_indicator handling ──────────────────────────────────────────


def test_adversarial_drops_interval_indicator_before_fitting():
    """The stacked-interval ``interval_indicator`` column would let
    the classifier trivially split train (interval=1) vs test
    (interval=2). Adversarial validation must drop this column so the
    AUC reflects real-feature covariate shift, not bookkeeping.

    Setup: two independent same-distribution feature draws (a/b/c),
    plus an interval_indicator column. If the column is kept, AUC
    would be 1.0 (trivially separable). If dropped, AUC should be
    well below 0.6 — the gauntlet's gate-4 threshold.
    """
    rng = np.random.default_rng(33)
    n = 400
    X_train = pd.DataFrame({
        "a": rng.standard_normal(n),
        "b": rng.standard_normal(n),
        "c": rng.standard_normal(n),
        "interval_indicator": np.full(n, 1, dtype=int),
    })
    X_test = pd.DataFrame({
        "a": rng.standard_normal(n),
        "b": rng.standard_normal(n),
        "c": rng.standard_normal(n),
        "interval_indicator": np.full(n, 2, dtype=int),
    })
    auc = adversarial_auc(X_train, X_test, random_state=0)
    # Should be well below the 0.6 gauntlet threshold. Independent
    # same-dist draws can drift modestly via small-sample noise.
    assert auc < 0.6, f"interval_indicator must be dropped, but AUC={auc:.4f}"


# ── Edge cases ────────────────────────────────────────────────────────────


def test_adversarial_returns_nan_on_empty_train():
    auc = adversarial_auc(
        pd.DataFrame(columns=["a"]),
        pd.DataFrame({"a": [1.0, 2.0, 3.0]}),
    )
    assert math.isnan(auc)


def test_adversarial_returns_nan_on_empty_test():
    auc = adversarial_auc(
        pd.DataFrame({"a": [1.0, 2.0, 3.0]}),
        pd.DataFrame(columns=["a"]),
    )
    assert math.isnan(auc)


def test_adversarial_returns_nan_when_all_features_dropped():
    """A stacked dataset with ONLY ``interval_indicator`` as a column
    (no real features) leaves nothing to discriminate on after the
    column is dropped."""
    X_train = pd.DataFrame({"interval_indicator": np.full(50, 1, dtype=int)})
    X_test = pd.DataFrame({"interval_indicator": np.full(50, 2, dtype=int)})
    auc = adversarial_auc(X_train, X_test, random_state=0)
    assert math.isnan(auc)
