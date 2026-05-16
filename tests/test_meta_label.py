"""Unit tests for the M5g.4 meta-label module.

Pure-function tests on synthetic data. Each test fits a small
LogisticRegressionCV (saga, multiclass=False, ~6 features) and
finishes in <2s.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from blackheart_train.meta_label import (
    META_LABEL_CONFIDENCE_THRESHOLD,
    META_LABEL_FEATURE_COLUMNS,
    META_LABEL_TRAIN_FRACTION,
    MetaLabel,
    build_meta_features,
    build_meta_target,
    fit_meta_label,
    gating_metrics,
    predict_meta_pwin,
)


# ── Constants ─────────────────────────────────────────────────────────────


def test_threshold_and_split_constants_are_pinned():
    """Operator-controlled knobs — pin so accidental edits surface."""
    assert META_LABEL_CONFIDENCE_THRESHOLD == 0.55
    assert META_LABEL_TRAIN_FRACTION == 0.20
    # Canonical feature ordering — scaler depends on this.
    assert META_LABEL_FEATURE_COLUMNS == (
        "confidence", "spread", "disagreement",
        "direction_0", "direction_1", "direction_2",
    )


# ── build_meta_features ───────────────────────────────────────────────────


def _make_probas_stack(n_models: int, n_rows: int, n_classes: int, seed: int = 0):
    """Synthetic per-base-model proba stack. Each row is a Dirichlet
    sample so rows sum to 1; per-model variation gives non-zero
    disagreement."""
    rng = np.random.default_rng(seed)
    alpha = np.ones(n_classes) * 0.5
    return np.stack([
        rng.dirichlet(alpha, size=n_rows) for _ in range(n_models)
    ], axis=0)


def test_build_meta_features_shape_and_columns():
    stack = _make_probas_stack(3, 100, 3, seed=1)
    X = build_meta_features(stack)
    assert X.shape == (100, 6)
    assert list(X.columns) == list(META_LABEL_FEATURE_COLUMNS)


def test_build_meta_features_confidence_matches_max_mean_proba():
    stack = _make_probas_stack(3, 50, 3, seed=2)
    X = build_meta_features(stack)
    mean_proba = stack.mean(axis=0)
    np.testing.assert_allclose(X["confidence"].to_numpy(), mean_proba.max(axis=1))


def test_build_meta_features_direction_is_one_hot():
    stack = _make_probas_stack(3, 50, 3, seed=3)
    X = build_meta_features(stack)
    one_hot = X[["direction_0", "direction_1", "direction_2"]].to_numpy()
    # Exactly one 1 per row, rest zeros
    assert np.all(one_hot.sum(axis=1) == 1.0)
    assert set(np.unique(one_hot)) == {0.0, 1.0}
    # The 1 is in the argmax column
    mean_proba = stack.mean(axis=0)
    assert np.all(one_hot.argmax(axis=1) == mean_proba.argmax(axis=1))


def test_build_meta_features_disagreement_zero_when_models_agree():
    """If all models predict the same proba, disagreement is ~0."""
    base = np.array([[0.7, 0.2, 0.1]] * 20)
    stack = np.stack([base, base, base], axis=0)
    X = build_meta_features(stack)
    np.testing.assert_allclose(X["disagreement"].to_numpy(), 0.0, atol=1e-12)


def test_build_meta_features_rejects_bad_shape():
    """A 2D probas_stack (one model, n × C) is not the contract — must
    fail clearly so a future caller can't slip past with the wrong shape."""
    bad = np.ones((50, 3))   # missing the model axis
    with pytest.raises(ValueError, match="must be"):
        build_meta_features(bad)


# ── build_meta_target ─────────────────────────────────────────────────────


def test_build_meta_target_is_argmax_correct():
    stack = _make_probas_stack(3, 50, 3, seed=4)
    mean_proba = stack.mean(axis=0)
    pred = mean_proba.argmax(axis=1)
    # Construct y_true with half matches, half mismatches
    y = np.where(np.arange(50) % 2 == 0, pred, (pred + 1) % 3)
    target = build_meta_target(stack, y)
    np.testing.assert_array_equal(target, np.arange(50) % 2 == 0)


def test_build_meta_target_returns_int_array():
    stack = _make_probas_stack(3, 10, 3, seed=5)
    target = build_meta_target(stack, np.zeros(10, dtype=int))
    assert target.dtype.kind == "i"
    assert set(np.unique(target)) <= {0, 1}


# ── fit_meta_label / predict_meta_pwin ────────────────────────────────────


def test_fit_meta_label_learns_separable_signal():
    """Construct meta-features where confidence directly predicts win.
    The meta-label should recover this relationship — predict_pwin
    correlates with confidence on a fresh sample."""
    rng = np.random.default_rng(11)
    n = 500
    confidence = rng.uniform(0.4, 0.9, size=n)
    # Win iff confidence > 0.65, with mild noise
    y_win = (confidence > 0.65).astype(int)
    flip_mask = rng.uniform(size=n) < 0.1
    y_win = np.where(flip_mask, 1 - y_win, y_win)
    X = pd.DataFrame({
        "confidence": confidence,
        "spread": rng.uniform(0.1, 0.8, size=n),
        "disagreement": rng.uniform(0.0, 0.3, size=n),
        "direction_0": rng.integers(0, 2, size=n).astype(float),
        "direction_1": rng.integers(0, 2, size=n).astype(float),
        "direction_2": rng.integers(0, 2, size=n).astype(float),
    })
    meta = fit_meta_label(X, y_win, n_classes=3, random_state=0)
    assert isinstance(meta, MetaLabel)
    assert meta.n_classes == 3
    pwin = predict_meta_pwin(meta, X)
    assert pwin.shape == (n,)
    # P(win) should monotonically track confidence — high-confidence
    # rows score higher than low-confidence on average.
    high_conf = pwin[confidence > 0.75].mean()
    low_conf = pwin[confidence < 0.5].mean()
    assert high_conf > low_conf + 0.1


def test_fit_meta_label_raises_on_single_class():
    """If the meta-train target is single-class (all wins or all losses),
    LogisticRegressionCV can't fit. We surface ValueError so the
    training loop can decide to disable gating for the fold gracefully."""
    rng = np.random.default_rng(7)
    n = 100
    X = pd.DataFrame({
        col: rng.standard_normal(n).astype(float)
        for col in META_LABEL_FEATURE_COLUMNS
    })
    y_all_win = np.ones(n, dtype=int)
    with pytest.raises(ValueError, match="only one class"):
        fit_meta_label(X, y_all_win, n_classes=3, random_state=0)


# ── gating_metrics ────────────────────────────────────────────────────────


def test_gating_metrics_uplift_when_meta_label_filters_misses():
    """Construct a case where the meta-label perfectly identifies
    wins. Gated accuracy should be 1.0 and uplift should be strictly
    positive (vs ungated)."""
    n = 100
    primary_pred = np.random.default_rng(0).integers(0, 3, size=n)
    y_true = np.random.default_rng(1).integers(0, 3, size=n)
    correct = (primary_pred == y_true).astype(int)
    # Perfect oracle meta-label: P(win) = 1 if correct, 0 otherwise
    meta_pwin = correct.astype(float)
    m = gating_metrics(
        primary_pred=primary_pred, meta_pwin=meta_pwin, y_true=y_true,
    )
    assert m["gated_accuracy"] == 1.0
    assert m["gated_accuracy_uplift"] > 0.0
    # Selectivity = fraction of correct predictions
    assert m["gated_selectivity"] == correct.mean()
    assert m["gated_n_kept"] == correct.sum()
    assert m["gated_n_total"] == n


def test_gating_metrics_nan_when_nothing_kept():
    """If every meta_pwin is below threshold, gated metrics are NaN
    (no kept rows to average over). MR3: the ungated baseline lives
    in the upstream metrics dict under ``accuracy``; ``gating_metrics``
    no longer duplicates it."""
    n = 50
    primary_pred = np.zeros(n, dtype=int)
    y_true = np.zeros(n, dtype=int)
    meta_pwin = np.zeros(n)   # all below 0.55 threshold
    m = gating_metrics(
        primary_pred=primary_pred, meta_pwin=meta_pwin, y_true=y_true,
    )
    assert m["gated_n_kept"] == 0.0
    assert math.isnan(m["gated_accuracy"])
    assert math.isnan(m["gated_accuracy_uplift"])
    # MR3 fix: ungated_accuracy is NOT returned — bit-equal to the
    # upstream _evaluate's ``accuracy`` key. Caller reads ``accuracy``
    # from the merged metrics dict.
    assert "ungated_accuracy" not in m


def test_gating_metrics_threshold_is_strict():
    """Threshold is strict >, not >= — a prediction exactly at the
    threshold is filtered out. Pin to keep the contract honest."""
    primary_pred = np.array([0, 1])
    y_true = np.array([0, 1])
    meta_pwin = np.array([0.55, 0.56])
    m = gating_metrics(
        primary_pred=primary_pred, meta_pwin=meta_pwin, y_true=y_true,
        threshold=0.55,
    )
    # Only the 0.56 row is kept
    assert m["gated_n_kept"] == 1.0


# ── Round-trip: fit → predict on the same features ──────────────────────


def test_predict_meta_pwin_rejects_scrambled_columns():
    """MR1 fix: passing X_meta with columns in a different order than
    fit_meta_label saw raises a clear error rather than silently
    scaling the wrong columns via the scaler's positional state."""
    stack = _make_probas_stack(3, 200, 3, seed=12)
    X = build_meta_features(stack)
    rng = np.random.default_rng(13)
    y_win = rng.integers(0, 2, size=200)
    meta = fit_meta_label(X, y_win, n_classes=3, random_state=0)

    # Reverse the column ordering — predict must refuse.
    X_scrambled = X[list(X.columns)[::-1]]
    with pytest.raises(ValueError, match="do not match"):
        predict_meta_pwin(meta, X_scrambled)


def test_predict_meta_pwin_accepts_same_canonical_ordering():
    """Sanity check: a freshly-built X_meta from a different probas
    stack still matches the canonical ordering, so predict accepts."""
    stack_fit = _make_probas_stack(3, 200, 3, seed=14)
    X_fit = build_meta_features(stack_fit)
    rng = np.random.default_rng(15)
    y_win = rng.integers(0, 2, size=200)
    meta = fit_meta_label(X_fit, y_win, n_classes=3, random_state=0)

    stack_predict = _make_probas_stack(3, 100, 3, seed=16)
    X_predict = build_meta_features(stack_predict)
    pwin = predict_meta_pwin(meta, X_predict)
    assert pwin.shape == (100,)


def test_fit_then_predict_pwin_is_pinned_to_canonical_columns():
    """The scaler captures the feature ordering at fit time —
    predict_meta_pwin must hand it columns in the same order or the
    scaling becomes meaningless. build_meta_features guarantees this;
    re-confirm by asserting the column ordering of a freshly-built
    feature frame matches META_LABEL_FEATURE_COLUMNS."""
    stack = _make_probas_stack(3, 200, 3, seed=42)
    X = build_meta_features(stack)
    # Construct a target with both classes present
    rng = np.random.default_rng(99)
    y_win = rng.integers(0, 2, size=200)
    meta = fit_meta_label(X, y_win, n_classes=3, random_state=0)
    pwin = predict_meta_pwin(meta, X)
    assert pwin.shape == (200,)
    assert (pwin >= 0).all() and (pwin <= 1).all()
    # The order matters — passing the columns reversed should produce
    # a different (and meaningless) result. We don't assert the value
    # but we DO assert the column tuple matches the canonical ordering.
    assert list(X.columns) == list(META_LABEL_FEATURE_COLUMNS)
