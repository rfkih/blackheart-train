"""Unit tests for the conditional-invariance transferability metric
(replaces adversarial AUC as gauntlet gate 4 per
``project_v2_adversarial_auc.md``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from blackheart_train.conditional_invariance import (
    DEFAULT_MIN_BIN_SAMPLES,
    DEFAULT_N_BINS,
    PASS_THRESHOLD,
    conditional_invariance,
    passes,
)


def _make_xy(
    n: int = 1000,
    seed: int = 0,
    kind: str = "binary",
    shift: float = 0.0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic (X, y) where ``y = logit(0.8*a + 0.4*b + shift)``.

    ``shift`` lets the test perturb the train→test conditional relation
    so we can verify the metric notices.
    """
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    if kind == "binary":
        logit = 0.8 * a + 0.4 * b + shift
        prob = 1.0 / (1.0 + np.exp(-logit))
        y = pd.Series((rng.uniform(size=n) < prob).astype(int))
    else:
        y = pd.Series(0.8 * a + 0.4 * b + shift + 0.1 * rng.standard_normal(n))
    X = pd.DataFrame({"a": a, "b": b})
    return X, y


# ── Happy path ────────────────────────────────────────────────────────────


def test_binary_no_shift_low_divergence():
    """Same DGP for train and test → max_abs_diff should be small
    (well under the 0.15 threshold for binary)."""
    X_tr, y_tr = _make_xy(2000, seed=1, kind="binary", shift=0.0)
    X_te, y_te = _make_xy(2000, seed=2, kind="binary", shift=0.0)
    r = conditional_invariance(X_tr, y_tr, X_te, y_te, objective="binary")
    assert r.skipped_reason is None
    assert r.max_abs_diff < 0.15, (
        f"expected low divergence on same DGP, got {r.max_abs_diff:.4f}"
    )
    assert r.n_pairs_evaluated > 0
    assert passes(r, "binary") is True


def test_binary_strong_shift_flags_divergence():
    """Train and test from different conditional distributions → the
    metric should pick up a real shift."""
    X_tr, y_tr = _make_xy(2000, seed=3, kind="binary", shift=0.0)
    # Shift the conditional relation: bias toward y=0 in test.
    X_te, y_te = _make_xy(2000, seed=4, kind="binary", shift=-2.0)
    r = conditional_invariance(X_tr, y_tr, X_te, y_te, objective="binary")
    assert r.skipped_reason is None
    # A shift this large (logit -2) should flip many bins by > 15pp.
    assert r.max_abs_diff > 0.15, (
        f"expected high divergence on shifted test, got {r.max_abs_diff:.4f}"
    )
    assert passes(r, "binary") is False


def test_regression_no_shift_low_divergence():
    X_tr, y_tr = _make_xy(2000, seed=5, kind="regression", shift=0.0)
    X_te, y_te = _make_xy(2000, seed=6, kind="regression", shift=0.0)
    r = conditional_invariance(X_tr, y_tr, X_te, y_te, objective="regression")
    assert r.skipped_reason is None
    assert r.max_abs_diff < 0.5  # half a std
    assert passes(r, "regression") is True


def test_regression_shift_flagged():
    X_tr, y_tr = _make_xy(2000, seed=7, kind="regression", shift=0.0)
    # Add a constant offset to y_test = whole-distribution shift,
    # which shows up uniformly across bins.
    X_te, y_te = _make_xy(2000, seed=8, kind="regression", shift=3.0)
    r = conditional_invariance(X_tr, y_tr, X_te, y_te, objective="regression")
    assert r.skipped_reason is None
    # 3.0 absolute shift on y with train_std around 1 → ~3 std diff,
    # divided by std_train → max_abs_diff >> 0.5.
    assert r.max_abs_diff > 0.5
    assert passes(r, "regression") is False


# ── Edge cases ─────────────────────────────────────────────────────────────


def test_empty_train_returns_skipped_reason():
    X_te, y_te = _make_xy(500, seed=9, kind="binary")
    r = conditional_invariance(
        pd.DataFrame(columns=["a", "b"]), pd.Series([], dtype=int),
        X_te, y_te,
        objective="binary",
    )
    assert r.skipped_reason == "empty_train_or_test"
    assert not passes(r, "binary")


def test_no_numeric_features_skipped():
    """All-string columns shouldn't blow up; returns no_numeric_features."""
    n = 500
    X_tr = pd.DataFrame({"s": ["x"] * n}).astype(str)
    X_te = pd.DataFrame({"s": ["x"] * n}).astype(str)
    y_tr = pd.Series(np.zeros(n, dtype=int))
    y_te = pd.Series(np.zeros(n, dtype=int))
    r = conditional_invariance(X_tr, y_tr, X_te, y_te, objective="binary")
    assert r.skipped_reason == "no_numeric_features"


def test_degenerate_train_label_regression_skipped():
    """If y_train has zero variance, divergence is undefined for
    regression — surface the skip rather than emit a meaningless 0."""
    n = 500
    X_tr, _ = _make_xy(n, seed=10, kind="regression")
    X_te, y_te = _make_xy(n, seed=11, kind="regression")
    y_tr = pd.Series(np.full(n, 5.0))  # constant
    r = conditional_invariance(X_tr, y_tr, X_te, y_te, objective="regression")
    assert r.skipped_reason == "degenerate_train_label"


def test_invalid_objective_rejected():
    X_tr, y_tr = _make_xy(100, kind="binary")
    X_te, y_te = _make_xy(100, kind="binary")
    with pytest.raises(ValueError, match="objective"):
        conditional_invariance(X_tr, y_tr, X_te, y_te, objective="invalid")


def test_n_bins_must_be_at_least_2():
    X_tr, y_tr = _make_xy(100, kind="binary")
    X_te, y_te = _make_xy(100, kind="binary")
    with pytest.raises(ValueError, match="n_bins"):
        conditional_invariance(X_tr, y_tr, X_te, y_te, objective="binary", n_bins=1)


def test_per_feature_max_diff_populated():
    """If one specific feature drives the shift, per_feature_max_diff
    should pinpoint it. Construct synthetic data where feature 'a' is
    invariant but 'b' shifts a lot in y."""
    rng = np.random.default_rng(42)
    n = 2000
    a_tr = rng.standard_normal(n)
    b_tr = rng.standard_normal(n)
    a_te = rng.standard_normal(n)
    b_te = rng.standard_normal(n)
    # Train: y depends only on a (sign of a). Test: y depends on a
    # AND a strong contribution from b — so binning on a → similar
    # means, but binning on b → very different means between train
    # (close to 0.5) and test (closer to sign of b).
    y_tr_arr = (a_tr > 0).astype(int)
    y_te_arr = ((a_te + 3.0 * b_te) > 0).astype(int)
    X_tr = pd.DataFrame({"a": a_tr, "b": b_tr})
    X_te = pd.DataFrame({"a": a_te, "b": b_te})
    r = conditional_invariance(
        X_tr, pd.Series(y_tr_arr), X_te, pd.Series(y_te_arr),
        objective="binary",
    )
    assert "a" in r.per_feature_max_diff
    assert "b" in r.per_feature_max_diff
    # 'b' moved between train and test; 'a' did not.
    assert r.per_feature_max_diff["b"] > r.per_feature_max_diff["a"]


# ── Pass threshold table ───────────────────────────────────────────────────


def test_pass_thresholds_exist_for_supported_objectives():
    assert "binary" in PASS_THRESHOLD
    assert "regression" in PASS_THRESHOLD
    # Multiclass deliberately omitted; gauntlet treats as SKIP.
    assert "multiclass" not in PASS_THRESHOLD


def test_defaults():
    """Pin defaults so reproducibility-sensitive consumers know what
    they're getting."""
    assert DEFAULT_N_BINS == 5
    assert DEFAULT_MIN_BIN_SAMPLES == 20
