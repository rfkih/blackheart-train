"""Unit tests for the M5g.7 feature selection module.

Pure-function tests over synthetic feature matrices. Each call is a
single MI computation + correlation matrix; the whole file runs in
under 5 seconds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from blackheart_train.feature_selection import (
    ALWAYS_KEEP,
    CORR_THRESHOLD_DEFAULT,
    MAX_FEATURES_DEFAULT,
    select_features,
)


# ── Pinned constants ──────────────────────────────────────────────────────


def test_constants_are_pinned():
    """Operator-controlled knobs — pin so accidental edits surface."""
    assert CORR_THRESHOLD_DEFAULT == 0.7
    assert MAX_FEATURES_DEFAULT == 8
    assert "interval_indicator" in ALWAYS_KEEP


# ── Correlation pruning ───────────────────────────────────────────────────


def test_select_drops_one_of_a_high_corr_pair():
    """Two near-identical features → only one survives. The MI-tiebreak
    decides which: the more-informative one keeps."""
    rng = np.random.default_rng(0)
    n = 500
    a = rng.standard_normal(n)
    a_dup = a + rng.normal(0, 0.01, n)   # ~1.0 correlation with a
    b = rng.standard_normal(n)
    # Label: a carries signal, a_dup carries the same signal +
    # tiny noise → MI(a_dup, y) ≈ MI(a, y). Tiebreak drops alphabetically
    # later (a_dup > a lexicographically).
    y = (a > 0).astype(int)
    X = pd.DataFrame({"a": a, "a_dup": a_dup, "b": b})
    selected = select_features(X, pd.Series(y), max_features=8)
    # One of (a, a_dup) is dropped — they're collinear.
    has_a = "a" in selected
    has_a_dup = "a_dup" in selected
    assert has_a ^ has_a_dup
    assert "b" in selected   # independent feature kept


def test_select_does_not_prune_when_corr_below_threshold():
    """Two independent features stay — they're not correlated."""
    rng = np.random.default_rng(1)
    n = 500
    X = pd.DataFrame({
        "a": rng.standard_normal(n),
        "b": rng.standard_normal(n),
        "c": rng.standard_normal(n),
    })
    y = (X["a"] > 0).astype(int)
    selected = select_features(X, y, max_features=8)
    assert set(selected) == {"a", "b", "c"}


# ── Cap at max_features ───────────────────────────────────────────────────


def test_select_caps_at_max_features():
    """With more uncorrelated features than max_features, top-K by MI
    survives."""
    rng = np.random.default_rng(2)
    n = 500
    # Build 10 uncorrelated features
    X = pd.DataFrame({
        f"f{i}": rng.standard_normal(n) for i in range(10)
    })
    # Label correlated mostly with f0 + small signal from f1, f2
    y = ((X["f0"] + 0.3 * X["f1"] + 0.1 * X["f2"]) > 0).astype(int)
    selected = select_features(X, y, max_features=3)
    assert len(selected) == 3
    # f0 (strongest signal) must be in the top 3.
    assert "f0" in selected


# ── Always-keep ───────────────────────────────────────────────────────────


def test_select_always_keeps_interval_indicator():
    """``interval_indicator`` is structural — survives even if MI is
    low (constant within a batch, e.g.)."""
    rng = np.random.default_rng(3)
    n = 500
    X = pd.DataFrame({
        "interval_indicator": np.full(n, 1, dtype=int),   # constant
        "real_signal": rng.standard_normal(n),
        "noise_a": rng.standard_normal(n),
        "noise_b": rng.standard_normal(n),
    })
    y = (X["real_signal"] > 0).astype(int)
    selected = select_features(X, y, max_features=2)
    # interval_indicator must survive even though it's constant
    # (MI = 0) and we only kept 2 features total.
    assert "interval_indicator" in selected
    # real_signal should fill the second slot
    assert "real_signal" in selected


def test_select_always_keep_doesnt_inflate_budget():
    """``max_features`` is the TOTAL ceiling, including always-keep.
    With max_features=2 and 1 always-keep, only 1 candidate slot."""
    rng = np.random.default_rng(4)
    n = 300
    X = pd.DataFrame({
        "interval_indicator": np.ones(n, dtype=int),
        "a": rng.standard_normal(n),
        "b": rng.standard_normal(n),
        "c": rng.standard_normal(n),
    })
    y = (X["a"] > 0).astype(int)
    selected = select_features(X, y, max_features=2)
    assert len(selected) == 2
    assert "interval_indicator" in selected


# ── Edge cases ────────────────────────────────────────────────────────────


def test_select_rejects_zero_max_features():
    with pytest.raises(ValueError, match="max_features"):
        select_features(
            pd.DataFrame({"a": [1.0, 2.0]}),
            pd.Series([0, 1]),
            max_features=0,
        )


def test_select_returns_empty_on_empty_dataframe():
    selected = select_features(pd.DataFrame(), pd.Series([], dtype=int))
    assert selected == []


def test_select_only_always_keep_columns():
    """If X has only always-keep columns, return them all without
    running MI."""
    X = pd.DataFrame({"interval_indicator": [1, 2, 1, 2]})
    y = pd.Series([0, 1, 1, 0])
    selected = select_features(X, y, max_features=8)
    assert selected == ["interval_indicator"]


# ── Determinism ──────────────────────────────────────────────────────────


def test_select_deterministic_under_same_seed():
    """Same inputs + same random_state → same selection across runs."""
    rng = np.random.default_rng(7)
    n = 600
    X = pd.DataFrame({
        f"f{i}": rng.standard_normal(n) for i in range(8)
    })
    y = (X["f0"] + 0.5 * X["f3"] > 0).astype(int)
    a = select_features(X, y, max_features=4, random_state=42)
    b = select_features(X, y, max_features=4, random_state=42)
    assert a == b


def test_select_preserves_input_column_order_for_survivors():
    """Output preserves the input column order for surviving features
    — same trained model sees same column layout across runs."""
    rng = np.random.default_rng(8)
    n = 400
    X = pd.DataFrame({
        "z_late": rng.standard_normal(n),
        "a_early": rng.standard_normal(n),
        "m_middle": rng.standard_normal(n),
    })
    y = (X["a_early"] + X["m_middle"] > 0).astype(int)
    selected = select_features(X, y, max_features=8)
    # All three should survive (uncorrelated). Order must match input.
    assert selected == ["z_late", "a_early", "m_middle"]
