"""Unit tests for the hyperparam grid search.

Pure-function tests — uses synthetic data so search runs in
sub-second time. DB-touching end-to-end coverage lives in the CLI
smoke run.
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from blackheart_train.loader import LoadedDataset
from blackheart_train.search import (
    _BINARY_GRID,
    _MULTICLASS_GRID,
    _REGRESSION_GRID,
    SearchError,
    grid_for,
    grid_search_one,
    primary_metric,
    search_result_to_dict,
    tuned_spec,
)
from blackheart_train.specs import get_spec


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_synthetic_dataset(
    *,
    n: int = 2000,
    label_kind: str = "binary",
    seed: int = 0,
) -> LoadedDataset:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-12-01", periods=n, freq="1h")
    # Two informative features + one noise so search has signal to find.
    sig_a = rng.standard_normal(n)
    sig_b = rng.standard_normal(n)
    noise = rng.standard_normal(n)
    if label_kind == "binary":
        logit = 0.6 * sig_a + 0.4 * sig_b
        prob = 1.0 / (1.0 + np.exp(-logit))
        y_arr = (rng.uniform(size=n) < prob).astype("float64")
    elif label_kind == "multiclass":
        # Generate -1/0/+1 from signal + bucket: ≤−0.5 → -1, [−0.5, 0.5] → 0,
        # ≥ 0.5 → +1. Three buckets, mildly imbalanced toward 0 — good enough
        # for a search-end-to-end smoke. Triple-barrier shape is irrelevant
        # at this layer.
        signal = 0.6 * sig_a + 0.4 * sig_b + 0.3 * rng.standard_normal(n)
        y_arr = np.where(signal < -0.5, -1.0, np.where(signal > 0.5, 1.0, 0.0))
    else:
        y_arr = 0.6 * sig_a + 0.4 * sig_b + 0.1 * rng.standard_normal(n)
    X = pd.DataFrame({"a": sig_a, "b": sig_b, "noise": noise}, index=idx)
    y = pd.Series(y_arr, index=idx, name="y")
    label_map = {
        "binary": "label_regime_risk_on_48h",
        "multiclass": "label_triple_barrier",
        "regression": "label_return_7d",
    }
    return LoadedDataset(
        X=X, y=y,
        feature_names=("a", "b", "noise"),
        n_bar_slots_total=n,
        n_bar_slots_dropped_nan=0,
        per_feature_non_null={"a": n, "b": n, "noise": n},
        per_feature_pct_non_null={"a": 1.0, "b": 1.0, "noise": 1.0},
        label_feature=label_map[label_kind],
        label_version=1,
    )


# ── Grid metadata ─────────────────────────────────────────────────────────


def test_grid_for_returns_objective_specific_grid():
    assert grid_for("binary") is _BINARY_GRID
    assert grid_for("regression") is _REGRESSION_GRID
    assert grid_for("multiclass") is _MULTICLASS_GRID


def test_primary_metric_is_per_objective():
    assert primary_metric("binary") == "auc"
    assert primary_metric("regression") == "pearson_r"
    # M5g.1: macro AUC OVR is the selection metric for directional —
    # accuracy is dominated by the 59/38 majority and rewards collapse.
    assert primary_metric("multiclass") == "macro_auc_ovr"


def test_grids_have_distinct_combinations():
    """Catch a copy-paste error where two grid entries are identical."""
    for grid in (_BINARY_GRID, _REGRESSION_GRID, _MULTICLASS_GRID):
        seen = {tuple(sorted(g.items())) for g in grid}
        assert len(seen) == len(grid), f"duplicate grid points: {grid}"


# ── Tuned spec ────────────────────────────────────────────────────────────


def test_tuned_spec_merges_overrides():
    spec = get_spec("regime_btc_v1")
    new = tuned_spec(spec, {"num_leaves": 99, "learning_rate": 0.5})
    assert new.hyperparams["num_leaves"] == 99
    assert new.hyperparams["learning_rate"] == 0.5
    # Untouched keys preserved
    assert new.hyperparams["random_state"] == spec.hyperparams["random_state"]
    # Original spec untouched
    assert spec.hyperparams["num_leaves"] != 99


def test_tuned_spec_returns_new_instance():
    spec = get_spec("regime_btc_v1")
    new = tuned_spec(spec, {"num_leaves": 99})
    assert new is not spec
    assert new.hyperparams is not spec.hyperparams


# ── End-to-end search on synthetic data ──────────────────────────────────


def test_binary_search_picks_best_and_records_runs():
    ds = _make_synthetic_dataset(n=2000, label_kind="binary", seed=42)
    spec = get_spec("regime_btc_v1")
    # Mini grid so the test runs in <2s.
    grid = (
        {"num_leaves": 15, "learning_rate": 0.05, "min_child_samples": 50},
        {"num_leaves": 31, "learning_rate": 0.05, "min_child_samples": 50},
    )
    result = grid_search_one(ds, spec, grid=grid)
    assert result.spec_name == "regime_btc_v1"
    assert result.primary_metric == "auc"
    assert len(result.runs) == len(grid)
    assert result.best_overrides in grid
    # Baseline must be a real number (synthetic data has signal)
    assert result.baseline_metric is not None
    assert not math.isnan(result.baseline_metric)
    # Best must dominate each run on the primary metric
    metric = result.primary_metric
    for r in result.runs:
        assert result.best_metric >= r.metrics[metric] - 1e-9


def test_regression_search_picks_best():
    ds = _make_synthetic_dataset(n=2000, label_kind="regression", seed=7)
    spec = get_spec("flow_btc_v1")
    grid = (
        {"num_leaves": 15, "learning_rate": 0.05, "min_child_samples": 50},
        {"num_leaves": 31, "learning_rate": 0.05, "min_child_samples": 50},
    )
    result = grid_search_one(ds, spec, grid=grid)
    assert result.primary_metric == "pearson_r"
    assert result.best_overrides in grid
    assert not math.isnan(result.best_metric)


def test_multiclass_search_picks_best():
    """M5g.1: search end-to-end on a 3-class synthetic dataset.
    Confirms macro_auc_ovr is selected, runs complete, and the best
    run dominates the others on macro-AUC."""
    ds = _make_synthetic_dataset(n=2000, label_kind="multiclass", seed=11)
    spec = get_spec("directional_btc_1h_v1")
    grid = (
        {"num_leaves": 31, "learning_rate": 0.05, "min_child_samples": 50},
        {"num_leaves": 63, "learning_rate": 0.05, "min_child_samples": 50},
    )
    result = grid_search_one(ds, spec, grid=grid)
    assert result.spec_name == "directional_btc_1h_v1"
    assert result.primary_metric == "macro_auc_ovr"
    assert len(result.runs) == len(grid)
    assert result.best_overrides in grid
    assert not math.isnan(result.best_metric)
    # Best must dominate every run on the primary metric.
    for r in result.runs:
        assert result.best_metric >= r.metrics["macro_auc_ovr"] - 1e-9


# ── Result serialisation ──────────────────────────────────────────────────


def test_search_result_to_dict_includes_improvement():
    ds = _make_synthetic_dataset(n=2000, label_kind="binary", seed=42)
    spec = get_spec("regime_btc_v1")
    grid = (
        {"num_leaves": 15, "learning_rate": 0.05, "min_child_samples": 50},
    )
    result = grid_search_one(ds, spec, grid=grid)
    d = search_result_to_dict(result)
    assert d["spec_name"] == "regime_btc_v1"
    assert d["primary_metric"] == "auc"
    assert "best_overrides" in d
    assert "improvement_vs_baseline" in d
    assert len(d["runs"]) == 1


# ── Defensive ─────────────────────────────────────────────────────────────


def test_search_picks_sensible_point_over_degenerate_point():
    """When the grid mixes a degenerate config (likely NaN or near-zero
    correlation) and a sensible one, the sensible one must win — even
    though _nan_to_neginf demotes NaN to negative infinity, we want to
    confirm the live training path actually exercises the fallback."""
    ds = _make_synthetic_dataset(n=2000, label_kind="regression", seed=11)
    spec = get_spec("positioning_btc_v1")
    grid = (
        # Effectively forces constant output: min_child_samples > training size.
        {"num_leaves": 2, "learning_rate": 0.001, "min_child_samples": 10_000, "reg_lambda": 100.0},
        # Sensible point — should win.
        {"num_leaves": 31, "learning_rate": 0.05, "min_child_samples": 50},
    )
    result = grid_search_one(ds, spec, grid=grid)
    # The sensible point should always beat the degenerate one.
    assert result.best_overrides == grid[1]


def test_search_raises_when_every_grid_run_is_degenerate(monkeypatch):
    """All-NaN grid means the user has no usable result. The loop must
    raise SearchError instead of silently picking grid[0]. We monkeypatch
    fit_and_evaluate to return NaN deterministically — coaxing LightGBM
    into actual NaN output is brittle across versions, and the SearchError
    logic is what this test really cares about.
    """
    import blackheart_train.search as search_mod

    def _stub_fit_and_evaluate(*_args, **_kwargs):
        return object(), {"rmse": 1.0, "mae": 1.0, "pearson_r": float("nan")}

    monkeypatch.setattr(search_mod, "fit_and_evaluate", _stub_fit_and_evaluate)

    ds = _make_synthetic_dataset(n=500, label_kind="regression", seed=3)
    spec = get_spec("positioning_btc_v1")
    grid = (
        {"num_leaves": 31, "learning_rate": 0.05, "min_child_samples": 50},
        {"num_leaves": 15, "learning_rate": 0.01, "min_child_samples": 50},
    )
    with pytest.raises(SearchError, match="degenerate"):
        grid_search_one(ds, spec, grid=grid)
