"""Tests for R2.S3 — stacking meta-learner.

Strategy: inject a stub OOF collector so tests don't pay walk-forward
fit time. The stub returns synthetic OOF predictions that follow a
known relationship to the label, so the meta-learner has signal to fit
and we can verify the blender actually improves over the best base.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from blackheart_train.bayesian_search import (
    BayesianSearchResult,
    BayesianTrialResult,
)
from blackheart_train.loader import LoadedDataset
from blackheart_train.specs import get_spec
from blackheart_train.stacking import (
    Stacker,
    StackingError,
    assemble_oof_matrix,
    predict_with_stacker,
    select_top_k,
    stacker_to_dict,
    top_k_to_specs,
    train_stacker,
    train_stacker_from_oof_matrix,
)


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_dataset(n: int = 1500, *, seed: int = 0, binary: bool = True) -> LoadedDataset:
    """Synthetic dataset whose label is correlated with a hidden signal.

    Each base model's OOF predictions (from the stub collector) are
    noisy estimates of the same signal — so a meta-learner that averages
    them should do better than any single estimate.
    """
    idx = pd.date_range("2024-12-01", periods=n, freq="1h")
    rng = np.random.default_rng(seed)
    signal = rng.standard_normal(n)
    if binary:
        # P(class=1) = sigmoid(signal)
        prob = 1.0 / (1.0 + np.exp(-signal))
        label = (rng.random(n) < prob).astype("float64")
    else:
        # Regression: y = signal + noise
        label = signal + 0.5 * rng.standard_normal(n)
    X = pd.DataFrame({"f0": signal}, index=idx)
    y = pd.Series(label, index=idx, name="y")
    return LoadedDataset(
        X=X, y=y, feature_names=("f0",),
        n_bar_slots_total=n, n_bar_slots_dropped_nan=0,
        per_feature_non_null={"f0": n}, per_feature_pct_non_null={"f0": 1.0},
        label_feature="label_regime_risk_on_48h" if binary else "label_return_24h",
        label_version=1,
    ), signal


def _make_sweep(n_trials: int = 6, seed: int = 0) -> BayesianSearchResult:
    """Fake sweep result with n COMPLETE trials of varying scores."""
    rng = np.random.default_rng(seed)
    runs = []
    for i in range(n_trials):
        runs.append(BayesianTrialResult(
            trial_number=i,
            overrides={"num_leaves": 15 + 8 * i, "learning_rate": 0.05},
            metrics={"auc": 0.55 + 0.01 * i},
            score=0.55 + 0.01 * i + float(rng.standard_normal() * 0.005),
            state="COMPLETE",
        ))
    return BayesianSearchResult(
        spec_name="test_spec",
        primary_metric="auc",
        best_overrides=runs[-1].overrides,
        best_metric=runs[-1].score,
        baseline_metric=0.5,
        runs=runs,
        n_trials_run=n_trials,
        n_trials_completed=n_trials,
        n_trials_pruned=0,
        n_trials_failed=0,
        wall_seconds=10.0,
        seed=seed,
    )


def _stub_collector_binary(signal: np.ndarray, ds: LoadedDataset, *, base_noise: float = 0.5):
    """Factory: returns a stub OOF collector that emits a probability
    derived from the hidden signal + per-base noise. Different specs get
    different noise realisations so the meta has a real diversification
    benefit to learn.

    R2 Bug-#12 fix: previously used ``hash(spec.name)`` which is
    randomised per process (PYTHONHASHSEED). Now uses ``zlib.crc32`` —
    deterministic across invocations + processes — so test assertions
    don't drift between runs.
    """
    import zlib
    call_state = {"n": 0}

    def collector(ds_in, spec, n_folds):
        seed = zlib.crc32(spec.name.encode("utf-8"))
        rng = np.random.default_rng(seed)
        # OOF preds cover the entire window except the first 200 bars
        # (mimics walk-forward's warmup-skip).
        ts = ds.X.index[200:]
        noisy_signal = signal[200:] + base_noise * rng.standard_normal(len(ts))
        prob = 1.0 / (1.0 + np.exp(-noisy_signal))
        call_state["n"] += 1
        info = {"primary_mean": 0.58, "n_folds_valid": 6}
        return prob, np.array(ts), info

    collector.call_state = call_state
    return collector


def _stub_collector_regression(signal: np.ndarray, ds: LoadedDataset, *, base_noise: float = 0.5):
    import zlib

    def collector(ds_in, spec, n_folds):
        seed = zlib.crc32(spec.name.encode("utf-8"))
        rng = np.random.default_rng(seed)
        ts = ds.X.index[200:]
        pred = signal[200:] + base_noise * rng.standard_normal(len(ts))
        return pred, np.array(ts), {"primary_mean": 0.4, "n_folds_valid": 6}
    return collector


# ── select_top_k ──────────────────────────────────────────────────────────


def test_select_top_k_returns_sorted_completed():
    sweep = _make_sweep(n_trials=6)
    top3 = select_top_k(sweep, 3)
    assert len(top3) == 3
    scores = [t.score for t in top3]
    assert scores == sorted(scores, reverse=True), scores


def test_select_top_k_excludes_pruned_and_failed():
    sweep = _make_sweep(n_trials=4)
    sweep.runs[0].state = "PRUNED"
    sweep.runs[1].state = "FAIL"
    # Only 2 completed remain — requesting 3 should raise.
    with pytest.raises(StackingError, match="COMPLETE"):
        select_top_k(sweep, 3)


def test_select_top_k_validates_k():
    sweep = _make_sweep(n_trials=4)
    with pytest.raises(ValueError, match="k must be"):
        select_top_k(sweep, 0)


def test_select_top_k_breaks_ties_deterministically():
    """R2 Bug-#8 fix: tied scores break on trial_number (ascending),
    not on insertion order, so reruns of the same sweep produce
    identical top-k picks even when many trials have the same score.
    """
    sweep = _make_sweep(n_trials=6)
    # Force three trials to tie on the highest score; expect the lowest
    # trial_number to be picked first.
    for i in (1, 3, 5):
        sweep.runs[i].score = 0.99
    top2 = select_top_k(sweep, 2)
    nums = [t.trial_number for t in top2]
    # Lowest trial_number wins among ties.
    assert nums == [1, 3], nums


def test_top_k_to_specs_merges_overrides_and_tags_name():
    sweep = _make_sweep(n_trials=4)
    top_k = select_top_k(sweep, 2)
    base_spec = get_spec("regime_btc_v1")
    specs = top_k_to_specs(base_spec, top_k)
    assert len(specs) == 2
    # Each tuned spec has a distinct name + carries the override.
    assert specs[0].name != specs[1].name
    assert specs[0].name.startswith(base_spec.name)
    assert "trial" in specs[0].name
    assert specs[0].hyperparams["num_leaves"] == top_k[0].overrides["num_leaves"]


# ── assemble_oof_matrix ───────────────────────────────────────────────────


def test_assemble_oof_matrix_aligns_on_intersection():
    ds, signal = _make_dataset(n=1500, binary=True)
    spec = get_spec("regime_btc_v1")
    sweep = _make_sweep(n_trials=4)
    top_k = select_top_k(sweep, 3)
    specs = top_k_to_specs(spec, top_k)
    collector = _stub_collector_binary(signal, ds)

    X_meta, y_meta, ts = assemble_oof_matrix(
        ds, specs, n_folds=6, collector=collector,
    )
    # Each base produced 1300 OOF rows (1500 total - 200 warmup), all
    # aligned → 1300 intersected samples × 3 base columns.
    assert X_meta.shape == (1300, 3)
    assert y_meta.shape == (1300,)
    assert len(ts) == 1300


def test_assemble_oof_matrix_raises_on_empty_intersection():
    ds, signal = _make_dataset(n=1500)
    spec = get_spec("regime_btc_v1")
    sweep = _make_sweep(n_trials=2)
    top_k = select_top_k(sweep, 2)
    specs = top_k_to_specs(spec, top_k)

    def disjoint_collector(ds_in, spec, n_folds):
        # Each spec covers a different half of the window — no overlap.
        rng = np.random.default_rng(hash(spec.name) % (2**32))
        if spec.name.endswith("trial0"):
            ts = ds.X.index[:600]
        else:
            ts = ds.X.index[900:]
        preds = rng.random(len(ts))
        return preds, np.array(ts), {"primary_mean": 0.5}

    with pytest.raises(StackingError, match="intersection"):
        assemble_oof_matrix(ds, specs, n_folds=6, collector=disjoint_collector)


def test_assemble_oof_matrix_rejects_duplicate_spec_names():
    """R2 Bug-#6 fix: explicit guard against duplicate spec names.
    Without it, pd.concat would silently merge or raise a cryptic
    pandas error. With it, callers get a clear actionable message.
    """
    ds, signal = _make_dataset(n=1500)
    spec = get_spec("regime_btc_v1")
    sweep = _make_sweep(n_trials=2)
    top_k = select_top_k(sweep, 2)
    specs = top_k_to_specs(spec, top_k)
    # Force a name collision.
    specs[1] = replace(specs[1], name=specs[0].name)

    with pytest.raises(StackingError, match="duplicate spec names"):
        assemble_oof_matrix(
            ds, specs, n_folds=6,
            collector=_stub_collector_binary(signal, ds),
        )


def test_assemble_oof_matrix_raises_when_base_emits_nothing():
    ds, signal = _make_dataset(n=1500)
    spec = get_spec("regime_btc_v1")
    sweep = _make_sweep(n_trials=2)
    top_k = select_top_k(sweep, 2)
    specs = top_k_to_specs(spec, top_k)

    def empty_collector(ds_in, spec, n_folds):
        return np.array([]), np.array([]), {"n_folds_valid": 0}

    with pytest.raises(StackingError, match="no OOF predictions"):
        assemble_oof_matrix(ds, specs, n_folds=6, collector=empty_collector)


# ── train_stacker_from_oof_matrix ─────────────────────────────────────────


def test_train_stacker_binary_fits_and_predicts():
    """The meta-learner trained on diverse base OOF predictions should
    achieve better-than-random AUC on its own training set (in-sample
    fit — OOS is the gauntlet's job).
    """
    n = 1000
    k = 4
    rng = np.random.default_rng(0)
    signal = rng.standard_normal(n)
    prob = 1.0 / (1.0 + np.exp(-signal))
    y = (rng.random(n) < prob).astype("float64")
    # Each base is a noisy estimate of P(class=1).
    X_meta = np.stack([
        1.0 / (1.0 + np.exp(-(signal + 0.5 * rng.standard_normal(n))))
        for _ in range(k)
    ], axis=1)

    stacker = train_stacker_from_oof_matrix(
        X_meta, y, objective="binary",
        base_spec_names=tuple(f"base_{i}" for i in range(k)),
    )
    assert isinstance(stacker, Stacker)
    assert stacker.objective == "binary"
    assert len(stacker.base_spec_names) == k
    assert "auc" in stacker.train_metrics
    assert stacker.train_metrics["auc"] > 0.6, stacker.train_metrics


def test_train_stacker_regression_fits():
    n = 1000
    k = 3
    rng = np.random.default_rng(0)
    signal = rng.standard_normal(n)
    y = signal + 0.3 * rng.standard_normal(n)
    X_meta = np.stack([
        signal + 0.4 * rng.standard_normal(n) for _ in range(k)
    ], axis=1)
    stacker = train_stacker_from_oof_matrix(
        X_meta, y, objective="regression",
        base_spec_names=tuple(f"base_{i}" for i in range(k)),
    )
    assert "rmse" in stacker.train_metrics
    assert stacker.train_metrics["pearson_r"] > 0.7


def test_train_stacker_rejects_single_class_binary():
    n = 500
    k = 3
    rng = np.random.default_rng(0)
    X_meta = rng.random((n, k))
    y = np.zeros(n)   # all class 0
    with pytest.raises(StackingError, match="single class"):
        train_stacker_from_oof_matrix(
            X_meta, y, objective="binary",
            base_spec_names=tuple(f"b{i}" for i in range(k)),
        )


def test_train_stacker_rejects_nan_inputs():
    X = np.full((100, 2), np.nan)
    y = np.zeros(100)
    with pytest.raises(StackingError, match="NaN"):
        train_stacker_from_oof_matrix(
            X, y, objective="binary", base_spec_names=("a", "b"),
        )


def test_train_stacker_rejects_too_few_samples():
    X = np.random.default_rng(0).random((8, 3))
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype="float64")
    with pytest.raises(StackingError, match="samples"):
        train_stacker_from_oof_matrix(
            X, y, objective="binary",
            base_spec_names=("a", "b", "c"),
        )


def test_train_stacker_rejects_column_mismatch():
    X = np.random.default_rng(0).random((100, 2))
    y = np.random.default_rng(1).integers(0, 2, 100).astype("float64")
    with pytest.raises(StackingError, match="base_spec_names"):
        train_stacker_from_oof_matrix(
            X, y, objective="binary", base_spec_names=("a", "b", "c"),
        )


# ── End-to-end train_stacker ──────────────────────────────────────────────


def test_train_stacker_end_to_end_binary():
    ds, signal = _make_dataset(n=1500, binary=True)
    spec = get_spec("regime_btc_v1")
    sweep = _make_sweep(n_trials=6)
    collector = _stub_collector_binary(signal, ds)

    stacker = train_stacker(
        ds, spec, sweep, k=3, n_folds=6, collector=collector,
    )
    assert isinstance(stacker, Stacker)
    assert len(stacker.base_spec_names) == 3
    assert stacker.n_meta_train_samples > 500
    # Stub each base predicts P(class=1) with the right sign — meta-learner
    # should easily beat 50/50.
    assert stacker.train_metrics["auc"] > 0.55


def test_train_stacker_end_to_end_regression():
    spec = get_spec("regime_btc_v1")
    # Repurpose the spec to regression for the test — modeling, not infra.
    spec_reg = replace(spec, objective="regression", label_feature="label_return_24h")
    ds, signal = _make_dataset(n=1500, binary=False)
    sweep = _make_sweep(n_trials=4)
    collector = _stub_collector_regression(signal, ds)

    stacker = train_stacker(
        ds, spec_reg, sweep, k=3, n_folds=6, collector=collector,
    )
    assert stacker.objective == "regression"
    assert stacker.train_metrics["pearson_r"] > 0.5


def test_train_stacker_rejects_multiclass():
    spec = get_spec("regime_btc_v1")
    spec_mc = replace(spec, objective="multiclass")
    ds, _ = _make_dataset(n=1500)
    sweep = _make_sweep(n_trials=4)
    with pytest.raises(StackingError, match="binary \\+ regression"):
        train_stacker(ds, spec_mc, sweep, k=2)


# ── predict_with_stacker ──────────────────────────────────────────────────


def test_predict_with_stacker_binary_shape():
    n, k = 200, 3
    X = np.random.default_rng(0).random((n, k))
    y = np.random.default_rng(1).integers(0, 2, n).astype("float64")
    stacker = train_stacker_from_oof_matrix(
        X, y, objective="binary",
        base_spec_names=tuple(f"b{i}" for i in range(k)),
    )
    preds = predict_with_stacker(stacker, X)
    assert preds.shape == (n,)
    # Binary path returns probabilities in [0, 1].
    assert preds.min() >= 0.0 and preds.max() <= 1.0


def test_predict_with_stacker_rejects_shape_mismatch():
    n, k = 200, 3
    X = np.random.default_rng(0).random((n, k))
    y = np.random.default_rng(1).integers(0, 2, n).astype("float64")
    stacker = train_stacker_from_oof_matrix(
        X, y, objective="binary",
        base_spec_names=tuple(f"b{i}" for i in range(k)),
    )
    with pytest.raises(ValueError, match="shape"):
        predict_with_stacker(stacker, np.random.default_rng(2).random((n, k + 1)))


def test_predict_with_stacker_validates_column_names_when_supplied():
    """R2 Bug-#4 fix: when column_names is passed, it must match
    stacker.base_spec_names exactly. A reversed order should raise
    before the meta sees the wrong-order data.
    """
    n, k = 200, 3
    X = np.random.default_rng(0).random((n, k))
    y = np.random.default_rng(1).integers(0, 2, n).astype("float64")
    names = ("alpha", "beta", "gamma")
    stacker = train_stacker_from_oof_matrix(
        X, y, objective="binary", base_spec_names=names,
    )
    # Correct order: no error.
    predict_with_stacker(stacker, X, column_names=names)
    # Reversed order: raises.
    with pytest.raises(ValueError, match="column_names mismatch"):
        predict_with_stacker(stacker, X, column_names=names[::-1])
    # Right count, wrong name set: also raises.
    with pytest.raises(ValueError, match="column_names mismatch"):
        predict_with_stacker(stacker, X, column_names=("a", "b", "c"))


def test_stacker_to_dict_is_json_safe():
    import json
    n, k = 200, 3
    X = np.random.default_rng(0).random((n, k))
    y = np.random.default_rng(1).integers(0, 2, n).astype("float64")
    stacker = train_stacker_from_oof_matrix(
        X, y, objective="binary",
        base_spec_names=("a", "b", "c"),
    )
    blob = stacker_to_dict(stacker)
    json.dumps(blob)   # must not raise
    assert blob["objective"] == "binary"
    assert blob["base_spec_names"] == ["a", "b", "c"]
    assert "meta_model_class" in blob
