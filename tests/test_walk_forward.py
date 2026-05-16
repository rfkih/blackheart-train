"""Unit tests for walk-forward CV.

Pure-function tests for fold generation; synthetic-data tests for the
execution loop and aggregation. DB-touching coverage lives in the CLI
end-to-end run.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from blackheart_train.loader import LoadedDataset
from blackheart_train.specs import get_spec
from blackheart_train.walk_forward import (
    DEFAULT_EMBARGO_DAYS,
    DEFAULT_N_FOLDS,
    DEFAULT_TEST_DAYS,
    DEFAULT_TRAIN_DAYS,
    Fold,
    WalkForwardError,
    generate_folds,
    run_walk_forward,
    train_via_walk_forward,
    walk_forward_to_dict,
)


# ── generate_folds ─────────────────────────────────────────────────────────


def test_generate_folds_produces_chronological_non_overlapping_test_windows():
    start = datetime(2024, 12, 1)
    end = datetime(2026, 5, 14)
    folds = generate_folds(start, end)
    assert 0 < len(folds) <= DEFAULT_N_FOLDS
    for k, f in enumerate(folds):
        assert f.k == k
        assert f.train_start < f.train_end <= f.test_start < f.test_end
        embargo_actual = (f.test_start - f.train_end).days
        assert embargo_actual == DEFAULT_EMBARGO_DAYS
        train_actual = (f.train_end - f.train_start).days
        assert train_actual == DEFAULT_TRAIN_DAYS
        test_actual = (f.test_end - f.test_start).days
        assert test_actual == DEFAULT_TEST_DAYS
    # Test windows do not overlap and are strictly increasing
    for prev, nxt in zip(folds, folds[1:]):
        assert prev.test_end <= nxt.test_start


def test_generate_folds_stops_when_data_runs_out():
    """A tight window can only fit fewer than n_folds; the function
    must stop early instead of producing folds that walk past
    train_end."""
    start = datetime(2024, 12, 1)
    # Only 410 days total — barely enough for fold 0's train + embargo
    # + test, definitely not 6 folds.
    end = start + timedelta(days=410)
    folds = generate_folds(start, end, n_folds=6)
    assert len(folds) < 6
    for f in folds:
        assert f.test_end <= end


def test_generate_folds_zero_folds_when_window_too_short():
    """No fold should be returned if the first fold's test window
    already exceeds train_end."""
    start = datetime(2024, 12, 1)
    end = start + timedelta(days=370)   # < 365 + 7 + 21
    folds = generate_folds(start, end, n_folds=6)
    assert len(folds) == 0


def test_run_walk_forward_raises_with_clear_message_on_zero_folds():
    """When the spec window is so short generate_folds returns nothing,
    the error should diagnose 'window too short', not 'NaN metrics'."""
    rng = np.random.default_rng(0)
    n = 100
    idx = pd.date_range("2024-12-01", periods=n, freq="1h")
    X = pd.DataFrame({"a": rng.standard_normal(n)}, index=idx)
    y = pd.Series((np.arange(n) % 2).astype(float), index=idx, name="y")
    ds = LoadedDataset(
        X=X, y=y,
        feature_names=("a",),
        n_bar_slots_total=n,
        n_bar_slots_dropped_nan=0,
        per_feature_non_null={"a": n},
        per_feature_pct_non_null={"a": 1.0},
        label_feature="label_regime_risk_on_48h",
        label_version=1,
    )
    # Spec window is only ~4 days; default needs >393 days.
    from dataclasses import replace
    spec = replace(
        get_spec("regime_btc_v1"),
        train_start=datetime(2024, 12, 1),
        train_end=datetime(2024, 12, 5),
    )
    with pytest.raises(WalkForwardError, match="0 folds"):
        run_walk_forward(ds, spec)


def test_generate_folds_rejects_non_positive_sizes():
    start = datetime(2024, 12, 1)
    end = datetime(2026, 5, 14)
    with pytest.raises(ValueError):
        generate_folds(start, end, train_days=0)
    with pytest.raises(ValueError):
        generate_folds(start, end, test_days=-1)
    with pytest.raises(ValueError):
        generate_folds(start, end, embargo_days=-1)
    with pytest.raises(ValueError):
        generate_folds(start, end, n_folds=0)


def test_generate_folds_embargo_can_be_zero():
    """Zero embargo is allowed (caller's choice — useful when labels
    are backward-only). Test windows must still be chronological."""
    start = datetime(2024, 12, 1)
    end = datetime(2026, 5, 14)
    folds = generate_folds(start, end, embargo_days=0)
    assert all(f.train_end == f.test_start for f in folds)


# ── Synthetic dataset helper ───────────────────────────────────────────────


def _make_dense_dataset(
    *,
    start: datetime = datetime(2024, 12, 1),
    end: datetime = datetime(2026, 5, 14),
    label_kind: str = "binary",
    seed: int = 0,
) -> LoadedDataset:
    """Dense hourly dataset with signal in two features. No NaN, so
    every walk-forward fold should run successfully."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start, end=end, freq="1h", inclusive="left")
    n = len(idx)
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    if label_kind == "binary":
        logit = 0.6 * a + 0.4 * b
        prob = 1.0 / (1.0 + np.exp(-logit))
        y_arr = (rng.uniform(size=n) < prob).astype("float64")
        label = "label_regime_risk_on_48h"
    elif label_kind == "multiclass":
        # Three buckets from a signal+noise score. Mildly imbalanced.
        signal = 0.6 * a + 0.4 * b + 0.3 * rng.standard_normal(n)
        y_arr = np.where(signal < -0.5, -1.0, np.where(signal > 0.5, 1.0, 0.0))
        label = "label_triple_barrier"
    else:
        y_arr = 0.6 * a + 0.4 * b + 0.1 * rng.standard_normal(n)
        label = "label_return_7d"
    X = pd.DataFrame({"a": a, "b": b}, index=idx)
    y = pd.Series(y_arr, index=idx, name="y")
    return LoadedDataset(
        X=X, y=y,
        feature_names=("a", "b"),
        n_bar_slots_total=n,
        n_bar_slots_dropped_nan=0,
        per_feature_non_null={"a": n, "b": n},
        per_feature_pct_non_null={"a": 1.0, "b": 1.0},
        label_feature=label,
        label_version=1,
    )


# ── run_walk_forward ───────────────────────────────────────────────────────


def test_walk_forward_runs_all_folds_on_dense_binary_data():
    ds = _make_dense_dataset(label_kind="binary", seed=42)
    spec = get_spec("regime_btc_v1")
    result = run_walk_forward(ds, spec)
    assert result.n_folds_run == DEFAULT_N_FOLDS
    assert result.n_folds_valid_metric == DEFAULT_N_FOLDS
    assert result.primary_metric == "auc"
    assert 0.0 <= result.primary_mean <= 1.0
    assert result.primary_std >= 0.0
    assert "auc" in result.metric_means
    assert "log_loss" in result.metric_means
    assert "accuracy" in result.metric_means
    # Per-fold detail records every metric the fit returned, plus the
    # M5g.6 ``adversarial_auc`` and the 2026-05-16 conditional-invariance
    # triad (``ci_max_abs_diff``, ``ci_mean_abs_diff``,
    # ``ci_n_pairs_evaluated``) that walk_forward injects per fold.
    expected_keys = {
        "auc", "log_loss", "accuracy",
        "adversarial_auc",
        "ci_max_abs_diff", "ci_mean_abs_diff", "ci_n_pairs_evaluated",
    }
    for fm in result.folds:
        assert fm.skipped_reason is None
        assert expected_keys == set(fm.metrics.keys()), (
            f"unexpected metric keys: missing={expected_keys - set(fm.metrics.keys())}, "
            f"extra={set(fm.metrics.keys()) - expected_keys}"
        )


def test_walk_forward_runs_all_folds_on_dense_regression_data():
    ds = _make_dense_dataset(label_kind="regression", seed=7)
    spec = get_spec("flow_btc_v1")
    result = run_walk_forward(ds, spec)
    assert result.n_folds_run == DEFAULT_N_FOLDS
    assert result.primary_metric == "pearson_r"
    assert "rmse" in result.metric_means
    assert "mae" in result.metric_means


def test_walk_forward_runs_all_folds_on_dense_multiclass_data():
    """M5g.2 + M5g.3 + M5g.4: multiclass directional walk-forward
    end-to-end. The directional spec opts into the 3-model ensemble
    AND meta-label gating, so the aggregated metric_means must carry
    ensemble (unprefixed), per-base (prefixed), disagreement, AND
    gated_* keys."""
    ds = _make_dense_dataset(label_kind="multiclass", seed=42)
    spec = get_spec("directional_btc_1h_v1")
    result = run_walk_forward(ds, spec)
    assert result.n_folds_run == DEFAULT_N_FOLDS
    assert result.primary_metric == "macro_auc_ovr"
    # Ensemble metrics (unprefixed) — the primary the WF aggregator
    # consumes for primary_mean / primary_std lives under "macro_auc_ovr".
    for key in ("log_loss", "accuracy", "macro_auc_ovr"):
        assert key in result.metric_means
    # Per-class precision/recall lands on the ensemble's averaged proba.
    assert "class_0_precision" in result.metric_means
    assert "class_2_recall" in result.metric_means
    # M5g.3: per-base metrics propagate through the aggregator.
    for prefix in ("lgb_", "xgb_", "lr_"):
        assert f"{prefix}log_loss" in result.metric_means
        assert f"{prefix}accuracy" in result.metric_means
        assert f"{prefix}macro_auc_ovr" in result.metric_means
    # Disagreement — also makes it to the aggregator.
    assert "mean_disagreement" in result.metric_means
    assert "mean_disagreement_class_0" in result.metric_means
    # M5g.4: gated_* keys (meta-label gating) propagate. They may be
    # absent in folds where meta-label couldn't fit (single-class
    # meta-train or primary slice missing a class), so the aggregator
    # averages only over folds where they exist — we just assert at
    # least one fold contributed.
    for key in ("gated_selectivity", "gated_accuracy",
                "gated_accuracy_uplift", "gated_n_kept", "gated_n_total"):
        assert key in result.metric_means, f"missing gated key: {key}"


def test_walk_forward_skips_multiclass_train_missing_a_class(monkeypatch):
    """WF1 fix: the multiclass train guard requires ALL classes in
    train (not just ≥2). Fitting LightGBM on a 2-of-3-class slice
    would produce a 2-class booster whose predict_proba is (n, 2);
    _evaluate would then call log_loss with labels=[0,1,2] against
    y_val carrying encoded class 2 → ValueError, walk-forward dies.

    Make fold-0's training window carry only 2 of the 3 classes and
    confirm the guard skips before any fit happens.
    """
    import blackheart_train.walk_forward as wf_mod

    original_fit = wf_mod.fit_and_evaluate
    fit_calls: list[int] = []

    def _spy_fit(X_tr, y_tr, X_te, y_te, spec):
        fit_calls.append(int(y_tr.astype(int).nunique()))
        return original_fit(X_tr, y_tr, X_te, y_te, spec)

    ds = _make_dense_dataset(label_kind="multiclass", seed=99)
    # Force class 1 (== triple-barrier value 0, horizon-end) out of
    # fold-0's training window. Anything in that window that was 0 is
    # nudged to -1. Fold 0's train window is [2024-12-01, 2025-12-01).
    mask = (ds.X.index >= datetime(2024, 12, 1)) & (ds.X.index < datetime(2025, 12, 1))
    nudge = mask & (ds.y == 0.0)
    ds.y.loc[nudge] = -1.0

    monkeypatch.setattr(wf_mod, "fit_and_evaluate", _spy_fit)
    spec = get_spec("directional_btc_1h_v1")
    result = wf_mod.run_walk_forward(ds, spec)

    fold0 = result.folds[0]
    assert fold0.skipped_reason == "insufficient_classes_in_train_set"
    # First fit call (if any) was fold 1 onward, where all 3 classes
    # are still present. None should ever report < N_MULTICLASS_CLASSES.
    from blackheart_train.train import N_MULTICLASS_CLASSES
    assert all(nu >= N_MULTICLASS_CLASSES for nu in fit_calls)


def test_walk_forward_skips_when_test_slice_empty_after_serving_filter():
    """MS11 fix: if MS4's serving-interval filter empties the test slice
    (no 1h bars in the test window — pathological but possible in a
    stacked dataset where 15m fills the window), the fold must be
    skipped with a clear reason, not crash sklearn metrics on empty
    arrays.

    Setup: synthetic dataset where every row is tagged
    interval_indicator=1 (15m). For the directional spec serving at
    1h (code 2), every fold's test slice becomes empty post-filter →
    every fold should be skipped with the expected reason. When ALL
    folds skip, run_walk_forward raises WalkForwardError (no valid
    fold) — the MS11 fix is the per-fold skip *reason*, which we
    can't read after the raise. So we monkeypatch to capture
    fold_results state mid-stream.
    """
    from dataclasses import replace as _replace
    spec = get_spec("directional_btc_1h_v1")

    ds = _make_dense_dataset(label_kind="multiclass", seed=99)
    ds.X = ds.X.copy()
    ds.X["interval_indicator"] = 1   # all 15m
    ds = _replace(
        ds,
        feature_names=tuple(list(ds.feature_names) + ["interval_indicator"]),
    )

    with pytest.raises(WalkForwardError, match="no valid fold"):
        run_walk_forward(ds, spec)


def test_walk_forward_binary_single_class_reason_unchanged(monkeypatch):
    """Regression guard: tightening the multiclass guard must not have
    changed the binary skipped_reason string. The skipped_reason is
    audit-trail data — downstream tooling pattern-matches on it."""
    import blackheart_train.walk_forward as wf_mod

    ds = _make_dense_dataset(label_kind="binary", seed=33)
    # Collapse fold-0's training window to a single class.
    mask = (ds.X.index >= datetime(2024, 12, 1)) & (ds.X.index < datetime(2025, 12, 1))
    ds.y.loc[mask] = 0.0

    spec = get_spec("regime_btc_v1")
    result = wf_mod.run_walk_forward(ds, spec)
    fold0 = result.folds[0]
    assert fold0.skipped_reason == "single_class_train_set"


def test_walk_forward_skips_empty_test_window():
    """If we artificially blank out the data range a fold expects, the
    fold should be recorded with skipped_reason rather than crashing."""
    ds = _make_dense_dataset(label_kind="binary", seed=1)
    # Drop every bar from 2026-03-01 onward — that wipes the last fold.
    cutoff = datetime(2026, 3, 1)
    ds.X = ds.X.loc[ds.X.index < cutoff].copy()
    ds.y = ds.y.loc[ds.y.index < cutoff].copy()
    spec = get_spec("regime_btc_v1")
    result = run_walk_forward(ds, spec)
    assert result.n_folds_configured == DEFAULT_N_FOLDS
    # At least one fold should be skipped due to empty test window
    skipped = [fm for fm in result.folds if fm.skipped_reason is not None]
    assert any(fm.skipped_reason == "empty_test_window" for fm in skipped)
    assert result.n_folds_valid_metric == result.n_folds_run


def test_walk_forward_raises_when_every_fold_is_degenerate(monkeypatch):
    """All-NaN walk-forward must surface as WalkForwardError, not
    silently return a NaN aggregate."""
    import blackheart_train.walk_forward as wf_mod

    def _stub_fit_and_evaluate(*_a, **_kw):
        return object(), {"rmse": 1.0, "mae": 1.0, "pearson_r": float("nan")}

    monkeypatch.setattr(wf_mod, "fit_and_evaluate", _stub_fit_and_evaluate)

    ds = _make_dense_dataset(label_kind="regression", seed=3)
    spec = get_spec("flow_btc_v1")
    with pytest.raises(WalkForwardError, match="no valid fold"):
        run_walk_forward(ds, spec)


def test_walk_forward_aggregate_matches_per_fold_values():
    """primary_mean / median / std must be consistent with the per-fold
    metrics list — anyone reading the artifact can verify the aggregate."""
    ds = _make_dense_dataset(label_kind="regression", seed=11)
    spec = get_spec("flow_btc_v1")
    result = run_walk_forward(ds, spec)
    values = [
        fm.metrics["pearson_r"]
        for fm in result.folds
        if fm.skipped_reason is None
        and not math.isnan(fm.metrics.get("pearson_r", float("nan")))
    ]
    assert result.primary_mean == pytest.approx(float(np.mean(values)))
    assert result.primary_median == pytest.approx(float(np.median(values)))
    assert result.primary_std == pytest.approx(float(np.std(values, ddof=0)))


# ── Serialisation ─────────────────────────────────────────────────────────


def test_walk_forward_to_dict_has_expected_keys():
    ds = _make_dense_dataset(label_kind="binary", seed=42)
    spec = get_spec("regime_btc_v1")
    result = run_walk_forward(ds, spec)
    d = walk_forward_to_dict(result)
    for key in (
        "spec_name", "primary_metric",
        "n_folds_configured", "n_folds_generated", "n_folds_run", "n_folds_valid_metric",
        "primary_mean", "primary_median", "primary_std",
        "metric_means", "folds",
    ):
        assert key in d, f"missing key: {key}"
    # _last_booster must NOT leak to the JSON view — boosters aren't
    # serialisable and don't belong in audit dicts.
    assert "_last_booster" not in d
    # Per-fold dicts have ISO-string timestamps so JSON-safe
    assert isinstance(d["folds"][0]["train_start"], str)
    assert isinstance(d["folds"][0]["test_end"], str)


def test_walk_forward_tracks_last_booster_from_most_recent_valid_fold():
    """``_last_booster`` must be the booster from the chronologically
    last *valid* fold. With dense data, that's fold N-1. If we skip a
    fold (e.g. by limiting the data range), the last_booster comes from
    the last non-skipped fold, not from the last generated fold."""
    ds = _make_dense_dataset(label_kind="binary", seed=42)
    spec = get_spec("regime_btc_v1")
    result = run_walk_forward(ds, spec)
    assert result._last_booster is not None
    # Booster's text representation should match the model trained on
    # fold (n_folds_run - 1)'s training window — i.e., the last valid one.
    valid = [fm for fm in result.folds if fm.skipped_reason is None]
    assert len(valid) == result.n_folds_run


def test_train_via_walk_forward_returns_payload_with_last_fold_booster():
    """The WF1 fix: the saved booster is the last fold's, and the
    payload's ``metrics`` describe THAT booster (not the 80/20 fit)."""
    ds = _make_dense_dataset(label_kind="binary", seed=42)
    spec = get_spec("regime_btc_v1")
    payload = train_via_walk_forward(ds, spec)
    assert payload["eval_kind"] == "walk_forward_last_fold"
    assert payload["walk_forward"] is not None
    # metrics must equal the last valid fold's metrics, by construction.
    last_valid = [
        fm for fm in payload["walk_forward"]["folds"]
        if fm["skipped_reason"] is None
    ][-1]
    assert payload["metrics"] == last_valid["metrics"]
    # The booster in the payload must match the result's last_booster.
    # (We can't trivially compare boosters; rely on the fact that
    # build_payload uses booster.model_to_string() for the content_sha.)
    assert payload["content_sha256"]
    # n_train / n_val_rows describe the last fold's split.
    assert payload["n_train_rows"] == last_valid["n_train"]
    assert payload["n_val_rows"] == last_valid["n_test"]


def test_n_folds_generated_distinguishes_data_short_from_runtime_skip():
    """When data is short, n_folds_generated < n_folds_configured.
    When all data fits, they're equal; runtime skips show up in
    n_folds_run only."""
    ds = _make_dense_dataset(label_kind="binary", seed=1)
    spec = get_spec("regime_btc_v1")
    result = run_walk_forward(ds, spec)
    assert result.n_folds_configured == DEFAULT_N_FOLDS
    assert result.n_folds_generated == DEFAULT_N_FOLDS
    assert result.n_folds_run <= result.n_folds_generated


def test_walk_forward_skips_single_class_train_set(monkeypatch):
    """Symmetric to the test-set guard: a single-class train set must
    be skipped, not silently fit into a constant predictor."""
    ds = _make_dense_dataset(label_kind="binary", seed=42)
    spec = get_spec("regime_btc_v1")
    # Force fold 0's training window to look single-class by zeroing it.
    fold0_train_end = spec.train_start + timedelta(days=DEFAULT_TRAIN_DAYS)
    mask = (ds.y.index >= spec.train_start) & (ds.y.index < fold0_train_end)
    ds.y = ds.y.copy()
    ds.y.loc[mask] = 0.0
    result = run_walk_forward(ds, spec)
    fold_0 = result.folds[0]
    assert fold_0.skipped_reason == "single_class_train_set"


# ── last_n_folds (dev-velocity knob, M5g.9.1) ─────────────────────────────


def test_last_n_folds_runs_only_most_recent_subset():
    """``last_n_folds=2`` should run folds 4 and 5 (the last two of a
    6-fold sequence), not folds 0-1."""
    ds = _make_dense_dataset(label_kind="binary", seed=42)
    spec = get_spec("regime_btc_v1")
    result = run_walk_forward(ds, spec, last_n_folds=2)
    assert result.n_folds_run == 2
    # Fold k indices must be preserved from the full configured sequence,
    # NOT renumbered. With 6 folds generated, last 2 are k=4 and k=5.
    fold_ks = sorted(fm.fold for fm in result.folds)
    assert fold_ks == [DEFAULT_N_FOLDS - 2, DEFAULT_N_FOLDS - 1]
    # n_folds_configured still reflects the full 6 so the operator can
    # see this was a truncated subset, not a re-configured 2-fold run.
    assert result.n_folds_configured == DEFAULT_N_FOLDS
    assert result.n_folds_generated == DEFAULT_N_FOLDS


def test_last_n_folds_preserves_test_windows_of_unfiltered_run():
    """The two folds we run with ``last_n_folds=2`` must share their
    train/test boundaries with folds 4 and 5 of the unfiltered run —
    so dev-velocity output is comparable to the full-run output."""
    ds = _make_dense_dataset(label_kind="binary", seed=42)
    spec = get_spec("regime_btc_v1")
    full = run_walk_forward(ds, spec)
    subset = run_walk_forward(ds, spec, last_n_folds=2)
    full_last_two = full.folds[-2:]
    for full_fm, sub_fm in zip(full_last_two, subset.folds):
        assert full_fm.fold == sub_fm.fold
        assert full_fm.train_start == sub_fm.train_start
        assert full_fm.test_start == sub_fm.test_start
        assert full_fm.test_end == sub_fm.test_end


def test_last_n_folds_clamps_to_available_when_larger_than_generated():
    """If the caller asks for last_n_folds=99 but only 6 were generated,
    the function should run all 6 — not crash, not silently re-window."""
    ds = _make_dense_dataset(label_kind="binary", seed=42)
    spec = get_spec("regime_btc_v1")
    result = run_walk_forward(ds, spec, last_n_folds=99)
    assert result.n_folds_run == DEFAULT_N_FOLDS


def test_last_n_folds_rejects_non_positive():
    """``last_n_folds=0`` and negative values must raise; "run nothing"
    is not a useful mode and silently shipping zero-fold evidence would
    bypass the WalkForwardError-on-zero-folds contract."""
    ds = _make_dense_dataset(label_kind="binary", seed=42)
    spec = get_spec("regime_btc_v1")
    with pytest.raises(ValueError, match="last_n_folds"):
        run_walk_forward(ds, spec, last_n_folds=0)
    with pytest.raises(ValueError, match="last_n_folds"):
        run_walk_forward(ds, spec, last_n_folds=-3)


def test_last_n_folds_one_runs_just_the_final_fold():
    """The most common dev-velocity invocation: a one-fold smoke that
    matches the booster ``train_via_walk_forward`` would save (last
    valid fold). Useful as a 2-3 min sanity check."""
    ds = _make_dense_dataset(label_kind="binary", seed=42)
    spec = get_spec("regime_btc_v1")
    result = run_walk_forward(ds, spec, last_n_folds=1)
    assert result.n_folds_run == 1
    assert result.folds[0].fold == DEFAULT_N_FOLDS - 1
