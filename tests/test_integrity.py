"""Unit tests for the data integrity checks.

Pure-function tests — no DB. We synthesise LoadedDataset instances and
ModelSpec stubs in-memory so each check can be exercised under the
exact failure mode it's defined to catch.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from blackheart_train.integrity import (
    IntegrityError,
    check_dataset,
    compute_data_fingerprint,
)
from blackheart_train.loader import LoadedDataset
from blackheart_train.specs import get_spec


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_dataset(
    n: int = 2000,
    *,
    features: dict[str, np.ndarray] | None = None,
    label: np.ndarray | None = None,
    label_feature: str = "label_regime_risk_on_48h",
    label_version: int = 1,
    start: str = "2024-12-01",
) -> LoadedDataset:
    """Build a synthetic LoadedDataset for testing checks in isolation."""
    idx = pd.date_range(start=start, periods=n, freq="1h")
    if features is None:
        rng = np.random.default_rng(seed=0)
        features = {
            "f0": rng.standard_normal(n),
            "f1": rng.standard_normal(n),
        }
    if label is None:
        label = (np.arange(n) % 2).astype("float64")   # balanced binary
    X = pd.DataFrame({k: v for k, v in features.items()}, index=idx)
    y = pd.Series(label, index=idx, name="y")
    return LoadedDataset(
        X=X, y=y,
        feature_names=tuple(features.keys()),
        n_bar_slots_total=n,
        n_bar_slots_dropped_nan=0,
        per_feature_non_null={k: n for k in features},
        per_feature_pct_non_null={k: 1.0 for k in features},
        label_feature=label_feature,
        label_version=label_version,
    )


# ── Individual checks ──────────────────────────────────────────────────────


def test_min_rows_fail_below_threshold():
    ds = _make_dataset(n=200)
    spec = get_spec("regime_btc_v1")
    report = check_dataset(ds, spec, min_rows=1000)
    assert report.verdict == "FAIL"
    fail = next(c for c in report.checks if c.name == "min_rows")
    assert fail.severity == "FAIL"


def test_min_rows_pass_above_threshold():
    ds = _make_dataset(n=2000)
    spec = get_spec("regime_btc_v1")
    report = check_dataset(ds, spec, min_rows=1000)
    assert any(c.name == "min_rows" and c.severity == "PASS" for c in report.checks)


def test_binary_class_balance_warns_on_skew():
    n = 2000
    label = np.concatenate([np.zeros(1900), np.ones(100)])   # 5/95
    ds = _make_dataset(n=n, label=label)
    spec = get_spec("regime_btc_v1")
    report = check_dataset(ds, spec)
    bc = next(c for c in report.checks if c.name == "binary_class_balance")
    assert bc.severity == "WARN"
    assert bc.details["min_pct"] < 0.15


def test_binary_class_balance_fail_on_single_class():
    n = 2000
    label = np.zeros(n)
    ds = _make_dataset(n=n, label=label)
    spec = get_spec("regime_btc_v1")
    report = check_dataset(ds, spec)
    assert report.verdict == "FAIL"
    bc = next(c for c in report.checks if c.name == "binary_class_balance")
    assert bc.severity == "FAIL"


def test_train_val_class_balance_warns_on_regime_shift():
    """Class-0 dominates training period, class-1 dominates val period —
    exactly the silent-killer pattern this check exists to catch."""
    n = 2000
    val_fraction = 0.2
    n_val = int(round(n * val_fraction))
    n_train = n - n_val
    label = np.concatenate([np.zeros(n_train), np.ones(n_val)])
    ds = _make_dataset(n=n, label=label)
    spec = get_spec("regime_btc_v1")
    report = check_dataset(ds, spec)
    tv = next(c for c in report.checks if c.name == "train_val_class_balance")
    assert tv.severity == "WARN"
    assert tv.details["drift_pp"] > 0.10


def test_constant_features_warns():
    n = 2000
    features = {
        "f_normal": np.random.default_rng(0).standard_normal(n),
        "f_constant": np.full(n, 3.14),
    }
    ds = _make_dataset(n=n, features=features)
    spec = get_spec("regime_btc_v1")
    report = check_dataset(ds, spec)
    cf = next(c for c in report.checks if c.name == "constant_features")
    assert cf.severity == "WARN"
    assert cf.details["features"] == ["f_constant"]


def test_stale_tail_warns_on_old_data():
    """Build a dataset that ends 30 days before spec.train_end so the
    stale_tail check trips."""
    spec = get_spec("regime_btc_v1")
    n = 2000
    # Start ~30 days before train_end so the last bar is 30 days early.
    start = spec.train_end - pd.Timedelta(days=30) - pd.Timedelta(hours=n)
    ds = _make_dataset(n=n, start=start.isoformat())
    report = check_dataset(ds, spec)
    st = next(c for c in report.checks if c.name == "stale_tail")
    assert st.severity == "WARN"


def test_regression_outliers_warns():
    spec = replace(get_spec("flow_btc_v1"))
    n = 2000
    rng = np.random.default_rng(0)
    label = rng.standard_normal(n)
    label[0] = 20.0   # ~20σ outlier
    ds = _make_dataset(n=n, label=label, label_feature="label_return_7d")
    report = check_dataset(ds, spec)
    rl = next(c for c in report.checks if c.name == "regression_label_distribution")
    assert rl.severity == "WARN"
    assert rl.details["extreme_count"] >= 1


def test_regression_zero_variance_label_fails():
    spec = get_spec("flow_btc_v1")
    n = 2000
    ds = _make_dataset(n=n, label=np.zeros(n), label_feature="label_return_7d")
    report = check_dataset(ds, spec)
    assert report.verdict == "FAIL"


# ── Multiclass (M5g.1) ────────────────────────────────────────────────────


def test_multiclass_class_balance_pass_on_realistic_skew():
    """Mirrors the production triple-barrier distribution (59/3/38).
    3% minority is below the WARN threshold (5%) but well above the
    FAIL threshold (1%) — should WARN, not FAIL."""
    spec = get_spec("directional_btc_1h_v1")
    n = 2000
    label = np.concatenate([
        np.full(int(n * 0.59), -1.0),
        np.full(int(n * 0.03), 0.0),
        np.full(n - int(n * 0.59) - int(n * 0.03), 1.0),
    ])
    ds = _make_dataset(n=n, label=label, label_feature="label_triple_barrier")
    report = check_dataset(ds, spec)
    mc = next(c for c in report.checks if c.name == "multiclass_class_balance")
    assert mc.severity == "WARN"   # 3% < 5% warn threshold
    assert report.verdict in {"WARN", "PASS"}   # no FAIL


def test_multiclass_class_balance_fail_when_class_too_sparse():
    """A single-digit row count in the minority class is unfittable
    even with class_weight=balanced."""
    spec = get_spec("directional_btc_1h_v1")
    n = 2000
    # only 10 rows of class 0 → below min_per_class_rows=50
    label = np.concatenate([
        np.full(1000, -1.0),
        np.full(10, 0.0),
        np.full(990, 1.0),
    ])
    ds = _make_dataset(n=n, label=label, label_feature="label_triple_barrier")
    report = check_dataset(ds, spec)
    mc = next(c for c in report.checks if c.name == "multiclass_class_balance")
    assert mc.severity == "FAIL"
    assert report.verdict == "FAIL"


def test_multiclass_class_balance_pass_on_balanced():
    """Equal-thirds distribution — no warning, no failure."""
    spec = get_spec("directional_btc_1h_v1")
    n = 2100   # divisible by 3
    label = np.concatenate([
        np.full(n // 3, -1.0),
        np.full(n // 3, 0.0),
        np.full(n // 3, 1.0),
    ])
    ds = _make_dataset(n=n, label=label, label_feature="label_triple_barrier")
    report = check_dataset(ds, spec)
    mc = next(c for c in report.checks if c.name == "multiclass_class_balance")
    assert mc.severity == "PASS"


# ── Fingerprint ─────────────────────────────────────────────────────────────


def test_fingerprint_is_deterministic_for_same_data():
    ds_a = _make_dataset(n=500)
    ds_b = _make_dataset(n=500)   # built identically with same seed
    assert compute_data_fingerprint(ds_a) == compute_data_fingerprint(ds_b)


def test_fingerprint_changes_when_a_value_changes():
    ds_a = _make_dataset(n=500)
    fp_a = compute_data_fingerprint(ds_a)
    # mutate a single cell
    ds_a.X.iloc[0, 0] = ds_a.X.iloc[0, 0] + 1e-9
    fp_b = compute_data_fingerprint(ds_a)
    assert fp_a != fp_b


def test_fingerprint_changes_when_timestamp_index_shifts():
    ds_a = _make_dataset(n=500, start="2024-12-01")
    ds_b = _make_dataset(n=500, start="2024-12-02")
    assert compute_data_fingerprint(ds_a) != compute_data_fingerprint(ds_b)


# ── Aggregation + caller contract ─────────────────────────────────────────


def test_verdict_is_worst_of():
    """Multiple WARN + one FAIL should yield FAIL overall."""
    n = 2000
    features = {
        "f_normal": np.random.default_rng(0).standard_normal(n),
        "f_constant": np.full(n, 1.0),   # WARN
    }
    # Single-class label → FAIL on binary_class_balance
    ds = _make_dataset(n=n, features=features, label=np.zeros(n))
    spec = get_spec("regime_btc_v1")
    report = check_dataset(ds, spec)
    assert report.verdict == "FAIL"


def test_integrity_error_is_raisable():
    with pytest.raises(IntegrityError):
        raise IntegrityError("test")
