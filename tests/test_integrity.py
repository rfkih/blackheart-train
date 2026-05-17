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
    compute_dataset_sha,
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


# ── R1.B Label-leakage detection ───────────────────────────────────────────


def test_label_leakage_fail_on_mirror_feature():
    """A feature that IS the label (perfect Pearson ρ=1.0) must FAIL.

    This is the canonical leakage pattern: the loader joined the label
    column into the feature set by accident, or a derived feature copied
    the label one bar early. Either way the check has to catch it before
    LightGBM time burns.
    """
    n = 2000
    label = (np.arange(n) % 2).astype("float64")
    features = {
        "f_clean": np.random.default_rng(0).standard_normal(n),
        # Leaks the label directly — |ρ| = 1.0
        "f_leak": label.copy(),
    }
    ds = _make_dataset(n=n, features=features, label=label)
    spec = get_spec("regime_btc_v1")
    report = check_dataset(ds, spec)

    assert report.verdict == "FAIL", report.verdict
    leak = next(c for c in report.checks if c.name == "label_leakage")
    assert leak.severity == "FAIL"
    assert leak.details["max_score_feature"] == "f_leak"
    assert leak.details["max_score"] >= 0.95
    assert "f_leak" in leak.details["leaking_features"]
    # leakage_report on the IntegrityReport mirror is populated for both
    # PASS and FAIL — checked here on the FAIL path so the CLI can stamp
    # the experiment_run.leakage_report column either way.
    assert report.leakage_report is not None
    assert report.leakage_report["severity"] == "FAIL"


def test_label_leakage_pass_on_moderate_correlation():
    """A feature that's strongly correlated but NOT the label should PASS.

    A real predictor (e.g. rsi_14 against a forward-return label) can run
    |ρ| up to ~0.6–0.7. The check needs headroom to distinguish "real
    signal" from "the label moved into the feature matrix."
    """
    n = 2000
    label = (np.arange(n) % 2).astype("float64")
    rng = np.random.default_rng(42)
    # Correlated signal: label * 0.5 + noise — |ρ| around 0.55.
    f_signal = label * 0.5 + rng.standard_normal(n) * 0.7
    features = {
        "f_clean": rng.standard_normal(n),
        "f_signal": f_signal,
    }
    ds = _make_dataset(n=n, features=features, label=label)
    spec = get_spec("regime_btc_v1")
    report = check_dataset(ds, spec)

    leak = next(c for c in report.checks if c.name == "label_leakage")
    assert leak.severity == "PASS", (leak.message, leak.details)
    # max_score is recorded even on PASS so the operator can track the
    # highest non-leaking correlation across runs.
    assert leak.details["max_score"] < 0.95


def test_allow_leakage_demotes_fail_to_warn():
    """``run_integrity_or_raise(allow_leakage=True)`` should turn the
    leakage FAIL into a WARN, keep the leakage_report intact, and NOT raise.
    """
    from blackheart_train.train import run_integrity_or_raise

    n = 2000
    label = (np.arange(n) % 2).astype("float64")
    features = {
        "f_clean": np.random.default_rng(7).standard_normal(n),
        "f_leak": label.copy(),
    }
    ds = _make_dataset(n=n, features=features, label=label)
    spec = get_spec("regime_btc_v1")

    # Default: must raise.
    with pytest.raises(IntegrityError):
        run_integrity_or_raise(ds, spec)

    # With allow_leakage=True: returns the report, leakage demoted to WARN.
    report = run_integrity_or_raise(ds, spec, allow_leakage=True)
    leak = next(c for c in report.checks if c.name == "label_leakage")
    assert leak.severity == "WARN"
    assert "[--allow-leakage]" in leak.message
    # The overall verdict reflects the demotion — should be at most WARN
    # (other checks may also WARN, but no FAIL remains).
    assert report.verdict in ("WARN", "PASS")
    # The audit-trail data is preserved.
    assert report.leakage_report is not None
    assert report.leakage_report["max_score_feature"] == "f_leak"


def test_label_leakage_skipped_for_constant_features_only():
    """If every feature has zero variance, the leakage check has nothing
    to compute and should report PASS with a clear message (the constant-
    features check independently surfaces them as WARN).
    """
    n = 2000
    label = (np.arange(n) % 2).astype("float64")
    features = {
        "f_const_a": np.full(n, 1.0),
        "f_const_b": np.full(n, 2.0),
    }
    ds = _make_dataset(n=n, features=features, label=label)
    spec = get_spec("regime_btc_v1")
    report = check_dataset(ds, spec)
    leak = next(c for c in report.checks if c.name == "label_leakage")
    assert leak.severity == "PASS"
    # No features had non-zero variance, so the corrs map was empty.
    assert "skipped" in leak.message


# ── R1 close-out: compute_dataset_sha ─────────────────────────────────────


def test_dataset_sha_stable_for_same_shape_and_window():
    """Two datasets with the same (symbol, interval, label, shape, range,
    features) produce the same dataset_sha even if the cell values differ.
    That's the contract — dataset_sha is the *coarse* fingerprint.
    """
    n = 1500
    rng_a = np.random.default_rng(1)
    rng_b = np.random.default_rng(2)
    ds_a = _make_dataset(
        n=n, features={"f0": rng_a.standard_normal(n), "f1": rng_a.standard_normal(n)},
    )
    ds_b = _make_dataset(
        n=n, features={"f0": rng_b.standard_normal(n), "f1": rng_b.standard_normal(n)},
    )
    spec = get_spec("regime_btc_v1")
    sha_a = compute_dataset_sha(ds_a, spec)
    sha_b = compute_dataset_sha(ds_b, spec)
    assert sha_a == sha_b, "different values must NOT change the coarse sha"
    # data_fingerprint, in contrast, MUST differ — it's bit-exact.
    assert compute_data_fingerprint(ds_a) != compute_data_fingerprint(ds_b)


def test_dataset_sha_changes_when_shape_changes():
    n = 1500
    ds_full = _make_dataset(n=n)
    ds_short = _make_dataset(n=n - 100)
    spec = get_spec("regime_btc_v1")
    assert compute_dataset_sha(ds_full, spec) != compute_dataset_sha(ds_short, spec)


def test_dataset_sha_changes_when_symbol_changes():
    ds = _make_dataset(n=1500)
    spec_btc = get_spec("regime_btc_v1")
    spec_eth = replace(spec_btc, symbol="ETHUSDT")
    assert compute_dataset_sha(ds, spec_btc) != compute_dataset_sha(ds, spec_eth)


def test_dataset_sha_changes_when_features_change():
    n = 1500
    ds_a = _make_dataset(n=n, features={"f0": np.zeros(n), "f1": np.zeros(n)})
    ds_b = _make_dataset(n=n, features={"f0": np.zeros(n), "f2": np.zeros(n)})
    spec = get_spec("regime_btc_v1")
    assert compute_dataset_sha(ds_a, spec) != compute_dataset_sha(ds_b, spec)


def test_dataset_sha_independent_of_feature_order():
    """Column order changes (e.g. a registry-query reorder) must NOT churn
    the sha — feature_names is sorted in the hash by design.
    """
    n = 1500
    arr0 = np.random.default_rng(0).standard_normal(n)
    arr1 = np.random.default_rng(1).standard_normal(n)
    ds_ab = _make_dataset(n=n, features={"f_a": arr0, "f_b": arr1})
    ds_ba = _make_dataset(n=n, features={"f_b": arr1, "f_a": arr0})
    spec = get_spec("regime_btc_v1")
    # Note: data_fingerprint WOULD differ here (column order is part of
    # the byte stream). dataset_sha does NOT.
    assert compute_dataset_sha(ds_ab, spec) == compute_dataset_sha(ds_ba, spec)


def test_label_leakage_multiclass_path():
    """For multiclass labels, the check uses normalized MI instead of
    Pearson. Confirm the multiclass branch flags a perfect-info feature.
    """
    n = 2000
    # Triple-barrier-style 3-class label: 0, 1, 2 cycling.
    label = (np.arange(n) % 3).astype("float64")
    features = {
        "f_noise": np.random.default_rng(0).standard_normal(n),
        # Encodes the label exactly — MI / H(y) should hit 1.0.
        "f_leak": label.copy(),
    }
    ds = _make_dataset(n=n, features=features, label=label)
    spec = replace(get_spec("regime_btc_v1"), objective="multiclass")
    report = check_dataset(ds, spec)
    leak = next(c for c in report.checks if c.name == "label_leakage")
    assert leak.severity == "FAIL", (leak.message, leak.details)
    assert leak.details["method"] == "mutual_info_norm"
    assert leak.details["max_score_feature"] == "f_leak"
    assert leak.details["max_score"] >= 0.95
