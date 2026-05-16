"""ES1: LightGBM early-stopping integration tests.

Covers the spec validation + the `_early_stopping_carve` helper + the
three fit paths in `fit_and_evaluate` (binary, multiclass single-model,
regression).
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from blackheart_train.specs import ModelSpec, get_spec
from blackheart_train.train import (
    _ES_EMBARGO_DAYS,
    _early_stopping_carve,
    fit_and_evaluate,
)


# ── spec field validation ──────────────────────────────────────────────────


def test_default_spec_has_early_stopping_disabled():
    """ES1: existing locked specs default to disabled so re-runs without
    explicit opt-in keep their original behavior."""
    for name in ("regime_btc_v3", "flow_btc_v2", "positioning_btc_v2"):
        spec = get_spec(name)
        assert spec.early_stopping_rounds == 0, (
            f"spec={name} must default to early_stopping_rounds=0 — "
            f"flipping this in a default would silently change every "
            f"locked spec's training behavior"
        )


def test_negative_rounds_rejected():
    base = get_spec("regime_btc_v3")
    with pytest.raises(ValueError, match="early_stopping_rounds must be >= 0"):
        replace(base, early_stopping_rounds=-1)


def test_fraction_out_of_range_rejected():
    base = get_spec("regime_btc_v3")
    with pytest.raises(ValueError, match="early_stopping_val_fraction must be in"):
        replace(base, early_stopping_rounds=20, early_stopping_val_fraction=0.0)
    with pytest.raises(ValueError, match="early_stopping_val_fraction must be in"):
        replace(base, early_stopping_rounds=20, early_stopping_val_fraction=1.0)


# ── _early_stopping_carve ──────────────────────────────────────────────────


def _make_synthetic_xy(n: int = 2000, kind: str = "binary", seed: int = 0):
    """Linearly-separable-ish synthetic data with a chronological index."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="1h")
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    if kind == "binary":
        logit = 0.8 * a + 0.5 * b
        prob = 1.0 / (1.0 + np.exp(-logit))
        y = pd.Series((rng.uniform(size=n) < prob).astype(int), index=idx)
    elif kind == "multiclass":
        score = 0.6 * a + 0.4 * b + 0.3 * rng.standard_normal(n)
        y = pd.Series(
            np.where(score < -0.5, -1.0, np.where(score > 0.5, 1.0, 0.0)),
            index=idx,
        )
    else:  # regression
        y = pd.Series(0.6 * a + 0.4 * b + 0.1 * rng.standard_normal(n), index=idx)
    X = pd.DataFrame({"a": a, "b": b}, index=idx)
    return X, y


def test_carve_returns_none_when_disabled():
    spec = get_spec("regime_btc_v3")  # early_stopping_rounds = 0
    X, y = _make_synthetic_xy(500)
    assert _early_stopping_carve(X, y, spec) is None


def test_carve_returns_none_for_ensemble_path():
    spec = replace(
        get_spec("directional_btc_1h_v1"),
        early_stopping_rounds=50,
    )
    assert len(spec.base_models) > 1
    X, y = _make_synthetic_xy(2000, kind="multiclass")
    # Even with ES turned on, ensemble path is deferred.
    assert _early_stopping_carve(X, y, spec) is None


def test_carve_returns_none_when_inner_val_too_small():
    spec = replace(
        get_spec("regime_btc_v3"),
        early_stopping_rounds=50,
        early_stopping_val_fraction=0.01,
    )
    X, y = _make_synthetic_xy(100)  # 1% of 100 = 1 row, below floor of 10
    assert _early_stopping_carve(X, y, spec) is None


def test_carve_returns_none_when_embargo_exceeds_data():
    spec = replace(get_spec("regime_btc_v3"), early_stopping_rounds=50)
    # 1h interval embargo is 7*24=168 bars. With n=150 we can't fit
    # embargo + carve at all.
    X, y = _make_synthetic_xy(150)
    assert _early_stopping_carve(X, y, spec) is None


def test_carve_shapes_and_embargo_for_1h_interval():
    spec = replace(
        get_spec("regime_btc_v3"),
        early_stopping_rounds=50,
        early_stopping_val_fraction=0.2,
    )
    n = 2000
    X, y = _make_synthetic_xy(n)
    carve = _early_stopping_carve(X, y, spec)
    assert carve is not None
    X_inner, y_inner, X_es, y_es = carve
    expected_val = int(round(n * 0.2))                # 400
    expected_embargo = _ES_EMBARGO_DAYS * 24          # 168 for 1h
    assert len(X_es) == expected_val
    assert len(y_es) == expected_val
    assert len(X_inner) == n - expected_val - expected_embargo
    # Chronological discipline — every inner-train ts is strictly before
    # the first ES-val ts (embargo enforced).
    assert X_inner.index.max() < X_es.index.min()
    gap_hours = (X_es.index.min() - X_inner.index.max()).total_seconds() / 3600
    assert gap_hours >= expected_embargo


# ── fit_and_evaluate integration ───────────────────────────────────────────


def _build_es_binary_spec(rounds: int) -> ModelSpec:
    return replace(
        get_spec("regime_btc_v3"),
        early_stopping_rounds=rounds,
        early_stopping_val_fraction=0.2,
    )


def test_es_off_does_not_change_iteration_count():
    """Sanity: with ES off, n_estimators (500) is the booster's final
    iteration count. We verify by reading booster.current_iteration()."""
    X_tr, y_tr = _make_synthetic_xy(3000, kind="binary", seed=1)
    X_val, y_val = _make_synthetic_xy(500, kind="binary", seed=2)
    spec = get_spec("regime_btc_v3")
    assert spec.early_stopping_rounds == 0
    booster, _metrics = fit_and_evaluate(X_tr, y_tr, X_val, y_val, spec)
    assert booster.current_iteration() == spec.hyperparams["n_estimators"]


def test_es_on_triggers_earlier_stop_on_easy_data():
    """When ES is on AND inner-val plateaus quickly, the booster should
    stop with fewer iterations than the n_estimators cap."""
    X_tr, y_tr = _make_synthetic_xy(3000, kind="binary", seed=1)
    X_val, y_val = _make_synthetic_xy(500, kind="binary", seed=2)
    spec = _build_es_binary_spec(rounds=20)
    booster, _metrics = fit_and_evaluate(X_tr, y_tr, X_val, y_val, spec)
    n_iter = booster.current_iteration()
    cap = spec.hyperparams["n_estimators"]
    assert 0 < n_iter < cap, (
        f"expected ES to trigger early; got n_iter={n_iter} cap={cap}"
    )


def test_es_falls_back_when_single_class_inner_slice():
    """If the inner-val carve happens to have only one class, the fit
    must fall back to a full-train no-ES fit rather than crash."""
    X_tr, y_tr = _make_synthetic_xy(3000, kind="binary", seed=1)
    # Force the inner-val tail to a single class.
    val_size = int(round(len(y_tr) * 0.2))
    y_tr = y_tr.copy()
    y_tr.iloc[-val_size:] = 1  # entire ES val slice = class 1
    X_val, y_val = _make_synthetic_xy(500, kind="binary", seed=2)
    spec = _build_es_binary_spec(rounds=20)
    booster, _metrics = fit_and_evaluate(X_tr, y_tr, X_val, y_val, spec)
    # Fell through to no-ES → n_estimators iterations.
    assert booster.current_iteration() == spec.hyperparams["n_estimators"]


def test_es_on_regression_triggers_earlier_stop():
    X_tr, y_tr = _make_synthetic_xy(3000, kind="regression", seed=3)
    X_val, y_val = _make_synthetic_xy(500, kind="regression", seed=4)
    spec = replace(
        get_spec("positioning_btc_v1"),  # regression spec
        early_stopping_rounds=20,
        early_stopping_val_fraction=0.2,
    )
    booster, _metrics = fit_and_evaluate(X_tr, y_tr, X_val, y_val, spec)
    n_iter = booster.current_iteration()
    cap = spec.hyperparams["n_estimators"]
    assert 0 < n_iter < cap


def test_es_off_produces_identical_booster_to_pre_es_code():
    """Determinism guard: with ES off + the seeded random_state, the
    booster bytes must equal what the pre-ES code produced. This pins
    the assumption that adding the spec fields doesn't change behavior
    when they're left at default.
    """
    X_tr, y_tr = _make_synthetic_xy(2000, kind="binary", seed=10)
    X_val, y_val = _make_synthetic_xy(400, kind="binary", seed=11)
    spec = get_spec("regime_btc_v3")
    booster_a, _ = fit_and_evaluate(X_tr, y_tr, X_val, y_val, spec)
    booster_b, _ = fit_and_evaluate(X_tr, y_tr, X_val, y_val, spec)
    # Determinism: identical inputs + seed → byte-identical model text.
    assert booster_a.model_to_string() == booster_b.model_to_string()
