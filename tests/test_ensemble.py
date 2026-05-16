"""Unit tests for the M5g.3 ensemble.

Pure-function tests on synthetic data. The ensemble's three base
models (LightGBM + XGBoost + Logistic-L1) each fit in <2s on a
2000-row 3-feature synthetic set, so the whole file runs in <30s.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from blackheart_train.ensemble import (
    BASE_MODEL_ORDER,
    Ensemble,
    base_model_disagreement,
    evaluate_ensemble,
    fit_ensemble,
    predict_proba_ensemble,
)
from blackheart_train.specs import get_spec
from blackheart_train.train import N_MULTICLASS_CLASSES, encode_multiclass


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_multiclass_dataset(*, n: int = 2000, seed: int = 0):
    """Synthetic 3-class dataset with two informative features. All
    three classes present in roughly the proportions the threshold
    bands produce. Returns (X, y_encoded)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-12-01", periods=n, freq="1h")
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    noise = rng.standard_normal(n)
    signal = 0.6 * a + 0.4 * b + 0.3 * rng.standard_normal(n)
    y = np.where(signal < -0.5, -1.0, np.where(signal > 0.5, 1.0, 0.0))
    X = pd.DataFrame({"a": a, "b": b, "noise": noise}, index=idx)
    y_ser = pd.Series(y, index=idx, name="y")
    return X, encode_multiclass(y_ser)


def _split(X: pd.DataFrame, y_enc: pd.Series, val_frac: float = 0.2):
    n = len(X)
    n_val = max(1, int(round(n * val_frac)))
    n_tr = n - n_val
    return X.iloc[:n_tr], y_enc.iloc[:n_tr], X.iloc[n_tr:], y_enc.iloc[n_tr:]


# ── fit_ensemble ──────────────────────────────────────────────────────────


def test_fit_ensemble_includes_every_requested_kind():
    """Ensemble holds exactly the base models the spec requested,
    in :data:`BASE_MODEL_ORDER`."""
    X, y_enc = _make_multiclass_dataset(seed=1)
    X_tr, y_tr, _, _ = _split(X, y_enc)
    spec = get_spec("directional_btc_1h_v1")
    ensemble = fit_ensemble(X_tr, y_tr, spec)
    kinds = [fbm.kind for fbm in ensemble.models]
    assert kinds == list(BASE_MODEL_ORDER)


def test_fit_ensemble_respects_partial_base_model_request():
    """A spec asking for only ``("lightgbm", "logreg_l1")`` skips
    XGBoost — the result has 2 base models in BASE_MODEL_ORDER order."""
    from dataclasses import replace
    X, y_enc = _make_multiclass_dataset(seed=2)
    X_tr, y_tr, _, _ = _split(X, y_enc)
    spec = get_spec("directional_btc_1h_v1")
    spec_two = replace(spec, base_models=("lightgbm", "logreg_l1"))
    ensemble = fit_ensemble(X_tr, y_tr, spec_two)
    kinds = [fbm.kind for fbm in ensemble.models]
    assert kinds == ["lightgbm", "logreg_l1"]


def test_fit_ensemble_rejects_unknown_base_model():
    from dataclasses import replace
    X, y_enc = _make_multiclass_dataset(seed=3)
    X_tr, y_tr, _, _ = _split(X, y_enc)
    spec = get_spec("directional_btc_1h_v1")
    spec_bad = replace(spec, base_models=("lightgbm", "transformer"))
    with pytest.raises(ValueError, match="unknown base models"):
        fit_ensemble(X_tr, y_tr, spec_bad)


def test_fit_ensemble_logreg_carries_scaler():
    """The linear base model needs scaled inputs; the scaler must be
    persisted on the FittedBaseModel so predict-time symmetry holds."""
    X, y_enc = _make_multiclass_dataset(seed=4)
    X_tr, y_tr, _, _ = _split(X, y_enc)
    spec = get_spec("directional_btc_1h_v1")
    ensemble = fit_ensemble(X_tr, y_tr, spec)
    lr = next(fbm for fbm in ensemble.models if fbm.kind == "logreg_l1")
    assert lr.scaler is not None
    # The tree models do NOT have a scaler — they're scale-invariant.
    for kind in ("lightgbm", "xgboost"):
        fbm = next(f for f in ensemble.models if f.kind == kind)
        assert fbm.scaler is None


# ── predict_proba_ensemble ────────────────────────────────────────────────


def test_predict_proba_ensemble_shape_and_normalisation():
    """Output shape is (n_val, n_classes) and each row sums to ~1 (it's
    the mean of three probability rows that each sum to 1)."""
    X, y_enc = _make_multiclass_dataset(seed=5)
    X_tr, y_tr, X_val, _ = _split(X, y_enc)
    spec = get_spec("directional_btc_1h_v1")
    ensemble = fit_ensemble(X_tr, y_tr, spec)
    proba = predict_proba_ensemble(ensemble, X_val)
    assert proba.shape == (len(X_val), N_MULTICLASS_CLASSES)
    row_sums = proba.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-9)


def test_predict_proba_ensemble_raises_on_empty():
    empty = Ensemble(models=[])
    with pytest.raises(ValueError, match="no fitted base models"):
        predict_proba_ensemble(empty, pd.DataFrame({"a": [1.0]}))


# ── base_model_disagreement ───────────────────────────────────────────────


def test_disagreement_zero_when_all_models_agree():
    """If every base model predicts the same proba row, disagreement is
    ~0 — population std (ddof=0) of identical values is mathematically 0
    but numpy's std implementation goes through a sum-of-squared-deviations
    pathway that leaks tiny rounding (≤ a few × 1e-16). Use ``allclose``
    with a tolerance well inside that floor."""
    # 3 models × 5 rows × 3 classes, all identical
    p = np.array([[0.7, 0.2, 0.1]] * 5)
    stack = np.stack([p, p, p], axis=0)
    dis = base_model_disagreement(stack)
    assert dis.shape == (5, 3)
    assert np.allclose(dis, 0.0, atol=1e-12)


def test_disagreement_positive_when_models_disagree():
    """Three different probas → strictly positive disagreement."""
    p1 = np.array([[0.9, 0.05, 0.05]] * 5)
    p2 = np.array([[0.05, 0.9, 0.05]] * 5)
    p3 = np.array([[0.05, 0.05, 0.9]] * 5)
    stack = np.stack([p1, p2, p3], axis=0)
    dis = base_model_disagreement(stack)
    # All three classes show identical disagreement (symmetric case).
    assert np.all(dis > 0.0)
    assert np.allclose(dis[:, 0], dis[:, 1], atol=1e-12)


# ── evaluate_ensemble ─────────────────────────────────────────────────────


def test_evaluate_ensemble_carries_per_base_and_ensemble_keys():
    """Per-base metrics use prefixed keys (``lgb_*``, ``xgb_*``,
    ``lr_*``); ensemble metrics are unprefixed; disagreement adds
    ``mean_disagreement[_class_i]``."""
    X, y_enc = _make_multiclass_dataset(seed=7)
    X_tr, y_tr, X_val, y_val = _split(X, y_enc)
    spec = get_spec("directional_btc_1h_v1")
    ensemble = fit_ensemble(X_tr, y_tr, spec)
    proba, m = evaluate_ensemble(
        ensemble, X_val, y_val, n_classes=N_MULTICLASS_CLASSES,
    )
    # Per-base
    for prefix in ("lgb_", "xgb_", "lr_"):
        assert f"{prefix}log_loss" in m
        assert f"{prefix}accuracy" in m
        assert f"{prefix}macro_auc_ovr" in m
    # Ensemble (unprefixed) — same three keys
    for key in ("log_loss", "accuracy", "macro_auc_ovr"):
        assert key in m
    # Disagreement
    assert "mean_disagreement" in m
    for i in range(N_MULTICLASS_CLASSES):
        assert f"mean_disagreement_class_{i}" in m
    # Disagreement values are non-negative
    assert m["mean_disagreement"] >= 0.0
    # proba comes back so caller can re-use without re-predicting
    assert proba.shape == (len(X_val), N_MULTICLASS_CLASSES)


def test_evaluate_ensemble_handles_missing_class_in_val_slice():
    """If a val slice happens to be missing one class, macro AUC OVR
    falls back to NaN (the contract from train._evaluate). Per-base
    macro_auc_ovr falls back the same way — there's no class-coverage
    that helps the AUC OVR computation per model."""
    X, y_enc = _make_multiclass_dataset(seed=8)
    X_tr, y_tr, X_val, y_val = _split(X, y_enc)
    # Drop every class-1 row from the val slice.
    keep = y_val != 1
    X_val = X_val.loc[keep]
    y_val = y_val.loc[keep]
    spec = get_spec("directional_btc_1h_v1")
    ensemble = fit_ensemble(X_tr, y_tr, spec)
    _, m = evaluate_ensemble(
        ensemble, X_val, y_val, n_classes=N_MULTICLASS_CLASSES,
    )
    # macro AUC OVR can't be computed when y_true is missing a class
    for key in ("macro_auc_ovr", "lgb_macro_auc_ovr",
                "xgb_macro_auc_ovr", "lr_macro_auc_ovr"):
        assert math.isnan(m[key]), f"{key} should be NaN; got {m[key]}"
    # Accuracy and log_loss are still defined
    for key in ("accuracy", "lgb_accuracy", "log_loss", "lgb_log_loss"):
        assert not math.isnan(m[key])


def test_evaluate_ensemble_raises_on_empty():
    empty = Ensemble(models=[])
    with pytest.raises(ValueError, match="empty ensemble"):
        evaluate_ensemble(
            empty,
            pd.DataFrame({"a": [1.0]}),
            pd.Series([0], dtype=int),
            n_classes=N_MULTICLASS_CLASSES,
        )


# ── Spec-side ─────────────────────────────────────────────────────────────


def test_directional_spec_requests_three_base_models():
    """Spec is the source of truth — bumping or trimming the ensemble
    happens here and propagates to the training path."""
    spec = get_spec("directional_btc_1h_v1")
    assert spec.base_models == ("lightgbm", "xgboost", "logreg_l1")


def test_phase2_specs_keep_single_base_model_default():
    """Existing modulator specs (regime/positioning/flow v1/v2) must
    not have been disturbed — they're still single-LightGBM."""
    for name in (
        "regime_btc_v1", "positioning_btc_v1", "flow_btc_v1",
        "regime_btc_v2", "positioning_btc_v2", "flow_btc_v2",
    ):
        assert get_spec(name).base_models == ("lightgbm",)


# ── Audit-fix tests (EB1 / EB2 / EB3) ────────────────────────────────────


def test_spec_rejects_empty_base_models():
    """EB2/EB3 guard: an empty base_models tuple is meaningless and
    must fail at ModelSpec construction."""
    from blackheart_train.specs import ModelSpec
    from datetime import datetime as _dt
    with pytest.raises(ValueError, match="base_models cannot be empty"):
        ModelSpec(
            name="x", purpose="directional", label_feature="label_triple_barrier",
            label_version=1, objective="multiclass", symbol="BTCUSDT", interval="1h",
            train_start=_dt(2024, 12, 1), train_end=_dt(2026, 5, 14),
            base_models=(),
        )


def test_spec_rejects_base_models_not_starting_with_lightgbm():
    """EB2 fix: LightGBM must be the first base model — the artifact
    only persists LightGBM today, and the booster extractor looks for
    it. Other orderings or non-LightGBM-led tuples are refused at spec
    construction so the bug can't reach training."""
    from blackheart_train.specs import ModelSpec
    from datetime import datetime as _dt
    base = dict(
        name="x", purpose="directional", label_feature="label_triple_barrier",
        label_version=1, objective="multiclass", symbol="BTCUSDT", interval="1h",
        train_start=_dt(2024, 12, 1), train_end=_dt(2026, 5, 14),
    )
    # EB2: xgboost-only — would crash booster extraction
    with pytest.raises(ValueError, match="must start with 'lightgbm'"):
        ModelSpec(base_models=("xgboost",), **base)
    # EB3: leading non-LightGBM with LightGBM elsewhere — spec lies
    # about which model is the primary
    with pytest.raises(ValueError, match="must start with 'lightgbm'"):
        ModelSpec(base_models=("xgboost", "lightgbm"), **base)
    with pytest.raises(ValueError, match="must start with 'lightgbm'"):
        ModelSpec(base_models=("logreg_l1",), **base)


# ── M5g.3 phase 2 — full ensemble persistence ────────────────────────────


def _make_ds_and_integrity(X: pd.DataFrame, y_enc: pd.Series):
    """Helper: build a minimal LoadedDataset + IntegrityReport pair
    around ``(X, y_enc)``. Uses a registry-only label so the deployment-
    ready branch isolates the ensemble/meta-label flags."""
    from blackheart_train.integrity import CheckResult, IntegrityReport
    from blackheart_train.loader import LoadedDataset
    n = len(X)
    ds = LoadedDataset(
        X=X,
        y=pd.Series(y_enc.to_numpy(), index=X.index, dtype="float64"),
        feature_names=tuple(X.columns),
        n_bar_slots_total=n, n_bar_slots_dropped_nan=0,
        per_feature_non_null={c: n for c in X.columns},
        per_feature_pct_non_null={c: 1.0 for c in X.columns},
        label_feature="label_regime_risk_on_48h", label_version=1,
    )
    integrity = IntegrityReport(
        verdict="PASS",
        checks=[CheckResult(name="x", severity="PASS", message="ok")],
        data_fingerprint="deadbeef" * 8,
    )
    return ds, integrity


def test_ensemble_content_signature_deterministic():
    """Two fits of the same spec on the same data produce identical
    signatures. Required so content_sha256 is stable across re-runs —
    a re-train lands at the same artifact path."""
    from blackheart_train.ensemble import ensemble_content_signature
    X, y_enc = _make_multiclass_dataset(n=600, seed=11)
    spec = get_spec("directional_btc_1h_v1")
    ensemble_a = fit_ensemble(X, y_enc, spec)
    ensemble_b = fit_ensemble(X, y_enc, spec)
    assert ensemble_content_signature(ensemble_a) == ensemble_content_signature(ensemble_b)


def test_ensemble_content_signature_changes_with_data():
    """Different training data → different signature, so two ensembles
    with identical specs but distinct fit-time inputs get distinct
    content shas (addressing follows estimator identity, not just spec
    identity)."""
    from blackheart_train.ensemble import ensemble_content_signature
    X_a, y_a = _make_multiclass_dataset(n=600, seed=11)
    X_b, y_b = _make_multiclass_dataset(n=600, seed=22)
    spec = get_spec("directional_btc_1h_v1")
    ensemble_a = fit_ensemble(X_a, y_a, spec)
    ensemble_b = fit_ensemble(X_b, y_b, spec)
    assert ensemble_content_signature(ensemble_a) != ensemble_content_signature(ensemble_b)


def test_ensemble_content_signature_rejects_empty():
    from blackheart_train.ensemble import Ensemble, ensemble_content_signature
    with pytest.raises(ValueError, match="empty ensemble"):
        ensemble_content_signature(Ensemble(models=[]))


def test_ensemble_content_signature_strips_xgboost_version():
    """XGBoost's JSON dump stamps the library version at the top level
    (e.g. ``"version": [2, 1, 4]``). Including it in the signature
    would churn content_sha on every ``pip upgrade xgboost`` even when
    the trees are bit-identical. The signature must scrub it — two
    boosters with identical trees but different version stamps produce
    the same signature."""
    import json as _json
    from blackheart_train.ensemble import ensemble_content_signature

    X, y_enc = _make_multiclass_dataset(n=400, seed=33)
    spec = get_spec("directional_btc_1h_v1")
    ensemble = fit_ensemble(X, y_enc, spec)
    xgb_fbm = next(fbm for fbm in ensemble.models if fbm.kind == "xgboost")

    real_save_raw = xgb_fbm.model.get_booster().save_raw
    real_json = real_save_raw(raw_format="json")
    if isinstance(real_json, (bytes, bytearray)):
        real_dict = _json.loads(real_json.decode("utf-8"))
    else:
        real_dict = _json.loads(real_json)

    # First: capture the signature with the actual version stamp.
    sig_real = ensemble_content_signature(ensemble)

    # Now monkey the save_raw to return the same trees with a forged
    # version. The signature must match — proves the version field is
    # not contributing to the hash.
    forged = dict(real_dict, version=[99, 99, 99])
    forged_bytes = _json.dumps(forged).encode("utf-8")

    class _FakeBooster:
        def save_raw(self, raw_format: str = "json"):
            return forged_bytes

    class _FakeXGBClassifier:
        def __init__(self, fake_booster):
            self._fake = fake_booster
        def get_booster(self):
            return self._fake

    xgb_fbm.model = _FakeXGBClassifier(_FakeBooster())
    sig_forged = ensemble_content_signature(ensemble)
    assert sig_real == sig_forged


def test_build_payload_deployment_readiness_branches():
    """Three deployment_ready branches after M5g.3 phase 2:

    * Single-model + registry inputs/label → True (M5a baseline,
      unchanged).
    * Ensemble + ``meta_label_enabled=True`` → False. The Phase-2
      ensemble blocker is lifted (the full ensemble is persisted) but
      the meta-label model isn't yet on disk (M5g.4 phase 2). Promoting
      such an artifact would mean live inference runs the ensemble
      without the gate — every bar would trade — and the metrics in
      the payload describe gated behaviour. Same EB1 hazard, one level
      up the stack.
    * Ensemble + ``meta_label_enabled=False`` → True. The Phase-2
      unblock: full ensemble persisted, no gate to miss.
    """
    from dataclasses import replace as _replace
    import lightgbm as lgb
    from blackheart_train.train import build_payload

    X, y_enc = _make_multiclass_dataset(n=600, seed=99)
    n = len(X)
    ds, integrity = _make_ds_and_integrity(X, y_enc)

    # ── Single-model spec → True ────────────────────────────────────────
    spec_single = _replace(get_spec("regime_btc_v1"), derived_features=())
    binary_y = (y_enc.to_numpy() == 2).astype(int)
    booster_lgb = lgb.LGBMClassifier(
        objective="binary", n_estimators=2, verbosity=-1,
    )
    booster_lgb.fit(X, binary_y)
    payload_single = build_payload(
        ds, spec_single, booster_lgb.booster_, {"auc": 0.6}, integrity,
        n_train_rows=n * 4 // 5, n_val_rows=n // 5, eval_kind="holdout_80_20",
    )
    assert payload_single["payload_version"] == 2
    assert payload_single["booster"] is not None
    assert payload_single["ensemble"] is None
    assert payload_single["deployment_readiness"]["deployment_ready"] is True

    # ── Ensemble + meta-label → False ───────────────────────────────────
    spec_ensemble_meta = _replace(
        get_spec("directional_btc_1h_v1"),
        training_intervals=(), meta_label_enabled=True,
    )
    ensemble = fit_ensemble(X, y_enc, spec_ensemble_meta)
    payload_meta = build_payload(
        ds, spec_ensemble_meta, ensemble, {"macro_auc_ovr": 0.55}, integrity,
        n_train_rows=n * 4 // 5, n_val_rows=n // 5, eval_kind="holdout_80_20",
    )
    assert payload_meta["payload_version"] == 2
    assert payload_meta["booster"] is None
    assert payload_meta["ensemble"] is ensemble
    assert payload_meta["deployment_readiness"]["deployment_ready"] is False

    # ── Ensemble without meta-label → True ──────────────────────────────
    spec_ensemble_nometa = _replace(spec_ensemble_meta, meta_label_enabled=False)
    payload_nometa = build_payload(
        ds, spec_ensemble_nometa, ensemble, {"macro_auc_ovr": 0.55}, integrity,
        n_train_rows=n * 4 // 5, n_val_rows=n // 5, eval_kind="holdout_80_20",
    )
    assert payload_nometa["deployment_readiness"]["deployment_ready"] is True


def test_build_payload_rejects_mismatched_estimator_shape():
    """Spec/estimator-type mismatch must raise. The alternative is an
    artifact whose spec block claims an ensemble while only a single
    booster is on disk (or vice versa) — same metrics-vs-deployment
    divergence the EB1 audit closed."""
    from dataclasses import replace as _replace
    import lightgbm as lgb
    from blackheart_train.train import build_payload

    X, y_enc = _make_multiclass_dataset(n=300, seed=77)
    ds, integrity = _make_ds_and_integrity(X, y_enc)

    spec_ensemble = _replace(
        get_spec("directional_btc_1h_v1"), training_intervals=(),
    )
    booster_lgb = lgb.LGBMClassifier(
        objective="multiclass", n_estimators=2, verbosity=-1,
    )
    booster_lgb.fit(X, y_enc)

    # Ensemble spec + bare Booster → reject
    with pytest.raises(ValueError, match="declares .* base models"):
        build_payload(
            ds, spec_ensemble, booster_lgb.booster_, {"macro_auc_ovr": 0.55},
            integrity, n_train_rows=240, n_val_rows=60,
            eval_kind="holdout_80_20",
        )

    # Single-model spec + Ensemble → reject
    spec_single = _replace(spec_ensemble, base_models=("lightgbm",))
    ensemble = fit_ensemble(X, y_enc, spec_ensemble)
    with pytest.raises(ValueError, match="single-model"):
        build_payload(
            ds, spec_single, ensemble, {"macro_auc_ovr": 0.55},
            integrity, n_train_rows=240, n_val_rows=60,
            eval_kind="holdout_80_20",
        )


def test_build_payload_rejects_mismatched_ensemble_kinds():
    """Defense-in-depth: an Ensemble whose ``models`` don't match
    ``spec.base_models`` (e.g. only LightGBM fitted for a 3-model spec)
    must be refused at build_payload, not silently written into an
    artifact whose spec block claims more than what's persisted.

    ``fit_ensemble`` populates the Ensemble to match the spec in normal
    operation, so this guard is dormant — it covers test code, future
    refactors that build Ensembles by hand, or a partial fit that
    swallowed an exception."""
    from dataclasses import replace as _replace
    from blackheart_train.ensemble import Ensemble, FittedBaseModel
    from blackheart_train.train import build_payload
    import lightgbm as lgb

    X, y_enc = _make_multiclass_dataset(n=300, seed=88)
    ds, integrity = _make_ds_and_integrity(X, y_enc)

    spec_ensemble = _replace(
        get_spec("directional_btc_1h_v1"), training_intervals=(),
    )
    # Build a 1-model "Ensemble" that masquerades as the 3-model spec.
    lgb_clf = lgb.LGBMClassifier(
        objective="multiclass", n_estimators=2, verbosity=-1,
    )
    lgb_clf.fit(X, y_enc)
    fake_ensemble = Ensemble(models=[
        FittedBaseModel(kind="lightgbm", model=lgb_clf, scaler=None),
    ])

    with pytest.raises(ValueError, match="kinds .* refusing"):
        build_payload(
            ds, spec_ensemble, fake_ensemble, {"macro_auc_ovr": 0.55},
            integrity, n_train_rows=240, n_val_rows=60,
            eval_kind="holdout_80_20",
        )


def test_ensemble_payload_round_trip(tmp_path):
    """Fit → build_payload → write_artifact → read_artifact → predict_proba
    matches bit-for-bit on the round-tripped ensemble. Verifies the
    Ensemble (and its three base models + scaler) survives pickle, and
    that content_sha stays stable across the write/read boundary."""
    from dataclasses import replace as _replace
    from blackheart_train.artifacts import read_artifact, write_artifact
    from blackheart_train.ensemble import predict_proba_ensemble
    from blackheart_train.train import build_payload

    X, y_enc = _make_multiclass_dataset(n=600, seed=123)
    spec = _replace(
        get_spec("directional_btc_1h_v1"),
        training_intervals=(), meta_label_enabled=False,
    )
    ensemble = fit_ensemble(X, y_enc, spec)
    X_val = X.iloc[-100:]
    proba_pre = predict_proba_ensemble(ensemble, X_val)

    ds, integrity = _make_ds_and_integrity(X, y_enc)
    payload = build_payload(
        ds, spec, ensemble, {"macro_auc_ovr": 0.55}, integrity,
        n_train_rows=500, n_val_rows=100, eval_kind="holdout_80_20",
    )
    # The two production callers (train_with_dataset / train_via_walk_forward)
    # set this AFTER build_payload returns; mirror that here so the
    # written artifact matches what the real path produces.
    payload["walk_forward"] = None

    path, _ = write_artifact(payload, payload["content_sha256"], tmp_path)
    assert path.exists()

    loaded = read_artifact(payload["content_sha256"], tmp_path)
    assert loaded["payload_version"] == 2
    assert loaded["booster"] is None
    assert loaded["ensemble"] is not None

    proba_post = predict_proba_ensemble(loaded["ensemble"], X_val)
    np.testing.assert_allclose(proba_pre, proba_post, atol=1e-12)
