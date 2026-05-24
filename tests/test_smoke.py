"""Smoke tests — pure-function, no DB.

DB-touching tests are deferred to M5b; for M5a we verify the surface
the rest of the pipeline depends on (spec lookup, artifact roundtrip,
content-sha addressing, split semantics, metric shapes).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from blackheart_train.artifacts import (
    compute_content_sha,
    read_artifact,
    write_artifact,
)
from blackheart_train.specs import SPECS, get_spec
from blackheart_train.train import (
    _evaluate,
    decode_multiclass,
    encode_multiclass,
    filter_eval_to_serving_interval,
    split_chronological,
)


# ── Spec layer ─────────────────────────────────────────────────────────────


def test_locked_specs_exist():
    assert set(SPECS.keys()) == {
        "regime_btc_v1", "positioning_btc_v1", "flow_btc_v1",
        "regime_btc_v2", "positioning_btc_v2", "flow_btc_v2",
        # Phase 4 Session 2: registry-only twin of regime_btc_v2 — no
        # derived_features, label resolved via feature_registry (V77).
        "regime_btc_v3",
        # 2026-05-20: first per-symbol ML spec on ETHUSDT (V110 expanded
        # the registry-feature symbol scope to include ETHUSDT).
        "regime_eth_v1",
        # M5g.1: Phase 3 directional model (3-class triple-barrier).
        "directional_btc_1h_v1",
        # 2026-05-21: binary directional twin of v1 — resolves
        # ANTI_PATTERN 20fa437f (orchestrator model_registry validator
        # rejects objective='multiclass'). v2 is the registerable peer
        # the researcher loop drives into HYBRID sweeps.
        "directional_btc_1h_v2",
        # 2026-05-21: triple-barrier-binary directional spec (v3) +
        # V111 bar-level feature stack augmentations (v4 = v3 + 5 new
        # bar-level features; v5 = v4 + short-horizon asymmetric label;
        # v6 = v5 with looser k_tp/k_sl). All BTC-1h.
        "directional_btc_1h_v3",
        "directional_btc_1h_v4",
        "directional_btc_1h_v5",
        "directional_btc_1h_v6",
        # 2026-05-21 Path C: first sidecar-servable ETH directional spec
        # (HARD_RULE_BLOCK_INFERENCE_STAMPING resolution). Bar-only
        # feature stack (9 features, no macro/cross-asset) so the
        # inference sidecar can resolve every feature_value at
        # (symbol=ETHUSDT, interval=1h, ts=bar) exact-match.
        "directional_eth_1h_v1",
        # 2026-05-21 Path C continuation: regime variant of the same
        # bar-only ETH stack. Routes through the 5-gate modulator
        # gauntlet (AUC bar 0.52) instead of the 13-gate directional
        # gauntlet (AUC bar 0.55) which the bar-only stack cannot clear.
        "regime_eth_v2",
    }


def test_excluded_from_inputs_has_no_labels():
    """H9 (2026-05-16): labels are filtered exclusively by the
    schema-based ``label_direction='forward'`` predicate in
    ``loader._list_input_features``. Keeping labels in the manual
    EXCLUDED_FROM_INPUTS set as well meant two lists had to stay in
    sync — which caused the V77 ``label_regime_risk_on_24h`` leak
    earlier the same day (regime_btc_v3 trained on its own label,
    AUC=1.0 trivial). The cleanup removes labels from the manual set.

    This guard fails loudly if a future contributor adds a label
    string back to EXCLUDED_FROM_INPUTS — the right fix is to ensure
    the label has ``label_direction='forward'`` in its seed migration,
    not to re-introduce the parallel list. The Tier-C non-label
    exclusions (source-capped Binance/CoinGecko features) remain
    legitimate; they aren't labels and can't be schema-filtered.
    """
    from blackheart_train.specs import EXCLUDED_FROM_INPUTS
    label_like = {name for name in EXCLUDED_FROM_INPUTS if name.startswith("label_")}
    assert not label_like, (
        f"EXCLUDED_FROM_INPUTS regressed — labels {sorted(label_like)} should "
        f"be filtered via label_direction='forward' in the seed migration "
        f"(see V73/V74/V77), not via the manual exclusion list. See "
        f"specs.EXCLUDED_FROM_INPUTS docstring for the H9 rationale."
    )
    assert get_spec("regime_btc_v1").objective == "binary"
    assert get_spec("flow_btc_v1").objective == "regression"
    assert get_spec("directional_btc_1h_v1").objective == "multiclass"
    assert get_spec("directional_btc_1h_v1").purpose == "directional"
    assert get_spec("directional_btc_1h_v1").label_feature == "label_triple_barrier"
    # v2 specs declare derived features; v1 specs do not.
    assert get_spec("regime_btc_v1").derived_features == ()
    assert "eth_btc_corr_24h" in get_spec("regime_btc_v2").derived_features


def test_specs_have_distinct_hyperparam_dicts():
    """Two specs share the same default values but not the same dict
    instance — mutating one must not bleed into another. The training
    pipeline relies on this when forwarding hyperparams to LightGBM.
    """
    a = get_spec("regime_btc_v1").hyperparams
    b = get_spec("positioning_btc_v1").hyperparams
    assert a == b
    assert a is not b


# ── Split semantics ────────────────────────────────────────────────────────


def test_split_chronological_preserves_order():
    n = 100
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    X = pd.DataFrame({"a": range(n)}, index=idx)
    y = pd.Series(range(n), index=idx)
    X_tr, y_tr, X_val, y_val = split_chronological(X, y, 0.2)
    assert len(X_tr) == 80 and len(X_val) == 20
    assert int(X_tr["a"].max()) < int(X_val["a"].min())


def test_split_chronological_rejects_shuffled_index():
    idx = pd.date_range("2024-01-01", periods=10, freq="1h")
    shuffled = idx[::-1]   # monotonically *decreasing* — defensive check trips
    X = pd.DataFrame({"a": range(10)}, index=shuffled)
    y = pd.Series(range(10), index=shuffled)
    with pytest.raises(ValueError, match="monotonically increasing"):
        split_chronological(X, y, 0.2)


def test_split_chronological_rejects_invalid_fraction():
    idx = pd.date_range("2024-01-01", periods=10, freq="1h")
    X = pd.DataFrame({"a": range(10)}, index=idx)
    y = pd.Series(range(10), index=idx)
    with pytest.raises(ValueError, match="val_fraction"):
        split_chronological(X, y, 0.6)


# ── Metrics ────────────────────────────────────────────────────────────────


def test_evaluate_binary_well_formed():
    y_true = pd.Series([0, 1, 0, 1, 1, 0])
    y_pred = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3])
    m = _evaluate("binary", y_true, y_pred)
    assert {"auc", "log_loss", "accuracy"} == set(m.keys())
    assert 0.5 < m["auc"] <= 1.0
    assert m["accuracy"] == 1.0


def test_evaluate_regression_well_formed():
    y_true = pd.Series([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.05, 3.95])
    m = _evaluate("regression", y_true, y_pred)
    assert {"rmse", "mae", "pearson_r"} == set(m.keys())
    assert m["pearson_r"] > 0.99


# ── Multiclass (M5g.1 — directional model) ────────────────────────────────


def test_encode_multiclass_fixed_order():
    """Encoding is locked: -1 → 0, 0 → 1, +1 → 2. Live-inference and
    metric labelling both depend on this being stable."""
    y = pd.Series([-1, 0, 1, -1, 1, 0])
    enc = encode_multiclass(y)
    assert list(enc) == [0, 1, 2, 0, 2, 1]


def test_encode_multiclass_rejects_unknown_value():
    """A label value outside {-1, 0, +1} is an error — better than
    silently mis-encoding into class 0."""
    y = pd.Series([-1, 2, 0])   # +2 is not a triple-barrier class
    with pytest.raises(ValueError, match="unknown values"):
        encode_multiclass(y)


def test_encode_multiclass_rejects_nan_with_clean_error():
    """MG3 fix: NaN in the label column raises a clear ValueError
    instead of the cryptic ``IntCastingNaNError`` that ``.astype(int)``
    would produce. Loader should drop NaN rows before training, but if
    one slips through we want a useful diagnostic."""
    y = pd.Series([-1.0, float("nan"), 1.0])
    with pytest.raises(ValueError, match="NaN"):
        encode_multiclass(y)


def test_encode_multiclass_accepts_float_label_values():
    """label_triple_barrier in feature_values is stored as float64
    (-1.0 / 0.0 / 1.0). Encoding must accept these as the same as
    integer ±1/0."""
    y = pd.Series([-1.0, 0.0, 1.0, -1.0])
    enc = encode_multiclass(y)
    assert list(enc) == [0, 1, 2, 0]


def test_encode_multiclass_rejects_non_integer_float():
    """A non-integer float (e.g. 0.5) is not a valid triple-barrier
    value even after casting — must raise."""
    y = pd.Series([-1.0, 0.5, 1.0])
    with pytest.raises(ValueError, match="unknown values"):
        encode_multiclass(y)


def test_decode_multiclass_is_inverse_of_encode():
    enc = np.array([0, 1, 2, 0, 2, 1])
    dec = decode_multiclass(enc)
    assert list(dec) == [-1, 0, 1, -1, 1, 0]


def test_evaluate_multiclass_well_formed():
    """A perfectly-predicted (n=6, C=3) batch — log_loss near 0,
    accuracy = 1.0, macro AUC = 1.0, per-class precision/recall = 1.0."""
    y_true = pd.Series([0, 1, 2, 0, 1, 2])
    # One-hot-ish high-confidence probas — argmax matches y_true.
    y_pred = np.array([
        [0.90, 0.05, 0.05],
        [0.05, 0.90, 0.05],
        [0.05, 0.05, 0.90],
        [0.90, 0.05, 0.05],
        [0.05, 0.90, 0.05],
        [0.05, 0.05, 0.90],
    ])
    m = _evaluate("multiclass", y_true, y_pred)
    expected_keys = {
        "log_loss", "accuracy",
        "macro_precision", "macro_recall", "macro_f1",
        "macro_auc_ovr",
        "class_0_precision", "class_0_recall",
        "class_1_precision", "class_1_recall",
        "class_2_precision", "class_2_recall",
    }
    assert expected_keys == set(m.keys())
    assert m["accuracy"] == 1.0
    assert m["macro_auc_ovr"] == 1.0
    assert m["log_loss"] < 0.2
    for i in range(3):
        assert m[f"class_{i}_precision"] == 1.0
        assert m[f"class_{i}_recall"] == 1.0


# ── filter_eval_to_serving_interval (M5g.5 MS4) ──────────────────────────


def test_filter_eval_to_serving_interval_passthrough_when_no_indicator():
    """No ``interval_indicator`` column → identity. Non-stacked specs
    must be untouched."""
    from blackheart_train.specs import get_spec
    n = 100
    X = pd.DataFrame({"a": np.zeros(n), "b": np.zeros(n)},
                     index=pd.date_range("2025-01-01", periods=n, freq="1h"))
    y = pd.Series(np.zeros(n), index=X.index)
    spec = get_spec("regime_btc_v1")   # training_intervals=()
    X_out, y_out = filter_eval_to_serving_interval(X, y, spec)
    assert X_out is X and y_out is y


def test_filter_eval_to_serving_interval_passthrough_when_no_training_intervals():
    """``interval_indicator`` column present but spec has no
    training_intervals — still identity (caller has nothing to filter
    against)."""
    from blackheart_train.specs import get_spec
    n = 50
    X = pd.DataFrame({
        "a": np.zeros(n),
        "interval_indicator": np.array([2] * 25 + [1] * 25, dtype=int),
    }, index=pd.date_range("2025-01-01", periods=n, freq="1h"))
    y = pd.Series(np.zeros(n), index=X.index)
    spec = get_spec("regime_btc_v1")   # training_intervals=()
    X_out, y_out = filter_eval_to_serving_interval(X, y, spec)
    assert len(X_out) == n   # unchanged


def test_filter_eval_to_serving_interval_filters_when_stacked():
    """Stacked spec → only rows whose interval_indicator matches the
    serving cadence (1h → code 2) survive."""
    from blackheart_train.specs import get_spec
    n = 100
    indicator = np.array([2] * 50 + [1] * 50, dtype=int)   # 50 1h + 50 15m
    X = pd.DataFrame({
        "a": np.arange(n, dtype=float),
        "interval_indicator": indicator,
    }, index=pd.date_range("2025-01-01", periods=n, freq="1h"))
    y = pd.Series(np.arange(n, dtype=float), index=X.index)
    spec = get_spec("directional_btc_1h_v1")   # interval=1h, training_intervals=(1h,15m)
    X_out, y_out = filter_eval_to_serving_interval(X, y, spec)
    assert len(X_out) == 50
    assert (X_out["interval_indicator"] == 2).all()
    # y is filtered in lockstep
    assert len(y_out) == 50
    assert y_out.index.equals(X_out.index)


def test_filter_eval_to_serving_interval_empty_result_when_no_serving_rows():
    """If the slice has no serving-interval rows, return empty — the
    caller decides whether to skip the fold or error."""
    from blackheart_train.specs import get_spec
    n = 40
    X = pd.DataFrame({
        "a": np.zeros(n),
        "interval_indicator": np.full(n, 1, dtype=int),   # all 15m
    }, index=pd.date_range("2025-01-01", periods=n, freq="15min"))
    y = pd.Series(np.zeros(n), index=X.index)
    spec = get_spec("directional_btc_1h_v1")
    X_out, y_out = filter_eval_to_serving_interval(X, y, spec)
    assert len(X_out) == 0
    assert len(y_out) == 0


def test_evaluate_multiclass_handles_missing_class_in_val_slice():
    """If the val slice doesn't contain every class, macro AUC OVR is
    not well-defined — must return NaN rather than crash."""
    y_true = pd.Series([0, 0, 1, 1])   # class 2 absent
    y_pred = np.array([
        [0.8, 0.1, 0.1],
        [0.7, 0.2, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.8, 0.1],
    ])
    m = _evaluate("multiclass", y_true, y_pred)
    # log_loss + accuracy still well-defined
    assert m["accuracy"] == 1.0
    # AUC OVR requires all classes present in y_true
    import math as _m
    assert _m.isnan(m["macro_auc_ovr"])


# ── Content-addressed artifacts ────────────────────────────────────────────


def test_compute_content_sha_is_deterministic():
    a = compute_content_sha({"k": 1, "z": [1, 2, 3]})
    b = compute_content_sha({"z": [1, 2, 3], "k": 1})   # different insertion order
    assert a == b   # canonical JSON sorts keys


def test_compute_content_sha_distinguishes_content():
    a = compute_content_sha({"k": 1})
    b = compute_content_sha({"k": 2})
    assert a != b


def test_artifact_roundtrip_with_explicit_content_sha(tmp_path):
    content = {"model": "x", "features": ["a", "b"]}
    sha = compute_content_sha(content)
    payload = {**content, "content_sha256": sha, "metrics": {"auc": 0.99}}
    path, size = write_artifact(payload, sha, tmp_path)
    assert path.exists() and size > 0
    back = read_artifact(sha, tmp_path)
    # read_artifact backfills v1 → v2 keys (``payload_version=1``,
    # ``ensemble=None``) on payloads missing them. Original fields pass
    # through untouched.
    expected = {**payload, "payload_version": 1, "ensemble": None}
    assert back == expected


def test_artifact_write_rejects_payload_without_content_sha(tmp_path):
    sha = compute_content_sha({"k": 1})
    payload = {"k": 1}   # missing content_sha256
    with pytest.raises(ValueError, match="content_sha256"):
        write_artifact(payload, sha, tmp_path)


def test_artifact_write_rejects_mismatched_payload_sha(tmp_path):
    sha = compute_content_sha({"k": 1})
    payload = {"k": 1, "content_sha256": "deadbeef"}   # wrong sha embedded
    with pytest.raises(ValueError, match="content_sha256"):
        write_artifact(payload, sha, tmp_path)


def test_artifact_read_detects_filename_payload_mismatch(tmp_path):
    sha = compute_content_sha({"k": 1})
    payload = {"k": 1, "content_sha256": sha}
    write_artifact(payload, sha, tmp_path)
    # Manually corrupt: stuff a different content_sha into the payload bytes.
    import pickle as _pickle
    target = tmp_path / sha[:2] / f"{sha}.pkl"
    tampered = {"k": 1, "content_sha256": "0" * 64}
    target.write_bytes(_pickle.dumps(tampered, protocol=5))
    with pytest.raises(ValueError, match="content_sha mismatch"):
        read_artifact(sha, tmp_path)


def test_artifact_idempotent_rewrite(tmp_path):
    """Same content_sha → same filename, second write is a no-op."""
    sha = compute_content_sha({"k": 1})
    payload = {"k": 1, "content_sha256": sha}
    path1, size1 = write_artifact(payload, sha, tmp_path)
    mtime1 = path1.stat().st_mtime_ns
    path2, size2 = write_artifact(payload, sha, tmp_path)
    assert path1 == path2 and size1 == size2
    assert path2.stat().st_mtime_ns == mtime1   # file not rewritten


def test_read_artifact_backfills_v1_payloads(tmp_path):
    """Pre-Phase-2 payloads have no ``payload_version`` / ``ensemble``
    keys. read_artifact must backfill them so v2-aware consumers can
    branch on ``payload["payload_version"]`` without probing for key
    existence first. The on-disk pickle is unchanged."""
    import pickle as _pickle
    # Write a v1-shaped payload directly (no payload_version, no ensemble).
    sha = compute_content_sha({"k": "v1"})
    v1_payload = {
        "k": "v1",
        "content_sha256": sha,
        "booster": "fake-lgb-booster-string",   # stand-in
    }
    target = tmp_path / sha[:2] / f"{sha}.pkl"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(_pickle.dumps(v1_payload, protocol=5))

    loaded = read_artifact(sha, tmp_path)
    assert loaded["payload_version"] == 1
    assert loaded["ensemble"] is None
    # The booster key passes through untouched.
    assert loaded["booster"] == "fake-lgb-booster-string"

    # A v2 payload (both keys present) is returned verbatim — no
    # accidental downgrade.
    sha2 = compute_content_sha({"k": "v2"})
    v2_payload = {
        "k": "v2",
        "content_sha256": sha2,
        "payload_version": 2,
        "booster": None,
        "ensemble": "fake-ensemble",
    }
    target2 = tmp_path / sha2[:2] / f"{sha2}.pkl"
    target2.parent.mkdir(parents=True, exist_ok=True)
    target2.write_bytes(_pickle.dumps(v2_payload, protocol=5))
    loaded2 = read_artifact(sha2, tmp_path)
    assert loaded2["payload_version"] == 2
    assert loaded2["ensemble"] == "fake-ensemble"


def test_artifact_read_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_artifact("0" * 64, tmp_path)
