"""Unit tests for the M5d 5-gate sub-model gauntlet.

Pure-function tests over synthetic payloads. The gates are pure
functions of an artifact payload, so we exercise each verdict path by
crafting a minimal payload that triggers it. End-to-end coverage on
real-data artifacts lives in the CLI run.
"""
from __future__ import annotations

import math

import pytest

from blackheart_train.gauntlet import (
    GauntletError,
    _EDGE_THRESHOLD,
    _RANDOM_BASELINE,
    _STABILITY_THRESHOLD,
    _gate_fold_stability,
    _gate_generalization_edge,
    _gate_integrity_passed,
    _gate_saved_booster_above_random,
    _gate_transferability,
    _gate_walk_forward_complete,
    gauntlet_to_dict,
    run_gauntlet,
)


# ── Helpers ────────────────────────────────────────────────────────────────


def _payload(
    *,
    integrity_verdict: str = "PASS",
    primary_metric: str = "auc",
    primary_mean: float = 0.60,
    primary_median: float | None = None,
    primary_std: float = 0.05,
    metrics: dict | None = None,
    ci_max_abs_diff: float = 0.05,
    adversarial_auc: float = 0.99,
    n_folds: int = 6,
    n_valid: int | None = None,
) -> dict:
    """Build a minimal payload that satisfies all gates. Tests override
    fields to make individual gates fail.

    ``ci_max_abs_diff`` defaults BELOW the 0.15 binary transferability bar
    so a clean payload passes gate 6; ``adversarial_auc`` defaults high
    (≈1.0) to prove it is info-only and does NOT gate. ``primary_median``
    defaults to ``primary_mean`` (so a clean payload passes the median floor)
    but can be set independently to exercise the lucky-mean / random-median
    case."""
    if metrics is None:
        if primary_metric == "auc":
            metrics = {"auc": 0.60, "log_loss": 0.65, "accuracy": 0.58}
        else:
            metrics = {"rmse": 0.04, "mae": 0.03, "pearson_r": 0.10}
    if n_valid is None:
        n_valid = n_folds
    if primary_median is None:
        primary_median = primary_mean
    return {
        "content_sha256": "deadbeef" * 8,
        "data_fingerprint": "cafebabe" * 8,
        # Tests default to a modulator spec; multiclass / directional
        # tests override these explicitly.
        "spec": {"name": "regime_btc_v1", "purpose": "regime", "objective": "binary"},
        "integrity": {"verdict": integrity_verdict, "checks": []},
        "metrics": metrics,
        "eval_kind": "walk_forward_aggregate",
        "walk_forward": {
            "primary_metric": primary_metric,
            "n_folds_configured": n_folds,
            "n_folds_generated": n_folds,
            "n_folds_run": n_folds,
            "n_folds_valid_metric": n_valid,
            "primary_mean": primary_mean,
            "primary_median": primary_median,
            "primary_std": primary_std,
            "metric_means": {
                primary_metric: primary_mean,
                "ci_max_abs_diff": ci_max_abs_diff,
                "adversarial_auc": adversarial_auc,
            },
            "folds": [],
        },
    }


# ── Gate 1: integrity_passed ──────────────────────────────────────────────


def test_gate_integrity_passes_on_pass_verdict():
    g = _gate_integrity_passed(_payload(integrity_verdict="PASS"))
    assert g.verdict == "PASS"


def test_gate_integrity_passes_on_warn_verdict():
    """WARN means integrity surfaced concerns but didn't block training.
    Gate must not penalise; the WARN is recorded elsewhere."""
    g = _gate_integrity_passed(_payload(integrity_verdict="WARN"))
    assert g.verdict == "PASS"


def test_gate_integrity_fails_on_fail_verdict():
    g = _gate_integrity_passed(_payload(integrity_verdict="FAIL"))
    assert g.verdict == "FAIL"


def test_gate_integrity_fails_on_missing_block():
    p = _payload()
    del p["integrity"]
    g = _gate_integrity_passed(p)
    assert g.verdict == "FAIL"


# ── Gate 2: walk_forward_complete ─────────────────────────────────────────


def test_gate_wf_complete_passes_when_all_folds_valid():
    g = _gate_walk_forward_complete(_payload(n_folds=6, n_valid=6))
    assert g.verdict == "PASS"


def test_gate_wf_complete_fails_when_a_fold_skipped():
    g = _gate_walk_forward_complete(_payload(n_folds=6, n_valid=5))
    assert g.verdict == "FAIL"
    assert g.actual["n_folds_valid_metric"] == 5


def test_gate_wf_complete_fails_when_no_walk_forward_block():
    p = _payload()
    p["walk_forward"] = None
    g = _gate_walk_forward_complete(p)
    assert g.verdict == "FAIL"


# ── Gate 3: generalization_edge ───────────────────────────────────────────


def test_gate_edge_passes_above_threshold_binary():
    th = _EDGE_THRESHOLD["auc"]
    g = _gate_generalization_edge(_payload(primary_metric="auc", primary_mean=th + 0.01))
    assert g.verdict == "PASS"


def test_gate_edge_passes_exactly_at_threshold_binary():
    """≥, not >."""
    th = _EDGE_THRESHOLD["auc"]
    g = _gate_generalization_edge(_payload(primary_metric="auc", primary_mean=th))
    assert g.verdict == "PASS"


def test_gate_edge_fails_below_threshold_binary():
    th = _EDGE_THRESHOLD["auc"]
    g = _gate_generalization_edge(_payload(primary_metric="auc", primary_mean=th - 0.01))
    assert g.verdict == "FAIL"


def test_gate_edge_passes_above_threshold_regression():
    th = _EDGE_THRESHOLD["pearson_r"]
    p = _payload(primary_metric="pearson_r", primary_mean=th + 0.01,
                 metrics={"rmse": 1.0, "mae": 0.8, "pearson_r": 0.10})
    g = _gate_generalization_edge(p)
    assert g.verdict == "PASS"


def test_gate_edge_fails_when_mean_is_nan():
    g = _gate_generalization_edge(_payload(primary_mean=float("nan")))
    assert g.verdict == "FAIL"


def test_gate_edge_fails_when_mean_passes_but_median_below_threshold():
    """The lucky-fold case: a mean lifted above the bar by one good fold,
    while the typical (median) fold is at random, must FAIL."""
    th = _EDGE_THRESHOLD["auc"]
    g = _gate_generalization_edge(
        _payload(primary_metric="auc", primary_mean=th + 0.02,
                 primary_median=th - 0.02)
    )
    assert g.verdict == "FAIL"
    assert g.actual["median"] == round(th - 0.02, 4)


def test_gate_edge_passes_when_both_mean_and_median_clear():
    th = _EDGE_THRESHOLD["auc"]
    g = _gate_generalization_edge(
        _payload(primary_metric="auc", primary_mean=th + 0.02,
                 primary_median=th + 0.01)
    )
    assert g.verdict == "PASS"


# ── Gate 6: transferability (conditional invariance) ──────────────────────


def test_gate_transferability_passes_below_threshold_binary():
    """ci_max_abs_diff below the 0.15 binary bar => PASS; a high
    adversarial_auc must NOT flip it (info-only, not gated)."""
    g = _gate_transferability(_payload(ci_max_abs_diff=0.05, adversarial_auc=0.999))
    assert g.verdict == "PASS"
    assert g.actual["adversarial_auc"] == 0.999   # surfaced, not gated


def test_gate_transferability_fails_above_threshold_binary():
    """The regime_eth_v2 failure mode: ci_max_abs_diff 0.2949 ≥ 0.15."""
    g = _gate_transferability(_payload(ci_max_abs_diff=0.2949))
    assert g.verdict == "FAIL"


def test_gate_transferability_fails_when_ci_missing():
    """A payload with no conditional-invariance measurement cannot affirm
    transferability and must FAIL (re-train), not silently pass."""
    p = _payload()
    p["walk_forward"]["metric_means"].pop("ci_max_abs_diff")
    g = _gate_transferability(p)
    assert g.verdict == "FAIL"


def test_gate_transferability_not_gated_on_adversarial_auc():
    """adversarial_auc ≈ 1.0 (structural covariate shift for hourly bars)
    must NOT fail the gate so long as ci_max_abs_diff is clean."""
    g = _gate_transferability(_payload(ci_max_abs_diff=0.05, adversarial_auc=1.0))
    assert g.verdict == "PASS"


def test_gate_edge_skips_on_unknown_metric():
    p = _payload()
    p["walk_forward"]["primary_metric"] = "spearman_r"   # not in threshold table
    g = _gate_generalization_edge(p)
    assert g.verdict == "SKIP"


# ── Gate 4: fold_stability ────────────────────────────────────────────────


def test_gate_stability_passes_below_threshold_binary():
    th = _STABILITY_THRESHOLD["auc"]
    g = _gate_fold_stability(_payload(primary_std=th - 0.01))
    assert g.verdict == "PASS"


def test_gate_stability_fails_above_threshold_binary():
    th = _STABILITY_THRESHOLD["auc"]
    g = _gate_fold_stability(_payload(primary_std=th + 0.01))
    assert g.verdict == "FAIL"


def test_gate_stability_fails_on_nan_std():
    g = _gate_fold_stability(_payload(primary_std=float("nan")))
    assert g.verdict == "FAIL"


# ── Gate 5: saved_booster_above_random ────────────────────────────────────


def test_gate_above_random_passes_for_binary():
    p = _payload(primary_metric="auc", metrics={"auc": 0.55, "log_loss": 0.7, "accuracy": 0.55})
    g = _gate_saved_booster_above_random(p)
    assert g.verdict == "PASS"


def test_gate_above_random_fails_for_binary_below_05():
    p = _payload(primary_metric="auc", metrics={"auc": 0.48, "log_loss": 0.7, "accuracy": 0.48})
    g = _gate_saved_booster_above_random(p)
    assert g.verdict == "FAIL"


def test_gate_above_random_passes_for_regression():
    p = _payload(primary_metric="pearson_r",
                 metrics={"rmse": 0.04, "mae": 0.03, "pearson_r": 0.01})
    g = _gate_saved_booster_above_random(p)
    assert g.verdict == "PASS"


def test_gate_above_random_fails_for_regression_below_0():
    p = _payload(primary_metric="pearson_r",
                 metrics={"rmse": 0.04, "mae": 0.03, "pearson_r": -0.01})
    g = _gate_saved_booster_above_random(p)
    assert g.verdict == "FAIL"


def test_gate_above_random_fails_on_nan_value():
    p = _payload(primary_metric="auc",
                 metrics={"auc": float("nan"), "log_loss": 1.0, "accuracy": 0.5})
    g = _gate_saved_booster_above_random(p)
    assert g.verdict == "FAIL"


# ── Aggregation: run_gauntlet ─────────────────────────────────────────────


def test_run_gauntlet_overall_pass_on_clean_payload():
    p = _payload(integrity_verdict="PASS", primary_mean=0.60, primary_std=0.05,
                 metrics={"auc": 0.60, "log_loss": 0.7, "accuracy": 0.55})
    report = run_gauntlet(p)
    assert report.overall_verdict == "PASS"
    assert len(report.gates) == 6
    assert all(g.verdict == "PASS" for g in report.gates)


def test_run_gauntlet_overall_fail_when_one_gate_fails():
    p = _payload(primary_std=0.50)   # blow up stability
    report = run_gauntlet(p)
    assert report.overall_verdict == "FAIL"
    fail_gates = [g.name for g in report.gates if g.verdict == "FAIL"]
    assert "fold_stability" in fail_gates


def test_run_gauntlet_raises_when_walk_forward_missing():
    p = _payload()
    p["walk_forward"] = None
    with pytest.raises(GauntletError, match="walk_forward"):
        run_gauntlet(p)


def test_run_gauntlet_refuses_multiclass_objective():
    """MG5 fix: the 5-gate gauntlet is calibrated for HYBRID modulator
    sub-models (binary/regression). A multiclass directional payload
    must be refused up front with a clear pointer to the full 13-gate
    gauntlet — not silently FAIL with three SKIP gates whose rationale
    is the cryptic 'no edge threshold for metric macro_auc_ovr'."""
    p = _payload()
    p["spec"] = {
        "name": "directional_btc_1h_v1",
        "purpose": "directional",
        "objective": "multiclass",
    }
    with pytest.raises(GauntletError, match="13-gate gauntlet"):
        run_gauntlet(p)


def test_run_gauntlet_refuses_directional_purpose_even_if_binary():
    """Defence in depth — a hypothetical directional model with a
    binary objective is still wrong-gauntlet."""
    p = _payload()
    p["spec"] = {
        "name": "directional_x",
        "purpose": "directional",
        "objective": "binary",
    }
    with pytest.raises(GauntletError, match="13-gate gauntlet"):
        run_gauntlet(p)


def test_run_gauntlet_propagates_traceability_fields():
    p = _payload()
    report = run_gauntlet(p)
    assert report.artifact_content_sha == p["content_sha256"]
    assert report.data_fingerprint == p["data_fingerprint"]
    assert report.primary_metric == "auc"


# ── Serialisation ─────────────────────────────────────────────────────────


def test_gauntlet_to_dict_has_expected_keys():
    p = _payload()
    report = run_gauntlet(p)
    d = gauntlet_to_dict(report)
    for key in ("spec_name", "overall_verdict", "primary_metric",
                "artifact_content_sha", "data_fingerprint", "gates"):
        assert key in d, f"missing key: {key}"
    assert len(d["gates"]) == 6
    for gate_dict in d["gates"]:
        for key in ("name", "verdict", "threshold", "actual", "rationale"):
            assert key in gate_dict


# ── Realistic regression scenarios ────────────────────────────────────────


def test_real_world_m5c_regime_data_fails_gauntlet():
    """The M5c real-data run reported regime_btc_v1 with primary_mean=0.4566,
    primary_std=0.0942, and saved-booster AUC=0.5371. That should fail
    generalization_edge (mean < 0.52) but PASS the other 4 gates."""
    p = _payload(
        primary_metric="auc",
        primary_mean=0.4566,
        primary_std=0.0942,
        metrics={"auc": 0.5371, "log_loss": 1.20, "accuracy": 0.47},
    )
    report = run_gauntlet(p)
    assert report.overall_verdict == "FAIL"
    by_name = {g.name: g.verdict for g in report.gates}
    assert by_name["generalization_edge"] == "FAIL"
    assert by_name["fold_stability"] == "PASS"   # 0.094 < 0.15
    assert by_name["saved_booster_above_random"] == "PASS"   # 0.5371 > 0.5
    assert by_name["integrity_passed"] == "PASS"
    assert by_name["walk_forward_complete"] == "PASS"


def test_real_world_m5c_flow_data_fails_gauntlet():
    """flow_btc_v1: mean=-0.010, std=0.322, saved=last fold pearson=-0.221.
    Should fail edge (mean < 0.05), stability (std > 0.20), and
    saved_booster (-0.221 < 0)."""
    p = _payload(
        primary_metric="pearson_r",
        primary_mean=-0.010,
        primary_std=0.322,
        metrics={"rmse": 0.057, "mae": 0.048, "pearson_r": -0.221},
    )
    report = run_gauntlet(p)
    assert report.overall_verdict == "FAIL"
    by_name = {g.name: g.verdict for g in report.gates}
    assert by_name["generalization_edge"] == "FAIL"
    assert by_name["fold_stability"] == "FAIL"
    assert by_name["saved_booster_above_random"] == "FAIL"


def test_regime_eth_v2_broken_model_now_fails_gauntlet():
    """REGRESSION (project_ml_training_pipeline_forensics): the exact broken
    ETH regime model that passed the original 5-gate set and reached a live
    trading gate. Honest 6-fold mean AUC 0.5338, median 0.5109, std 0.1142;
    a single lucky last fold reported 0.6094; ci_max_abs_diff 0.2949 and
    adversarial_auc 0.9966. Under the old gauntlet this was PASS. It must now
    FAIL on BOTH the median floor (gate 3) AND transferability (gate 6),
    while adversarial_auc (info-only) is reported but does not itself gate."""
    p = _payload(
        primary_metric="auc",
        primary_mean=0.5338,
        primary_median=0.5109,
        primary_std=0.1142,
        ci_max_abs_diff=0.2949,
        adversarial_auc=0.9966,
        # last-fold (saved booster) metrics — the cherry-picked 0.6094 that
        # used to be the headline; now stored separately.
        metrics={"auc": 0.6094, "log_loss": 0.7638, "accuracy": 0.5714},
    )
    p["last_fold_metrics"] = {"auc": 0.6094, "log_loss": 0.7638, "accuracy": 0.5714}
    report = run_gauntlet(p)
    assert report.overall_verdict == "FAIL"
    by_name = {g.name: g.verdict for g in report.gates}
    # The two NEW guards that catch it:
    assert by_name["generalization_edge"] == "FAIL"   # median 0.5109 < 0.52
    assert by_name["transferability"] == "FAIL"       # ci_max 0.2949 ≥ 0.15
    # The saved booster's cherry-picked last fold still clears random — proof
    # that gate 5 alone (the old defence) was insufficient.
    assert by_name["saved_booster_above_random"] == "PASS"   # 0.6094 ≥ 0.5
