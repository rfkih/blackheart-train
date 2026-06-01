"""Unit tests for the M5h 13-gate directional gauntlet.

Each test feeds a synthetic payload and asserts the per-gate verdict
+ overall verdict. Payloads are minimal — only the fields the gauntlet
reads are populated.
"""
from __future__ import annotations

from typing import Any

import pytest

from blackheart_train.gauntlet_directional import (
    DirectionalGauntletError,
    directional_gauntlet_to_dict,
    run_directional_gauntlet,
)


# ── Payload builder ───────────────────────────────────────────────────────


def _make_directional_payload(
    *,
    objective: str = "multiclass",
    integrity_verdict: str = "PASS",
    primary_mean: float = 0.60,
    ci_lower_5: float = 0.55,
    adversarial_auc: float = 0.55,
    ci_max_abs_diff: float = 0.08,
    ci_mean_abs_diff: float = 0.04,
    cost_realistic_profitable: float = 1.0,
    cost_conservative_profitable: float = 1.0,
    cost_realistic_net_pnl_bps_per_trade: float = 25.0,
    regime_gate_6_pass: float = 1.0,
    regime_gate_6_caveat: float = 0.0,
    dsr_gate_7_pass: float = 1.0,
    dsr_value: float = 0.98,
    dsr_trial_discount_active: float = 1.0,
    n_folds: int = 6,
) -> dict[str, Any]:
    """Build a Tier-A-passing payload by default. Override individual
    knobs in each test to flip specific gates.

    Default ``objective='multiclass'`` matches the directional spec
    (label_triple_barrier). For the new gate-4 conditional-invariance
    coverage tests, override to ``binary`` or ``regression``.
    """
    return {
        "spec": {
            "name": "directional_btc_1h_v1",
            "objective": objective,
            "purpose": "directional",
        },
        "content_sha256": "dead" * 16,
        "data_fingerprint": "beef" * 16,
        "integrity": {"verdict": integrity_verdict},
        "walk_forward": {
            "primary_metric": "macro_auc_ovr",
            "primary_mean": primary_mean,
            "primary_std": 0.02,
            "n_folds_configured": n_folds,
            "n_folds_generated": n_folds,
            "n_folds_run": n_folds,
            "n_folds_valid_metric": n_folds,
            "metric_means": {
                "macro_auc_ovr": primary_mean,
                "macro_auc_ovr_ci_lower_5": ci_lower_5,
                "adversarial_auc": adversarial_auc,
                "ci_max_abs_diff": ci_max_abs_diff,
                "ci_mean_abs_diff": ci_mean_abs_diff,
                "cost_realistic_profitable": cost_realistic_profitable,
                "cost_conservative_profitable": cost_conservative_profitable,
                "cost_realistic_net_pnl_bps_per_trade": cost_realistic_net_pnl_bps_per_trade,
                "regime_gate_6_pass": regime_gate_6_pass,
                "regime_gate_6_caveat": regime_gate_6_caveat,
                "dsr_gate_7_pass": dsr_gate_7_pass,
                "dsr_value": dsr_value,
                "dsr_trial_discount_active": dsr_trial_discount_active,
            },
        },
        "metrics": {"macro_auc_ovr": primary_mean},
    }


# ── Overall verdict ───────────────────────────────────────────────────────


def test_all_tier_a_pass_gives_conditional_pass():
    """Tier A all PASS (7 gates), Tier B/C still SKIP (6 gates) → CONDITIONAL_PASS.
    Gate 4 (transferability) now evaluates multiclass via per-class P(y=k|bin)
    shift — ci_max_abs_diff=0.08 < 0.15 → PASS. 7 PASS + 6 Tier-B/C-SKIP."""
    payload = _make_directional_payload()
    report = run_directional_gauntlet(payload)
    assert report.overall_verdict == "CONDITIONAL_PASS"
    # 7 Tier-A PASS + 6 Tier-B/C SKIP = 13 gates
    assert report.n_pass == 7
    assert report.n_fail == 0
    assert report.n_skip == 6


def test_any_tier_a_fail_gives_overall_fail():
    """Single Tier A failure → overall FAIL (the binding gates are AND)."""
    # Drop bootstrap CI below random
    payload = _make_directional_payload(ci_lower_5=0.45)
    report = run_directional_gauntlet(payload)
    assert report.overall_verdict == "FAIL"
    assert any(g.name == "bootstrap_ci_lower" and g.verdict == "FAIL" for g in report.gates)


def test_missing_walk_forward_raises():
    """Without a walk_forward block we have no per-fold aggregates →
    raise rather than silently FAIL every binding gate."""
    payload = _make_directional_payload()
    payload["walk_forward"] = None
    with pytest.raises(DirectionalGauntletError, match="walk_forward"):
        run_directional_gauntlet(payload)


# ── Tier A per-gate tests ─────────────────────────────────────────────────


def test_gate_1_baseline_edge_pass_and_fail():
    pass_payload = _make_directional_payload(primary_mean=0.60)
    fail_payload = _make_directional_payload(primary_mean=0.52)
    r_pass = run_directional_gauntlet(pass_payload)
    r_fail = run_directional_gauntlet(fail_payload)
    g_pass = next(g for g in r_pass.gates if g.gate_number == 1)
    g_fail = next(g for g in r_fail.gates if g.gate_number == 1)
    assert g_pass.verdict == "PASS"
    assert g_fail.verdict == "FAIL"


def test_gate_1_handles_nan_primary_mean():
    """NaN primary_mean → FAIL (not a crash). The walk-forward aggregator
    can produce NaN if every fold's primary metric was filtered out."""
    payload = _make_directional_payload(primary_mean=float("nan"))
    report = run_directional_gauntlet(payload)
    g1 = next(g for g in report.gates if g.gate_number == 1)
    assert g1.verdict == "FAIL"


def test_gate_2_pit_no_leak():
    """integrity verdict PASS → gate 2 PASS; FAIL → gate 2 FAIL."""
    pass_payload = _make_directional_payload(integrity_verdict="PASS")
    fail_payload = _make_directional_payload(integrity_verdict="FAIL")
    warn_payload = _make_directional_payload(integrity_verdict="WARN")
    assert next(g for g in run_directional_gauntlet(pass_payload).gates
                if g.gate_number == 2).verdict == "PASS"
    assert next(g for g in run_directional_gauntlet(fail_payload).gates
                if g.gate_number == 2).verdict == "FAIL"
    # WARN is not FAIL — sub-model gauntlet treats both PASS and WARN
    # as acceptable provenance.
    assert next(g for g in run_directional_gauntlet(warn_payload).gates
                if g.gate_number == 2).verdict == "PASS"


def test_gate_3_bootstrap_ci_lower():
    """ci_lower_5 above 0.5 → PASS; at or below → FAIL."""
    above = _make_directional_payload(ci_lower_5=0.52)
    below = _make_directional_payload(ci_lower_5=0.49)
    g_above = next(g for g in run_directional_gauntlet(above).gates if g.gate_number == 3)
    g_below = next(g for g in run_directional_gauntlet(below).gates if g.gate_number == 3)
    assert g_above.verdict == "PASS"
    assert g_below.verdict == "FAIL"


def test_gate_4_passes_for_multiclass_when_ci_below_threshold():
    """Gate 4 evaluates multiclass via per-class P(y=k|bin) shift.
    ci_max_abs_diff < 0.15 → PASS; ≥ 0.15 → FAIL.
    Replaces the old ``test_gate_4_skips_for_multiclass`` — multiclass
    conditional invariance was implemented 2026-05-27.
    """
    pass_payload = _make_directional_payload(ci_max_abs_diff=0.08)
    fail_payload = _make_directional_payload(ci_max_abs_diff=0.20)
    g4_pass = next(g for g in run_directional_gauntlet(pass_payload).gates if g.gate_number == 4)
    g4_fail = next(g for g in run_directional_gauntlet(fail_payload).gates if g.gate_number == 4)
    assert g4_pass.verdict == "PASS"
    assert g4_fail.verdict == "FAIL"


def test_gate_4_binary_passes_below_threshold():
    """Binary objective with ci_max_abs_diff below 0.15 → PASS."""
    payload = _make_directional_payload(
        objective="binary", ci_max_abs_diff=0.08,
    )
    g4 = next(g for g in run_directional_gauntlet(payload).gates if g.gate_number == 4)
    assert g4.verdict == "PASS"
    assert "ci_max_abs_diff" in g4.actual


def test_gate_4_binary_fails_above_threshold():
    """Binary objective with ci_max_abs_diff above 0.15 → FAIL."""
    payload = _make_directional_payload(
        objective="binary", ci_max_abs_diff=0.20,
    )
    g4 = next(g for g in run_directional_gauntlet(payload).gates if g.gate_number == 4)
    assert g4.verdict == "FAIL"


def test_gate_4_regression_threshold():
    """Regression objective uses 0.5 std threshold."""
    ok = _make_directional_payload(objective="regression", ci_max_abs_diff=0.3)
    nope = _make_directional_payload(objective="regression", ci_max_abs_diff=0.6)
    g_ok = next(g for g in run_directional_gauntlet(ok).gates if g.gate_number == 4)
    g_nope = next(g for g in run_directional_gauntlet(nope).gates if g.gate_number == 4)
    assert g_ok.verdict == "PASS"
    assert g_nope.verdict == "FAIL"


def test_gate_4_surfaces_adversarial_auc_as_info():
    """Even when gate 4 PASSes on CI, the adversarial AUC value should
    still be surfaced in ``actual`` for operator visibility — proving
    the methodological pivot didn't drop the diagnostic."""
    payload = _make_directional_payload(
        objective="binary", ci_max_abs_diff=0.05, adversarial_auc=0.99,
    )
    g4 = next(g for g in run_directional_gauntlet(payload).gates if g.gate_number == 4)
    assert g4.verdict == "PASS"
    assert g4.actual.get("adversarial_auc") == 0.99
    assert "info-only" in g4.rationale


def test_gate_5_requires_both_realistic_and_conservative_profitable():
    """Both regime fractions must be 1.0 (every fold profitable).
    Partial profitability fails."""
    both_pass = _make_directional_payload(
        cost_realistic_profitable=1.0, cost_conservative_profitable=1.0,
    )
    realistic_only = _make_directional_payload(
        cost_realistic_profitable=1.0, cost_conservative_profitable=0.5,
    )
    conservative_only = _make_directional_payload(
        cost_realistic_profitable=0.83, cost_conservative_profitable=1.0,
    )
    assert next(g for g in run_directional_gauntlet(both_pass).gates
                if g.gate_number == 5).verdict == "PASS"
    assert next(g for g in run_directional_gauntlet(realistic_only).gates
                if g.gate_number == 5).verdict == "FAIL"
    assert next(g for g in run_directional_gauntlet(conservative_only).gates
                if g.gate_number == 5).verdict == "FAIL"


def test_gate_6_caveated_pass_is_failed_as_binding():
    """M5g.9.1: a regime sub-cut pass with caveat=1 is not a binding
    pass — gate 6 should FAIL until low_vol / high_vol both have
    enough trades."""
    caveated = _make_directional_payload(
        regime_gate_6_pass=1.0, regime_gate_6_caveat=1.0,
    )
    g6 = next(g for g in run_directional_gauntlet(caveated).gates
              if g.gate_number == 6)
    assert g6.verdict == "FAIL"
    assert "caveat" in g6.rationale


def test_gate_6_full_pass_with_no_caveat():
    """pass=1.0, caveat=0.0 → binding PASS."""
    clean = _make_directional_payload(
        regime_gate_6_pass=1.0, regime_gate_6_caveat=0.0,
    )
    g6 = next(g for g in run_directional_gauntlet(clean).gates if g.gate_number == 6)
    assert g6.verdict == "PASS"


def test_gate_7_dsr_pass_with_active_trial_discount():
    """dsr_gate_7_pass=1, dsr_trial_discount_active=1 → PASS without
    caveat in rationale."""
    payload = _make_directional_payload(
        dsr_gate_7_pass=1.0, dsr_value=0.97, dsr_trial_discount_active=1.0,
    )
    g7 = next(g for g in run_directional_gauntlet(payload).gates if g.gate_number == 7)
    assert g7.verdict == "PASS"
    assert "CAVEAT" not in g7.rationale


def test_gate_7_dsr_pass_with_inactive_trial_discount_carries_caveat():
    """MR-DSR3 wiring: gate 7 PASS without trial discount is still PASS
    but rationale flags the caveat. M5h's overall verdict logic doesn't
    downgrade this today, but the caveat is surfaced for the operator."""
    payload = _make_directional_payload(
        dsr_gate_7_pass=1.0, dsr_value=0.97, dsr_trial_discount_active=0.0,
    )
    g7 = next(g for g in run_directional_gauntlet(payload).gates if g.gate_number == 7)
    assert g7.verdict == "PASS"
    assert "CAVEAT" in g7.rationale


def test_gate_7_dsr_fail_when_below_threshold():
    payload = _make_directional_payload(dsr_gate_7_pass=0.0, dsr_value=0.5)
    g7 = next(g for g in run_directional_gauntlet(payload).gates if g.gate_number == 7)
    assert g7.verdict == "FAIL"


# ── Tier B / C are all SKIP ──────────────────────────────────────────────


def test_tier_b_and_c_gates_are_all_skip():
    """Gates 8-13 are deferred to phase 2 or external review — they
    must always SKIP at the current infra level. If a future change
    accidentally flips one to PASS without the underlying work, this
    test catches it."""
    payload = _make_directional_payload()
    report = run_directional_gauntlet(payload)
    skip_gates = [g for g in report.gates if g.tier in ("B", "C")]
    assert len(skip_gates) == 6
    for g in skip_gates:
        assert g.verdict == "SKIP"


# ── Serialisation ────────────────────────────────────────────────────────


def test_directional_gauntlet_to_dict_round_trip():
    """The serialised dict carries all 13 gate entries with their
    threshold + actual fields preserved."""
    payload = _make_directional_payload()
    report = run_directional_gauntlet(payload)
    d = directional_gauntlet_to_dict(report)
    assert d["kind"] == "directional_13_gate"
    assert d["overall_verdict"] == "CONDITIONAL_PASS"
    assert len(d["gates"]) == 13
    assert d["gates"][0]["gate_number"] == 1
    assert d["gates"][-1]["gate_number"] == 13
    # Tier annotations preserved
    tiers = [g["tier"] for g in d["gates"]]
    assert tiers == ["A"] * 7 + ["B"] * 3 + ["C"] * 3


# ── Failure modes — missing metric keys ──────────────────────────────────


def test_missing_metric_keys_surface_as_fail_with_actionable_rationale():
    """When a binding gate's metric key is missing from metric_means
    (caller forgot to enable the M5g.X feature that produces it), the
    gate must FAIL with a rationale that points to the missing
    pipeline step. Uses a BINARY objective so gate 4 actually
    evaluates conditional invariance — multiclass would SKIP gate 4
    regardless of missing metrics."""
    payload = _make_directional_payload(objective="binary")
    # Strip every Tier A metric we know about
    payload["walk_forward"]["metric_means"] = {"macro_auc_ovr": 0.55}
    report = run_directional_gauntlet(payload)
    assert report.overall_verdict == "FAIL"
    g3 = next(g for g in report.gates if g.gate_number == 3)
    assert g3.verdict == "FAIL"
    assert "M5g.6" in g3.rationale or "bootstrap" in g3.rationale.lower()
    g4 = next(g for g in report.gates if g.gate_number == 4)
    assert g4.verdict == "FAIL"
    assert (
        "ci_max_abs_diff" in g4.rationale
        or "conditional" in g4.rationale.lower()
    )
    g7 = next(g for g in report.gates if g.gate_number == 7)
    assert g7.verdict == "FAIL"
    assert "M5g.10" in g7.rationale or "DSR" in g7.rationale.upper()


# ── MR-M5h2: strict threshold boundary tests ──────────────────────────────


def test_gate_1_exact_threshold_is_fail():
    """MR-M5h2: blueprint specifies STRICT > 0.55. mean = exactly 0.55
    must FAIL, not PASS."""
    payload = _make_directional_payload(primary_mean=0.55)
    g1 = next(g for g in run_directional_gauntlet(payload).gates if g.gate_number == 1)
    assert g1.verdict == "FAIL"


def test_gate_3_exact_threshold_is_fail():
    """MR-M5h2: blueprint specifies STRICT > 0.50. ci_lower_5 = exactly
    0.50 must FAIL."""
    payload = _make_directional_payload(ci_lower_5=0.50)
    g3 = next(g for g in run_directional_gauntlet(payload).gates if g.gate_number == 3)
    assert g3.verdict == "FAIL"


def test_gate_4_exact_threshold_is_fail():
    """MR-M5h2 (updated 2026-05-16): the strict-boundary contract still
    holds, but the metric changed from adversarial_auc (deprecated,
    info-only) to conditional invariance. For binary, exactly at the
    PASS_THRESHOLD['binary']=0.15 must FAIL (strict < threshold)."""
    payload = _make_directional_payload(
        objective="binary", ci_max_abs_diff=0.15,
    )
    g4 = next(g for g in run_directional_gauntlet(payload).gates if g.gate_number == 4)
    assert g4.verdict == "FAIL"


# ── MR-M5h3: NaN handling consistent across binding gates ────────────────


def test_gate_5_handles_nan_profitability_fractions():
    """MR-M5h3: NaN in cost_realistic_profitable / cost_conservative_
    profitable must FAIL with the missing-key rationale, not crash."""
    payload = _make_directional_payload(
        cost_realistic_profitable=float("nan"),
        cost_conservative_profitable=1.0,
    )
    g5 = next(g for g in run_directional_gauntlet(payload).gates if g.gate_number == 5)
    assert g5.verdict == "FAIL"
    assert "missing or not finite" in g5.rationale


def test_gate_6_handles_nan_caveat_frac():
    """MR-M5h3: NaN caveat_frac (no fold actually ran the sub-cut)
    must FAIL gate 6 — caveated pass without finite caveat info can't
    be a binding pass."""
    payload = _make_directional_payload(
        regime_gate_6_pass=1.0,
        regime_gate_6_caveat=float("nan"),
    )
    g6 = next(g for g in run_directional_gauntlet(payload).gates if g.gate_number == 6)
    assert g6.verdict == "FAIL"


def test_gate_7_handles_nan_dsr_pass():
    """MR-M5h3: NaN dsr_gate_7_pass surfaces the missing-key path."""
    payload = _make_directional_payload(dsr_gate_7_pass=float("nan"))
    g7 = next(g for g in run_directional_gauntlet(payload).gates if g.gate_number == 7)
    assert g7.verdict == "FAIL"
    assert "missing or not finite" in g7.rationale


# ── MN-M5h1: additional coverage ──────────────────────────────────────────


def test_gate_5_mixed_profitability_fails():
    """Partial profitability across folds (e.g., 4/5 = 0.8) must FAIL.
    Confirms gate 5's 'every fold profitable' strict criterion."""
    payload = _make_directional_payload(
        cost_realistic_profitable=0.6,    # 3/5 folds profitable
        cost_conservative_profitable=1.0,
    )
    g5 = next(g for g in run_directional_gauntlet(payload).gates if g.gate_number == 5)
    assert g5.verdict == "FAIL"
    assert "0.60" in g5.rationale or "0.6" in g5.rationale


def test_overall_verdict_counts_are_consistent():
    """n_pass + n_fail + n_skip must equal the total 13 gates in
    every overall verdict path."""
    # All-PASS payload
    p_pass = _make_directional_payload()
    r_pass = run_directional_gauntlet(p_pass)
    assert r_pass.n_pass + r_pass.n_fail + r_pass.n_skip == 13
    # Tier A fail
    p_fail = _make_directional_payload(adversarial_auc=0.99)
    r_fail = run_directional_gauntlet(p_fail)
    assert r_fail.n_pass + r_fail.n_fail + r_fail.n_skip == 13


def test_dsr_value_uses_scientific_notation_for_tiny_values():
    """MN-M5h4: rationale should show 5.6e-12, not 0.000000, for very
    small DSR values. Operator reading 0.000000 might misread it as
    literal zero."""
    payload = _make_directional_payload(dsr_value=5.6e-12, dsr_gate_7_pass=0.0)
    g7 = next(g for g in run_directional_gauntlet(payload).gates if g.gate_number == 7)
    assert "e-" in g7.rationale or "e+" in g7.rationale, (
        f"expected scientific notation in '{g7.rationale}'"
    )


def test_dsr_value_uses_decimal_for_normal_magnitudes():
    """MN-M5h4: ordinary-magnitude values still use plain decimal."""
    payload = _make_directional_payload(dsr_value=0.7234, dsr_gate_7_pass=0.0)
    g7 = next(g for g in run_directional_gauntlet(payload).gates if g.gate_number == 7)
    assert "0.7234" in g7.rationale
    # No scientific notation for a value around 0.7.
    assert "e-" not in g7.rationale
