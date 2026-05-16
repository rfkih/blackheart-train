"""13-gate gauntlet for directional / multiclass specs (M5h, blueprint § 6-7).

The 5-gate :mod:`gauntlet` module is calibrated for HYBRID modulator
sub-models (regime / positioning / flow) — small consistent edge,
binary or regression. Directional standalone strategies must clear a
much higher bar: they trade autonomously and have to survive multiple
testing, cost stress, regime non-stationarity, and trial inflation.

The 13 gates fall into three tiers:

  Tier A — binding, computable from a single training run (gates 1-7):
    1. Baseline edge        — walk-forward primary mean ≥ 0.55
    2. PIT no-leak          — integrity verdict not FAIL
    3. Bootstrap CI lower   — macro_auc_ovr_ci_lower_5 ≥ 0.50
    4. Adversarial          — adversarial_auc ≤ 0.80 (covariate shift bound)
    5. Cost-regime profit   — realistic AND conservative profitable
    6. Regime sub-cut       — gate_6_pass = 1.0 AND caveat = 0.0
    7. DSR                  — dsr_gate_7_pass = 1.0

  Tier B — binding, but currently deferred to phase 2 (gates 8-10):
    8. Capacity stress      — does the strategy survive notional /
                              volume limits? Requires order-book data.
    9. Retraining stability — does a fresh retrain on shifted window
                              produce a similar booster? Requires a
                              second pipeline pass on the orchestrator.
    10. Shadow validation   — does paper-trade replay over an unseen
                              window match the model's confidence?
                              Requires M5c-style replay infra for
                              the directional spec.

  Tier C — manual, not machine-computable (gates 11-13):
    11. Reviewer approval   — paired-research adversarial audit
                              (quant-reviewer.md sub-agent).
    12. Operator approval   — explicit human gate before live trading.
    13. Cross-asset sanity  — does the model produce sensible signals
                              on ETHUSDT / SOLUSDT? Requires a
                              separately-fit cross-asset spec.

This module computes Tier A. Tier B is marked SKIP with rationale
"phase-2 infra not wired yet"; Tier C is SKIP with rationale
"requires external approval." The overall verdict is:

  overall_verdict =
    PASS    if all Tier A gates PASS AND no Tier B/C downgrades
    CONDITIONAL_PASS  if all Tier A gates PASS but Tier B/C have SKIPs
    FAIL    if any Tier A gate is FAIL

CONDITIONAL_PASS is the operator's signal that "the math says yes
but the human/infra layer hasn't validated yet." A model can move to
HYBRID-only with CONDITIONAL_PASS; INDEPENDENT trading requires PASS.

Determinism: each gate is a pure function of the payload. Same payload
in → same gate verdicts → same overall verdict.

Design choices (audit-pinned 2026-05-15):

* **Gate 6 caveat = FAIL, Gate 7 caveat = PASS-with-note: deliberate
  asymmetry.** Gate 6's caveat
  (``regime_gate_6_caveat=1``) fires when one regime had zero or
  under-min trades. That's evidence the strategy was NOT TESTED in
  that regime — a *strategy* gap; the operator can address it by
  widening eval windows or adding trend regimes. Gate 7's caveat
  (``dsr_trial_discount_active=0``) fires when no trial registry was
  wired to provide N>1 and V>0. That's an *infra* gap; the strategy
  itself isn't to blame. Punishing strategies for infra debt with a
  gate FAIL would create a perverse incentive (skip DSR rather than
  build a half-wired version). So Gate 6's caveat blocks the gate,
  while Gate 7's caveat surfaces a CAVEAT string in the rationale
  without blocking. Either side can be reconsidered when the trial
  registry lands in M5h-phase-2.
* **Gate 5 binding criterion = every fold profitable.** Blueprint
  § 7.2 says "profitable under realistic AND conservative." We read
  that as "every fold profitable in both regimes" — strict. An
  alternative read is "mean PnL positive across folds." Strict
  catches "great average, one catastrophic fold" patterns that mean-
  based readings miss.
* **Threshold strictness matches blueprint § 6-7 verbatim.** All
  comparisons are strict (``>``, ``<``), NOT inclusive. Borderline
  metrics exactly at the threshold fail.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)


GateVerdict = Literal["PASS", "FAIL", "SKIP"]
OverallVerdict = Literal["PASS", "CONDITIONAL_PASS", "FAIL"]


# ── Thresholds (per blueprint § 6-7) ──────────────────────────────────────


# Gate 1 — baseline directional edge. macro_auc_ovr > 0.50 is random;
# 0.55 is the blueprint's "meaningfully above random" threshold for
# multiclass directional. Lower than binary because 3-class baseline is
# 1/3 not 1/2 and partial-credit AUC dynamics are different.
_GATE_1_AUC_MIN: float = 0.55

# Gate 3 — bootstrap CI lower bound. Standard convention: the bottom
# of the 90% CI must clear random; if it doesn't, the point estimate
# being above random is plausibly luck.
_GATE_3_CI_LOWER_MIN: float = 0.50

# Gate 4 — transferability gate (replaces the original adversarial-AUC
# gate, 2026-05-16). The adversarial-AUC threshold (originally < 0.80)
# is now info-only because it measures P(features) shift, which is
# structurally ~1.0 for hourly bars with macro features on rolling WF
# (see ``project_v2_adversarial_auc.md``). The binding metric is now
# ``ci_max_abs_diff`` — the worst per-(feature, bin) shift in
# ``mean(y|bin)`` between train and test. For binary the threshold is
# a 15-percentage-point shift in P(y=1); for regression it's half a
# train-side standard deviation. Multiclass is skipped (gate emits SKIP
# with rationale "multiclass not yet supported") until the directional
# spec retry extends the conditional-invariance formulation.
#
# Both thresholds live in ``conditional_invariance.PASS_THRESHOLD`` so
# they can be tuned in one place.
_GATE_4_ADVERSARIAL_INFO_NOTE: str = (
    "adversarial_auc surfaced as info-only; conditional_invariance is "
    "the binding transferability metric"
)


# ── Errors ─────────────────────────────────────────────────────────────────


class DirectionalGauntletError(RuntimeError):
    """Raised when the payload is missing fields the 13-gate gauntlet
    requires (e.g. ``walk_forward`` is None — gates 1, 3-7 cannot run
    without per-fold metrics). Catch at the CLI to surface a clean
    error rather than a stack trace.
    """


# ── Records ────────────────────────────────────────────────────────────────


@dataclass
class GateResult:
    """One gate's outcome — same shape as the 5-gate module's
    GateResult so downstream JSON consumers don't have to branch."""

    name: str
    gate_number: int
    tier: Literal["A", "B", "C"]
    verdict: GateVerdict
    threshold: dict[str, Any] = field(default_factory=dict)
    actual: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""


@dataclass
class DirectionalGauntletReport:
    spec_name: str
    overall_verdict: OverallVerdict
    gates: list[GateResult]
    artifact_content_sha: str
    data_fingerprint: str
    n_pass: int
    n_fail: int
    n_skip: int


# ── Helpers ────────────────────────────────────────────────────────────────


def _wf_aggregate(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the walk-forward ``metric_means`` dict, or ``{}`` if
    walk_forward isn't present. Used by every binding gate that reads
    a per-fold-aggregated metric."""
    wf = payload.get("walk_forward") or {}
    return wf.get("metric_means") or {}


def _is_finite(x: Any) -> bool:
    """``x`` is a finite number (not None, not NaN, not inf)."""
    if x is None:
        return False
    try:
        x = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(x)


def _fmt_value(x: Any) -> str:
    """MN-M5h4: format a metric value for the rationale string.
    Uses scientific notation for very small magnitudes so the operator
    doesn't read ``dsr_value=0.000000`` and assume the value is
    literally zero when it's actually 5.6e-12."""
    if not _is_finite(x):
        return repr(x)
    f = float(x)
    if f != 0.0 and abs(f) < 1e-4:
        return f"{f:.3e}"
    return f"{f:.4f}"


# ── Tier A: binding, computable now ───────────────────────────────────────


def _gate_1_baseline_edge(payload: dict[str, Any]) -> GateResult:
    """Gate 1 — walk-forward primary metric mean clears the blueprint's
    above-random threshold."""
    wf = payload.get("walk_forward") or {}
    metric_name = wf.get("primary_metric", "")
    mean = wf.get("primary_mean")
    common = dict(
        name="baseline_edge", gate_number=1, tier="A",
        threshold={"min_mean": _GATE_1_AUC_MIN, "metric": metric_name},
    )
    if not _is_finite(mean):
        return GateResult(
            **common, verdict="FAIL",
            actual={"mean": mean},
            rationale=f"walk-forward primary_mean is {mean!r}",
        )
    # Strict per blueprint § 6: mean must be ABOVE the threshold,
    # not exactly at it.
    verdict = "PASS" if mean > _GATE_1_AUC_MIN else "FAIL"
    return GateResult(
        **common, verdict=verdict,
        actual={"mean": round(float(mean), 4)},
        rationale=(
            f"walk-forward {metric_name} mean={mean:.4f} "
            f"({'>' if verdict == 'PASS' else '≤'} {_GATE_1_AUC_MIN})"
        ),
    )


def _gate_2_pit_no_leak(payload: dict[str, Any]) -> GateResult:
    """Gate 2 — point-in-time leak check. The integrity block already
    encodes this; we surface the verdict in gate form."""
    integ = payload.get("integrity") or {}
    verdict_raw = integ.get("verdict")
    common = dict(name="pit_no_leak", gate_number=2, tier="A")
    if verdict_raw is None:
        return GateResult(
            **common, verdict="FAIL",
            rationale="payload has no integrity block",
        )
    verdict = "FAIL" if verdict_raw == "FAIL" else "PASS"
    return GateResult(
        **common, verdict=verdict,
        actual={"integrity_verdict": verdict_raw},
        rationale=f"integrity verdict={verdict_raw}",
    )


def _gate_3_bootstrap_ci_lower(payload: dict[str, Any]) -> GateResult:
    """Gate 3 — bootstrap CI lower 5% above the random baseline."""
    agg = _wf_aggregate(payload)
    ci_lower = agg.get("macro_auc_ovr_ci_lower_5")
    common = dict(
        name="bootstrap_ci_lower", gate_number=3, tier="A",
        threshold={"min_ci_lower_5": _GATE_3_CI_LOWER_MIN},
    )
    if not _is_finite(ci_lower):
        return GateResult(
            **common, verdict="FAIL",
            actual={"ci_lower_5": ci_lower},
            rationale=(
                f"macro_auc_ovr_ci_lower_5 missing or not finite "
                f"({ci_lower!r}); re-train with bootstrap enabled (M5g.6)"
            ),
        )
    # Strict per blueprint § 7.5: CI lower must be ABOVE 0.50 (random).
    verdict = "PASS" if ci_lower > _GATE_3_CI_LOWER_MIN else "FAIL"
    return GateResult(
        **common, verdict=verdict,
        actual={"ci_lower_5": round(float(ci_lower), 4)},
        rationale=(
            f"macro_auc_ovr_ci_lower_5={ci_lower:.4f} "
            f"({'>' if verdict == 'PASS' else '≤'} {_GATE_3_CI_LOWER_MIN})"
        ),
    )


def _gate_4_adversarial(payload: dict[str, Any]) -> GateResult:
    """Gate 4 — transferability (conditional invariance).

    Originally an adversarial-AUC bound (``< 0.80``). 2026-05-16: that
    measures ``P(features)`` shift, which is structurally ~1.0 for
    macro features on hourly walk-forward — see
    ``project_v2_adversarial_auc.md``. Replaced with conditional
    invariance: ``ci_max_abs_diff`` (worst per-(feature, bin) shift
    in ``mean(y|bin)`` between train and test) must clear the
    objective-specific threshold in
    ``conditional_invariance.PASS_THRESHOLD``.

    Adversarial AUC is still surfaced in ``actual`` for transparency
    (the operator can see covariate shift even though it doesn't gate).
    """
    from .conditional_invariance import PASS_THRESHOLD
    agg = _wf_aggregate(payload)
    ci_max = agg.get("ci_max_abs_diff")
    adv = agg.get("adversarial_auc")  # info-only; reported but not gated
    # The objective lives in the spec block, not the wf aggregate.
    spec_block = payload.get("spec") or {}
    objective = spec_block.get("objective")
    threshold = PASS_THRESHOLD.get(objective) if objective else None
    common = dict(
        name="transferability", gate_number=4, tier="A",
        threshold={
            "max_ci_max_abs_diff": threshold,
            "objective": objective,
            "info_note": _GATE_4_ADVERSARIAL_INFO_NOTE,
        },
    )
    # Multiclass: conditional invariance not yet supported (per the
    # module's "Non-goals" — needs a per-class formulation). Skip with
    # a clear rationale so the directional retry sees this immediately.
    if objective == "multiclass":
        return GateResult(
            **common, verdict="SKIP",
            actual={
                "adversarial_auc": (
                    round(float(adv), 4) if _is_finite(adv) else adv
                ),
                "ci_max_abs_diff": ci_max,
            },
            rationale=(
                "conditional invariance not yet implemented for multiclass; "
                "extend conditional_invariance.py before re-attempting "
                "directional gauntlet"
            ),
        )
    if threshold is None:
        return GateResult(
            **common, verdict="FAIL",
            actual={
                "objective": objective,
                "ci_max_abs_diff": ci_max,
                "adversarial_auc": adv,
            },
            rationale=(
                f"no PASS_THRESHOLD configured for objective={objective!r}; "
                f"add to conditional_invariance.PASS_THRESHOLD"
            ),
        )
    if not _is_finite(ci_max):
        return GateResult(
            **common, verdict="FAIL",
            actual={"ci_max_abs_diff": ci_max, "adversarial_auc": adv},
            rationale=(
                f"ci_max_abs_diff missing or not finite ({ci_max!r}); "
                f"re-train with conditional-invariance computed in walk-forward"
            ),
        )
    verdict = "PASS" if ci_max < threshold else "FAIL"
    return GateResult(
        **common, verdict=verdict,
        actual={
            "ci_max_abs_diff": round(float(ci_max), 4),
            "ci_mean_abs_diff": (
                round(float(agg["ci_mean_abs_diff"]), 4)
                if _is_finite(agg.get("ci_mean_abs_diff")) else None
            ),
            "adversarial_auc": (
                round(float(adv), 4) if _is_finite(adv) else adv
            ),
        },
        rationale=(
            f"ci_max_abs_diff={ci_max:.4f} "
            f"({'<' if verdict == 'PASS' else '≥'} {threshold}); "
            f"adversarial_auc={_fmt_value(adv)} (info-only)"
        ),
    )


def _gate_5_cost_regime_profitable(payload: dict[str, Any]) -> GateResult:
    """Gate 5 — net PnL profitable under BOTH realistic AND conservative
    regimes. Adversarial regime is informational only per blueprint
    § 7.2 (the +50 bps slippage stress is an upper-bound sanity check,
    not binding)."""
    agg = _wf_aggregate(payload)
    realistic = agg.get("cost_realistic_profitable")
    conservative = agg.get("cost_conservative_profitable")
    realistic_net = agg.get("cost_realistic_net_pnl_bps_per_trade")
    common = dict(
        name="cost_regime_profitable", gate_number=5, tier="A",
        threshold={
            "min_fraction_folds_profitable_realistic": 1.0,
            "min_fraction_folds_profitable_conservative": 1.0,
        },
    )
    if not (_is_finite(realistic) and _is_finite(conservative)):
        return GateResult(
            **common, verdict="FAIL",
            actual={
                "realistic_profitable_frac": realistic,
                "conservative_profitable_frac": conservative,
            },
            rationale=(
                "cost-regime metrics missing or not finite; re-train "
                "with M5g.8 cost-regime simulation enabled (meta-label "
                "gating active in the ensemble branch)"
            ),
        )
    # The walk-forward aggregator stores 0/1 floats as the FRACTION of
    # folds where each regime was profitable. Gate 5 binding criterion:
    # every fold must be profitable in realistic + conservative
    # (strict reading of blueprint § 7.2 — see module docstring's
    # "Design choices" for the rationale vs alternative readings).
    pass_realistic = realistic >= 1.0
    pass_conservative = conservative >= 1.0
    verdict = "PASS" if (pass_realistic and pass_conservative) else "FAIL"
    return GateResult(
        **common, verdict=verdict,
        actual={
            "realistic_profitable_frac": round(float(realistic), 4),
            "conservative_profitable_frac": round(float(conservative), 4),
            "realistic_net_bps_per_trade": (
                round(float(realistic_net), 2) if _is_finite(realistic_net) else None
            ),
        },
        rationale=(
            f"realistic={realistic:.2f}, conservative={conservative:.2f} "
            f"(both need 1.0 for PASS)"
        ),
    )


def _gate_6_regime_subcut(payload: dict[str, Any]) -> GateResult:
    """Gate 6 — per-regime PnL t-test passes in every fold AND caveat is
    not set (no regime had < min_trades). Caveat = pass-on-incomplete-
    evidence; we treat caveated-pass as FAIL here because the binding
    standard is "full evidence on all regimes."""
    agg = _wf_aggregate(payload)
    pass_frac = agg.get("regime_gate_6_pass")
    caveat_frac = agg.get("regime_gate_6_caveat")
    common = dict(
        name="regime_subcut", gate_number=6, tier="A",
        threshold={
            "min_fraction_folds_pass": 1.0,
            "max_caveat_frac": 0.0,   # M5g.9.1: caveated passes don't bind
        },
    )
    if not _is_finite(pass_frac):
        return GateResult(
            **common, verdict="FAIL",
            actual={"pass_frac": pass_frac, "caveat_frac": caveat_frac},
            rationale=(
                "regime_gate_6_pass missing or not finite; re-train "
                "with M5g.9 regime sub-cut enabled (requires "
                "btc_realized_vol_30d feature)"
            ),
        )
    # Gate 6's caveat fires on STRATEGY weakness (one regime empty or
    # under-min trades) — see module docstring's "Design choices" for
    # why caveats here block the gate while Gate 7 caveats don't.
    full_pass = pass_frac >= 1.0
    no_caveat = _is_finite(caveat_frac) and caveat_frac <= 0.0
    verdict = "PASS" if (full_pass and no_caveat) else "FAIL"
    return GateResult(
        **common, verdict=verdict,
        actual={
            "pass_frac": round(float(pass_frac), 4),
            "caveat_frac": (
                round(float(caveat_frac), 4) if caveat_frac is not None else None
            ),
        },
        rationale=(
            f"regime gate 6 pass_frac={pass_frac:.2f}, caveat_frac="
            f"{caveat_frac:.2f}" if caveat_frac is not None else
            f"regime gate 6 pass_frac={pass_frac:.2f}, caveat unset"
        ) + (
            "" if verdict == "PASS" else
            " (binding criterion: pass_frac=1.0 AND caveat_frac=0.0)"
        ),
    )


def _gate_7_dsr(payload: dict[str, Any]) -> GateResult:
    """Gate 7 — Deflated Sharpe Ratio passes its threshold. If trial
    discount isn't active (n_trials=1 or sr_variance=0), the gate is
    binding but flagged as missing the multiple-testing layer."""
    agg = _wf_aggregate(payload)
    dsr_pass = agg.get("dsr_gate_7_pass")
    dsr_value = agg.get("dsr_value")
    discount_active = agg.get("dsr_trial_discount_active")
    common = dict(
        name="dsr", gate_number=7, tier="A",
        threshold={"min_dsr_value": 0.95, "trial_discount_active": 1.0},
    )
    if not _is_finite(dsr_pass):
        return GateResult(
            **common, verdict="FAIL",
            actual={
                "dsr_value": dsr_value, "trial_discount_active": discount_active,
            },
            rationale=(
                "dsr_gate_7_pass missing or not finite; re-train with "
                "M5g.10 DSR enabled (requires meta-label gating active)"
            ),
        )
    verdict = "PASS" if dsr_pass >= 1.0 else "FAIL"
    caveat_note = ""
    if verdict == "PASS" and not (_is_finite(discount_active) and discount_active >= 1.0):
        # PSR survived but no real trial discount applied. Per the
        # asymmetry documented in the module docstring, Gate 7's caveat
        # is INFRA debt (trial registry not wired), not strategy
        # weakness — so we PASS-with-note rather than FAIL like Gate 6.
        caveat_note = (
            " | CAVEAT: trial_discount_active=0 — gate passed on "
            "PSR(SR*=0) only, no multiple-testing penalty applied "
            "(M5h needs trial registry wiring)"
        )
    return GateResult(
        **common, verdict=verdict,
        actual={
            "dsr_value": float(dsr_value) if _is_finite(dsr_value) else dsr_value,
            "trial_discount_active": (
                float(discount_active) if _is_finite(discount_active) else None
            ),
        },
        rationale=(
            f"dsr_value={_fmt_value(dsr_value)}, threshold=0.95" + caveat_note
        ),
    )


# ── Tier B (deferred to phase 2) ─────────────────────────────────────────


def _gate_8_capacity_stress(payload: dict[str, Any]) -> GateResult:
    return GateResult(
        name="capacity_stress", gate_number=8, tier="B",
        verdict="SKIP",
        rationale=(
            "phase-2: capacity stress test (notional / volume limits) "
            "requires order-book depth integration"
        ),
    )


def _gate_9_retraining_stability(payload: dict[str, Any]) -> GateResult:
    return GateResult(
        name="retraining_stability", gate_number=9, tier="B",
        verdict="SKIP",
        rationale=(
            "phase-2: retraining-stability check requires a second "
            "pipeline pass on a shifted train window (orchestrator infra)"
        ),
    )


def _gate_10_shadow_validation(payload: dict[str, Any]) -> GateResult:
    return GateResult(
        name="shadow_validation", gate_number=10, tier="B",
        verdict="SKIP",
        rationale=(
            "phase-2: paper-trade replay over unseen window — M5c-style "
            "replay infra needs directional-spec adaptation"
        ),
    )


# ── Tier C (manual / external) ───────────────────────────────────────────


def _gate_11_reviewer_approval(payload: dict[str, Any]) -> GateResult:
    return GateResult(
        name="reviewer_approval", gate_number=11, tier="C",
        verdict="SKIP",
        rationale=(
            "manual: requires quant-reviewer.md adversarial audit (paired-"
            "research sub-agent), not machine-computable"
        ),
    )


def _gate_12_operator_approval(payload: dict[str, Any]) -> GateResult:
    return GateResult(
        name="operator_approval", gate_number=12, tier="C",
        verdict="SKIP",
        rationale="manual: explicit human approval before INDEPENDENT live trading",
    )


def _gate_13_cross_asset_sanity(payload: dict[str, Any]) -> GateResult:
    return GateResult(
        name="cross_asset_sanity", gate_number=13, tier="C",
        verdict="SKIP",
        rationale=(
            "phase-2 / manual: cross-asset consistency on ETHUSDT/SOLUSDT "
            "requires separately-fit cross-asset specs"
        ),
    )


# ── Aggregator ────────────────────────────────────────────────────────────


_GATES_13 = (
    _gate_1_baseline_edge,
    _gate_2_pit_no_leak,
    _gate_3_bootstrap_ci_lower,
    _gate_4_adversarial,
    _gate_5_cost_regime_profitable,
    _gate_6_regime_subcut,
    _gate_7_dsr,
    _gate_8_capacity_stress,
    _gate_9_retraining_stability,
    _gate_10_shadow_validation,
    _gate_11_reviewer_approval,
    _gate_12_operator_approval,
    _gate_13_cross_asset_sanity,
)


def run_directional_gauntlet(payload: dict[str, Any]) -> DirectionalGauntletReport:
    """Run all 13 gates on a directional/multiclass artifact payload.

    Verdict logic:

    * Any Tier A FAIL → overall FAIL.
    * All Tier A PASS, Tier B/C have SKIPs → CONDITIONAL_PASS.
    * All 13 PASS (only possible once Tier B is built) → PASS.

    CONDITIONAL_PASS is the operator's signal that the math is green
    but the human / infra layer hasn't validated yet. HYBRID-mode
    deployment is allowed at CONDITIONAL_PASS; INDEPENDENT trading
    requires PASS.

    Raises :class:`DirectionalGauntletError` if the payload is missing
    walk_forward (gates 1, 3-7 need per-fold aggregates).
    """
    spec_block = payload.get("spec") or {}
    spec_name = spec_block.get("name", "<unknown>")
    if payload.get("walk_forward") is None:
        raise DirectionalGauntletError(
            f"13-gate gauntlet requires walk_forward block (spec={spec_name}); "
            "re-train with --walk-forward"
        )

    gates = [g(payload) for g in _GATES_13]
    tier_a = [g for g in gates if g.tier == "A"]
    n_pass = sum(1 for g in gates if g.verdict == "PASS")
    n_fail = sum(1 for g in gates if g.verdict == "FAIL")
    n_skip = sum(1 for g in gates if g.verdict == "SKIP")

    if any(g.verdict == "FAIL" for g in tier_a):
        overall: OverallVerdict = "FAIL"
    elif all(g.verdict == "PASS" for g in tier_a) and n_skip == 0:
        overall = "PASS"
    elif all(g.verdict in ("PASS", "SKIP") for g in tier_a):
        # Tier A is PASS-or-SKIP (no FAILs). Gate 4 emits SKIP for
        # multiclass on the conditional-invariance metric (infra gap,
        # not a strategy gap — see ``conditional_invariance.py`` and
        # ``project_v2_adversarial_auc.md``). Tier B/C also SKIP at
        # current infra level. Combined: CONDITIONAL_PASS. The
        # rationale on each SKIP gate identifies the infra gap.
        overall = "CONDITIONAL_PASS"
    else:
        # Defensive: shouldn't be reachable given the branches above.
        # Logged as a guardrail in case a future tier extension allows
        # a state combination we didn't anticipate.
        overall = "FAIL"

    report = DirectionalGauntletReport(
        spec_name=spec_name, overall_verdict=overall, gates=gates,
        artifact_content_sha=payload.get("content_sha256", ""),
        data_fingerprint=payload.get("data_fingerprint", ""),
        n_pass=n_pass, n_fail=n_fail, n_skip=n_skip,
    )

    for g in gates:
        logger.info(
            "13-gate | spec=%s gate=%d(%s) tier=%s verdict=%s rationale=%s",
            spec_name, g.gate_number, g.name, g.tier, g.verdict, g.rationale,
        )
    logger.info(
        "13-gate | spec=%s overall=%s pass=%d fail=%d skip=%d",
        spec_name, overall, n_pass, n_fail, n_skip,
    )
    return report


# ── Serialisation ─────────────────────────────────────────────────────────


def directional_gauntlet_to_dict(
    report: DirectionalGauntletReport,
) -> dict[str, Any]:
    """JSON view of the 13-gate report — used by the CLI summary and
    the artifact's ``gauntlet`` block.
    """
    return {
        "kind": "directional_13_gate",
        "spec_name": report.spec_name,
        "overall_verdict": report.overall_verdict,
        "n_pass": report.n_pass,
        "n_fail": report.n_fail,
        "n_skip": report.n_skip,
        "artifact_content_sha": report.artifact_content_sha,
        "data_fingerprint": report.data_fingerprint,
        "gates": [
            {
                "gate_number": g.gate_number,
                "name": g.name,
                "tier": g.tier,
                "verdict": g.verdict,
                "threshold": g.threshold,
                "actual": g.actual,
                "rationale": g.rationale,
            }
            for g in report.gates
        ],
    }
