"""Lighter 5-gate gauntlet for HYBRID sub-models (M5d).

Sub-models (regime / positioning / flow) modulate existing strategies;
they are not standalone trading systems. The blueprint's full 13-gate
gauntlet (capacity, cost stress, regime sub-cuts, retraining stability,
shadow validation, etc.) targets standalone strategies that must clear
a 10%/yr bar on their own. For sub-models we use a tighter set focused
on **edge + stability**:

  1. ``integrity_passed``         — training data wasn't degenerate
  2. ``walk_forward_complete``    — every configured fold produced a valid metric
  3. ``generalization_edge``      — walk-forward primary mean ≥ threshold
  4. ``fold_stability``           — walk-forward primary std ≤ threshold
  5. ``saved_booster_above_random``— the actual saved model isn't worse than random

All five are pure functions of an artifact payload — no DB queries, no
extra fits. Reviewer approval (Gate 8 in the full 13) is M5e's concern
and is not auto-computable.

Thresholds were calibrated by inspection of the M5b/M5c real results:
HYBRID modulators only need a small consistent edge to add value on top
of LSR/VCB/VBO. Tightening these turns the gate into "must beat
standalone strategies" which contradicts the modulator framing.

Determinism: each gate is a pure function of the payload, so the same
payload always produces the same verdict. The aggregate is the AND of
the gate verdicts — overall PASS only if every gate is PASS.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)


GateVerdict = Literal["PASS", "FAIL", "SKIP"]


# ── Thresholds (per-objective, all "modulator-grade") ──────────────────────


# Generalization edge — walk-forward mean must clear this to count as
# adding value over random.
_EDGE_THRESHOLD: dict[str, float] = {
    "auc": 0.52,         # AUC of 0.5 = random; 0.52 = small consistent edge
    "pearson_r": 0.05,   # Pearson r of 0 = random; 0.05 = correlated signal
}

# Fold stability — walk-forward std must stay below this. Catches
# "great mean, wild swings" models whose live behaviour is unreliable.
_STABILITY_THRESHOLD: dict[str, float] = {
    "auc": 0.15,         # 6-fold std of 0.15 means typical fold spread ±0.30
    "pearson_r": 0.20,
}

# Saved-booster baseline — the actual deployed model must clear random.
_RANDOM_BASELINE: dict[str, float] = {
    "auc": 0.5,
    "pearson_r": 0.0,
}


# ── Errors ─────────────────────────────────────────────────────────────────


class GauntletError(RuntimeError):
    """Raised by :func:`run_gauntlet` when the payload is missing
    required fields (e.g. ``walk_forward`` is None — gates 2-5 cannot be
    computed without it). Catch this at the CLI to convert into a clean
    error entry rather than a stack trace.
    """


# ── Records ────────────────────────────────────────────────────────────────


@dataclass
class GateResult:
    """One gate's outcome.

    ``threshold`` and ``actual`` are kept as dicts (not flat floats) so
    gates with multiple criteria can report them all. ``rationale`` is
    prose for the operator log; ``verdict`` is the machine-readable
    PASS / FAIL / SKIP that the aggregator reads.
    """

    name: str
    verdict: GateVerdict
    threshold: dict[str, Any] = field(default_factory=dict)
    actual: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""


@dataclass
class GauntletReport:
    spec_name: str
    overall_verdict: Literal["PASS", "FAIL"]
    gates: list[GateResult]
    artifact_content_sha: str
    data_fingerprint: str
    primary_metric: str   # echo from walk_forward so consumers don't have to look it up


# ── Individual gates ──────────────────────────────────────────────────────


def _gate_integrity_passed(payload: dict[str, Any]) -> GateResult:
    integ = payload.get("integrity")
    if not integ:
        return GateResult(
            name="integrity_passed", verdict="FAIL",
            rationale="payload has no integrity block",
        )
    verdict = integ.get("verdict")
    if verdict == "FAIL":
        # Should be unreachable in practice — training raises on FAIL —
        # but the gauntlet must still surface the verdict honestly if
        # someone hands us a tampered payload.
        return GateResult(
            name="integrity_passed", verdict="FAIL",
            actual={"integrity_verdict": verdict},
            rationale="integrity check failed during training",
        )
    return GateResult(
        name="integrity_passed", verdict="PASS",
        actual={"integrity_verdict": verdict},
        rationale=f"integrity verdict={verdict}",
    )


def _gate_walk_forward_complete(payload: dict[str, Any]) -> GateResult:
    wf = payload.get("walk_forward")
    if not wf:
        return GateResult(
            name="walk_forward_complete", verdict="FAIL",
            rationale="payload has no walk_forward block (re-train with --walk-forward)",
        )
    configured = wf.get("n_folds_configured", 0)
    generated = wf.get("n_folds_generated", 0)
    run = wf.get("n_folds_run", 0)
    valid = wf.get("n_folds_valid_metric", 0)
    actual = {
        "n_folds_configured": configured,
        "n_folds_generated": generated,
        "n_folds_run": run,
        "n_folds_valid_metric": valid,
    }
    if configured == generated == run == valid and configured > 0:
        return GateResult(
            name="walk_forward_complete", verdict="PASS",
            actual=actual,
            rationale=f"all {configured} configured folds generated, ran, and produced finite metric",
        )
    return GateResult(
        name="walk_forward_complete", verdict="FAIL",
        actual=actual,
        rationale=(
            f"fold counts diverge: configured={configured} generated={generated} "
            f"run={run} valid={valid} (need all equal and > 0)"
        ),
    )


def _gate_generalization_edge(payload: dict[str, Any]) -> GateResult:
    wf = payload.get("walk_forward")
    if not wf:
        return GateResult(
            name="generalization_edge", verdict="FAIL",
            rationale="no walk_forward block; gate cannot be computed",
        )
    metric_name = wf.get("primary_metric")
    mean = wf.get("primary_mean")
    threshold = _EDGE_THRESHOLD.get(metric_name)
    if threshold is None:
        return GateResult(
            name="generalization_edge", verdict="SKIP",
            actual={"metric": metric_name, "mean": mean},
            rationale=f"no edge threshold defined for metric '{metric_name}'",
        )
    if mean is None or math.isnan(mean):
        return GateResult(
            name="generalization_edge", verdict="FAIL",
            actual={"metric": metric_name, "mean": mean},
            threshold={"min_mean": threshold},
            rationale=f"walk-forward primary_mean is {mean!r}",
        )
    verdict = "PASS" if mean >= threshold else "FAIL"
    return GateResult(
        name="generalization_edge", verdict=verdict,
        actual={"metric": metric_name, "mean": round(float(mean), 4)},
        threshold={"min_mean": threshold},
        rationale=(
            f"walk-forward {metric_name} mean={mean:.4f} "
            f"({'≥' if verdict == 'PASS' else '<'} threshold {threshold})"
        ),
    )


def _gate_fold_stability(payload: dict[str, Any]) -> GateResult:
    wf = payload.get("walk_forward")
    if not wf:
        return GateResult(
            name="fold_stability", verdict="FAIL",
            rationale="no walk_forward block; gate cannot be computed",
        )
    metric_name = wf.get("primary_metric")
    std = wf.get("primary_std")
    threshold = _STABILITY_THRESHOLD.get(metric_name)
    if threshold is None:
        return GateResult(
            name="fold_stability", verdict="SKIP",
            actual={"metric": metric_name, "std": std},
            rationale=f"no stability threshold defined for metric '{metric_name}'",
        )
    if std is None or math.isnan(std):
        return GateResult(
            name="fold_stability", verdict="FAIL",
            actual={"metric": metric_name, "std": std},
            threshold={"max_std": threshold},
            rationale=f"walk-forward primary_std is {std!r}",
        )
    verdict = "PASS" if std <= threshold else "FAIL"
    return GateResult(
        name="fold_stability", verdict=verdict,
        actual={"metric": metric_name, "std": round(float(std), 4)},
        threshold={"max_std": threshold},
        rationale=(
            f"walk-forward {metric_name} std={std:.4f} "
            f"({'≤' if verdict == 'PASS' else '>'} threshold {threshold})"
        ),
    )


def _gate_saved_booster_above_random(payload: dict[str, Any]) -> GateResult:
    """The booster actually shipped in the artifact (whatever its
    ``eval_kind``) must clear the random baseline on its primary metric.

    For ``eval_kind='walk_forward_last_fold'`` this is the last fold's
    metric. For ``eval_kind='holdout_80_20'`` it's the 80/20 split's
    metric. Either way the metric describes the saved booster, by the
    M5c contract.
    """
    metrics = payload.get("metrics") or {}
    wf = payload.get("walk_forward")
    metric_name = (wf or {}).get("primary_metric")
    # If walk-forward isn't present, infer the primary metric from
    # the objective via the keys metrics actually carries.
    if metric_name is None:
        if "auc" in metrics:
            metric_name = "auc"
        elif "pearson_r" in metrics:
            metric_name = "pearson_r"
        else:
            return GateResult(
                name="saved_booster_above_random", verdict="FAIL",
                rationale="cannot infer primary metric from payload",
            )
    baseline = _RANDOM_BASELINE.get(metric_name)
    if baseline is None:
        return GateResult(
            name="saved_booster_above_random", verdict="SKIP",
            actual={"metric": metric_name},
            rationale=f"no random baseline defined for metric '{metric_name}'",
        )
    value = metrics.get(metric_name)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return GateResult(
            name="saved_booster_above_random", verdict="FAIL",
            actual={"metric": metric_name, "value": value},
            threshold={"min_value": baseline},
            rationale=f"saved booster's {metric_name} is {value!r}",
        )
    verdict = "PASS" if value >= baseline else "FAIL"
    return GateResult(
        name="saved_booster_above_random", verdict=verdict,
        actual={"metric": metric_name, "value": round(float(value), 4)},
        threshold={"min_value": baseline},
        rationale=(
            f"saved booster's {metric_name}={value:.4f} "
            f"({'≥' if verdict == 'PASS' else '<'} random baseline {baseline})"
        ),
    )


# ── Aggregator ────────────────────────────────────────────────────────────


_GATES = (
    _gate_integrity_passed,
    _gate_walk_forward_complete,
    _gate_generalization_edge,
    _gate_fold_stability,
    _gate_saved_booster_above_random,
)


def run_gauntlet(payload: dict[str, Any]) -> GauntletReport:
    """Run all 5 gates on the artifact payload. Overall verdict is PASS
    only if every gate is PASS — SKIP gates count as failing the
    overall AND because we cannot affirm a SKIP'd dimension. The
    operator can re-run with the missing inputs (e.g. ``--walk-forward``)
    or invoke the reviewer (M5e) for an explicit override.

    Required payload fields:
      * ``content_sha256``    — for traceability
      * ``data_fingerprint``  — likewise
      * ``integrity``         — gate 1
      * ``walk_forward``      — gates 2/3/4 (and gate 5 reads primary_metric from it)
      * ``metrics``           — gate 5
      * ``spec.name``         — surfaced in report
    """
    spec_block = payload.get("spec") or {}
    spec_name = spec_block.get("name", "<unknown>")
    # MG5 fix: the 5-gate gauntlet is for HYBRID modulator sub-models
    # (binary / regression). Running it on a directional / multiclass
    # model would silently SKIP gates 3/4/5 (their thresholds are keyed
    # on auc / pearson_r — multiclass uses macro_auc_ovr) and report a
    # confusing FAIL with rationale "no edge threshold for metric
    # macro_auc_ovr". Refuse up front with a pointer to the full
    # 13-gate gauntlet (M5h) that the directional path needs.
    spec_objective = spec_block.get("objective")
    spec_purpose = spec_block.get("purpose")
    if spec_objective == "multiclass" or spec_purpose == "directional":
        raise GauntletError(
            f"5-gate modulator gauntlet refuses spec={spec_name} "
            f"(purpose={spec_purpose!r}, objective={spec_objective!r}). "
            "Directional / multiclass models require the full 13-gate "
            "gauntlet (M5h); this 5-gate set is calibrated for HYBRID "
            "modulator sub-models only."
        )
    if payload.get("walk_forward") is None:
        raise GauntletError(
            f"gauntlet requires walk_forward block (spec={spec_name}); "
            "re-train with --walk-forward"
        )

    gates = [g(payload) for g in _GATES]
    overall = "PASS" if all(g.verdict == "PASS" for g in gates) else "FAIL"

    primary_metric = (payload.get("walk_forward") or {}).get("primary_metric", "")
    report = GauntletReport(
        spec_name=spec_name,
        overall_verdict=overall,
        gates=gates,
        artifact_content_sha=payload.get("content_sha256", ""),
        data_fingerprint=payload.get("data_fingerprint", ""),
        primary_metric=primary_metric,
    )

    for g in gates:
        logger.info(
            "gauntlet | spec=%s gate=%s verdict=%s rationale=%s",
            spec_name, g.name, g.verdict, g.rationale,
        )
    logger.info("gauntlet | spec=%s overall=%s", spec_name, overall)
    return report


# ── Serialisation ─────────────────────────────────────────────────────────


def gauntlet_to_dict(report: GauntletReport) -> dict[str, Any]:
    """JSON view of a GauntletReport — used by the CLI summary and the
    artifact's ``gauntlet`` block.
    """
    return {
        "spec_name": report.spec_name,
        "overall_verdict": report.overall_verdict,
        "primary_metric": report.primary_metric,
        "artifact_content_sha": report.artifact_content_sha,
        "data_fingerprint": report.data_fingerprint,
        "gates": [
            {
                "name": g.name,
                "verdict": g.verdict,
                "threshold": g.threshold,
                "actual": g.actual,
                "rationale": g.rationale,
            }
            for g in report.gates
        ],
    }
