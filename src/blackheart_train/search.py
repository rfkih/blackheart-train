"""Hyperparam grid search for one ModelSpec.

M5b scope (deliberately tight):

* **Grid only, no Bayesian.** Eight to ten predefined hyperparam
  combinations per objective. LightGBM converges in ~3 seconds on our
  training set, so even running 10 points × 3 sub-models = 30 fits
  finishes inside one minute.
* **Reuses the M5a chronological 80/20 split.** Walk-forward with
  embargo is M5c — keeping that out of M5b lets us validate the
  search/training/persist plumbing first.
* **Primary metrics:** AUC for binary, Pearson r for regression. Higher
  is better for both. NaN metrics (zero-variance prediction, degenerate
  fold) are treated as ``-inf`` so they never win the selection.
* **Load data once.** The whole search shares one ``LoadedDataset``;
  only :func:`train.fit_and_evaluate` runs per grid point. No DB calls
  inside the loop.

The :class:`SearchResult` records every run so M5d's reviewer gauntlet
can audit the search trajectory after the fact.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, replace
from typing import Any

from .loader import LoadedDataset
from .specs import ModelSpec
from .train import fit_and_evaluate, split_chronological

logger = logging.getLogger(__name__)


# ── Grids ──────────────────────────────────────────────────────────────────
#
# Conservative, intentionally small. The point of M5b is to demonstrate the
# search loop works and surfaces a hyperparam improvement over M5a's
# defaults, not to find the global optimum. M5c re-tunes per fold; M5d
# would broaden the grid if any sub-model shows promise.


_BINARY_GRID: tuple[dict[str, object], ...] = (
    {"num_leaves": 15, "learning_rate": 0.01, "min_child_samples": 50},
    {"num_leaves": 15, "learning_rate": 0.05, "min_child_samples": 50},
    {"num_leaves": 31, "learning_rate": 0.01, "min_child_samples": 50},
    {"num_leaves": 31, "learning_rate": 0.05, "min_child_samples": 50},
    {"num_leaves": 31, "learning_rate": 0.05, "min_child_samples": 20},
    {"num_leaves": 31, "learning_rate": 0.05, "min_child_samples": 100},
    {"num_leaves": 63, "learning_rate": 0.05, "min_child_samples": 50},
    {"num_leaves": 63, "learning_rate": 0.05, "min_child_samples": 100},
)


_REGRESSION_GRID: tuple[dict[str, object], ...] = (
    {"num_leaves": 15, "learning_rate": 0.01, "min_child_samples": 50},
    {"num_leaves": 15, "learning_rate": 0.05, "min_child_samples": 50},
    {"num_leaves": 31, "learning_rate": 0.01, "min_child_samples": 50},
    {"num_leaves": 31, "learning_rate": 0.05, "min_child_samples": 50},
    {"num_leaves": 31, "learning_rate": 0.05, "min_child_samples": 20},
    {"num_leaves": 31, "learning_rate": 0.05, "min_child_samples": 100},
    {"num_leaves": 63, "learning_rate": 0.05, "min_child_samples": 50},
    {"num_leaves": 63, "learning_rate": 0.05, "min_child_samples": 100},
)


# M5g.1: directional / multiclass. Grid is biased toward deeper trees and
# lower learning rates because the 3-class triple-barrier problem has
# more decision surfaces to fit than binary regime classification, and
# the rare horizon-end class needs the model to escape the easy bimodal
# local optimum. The proper fix (ensemble + meta-label) lands in
# M5g.3/4; the search-grid bias is just to help the foundation case
# pick a less collapsed point.
_MULTICLASS_GRID: tuple[dict[str, object], ...] = (
    {"num_leaves": 31, "learning_rate": 0.05, "min_child_samples": 50},
    {"num_leaves": 31, "learning_rate": 0.02, "min_child_samples": 50},
    {"num_leaves": 63, "learning_rate": 0.05, "min_child_samples": 50},
    {"num_leaves": 63, "learning_rate": 0.02, "min_child_samples": 50},
    {"num_leaves": 63, "learning_rate": 0.05, "min_child_samples": 20},
    {"num_leaves": 127, "learning_rate": 0.02, "min_child_samples": 50},
)


def grid_for(objective: str) -> tuple[dict[str, object], ...]:
    if objective == "binary":
        return _BINARY_GRID
    if objective == "multiclass":
        return _MULTICLASS_GRID
    return _REGRESSION_GRID


def primary_metric(objective: str) -> str:
    """Which metric the search optimises over. Higher is better.

    Multiclass uses macro-AUC OVR (one-vs-rest, macro-averaged) because:
    accuracy is dominated by the 59%/38% majority classes on the
    triple-barrier label and rewards the trivial-collapse model;
    macro-AUC OVR weights each class equally and penalises the model
    that ignores the 3% horizon-end class.
    """
    if objective == "binary":
        return "auc"
    if objective == "multiclass":
        return "macro_auc_ovr"
    return "pearson_r"


# ── Result records ─────────────────────────────────────────────────────────


@dataclass
class SearchRunResult:
    overrides: dict[str, object]
    metrics: dict[str, float]


@dataclass
class SearchResult:
    spec_name: str
    primary_metric: str
    best_overrides: dict[str, object]
    best_metric: float
    baseline_metric: float | None   # primary metric of the spec's default hyperparams
    runs: list[SearchRunResult] = field(default_factory=list)


# ── The search ─────────────────────────────────────────────────────────────


class SearchError(RuntimeError):
    """Raised when grid search cannot produce a usable result —
    typically every grid point's primary metric came back NaN. The
    caller (CLI or future orchestrator) converts this into a clean
    error rather than silently using whichever run happened to sort
    first.
    """


def _nan_to_neginf(x: float) -> float:
    """For 'higher is better' selection, treat NaN as the worst possible
    score so a degenerate fit never wins. RMSE-style 'lower is better'
    isn't used in M5b, but if it ever is the inverse applies — handle
    there, not here."""
    return -math.inf if math.isnan(x) else x


def _matches_default_hyperparams(
    overrides: dict[str, object], defaults: dict[str, object]
) -> bool:
    """True iff applying ``overrides`` on top of ``defaults`` would leave
    every searched key at its default value. Used to detect that a grid
    point IS the baseline so we don't fit it twice.
    """
    return all(defaults.get(k) == v for k, v in overrides.items())


def grid_search_one(
    ds: LoadedDataset,
    spec: ModelSpec,
    *,
    grid: tuple[dict[str, object], ...] | None = None,
) -> SearchResult:
    """Run the grid for one spec. Returns the best overrides + all runs.

    The dataset is taken pre-loaded so the caller can reuse it across
    multiple searches (e.g. when training all three sub-models in one
    CLI invocation we still pay the DB cost once per spec, not once per
    grid point).

    Baseline (default-hyperparam) metrics come from the grid point that
    matches spec.hyperparams when one exists; otherwise we run a
    separate baseline fit. This avoids the wasted ~3s of LightGBM time
    that the old code burned when the spec's defaults were already on
    the grid.

    Raises :class:`SearchError` if every grid run produced NaN on the
    primary metric — degenerate state we want to surface, not paper
    over by picking the first run silently.
    """
    if grid is None:
        grid = grid_for(spec.objective)
    metric_name = primary_metric(spec.objective)

    X_tr, y_tr, X_val, y_val = split_chronological(ds.X, ds.y, spec.val_fraction)

    runs: list[SearchRunResult] = []
    for i, overrides in enumerate(grid):
        merged = {**spec.hyperparams, **overrides}
        tuned = replace(spec, hyperparams=merged)
        _, metrics = fit_and_evaluate(X_tr, y_tr, X_val, y_val, tuned)
        runs.append(SearchRunResult(overrides=dict(overrides), metrics=metrics))
        logger.info(
            "search %d/%d | spec=%s %s=%.4f overrides=%s",
            i + 1, len(grid), spec.name, metric_name,
            metrics.get(metric_name, float("nan")), overrides,
        )

    # Baseline: prefer a grid run whose overrides leave defaults
    # unchanged. Falls back to a dedicated fit if no grid point matches.
    defaults = dict(spec.hyperparams)
    baseline_from_grid = next(
        (r for r in runs if _matches_default_hyperparams(r.overrides, defaults)),
        None,
    )
    if baseline_from_grid is not None:
        baseline = baseline_from_grid.metrics.get(metric_name, float("nan"))
        logger.info(
            "search baseline | spec=%s %s=%.4f (reused from grid)",
            spec.name, metric_name, baseline,
        )
    else:
        _, baseline_metrics = fit_and_evaluate(X_tr, y_tr, X_val, y_val, spec)
        baseline = baseline_metrics.get(metric_name, float("nan"))
        logger.info(
            "search baseline | spec=%s %s=%.4f (fresh fit)",
            spec.name, metric_name, baseline,
        )

    best = max(runs, key=lambda r: _nan_to_neginf(r.metrics.get(metric_name, float("nan"))))
    best_value = best.metrics.get(metric_name, float("nan"))

    # All-NaN guard: if max() landed on a NaN run (every run was NaN),
    # the "best" is meaningless. Caller deserves a clear error.
    if math.isnan(best_value):
        nan_count = sum(
            1 for r in runs if math.isnan(r.metrics.get(metric_name, float("nan")))
        )
        raise SearchError(
            f"grid search degenerate for spec={spec.name}: "
            f"{nan_count}/{len(runs)} runs produced NaN on primary metric '{metric_name}'"
        )

    logger.info(
        "search best | spec=%s %s=%.4f overrides=%s",
        spec.name, metric_name, best_value, best.overrides,
    )

    return SearchResult(
        spec_name=spec.name,
        primary_metric=metric_name,
        best_overrides=dict(best.overrides),
        best_metric=float(best_value),
        baseline_metric=float(baseline) if not math.isnan(baseline) else None,
        runs=runs,
    )


def tuned_spec(spec: ModelSpec, overrides: dict[str, object]) -> ModelSpec:
    """Return a new ModelSpec with ``overrides`` merged into hyperparams.

    Used by the CLI after :func:`grid_search_one` returns the best
    overrides — the final fit happens via ``train_one`` (or
    ``train_with_dataset``) on this tuned spec so the artifact's
    ``spec.hyperparams`` reflects the actually-trained values.
    """
    merged = {**spec.hyperparams, **overrides}
    return replace(spec, hyperparams=merged)


def search_result_to_dict(result: SearchResult) -> dict[str, Any]:
    """JSON-friendly view of a SearchResult — used by the CLI summary
    and the artifact's ``search`` block.
    """
    return {
        "spec_name": result.spec_name,
        "primary_metric": result.primary_metric,
        "best_overrides": result.best_overrides,
        "best_metric": result.best_metric,
        "baseline_metric": result.baseline_metric,
        "improvement_vs_baseline": (
            None
            if result.baseline_metric is None
            else round(result.best_metric - result.baseline_metric, 6)
        ),
        "runs": [
            {"overrides": r.overrides, "metrics": r.metrics}
            for r in result.runs
        ],
    }
