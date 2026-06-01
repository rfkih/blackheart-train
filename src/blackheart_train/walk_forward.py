"""Walk-forward CV with rolling refit and embargo.

M5c per blueprint § 7.1. Each fold is

    [train_start, train_end) → [embargo] → [test_start, test_end)

with ``test_start = train_end + embargo`` and ``test_end = test_start +
test_days``. Successive folds shift forward by ``test_days`` so test
windows are non-overlapping and chronological.

Why an embargo
--------------

Our labels read forward (``label_return_7d`` looks 7 days ahead;
``label_regime_risk_on_48h`` looks 48 bars ahead; etc.). A train bar at
time T and a test bar at time T+ε can share their forward windows —
the model sees the same future twice. A 7-day embargo blocks that leak
for every label whose horizon is ≤7 days. Labels with longer horizons
(e.g. ``label_return_7d`` at 168h ≡ 7 days exactly) are protected at
the boundary; if we add a 14-day or 30-day label in the future, the
embargo needs to grow with it.

Why calendar windows
--------------------

Bars are dropped by the NaN filter at different rates across the
training window (e.g. ``eth_btc_ratio_momentum_20d`` starts mid-2025
so the first half of the bar grid is sparse). Calendar windows reflect
"as-of date T" the way live trading does — the fold's training set is
"every bar we'd have at this calendar moment," not "the next N rows
from somewhere in the cleaned matrix." Index-position windows would be
simpler but blur the temporal contract.

Defaults
--------

Blueprint says 12-mo train / 7-day embargo / 3-mo test / 6 folds for
the full plan. Our 17-month dataset doesn't fit ~30 months of
non-overlapping span, so we scale the test window to 21 days (3 weeks)
while keeping train/embargo at blueprint values:

* train_days = 365 (12 months)
* embargo_days = 7
* test_days = 21
* n_folds = 6 → total span 365 + 7 + 6×21 = 498 days ≈ 16.6 months

Aggregation
-----------

Reports per-fold metrics plus mean / median / std over the *primary*
metric (AUC for binary, Pearson r for regression). NaN folds (e.g.
binary fold whose test set has one class) are excluded from
aggregation but kept in the per-fold list with ``skipped_reason``.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from .loader import LoadedDataset
from .search import primary_metric
from .specs import ModelSpec
from .train import (
    N_MULTICLASS_CLASSES,
    build_payload,
    filter_eval_to_serving_interval,
    fit_and_evaluate,
    run_integrity_or_raise,
)

logger = logging.getLogger(__name__)


# ── Defaults ───────────────────────────────────────────────────────────────


DEFAULT_TRAIN_DAYS = 365
DEFAULT_TEST_DAYS = 21
DEFAULT_EMBARGO_DAYS = 7
DEFAULT_N_FOLDS = 6
# Reduce to 14 days to unlock ~10 folds on 17-month datasets without
# extending the training window. Uncomment on the CLI call when you
# need more OOF evaluation coverage (lower statistical noise on gauntlet
# verdicts); keep 21 for registered models to preserve comparability.
_FINE_TEST_DAYS = 14

# Minimum embargo required per label, in calendar days. embargo_days
# must cover at least the label's forward-lookahead window to prevent
# train/test label leakage at fold boundaries. 7 days is safe for all
# current 24-bar × 1h = 24h labels; 168-bar labels (label_return_7d)
# require ≥7 days (exactly at the edge — use ≥8 if in doubt).
_LABEL_MIN_EMBARGO_DAYS: dict[str, int] = {
    "label_return_7d": 7,
    "label_return_24h": 1,
    "label_meanrev_24h": 1,
    "label_regime_risk_on_48h": 2,
    "label_regime_risk_on_24h": 1,
    "label_triple_barrier": 1,
    "label_long_win_tb_1h_v1": 1,
    "label_long_win_tb_short_v1": 1,
    "label_long_win_tb_loose_v1": 1,
    "label_short_win_tb_1h_v1": 1,
}


# ── Errors ─────────────────────────────────────────────────────────────────


class WalkForwardError(RuntimeError):
    """Raised when walk-forward can't produce any valid fold — e.g.
    every fold's test set has only one class on a binary objective, or
    every test window falls entirely on dropped rows.
    """


# ── Records ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Fold:
    """Pure boundaries — no data."""

    k: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime


@dataclass
class FoldMetrics:
    """One fold's result. ``skipped_reason`` is set iff the fold did
    not produce a fit (empty window, single-class test, etc.); in that
    case ``metrics`` is an empty dict.

    ``features_selected`` (M5g.7) is the per-fold output of
    :func:`feature_selection.select_features` when the spec opts in;
    empty list otherwise (no selection ran). Persisted so a reviewer
    can audit consistency across folds.

    ``oof_predictions`` and ``oof_timestamps`` (V79 / Phase 4
    D-execution): point-in-time-correct out-of-fold predictions for
    this fold's test window. Persisted so the backfill script can
    replay them into ``signal_history`` for honest backtest serving.

    * For binary specs, each prediction is ``P(class=1)``.
    * For multiclass specs, each row is a probability vector.
    * Empty list when the fold was skipped (no fit ran).

    Predictions arrive from ``booster.predict(X_te)`` after the fold's
    fit. Length matches ``n_test`` and aligns 1:1 with
    ``oof_timestamps`` (same order as ``X_te.index``).
    """

    fold: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    n_train: int
    n_test: int
    metrics: dict[str, float] = field(default_factory=dict)
    skipped_reason: str | None = None
    features_selected: list[str] = field(default_factory=list)
    oof_predictions: list[Any] = field(default_factory=list)
    oof_timestamps: list[datetime] = field(default_factory=list)
    # Top-10 LightGBM gain importances for this fold's booster.
    # Keyed by feature name; value is the raw gain (not normalised).
    # Empty when the estimator type doesn't expose feature_importance().
    feature_importances: dict[str, float] = field(default_factory=dict)


@dataclass
class WalkForwardResult:
    spec_name: str
    primary_metric: str
    # Three counts with three meanings — anyone reading the artifact can
    # tell whether a missing fold was "data ran out before generation"
    # or "skipped at run time" (e.g. empty test window, single class).
    n_folds_configured: int   # what the caller asked for
    n_folds_generated: int    # how many boundary tuples generate_folds returned
    n_folds_run: int          # how many actually fit a model
    n_folds_valid_metric: int # how many had finite primary metric
    primary_mean: float
    primary_median: float
    primary_std: float
    metric_means: dict[str, float]   # mean per metric across valid folds
    # Slope of primary-metric values across fold indices (OLS, 1 unit = 1 fold).
    # Negative slope means the metric is declining over time — a signal that
    # the model's edge may be decaying. NaN when < 2 valid folds.
    primary_fold_trend_slope: float = field(default=float("nan"))
    # Best fold by primary metric and its value — diagnostic, not used for
    # model selection (last-fold semantics are preserved for production).
    best_fold_k: int | None = field(default=None)
    best_primary_value: float | None = field(default=None)
    # True when filter_eval_to_serving_interval was active for this spec:
    # the model trains on all intervals but eval (and serving) use only
    # the spec's target interval. CI runs on full X_tr vs filtered X_te,
    # so conditional-invariance is measured across distributions.
    train_serve_filter_active: bool = field(default=False)
    folds: list[FoldMetrics] = field(default_factory=list)
    # Estimator from the most recent valid fold — a ``lgb.Booster`` for
    # single-model specs, or an ``ensemble.Ensemble`` for multi-model
    # specs (M5g.3 phase 2). Used by the walk-forward training path to
    # seed the artifact's saved estimator. Deliberately excluded from
    # :func:`walk_forward_to_dict` since neither shape is JSON-safe.
    # Field name is kept as ``_last_booster`` for back-compat with the
    # single-model path; the contained object's true type is checked at
    # the build_payload boundary.
    _last_booster: Any = field(default=None, repr=False, compare=False)


# ── Fold generation ───────────────────────────────────────────────────────


def generate_folds(
    train_start: datetime,
    train_end: datetime,
    *,
    train_days: int = DEFAULT_TRAIN_DAYS,
    test_days: int = DEFAULT_TEST_DAYS,
    embargo_days: int = DEFAULT_EMBARGO_DAYS,
    n_folds: int = DEFAULT_N_FOLDS,
) -> list[Fold]:
    """Compute fold boundaries within ``[train_start, train_end)``.

    Stops short of ``n_folds`` if data runs out — the caller surfaces
    this via ``WalkForwardResult.n_folds_configured`` (param) vs
    ``len(result.folds)`` (actual).
    """
    if train_days <= 0 or test_days <= 0 or embargo_days < 0 or n_folds <= 0:
        raise ValueError(
            f"all walk-forward sizes must be positive (embargo may be 0): "
            f"train_days={train_days} test_days={test_days} "
            f"embargo_days={embargo_days} n_folds={n_folds}"
        )

    train_delta = timedelta(days=train_days)
    test_delta = timedelta(days=test_days)
    embargo_delta = timedelta(days=embargo_days)

    # Fold 0's test window starts immediately after the first full
    # training window + embargo. Successive folds shift forward by
    # test_days so test windows do not overlap.
    first_test_start = train_start + train_delta + embargo_delta

    folds: list[Fold] = []
    for k in range(n_folds):
        test_start = first_test_start + k * test_delta
        test_end = test_start + test_delta
        if test_end > train_end:
            logger.info(
                "walk-forward stops at fold %d/%d | test_end=%s exceeds train_end=%s",
                k, n_folds, test_end.isoformat(), train_end.isoformat(),
            )
            break
        fold_train_end = test_start - embargo_delta
        fold_train_start = fold_train_end - train_delta
        folds.append(Fold(
            k=k,
            train_start=fold_train_start,
            train_end=fold_train_end,
            test_start=test_start,
            test_end=test_end,
        ))
    return folds


# ── Walk-forward execution ────────────────────────────────────────────────


def _slice_by_time(
    ds: LoadedDataset, start: datetime, end: datetime
) -> tuple:
    """Half-open ``[start, end)`` slice of the loaded dataset by ts.

    Returns ``(X_slice, y_slice)``. Boolean-mask ``.loc`` indexing
    already returns a fresh DataFrame/Series — no extra ``.copy()``
    needed. Both empty if no rows fall in range.
    """
    mask = (ds.X.index >= start) & (ds.X.index < end)
    return ds.X.loc[mask], ds.y.loc[mask]


def _capture_oof(
    booster: Any, X_te: Any
) -> tuple[list[Any], list[datetime]]:
    """V79: extract OOF predictions + their timestamps for one fold.

    Dispatches on estimator type:

    * ``lgb.Booster`` — call ``.predict(X)``. Returns 1D for binary,
      2D for multiclass.
    * ``Ensemble`` — route through ``predict_proba_ensemble``, which
      returns the deterministic averaged 2D proba.
    * Anything else (test stubs, future model types) — log a warning
      and return empty lists so the fold still records its metrics
      but the backfill script will skip it.

    Returns (predictions_list, timestamps_list). Empty when no usable
    predict path is available.
    """
    # Use duck-typing rather than `isinstance` to avoid hard imports of
    # optional heavy deps (lightgbm, ensemble module) at this module
    # level. The Ensemble dataclass has a ``models`` field; lgb.Booster
    # has a ``predict`` method.
    timestamps = list(X_te.index.to_pydatetime())
    if hasattr(booster, "predict") and not hasattr(booster, "models"):
        preds = np.asarray(booster.predict(X_te))
        return preds.tolist(), timestamps
    if hasattr(booster, "models"):
        from .ensemble import predict_proba_ensemble
        preds = np.asarray(predict_proba_ensemble(booster, X_te))
        return preds.tolist(), timestamps
    logger.warning(
        "walk-forward: cannot capture OOF predictions — estimator type "
        "%s has neither .predict nor .models. OOF list left empty for "
        "this fold; backfill will skip it.",
        type(booster).__name__,
    )
    return [], []


def _capture_feature_importances(booster: Any, top_n: int = 10) -> dict[str, float]:
    """Extract per-feature gain importances from a booster or ensemble.

    For an lgb.Booster: calls ``.feature_importance(importance_type='gain')``.
    For an Ensemble: averages gain importances across sub-models.
    Returns an empty dict when the estimator type doesn't expose importances
    (diagnostic only — never blocks training).
    """
    try:
        if hasattr(booster, "feature_importance") and not hasattr(booster, "models"):
            imps = booster.feature_importance(importance_type="gain")
            fnames = booster.feature_name()
            pairs = sorted(zip(fnames, imps.tolist()), key=lambda x: -x[1])
            return {k: round(float(v), 2) for k, v in pairs[:top_n]}
        if hasattr(booster, "models"):
            accum: dict[str, float] = {}
            n_sub = 0
            for m in booster.models:
                if hasattr(m, "feature_importance"):
                    imps = m.feature_importance(importance_type="gain")
                    fnames = m.feature_name()
                    for f, v in zip(fnames, imps.tolist()):
                        accum[f] = accum.get(f, 0.0) + float(v)
                    n_sub += 1
            if n_sub > 0:
                avg = {f: round(v / n_sub, 2) for f, v in accum.items()}
                pairs = sorted(avg.items(), key=lambda x: -x[1])
                return dict(pairs[:top_n])
    except Exception:
        pass
    return {}


def run_walk_forward(
    ds: LoadedDataset,
    spec: ModelSpec,
    *,
    train_days: int = DEFAULT_TRAIN_DAYS,
    test_days: int = DEFAULT_TEST_DAYS,
    embargo_days: int = DEFAULT_EMBARGO_DAYS,
    n_folds: int = DEFAULT_N_FOLDS,
    last_n_folds: int | None = None,
) -> WalkForwardResult:
    """Generate folds from ``spec.train_start``/``spec.train_end`` and
    run each: train on the fold's train window, evaluate on the test
    window. Aggregate.

    Raises :class:`WalkForwardError` if no fold produced a finite
    primary metric — degenerate state worth surfacing loudly rather
    than papering over with a NaN mean.

    ``last_n_folds`` (dev-velocity knob): when set, only the most
    recent N generated folds are run. Fold ``k`` indices are preserved
    from the full configured sequence so per-fold output stays
    interpretable (e.g. fold k=5 is still fold 5, not fold 0).
    Aggregates describe only the executed subset — operator must read
    ``n_folds_run`` vs ``n_folds_configured`` to know the scope.
    """
    folds = generate_folds(
        spec.train_start, spec.train_end,
        train_days=train_days,
        test_days=test_days,
        embargo_days=embargo_days,
        n_folds=n_folds,
    )
    n_folds_generated = len(folds)
    if last_n_folds is not None:
        if last_n_folds <= 0:
            raise ValueError(
                f"last_n_folds must be a positive int when set; got {last_n_folds}"
            )
        if folds and last_n_folds < len(folds):
            kept = folds[-last_n_folds:]
            logger.info(
                "walk-forward fold subset | last_n_folds=%d kept=%s of %d generated",
                last_n_folds, [f.k for f in kept], len(folds),
            )
            folds = kept
    # Zero-fold case gets its own message so the operator knows the
    # diagnosis is "window too short" rather than "every fold was NaN".
    if not folds:
        total_days = (spec.train_end - spec.train_start).days
        min_days = train_days + embargo_days + test_days
        raise WalkForwardError(
            f"walk-forward generated 0 folds for spec={spec.name}: "
            f"spec window is {total_days}d, minimum needed for one fold is "
            f"{min_days}d (train={train_days} + embargo={embargo_days} + test={test_days})"
        )

    metric_name = primary_metric(spec.objective)
    fold_results: list[FoldMetrics] = []
    n_runs = 0
    last_booster: Any = None
    best_booster_metric: float = float("-inf")
    best_fold_k: int | None = None
    best_primary_value: float | None = None
    # True when the spec uses interval filtering (stacked-interval training
    # but serving-interval eval). Set on first encounter; constant for spec.
    train_serve_filter_active = bool(spec.training_intervals)

    for fold in folds:
        X_tr, y_tr = _slice_by_time(ds, fold.train_start, fold.train_end)
        X_te, y_te = _slice_by_time(ds, fold.test_start, fold.test_end)

        if X_tr.empty or X_te.empty:
            reason = (
                "empty_train_window" if X_tr.empty else "empty_test_window"
            )
            logger.warning(
                "walk-forward fold %d skipped | spec=%s reason=%s n_train=%d n_test=%d",
                fold.k, spec.name, reason, len(X_tr), len(X_te),
            )
            fold_results.append(FoldMetrics(
                fold=fold.k,
                train_start=fold.train_start, train_end=fold.train_end,
                test_start=fold.test_start, test_end=fold.test_end,
                n_train=len(X_tr), n_test=len(X_te),
                skipped_reason=reason,
            ))
            continue

        if spec.objective in ("binary", "multiclass"):
            # Class-balance guard. The train-side threshold differs by
            # objective:
            #
            # * binary: skip if <2 classes — a constant predictor's AUC
            #   is undefined.
            # * multiclass (WF1 fix): skip if <N_MULTICLASS_CLASSES.
            #   LightGBM auto-infers ``num_class`` from y_train. A
            #   2-of-3-class train slice produces a 2-class booster
            #   whose ``predict_proba`` has shape (n, 2); ``_evaluate``
            #   then tries ``log_loss(y, p, labels=[0,1,2])`` against
            #   y_val that may carry encoded class 2 → raises
            #   ``"y_true contains values not in labels"`` and the
            #   whole walk-forward dies instead of skipping one fold.
            #
            # The test-side threshold stays at <2 for both: a single-
            # class test slice can't compute *any* discriminating
            # metric, but a 2-of-3-class test slice still gives finite
            # secondary metrics (accuracy, per-class precision/recall);
            # macro AUC OVR drops to NaN gracefully and the fold gets
            # filtered from primary aggregates by the existing logic.
            if spec.objective == "binary":
                min_train_classes = 2
                train_skip_reason = "single_class_train_set"
            else:
                min_train_classes = N_MULTICLASS_CLASSES
                train_skip_reason = "insufficient_classes_in_train_set"
            if y_tr.astype(int).nunique() < min_train_classes:
                logger.warning(
                    "walk-forward fold %d skipped | spec=%s reason=%s n_train=%d "
                    "(saw %d distinct classes, needed %d)",
                    fold.k, spec.name, train_skip_reason, len(X_tr),
                    int(y_tr.astype(int).nunique()), min_train_classes,
                )
                fold_results.append(FoldMetrics(
                    fold=fold.k,
                    train_start=fold.train_start, train_end=fold.train_end,
                    test_start=fold.test_start, test_end=fold.test_end,
                    n_train=len(X_tr), n_test=len(X_te),
                    skipped_reason=train_skip_reason,
                ))
                continue
            if y_te.astype(int).nunique() < 2:
                reason = "single_class_test_set"
                logger.warning(
                    "walk-forward fold %d skipped | spec=%s reason=%s n_test=%d",
                    fold.k, spec.name, reason, len(X_te),
                )
                fold_results.append(FoldMetrics(
                    fold=fold.k,
                    train_start=fold.train_start, train_end=fold.train_end,
                    test_start=fold.test_start, test_end=fold.test_end,
                    n_train=len(X_tr), n_test=len(X_te),
                    skipped_reason=reason,
                ))
                continue

        # M5g.7: per-fold feature selection. Runs on X_tr / y_tr
        # BEFORE projection-then-fit; the same selection then projects
        # X_te so the model sees the same columns at predict time.
        # Order matters: feature selection must happen before MS4's
        # row-filter so the selection sees the full train slice
        # (stacked-interval distributions across both 1h and 15m).
        features_selected: list[str] = []
        if spec.feature_selection_enabled:
            from .feature_selection import select_features
            features_selected = select_features(
                X_tr, y_tr,
                random_state=int(spec.hyperparams.get("random_state", 0) or 0),
            )
            X_tr = X_tr[features_selected]
            X_te = X_te[features_selected]
            logger.info(
                "walk-forward fold %d feature_selection | spec=%s kept %d/%d: %s",
                fold.k, spec.name, len(features_selected), len(ds.feature_names),
                features_selected,
            )

        # MS4: filter test slice to serving-interval rows for stacked
        # specs. n_test recorded below reflects the post-filter count
        # so the audit trail describes what was actually evaluated.
        X_te, y_te = filter_eval_to_serving_interval(X_te, y_te, spec)
        # MS11: filter can leave the test slice empty if the window
        # somehow contained no serving-interval bars (rare — would
        # need a fold whose test_start..test_end happens to fall
        # entirely in a gap of 1h data while still having 15m data).
        # Skip rather than crash sklearn metrics on an empty array.
        if X_te.empty:
            reason = "empty_test_window_after_interval_filter"
            logger.warning(
                "walk-forward fold %d skipped | spec=%s reason=%s",
                fold.k, spec.name, reason,
            )
            fold_results.append(FoldMetrics(
                fold=fold.k,
                train_start=fold.train_start, train_end=fold.train_end,
                test_start=fold.test_start, test_end=fold.test_end,
                n_train=len(X_tr), n_test=0,
                skipped_reason=reason,
            ))
            continue
        booster, metrics = fit_and_evaluate(X_tr, y_tr, X_te, y_te, spec)
        last_booster = booster   # track the most recent valid-fold model
        fold_metric_val = metrics.get(metric_name, float("nan"))
        if not math.isnan(fold_metric_val) and fold_metric_val > best_booster_metric:
            best_booster_metric = fold_metric_val
            best_fold_k = fold.k
            best_primary_value = fold_metric_val
        feature_importances = _capture_feature_importances(booster)
        # V79 / Phase 4 D-execution: capture out-of-fold predictions for
        # the backfill path. Dispatch on estimator type:
        #   * lgb.Booster (single-model) -> .predict(X) -> 1D for binary,
        #     2D for multiclass.
        #   * Ensemble (multi-model)     -> predict_proba_ensemble(...)
        #     -> always 2D (n_classes columns).
        # Predictions are point-in-time-correct: each fold's booster was
        # trained only on data up to ``fold.train_end``. The backfill
        # script handles shape per spec.objective.
        oof_predictions, oof_timestamps = _capture_oof(booster, X_te)
        # M5g.6: adversarial validation per fold. Originally Gate 4 of
        # M5h's 13-gate gauntlet, but ``project_v2_adversarial_auc.md``
        # (2026-05-16) closed the methodology question: adversarial AUC
        # measures ``P(features)`` shift, which is structurally ~1.0
        # for hourly bars with macro features on rolling WF. The right
        # transferability gate is ``conditional_invariance`` below
        # (``P(y | features)`` shift). Adversarial AUC stays in the
        # metrics dict as an info-only diagnostic — the gauntlet's
        # gate 4 now reads ``ci_max_abs_diff``, not this.
        from .adversarial import adversarial_auc
        adv_auc = adversarial_auc(
            X_tr, X_te,
            random_state=int(spec.hyperparams.get("random_state", 0) or 0),
        )
        metrics["adversarial_auc"] = adv_auc
        # Conditional invariance (replaces adversarial AUC as the
        # transferability signal). Extended to multiclass 2026-05-27 —
        # per-class P(y=k|bin) divergence, same 0.15 threshold as binary.
        from .conditional_invariance import conditional_invariance
        ci_result = conditional_invariance(
            X_tr, y_tr, X_te, y_te,
            objective=spec.objective,
        )
        metrics["ci_max_abs_diff"] = ci_result.max_abs_diff
        metrics["ci_mean_abs_diff"] = ci_result.mean_abs_diff
        metrics["ci_n_pairs_evaluated"] = float(ci_result.n_pairs_evaluated)
        if ci_result.skipped_reason is not None:
            metrics["ci_skipped_reason_flag"] = 1.0
        fold_results.append(FoldMetrics(
            fold=fold.k,
            train_start=fold.train_start, train_end=fold.train_end,
            test_start=fold.test_start, test_end=fold.test_end,
            n_train=len(X_tr), n_test=len(X_te),
            metrics=metrics,
            features_selected=features_selected,
            oof_predictions=oof_predictions,
            oof_timestamps=oof_timestamps,
            feature_importances=feature_importances,
        ))
        n_runs += 1
        logger.info(
            "walk-forward fold %d done | spec=%s n_train=%d n_test=%d %s=%.4f",
            fold.k, spec.name, len(X_tr), len(X_te),
            metric_name, metrics.get(metric_name, float("nan")),
        )

    valid = [
        fm for fm in fold_results
        if fm.metrics and not math.isnan(fm.metrics.get(metric_name, float("nan")))
    ]
    if not valid:
        raise WalkForwardError(
            f"walk-forward produced no valid fold for spec={spec.name}: "
            f"{n_runs}/{len(folds)} folds ran; "
            f"none had finite primary metric '{metric_name}'"
        )

    primary_values = [fm.metrics[metric_name] for fm in valid]
    primary_mean = float(np.mean(primary_values))
    primary_median = float(np.median(primary_values))
    primary_std = float(np.std(primary_values, ddof=0))

    # OLS trend slope: positive = improving, negative = decaying across folds.
    # NaN when < 2 valid folds (can't fit a line).
    if len(primary_values) >= 2:
        fold_indices = np.arange(len(primary_values), dtype=float)
        primary_fold_trend_slope = float(np.polyfit(fold_indices, primary_values, 1)[0])
    else:
        primary_fold_trend_slope = float("nan")

    # Mean of EACH reported metric across valid folds. Useful for the
    # secondary metrics (log_loss, accuracy for binary; rmse, mae for
    # regression) without making every consumer re-derive them.
    all_metric_names: set[str] = set()
    for fm in valid:
        all_metric_names.update(fm.metrics.keys())
    metric_means: dict[str, float] = {}
    for name in sorted(all_metric_names):
        finite = [
            fm.metrics[name] for fm in valid
            if name in fm.metrics and not math.isnan(fm.metrics[name])
        ]
        metric_means[name] = float(np.mean(finite)) if finite else float("nan")

    return WalkForwardResult(
        spec_name=spec.name,
        primary_metric=metric_name,
        n_folds_configured=n_folds,
        n_folds_generated=n_folds_generated,
        n_folds_run=n_runs,
        n_folds_valid_metric=len(valid),
        primary_mean=primary_mean,
        primary_median=primary_median,
        primary_std=primary_std,
        primary_fold_trend_slope=primary_fold_trend_slope,
        best_fold_k=best_fold_k,
        best_primary_value=best_primary_value,
        train_serve_filter_active=train_serve_filter_active,
        metric_means=metric_means,
        folds=fold_results,
        _last_booster=last_booster,
    )


# ── Serialisation ─────────────────────────────────────────────────────────


def walk_forward_to_dict(result: WalkForwardResult) -> dict[str, Any]:
    """JSON-friendly view — used by the CLI summary and the artifact's
    ``walk_forward`` block. Excludes ``_last_booster`` (not JSON-safe).
    """
    return {
        "spec_name": result.spec_name,
        "primary_metric": result.primary_metric,
        "n_folds_configured": result.n_folds_configured,
        "n_folds_generated": result.n_folds_generated,
        "n_folds_run": result.n_folds_run,
        "n_folds_valid_metric": result.n_folds_valid_metric,
        "primary_mean": result.primary_mean,
        "primary_median": result.primary_median,
        "primary_std": result.primary_std,
        "primary_fold_trend_slope": result.primary_fold_trend_slope,
        "best_fold_k": result.best_fold_k,
        "best_primary_value": result.best_primary_value,
        "train_serve_filter_active": result.train_serve_filter_active,
        "metric_means": result.metric_means,
        "folds": [
            {
                "fold": fm.fold,
                "train_start": fm.train_start.isoformat(),
                "train_end": fm.train_end.isoformat(),
                "test_start": fm.test_start.isoformat(),
                "test_end": fm.test_end.isoformat(),
                "n_train": fm.n_train,
                "n_test": fm.n_test,
                "metrics": fm.metrics,
                "skipped_reason": fm.skipped_reason,
                # M5g.7: per-fold selected feature list (empty when
                # spec.feature_selection_enabled is False).
                "features_selected": list(fm.features_selected),
                # V79 / Phase 4 D-execution: point-in-time-correct OOF
                # predictions for this fold's test window. The backfill
                # script reads these to populate signal_history.
                "oof_predictions": list(fm.oof_predictions),
                "oof_timestamps": [ts.isoformat() for ts in fm.oof_timestamps],
                # Top-10 gain importances for this fold's booster.
                "feature_importances": dict(fm.feature_importances),
            }
            for fm in result.folds
        ],
    }


# ── Walk-forward training path (WF1) ──────────────────────────────────────


def train_via_walk_forward(
    ds: LoadedDataset,
    spec: ModelSpec,
    *,
    train_days: int = DEFAULT_TRAIN_DAYS,
    test_days: int = DEFAULT_TEST_DAYS,
    embargo_days: int = DEFAULT_EMBARGO_DAYS,
    n_folds: int = DEFAULT_N_FOLDS,
    last_n_folds: int | None = None,
    allow_leakage: bool = False,
) -> dict[str, Any]:
    """Train via walk-forward and save the LAST valid fold's estimator.

    "Estimator" is a ``lgb.Booster`` for single-model specs and an
    ``ensemble.Ensemble`` for multi-model specs (M5g.3 phase 2).

    Why this rather than the 80/20 fit: the artifact's estimator is the
    one M5d's gauntlet validates and M5e promotes. If walk-forward
    metrics describe estimators that get thrown away, the gauntlet is
    auditing a different model than the one being shipped. Saving the
    last fold's estimator means the metrics in the payload describe the
    model in the payload.

    The "last fold" is the most recent train window — the closest
    approximation to "what we'd retrain on tomorrow" — which matches
    the deployment intent. Earlier folds remain as out-of-sample
    validation evidence in the ``walk_forward`` block.

    Same payload shape as :func:`train.train_with_dataset`, plus a
    populated ``walk_forward`` block. ``eval_kind`` is
    ``"walk_forward_last_fold"`` to make the metric semantics explicit
    for downstream consumers.
    """
    logger.info(
        "loaded | spec=%s bar_slots=%d dropped_nan=%d features=%d label=%s",
        spec.name, ds.n_bar_slots_total, ds.n_bar_slots_dropped_nan,
        len(ds.feature_names), spec.label_feature,
    )
    # Embargo validation: ensure embargo_days covers the label's forward
    # lookahead window. Unknown labels default to a conservative 1-day
    # minimum; callers adding new labels should extend _LABEL_MIN_EMBARGO_DAYS.
    min_embargo = _LABEL_MIN_EMBARGO_DAYS.get(spec.label_feature, 1)
    if embargo_days < min_embargo:
        raise ValueError(
            f"embargo_days={embargo_days} is too small for label "
            f"'{spec.label_feature}': minimum is {min_embargo} days. "
            f"A shorter embargo risks train/test label leakage at fold "
            f"boundaries — the label reads future bars that cross the "
            f"embargo gap. Increase embargo_days or shorten the label horizon."
        )
    integrity = run_integrity_or_raise(ds, spec, allow_leakage=allow_leakage)

    result = run_walk_forward(
        ds, spec,
        train_days=train_days,
        test_days=test_days,
        embargo_days=embargo_days,
        n_folds=n_folds,
        last_n_folds=last_n_folds,
    )

    valid_folds = [fm for fm in result.folds if fm.skipped_reason is None]
    # ``run_walk_forward`` already raises if no fold produced a finite
    # primary metric. The fold-list-non-empty contract therefore holds
    # whenever we reach this line, but assert it so a refactor can't
    # silently mis-use this entry point.
    assert valid_folds, "run_walk_forward must raise if no fold ran"
    assert result._last_booster is not None, "last booster must be set when a fold ran"

    last_fold = valid_folds[-1]
    logger.info(
        "trained via walk-forward | spec=%s eval_kind=walk_forward_last_fold "
        "last_fold_metrics=%s",
        spec.name, last_fold.metrics,
    )

    payload = build_payload(
        ds, spec,
        booster=result._last_booster,
        metrics=last_fold.metrics,
        integrity=integrity,
        n_train_rows=last_fold.n_train,
        n_val_rows=last_fold.n_test,
        eval_kind="walk_forward_last_fold",
    )
    payload["walk_forward"] = walk_forward_to_dict(result)
    return payload
