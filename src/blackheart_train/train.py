"""Train one ModelSpec end-to-end.

M5a scope: chronological 80/20 train/val split, fixed hyperparams from
the spec, single LightGBM fit, basic metrics. Walk-forward with rolling
refit is M5c.

Conventions:

* ``random_state`` flows from ``spec.hyperparams['random_state']`` so a
  re-run is bit-reproducible given the same data. ``subsample_freq=1``
  + a seeded ``random_state`` make the booster bytes deterministic.
* Validation set is the *most recent* slice of the chronologically-sorted
  matrix, never the earliest — standard time-series eval. We assert the
  index is monotonically increasing before slicing so a future caller
  with shuffled data fails loudly instead of leaking quietly.
* For binary objectives we predict ``proba`` and report AUC + log-loss +
  accuracy. For regression we report RMSE + MAE + Pearson r. M5d's
  reviewer gauntlet adds bootstrap CIs, regime sub-cuts, cost stress.
* The returned payload carries a content_sha256 computed over only the
  model-identity fields (spec + features + booster text). Run metadata
  (trained_at, metrics, row counts) is in the payload but does NOT
  affect the sha — re-training identical data lands at the same path.
"""
from __future__ import annotations

import logging
import math
from dataclasses import asdict
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import lightgbm as lgb

if TYPE_CHECKING:
    # Type-only import: ``Ensemble`` is referenced in annotations on
    # :func:`fit_and_evaluate` and :func:`build_payload`. The runtime
    # imports stay inside those functions so single-model paths don't
    # pay the cost of loading the ensemble module (and its transitive
    # xgboost / sklearn-CV dependencies) at module-import time.
    from .ensemble import Ensemble
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .artifacts import compute_content_sha
from .integrity import IntegrityError, check_dataset, recompute_verdict
from .loader import LoadedDataset, load_dataset, load_stacked_dataset
from .specs import ModelSpec

logger = logging.getLogger(__name__)


def filter_eval_to_serving_interval(
    X: pd.DataFrame, y: pd.Series, spec: ModelSpec,
) -> tuple[pd.DataFrame, pd.Series]:
    """If the dataset is stacked-interval, filter ``(X, y)`` to rows
    whose ``interval_indicator`` matches ``spec.interval`` — the
    serving cadence.

    Identity transform when the dataset isn't stacked (no
    ``interval_indicator`` column OR spec wasn't built for stacked
    training). Used at the eval boundary (train_with_dataset,
    run_walk_forward) so the metrics describe the cadence the model
    will actually serve at, not the mixed-interval training set.
    """
    if "interval_indicator" not in X.columns or not spec.training_intervals:
        return X, y
    from .loader import _INTERVAL_INDICATOR_ENCODING
    serving_code = _INTERVAL_INDICATOR_ENCODING[spec.interval]
    mask = X["interval_indicator"] == serving_code
    return X.loc[mask], y.loc[mask]


def split_chronological(
    X: pd.DataFrame, y: pd.Series, val_fraction: float
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    if not 0.0 < val_fraction < 0.5:
        raise ValueError(f"val_fraction must be in (0, 0.5); got {val_fraction}")
    # Defensive: position-slicing only gives a chronological split when
    # the index is already monotonically increasing. Fail loudly otherwise
    # so we don't leak future into past silently.
    if not X.index.is_monotonic_increasing:
        raise ValueError(
            "X.index must be monotonically increasing for chronological split "
            "(loader returns sorted; a caller shuffled the rows)."
        )
    n = len(X)
    n_val = max(1, int(round(n * val_fraction)))
    n_train = n - n_val
    return X.iloc[:n_train], y.iloc[:n_train], X.iloc[n_train:], y.iloc[n_train:]


def _evaluate(
    objective: str, y_true: pd.Series, y_pred: np.ndarray
) -> dict[str, float]:
    if objective == "binary":
        y_true_arr = y_true.astype(int).to_numpy()
        auc = float(roc_auc_score(y_true_arr, y_pred)) if len(set(y_true_arr)) > 1 else float("nan")
        ll = float(log_loss(y_true_arr, np.clip(y_pred, 1e-7, 1 - 1e-7)))
        acc = float(accuracy_score(y_true_arr, (y_pred >= 0.5).astype(int)))
        return {"auc": auc, "log_loss": ll, "accuracy": acc}
    if objective == "multiclass":
        # y_true already encoded to 0..C-1 indices by ``fit_and_evaluate``.
        # y_pred is the (n, C) proba matrix.
        #
        # WF3 docs: ``n_classes`` is read off y_pred.shape[1] — this is
        # the number of classes the booster *saw at fit time*. The
        # walk_forward path's train-slice guard enforces that this
        # equals ``N_MULTICLASS_CLASSES`` so y_true_arr never contains
        # an index outside ``all_labels``. If a future caller bypasses
        # that guard (fits on fewer classes than the encoding declares),
        # ``log_loss(..., labels=all_labels)`` will raise
        # ``"y_true contains values not in labels"`` — loud failure,
        # not silent metric corruption.
        y_true_arr = y_true.astype(int).to_numpy()
        y_pred = np.asarray(y_pred)
        y_hat = y_pred.argmax(axis=1)
        n_classes = y_pred.shape[1]
        all_labels = list(range(n_classes))
        # MG4 fix: clipping breaks the row-sums-to-1 invariant sklearn
        # expects (its log_loss would otherwise emit a UserWarning).
        # Re-normalise per row after the clip so each row sums to 1
        # again — the relative ranking of classes is preserved.
        clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        clipped = clipped / clipped.sum(axis=1, keepdims=True)
        ll = float(log_loss(y_true_arr, clipped, labels=all_labels))
        acc = float(accuracy_score(y_true_arr, y_hat))
        # Macro-averaged: each class contributes equally. With the
        # 59/3/38 triple-barrier skew the 3% class would vanish in a
        # micro average, defeating the purpose of looking at it.
        #
        # MG1 fix: pass labels=all_labels to every macro call. Without
        # it, when y_true is missing a class (regime-locked val slice
        # in walk-forward), sklearn computes macro over only the
        # classes present in y_true — different denominators across
        # runs, metric numbers stop being comparable.
        prec = float(precision_score(
            y_true_arr, y_hat, average="macro", zero_division=0, labels=all_labels,
        ))
        rec = float(recall_score(
            y_true_arr, y_hat, average="macro", zero_division=0, labels=all_labels,
        ))
        f1 = float(f1_score(
            y_true_arr, y_hat, average="macro", zero_division=0, labels=all_labels,
        ))
        # Multiclass AUC: one-vs-rest macro. Requires every class to
        # appear in y_true; if the val slice happens to be regime-locked
        # and missing a class, fall back to NaN rather than crash.
        if len(set(y_true_arr)) == n_classes:
            try:
                auc_macro = float(roc_auc_score(
                    y_true_arr, y_pred, multi_class="ovr", average="macro",
                    labels=all_labels,
                ))
            except ValueError:
                auc_macro = float("nan")
        else:
            auc_macro = float("nan")
        metrics: dict[str, float] = {
            "log_loss": ll,
            "accuracy": acc,
            "macro_precision": prec,
            "macro_recall": rec,
            "macro_f1": f1,
            "macro_auc_ovr": auc_macro,
        }
        # Per-class precision/recall — the reviewer wants to see whether
        # the rare class is being predicted at all, not just absorbed
        # into macro averages.
        per_class_prec = precision_score(
            y_true_arr, y_hat, average=None, zero_division=0, labels=all_labels,
        )
        per_class_rec = recall_score(
            y_true_arr, y_hat, average=None, zero_division=0, labels=all_labels,
        )
        for i, (p, r) in enumerate(zip(per_class_prec, per_class_rec)):
            metrics[f"class_{i}_precision"] = float(p)
            metrics[f"class_{i}_recall"] = float(r)
        return metrics
    y_true_arr = y_true.to_numpy()
    rmse = float(math.sqrt(mean_squared_error(y_true_arr, y_pred)))
    mae = float(mean_absolute_error(y_true_arr, y_pred))
    if np.std(y_true_arr) == 0.0 or np.std(y_pred) == 0.0:
        pearson = float("nan")
    else:
        pearson = float(np.corrcoef(y_true_arr, y_pred)[0, 1])
    return {"rmse": rmse, "mae": mae, "pearson_r": pearson}


def _build_estimator(spec: ModelSpec) -> lgb.LGBMModel:
    # dict() copies so accidental in-place mutation by LightGBM internals
    # can't bleed back into the spec's default factory output.
    params = dict(spec.hyperparams)
    if spec.objective == "binary":
        return lgb.LGBMClassifier(objective="binary", **params)
    if spec.objective == "multiclass":
        # num_class is inferred by LightGBM from the label vector at fit
        # time, so we don't hardcode it here. The directional spec
        # currently uses 3 classes (triple-barrier).
        return lgb.LGBMClassifier(objective="multiclass", **params)
    return lgb.LGBMRegressor(objective="regression", **params)


# ES1: Label-embargo lookup table. Mirrors ``walk_forward.DEFAULT_EMBARGO_DAYS``
# at 7 days — same rationale (cover any label whose forward horizon is
# ≤7 days). The inner-val carve happens within a single fold's train
# slice, so this gap stops the inner train's forward labels from
# overlapping the inner ES-val's labels.
_ES_EMBARGO_DAYS: int = 7


def _early_stopping_carve(
    X_tr: "pd.DataFrame",
    y_tr: "pd.Series",
    spec: ModelSpec,
) -> "tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series] | None":
    """Chronological tail-carve for LightGBM early stopping.

    Returns ``(X_inner, y_inner, X_es, y_es)`` when ES is enabled and
    every guard passes. Returns ``None`` when ES is disabled OR a guard
    fails — the caller falls through to a normal fit (no ES) and the
    spec still trains, just without the early-stopping safeguard.

    Guards:

    * spec opts in (``early_stopping_rounds > 0``)
    * single-model path only (``len(base_models) == 1``); ensemble specs
      are deferred — each base model would need its own ES wiring.
    * inner-val carve has at least 10 rows (sub-tens give noisy
      stop-point estimates).
    * embargo gap fits inside the train slice (``n - val_size -
      embargo_bars > 0``).

    Embargo is fixed at 7 days at the spec's serving interval. Same
    rationale as ``walk_forward.DEFAULT_EMBARGO_DAYS``: stops forward
    labels at the inner-train tail from overlapping the inner-val head
    for any label whose horizon is ≤7 days.

    Pure: no DB, no disk. y is sliced alongside X by integer position so
    objective-specific encoding (``encode_multiclass``, ``.astype(int)``)
    can happen on each half independently downstream.
    """
    if spec.early_stopping_rounds <= 0:
        return None
    if len(spec.base_models) > 1:
        return None
    n = len(X_tr)
    val_size = int(round(n * spec.early_stopping_val_fraction))
    if val_size < 10:
        logger.warning(
            "early stopping disabled for spec=%s: inner-val too small "
            "(%d rows, need >=10)",
            spec.name, val_size,
        )
        return None
    from .loader import _INTERVAL_HOURS
    hours_per_bar = _INTERVAL_HOURS.get(spec.interval, 1.0)
    embargo_bars = max(1, int(round(_ES_EMBARGO_DAYS * 24 / hours_per_bar)))
    inner_train_end = n - val_size - embargo_bars
    if inner_train_end <= 0:
        logger.warning(
            "early stopping disabled for spec=%s: not enough rows for "
            "embargo (n=%d val=%d embargo_bars=%d)",
            spec.name, n, val_size, embargo_bars,
        )
        return None
    X_inner = X_tr.iloc[:inner_train_end]
    y_inner = y_tr.iloc[:inner_train_end]
    X_es = X_tr.iloc[n - val_size:]
    y_es = y_tr.iloc[n - val_size:]
    return X_inner, y_inner, X_es, y_es


# Canonical encoding for the triple-barrier label values into LightGBM's
# 0..C-1 class index convention. The order is fixed (-1 → 0, 0 → 1,
# +1 → 2) so the per-class metrics in the artifact always mean the same
# thing across runs. The booster doesn't know about the original values;
# downstream consumers (live inference, this evaluator) translate back
# via :func:`decode_multiclass`.
_MULTICLASS_LABEL_ENCODING: dict[int, int] = {-1: 0, 0: 1, 1: 2}
_MULTICLASS_LABEL_DECODING: dict[int, int] = {v: k for k, v in _MULTICLASS_LABEL_ENCODING.items()}
# Public so other modules (walk_forward) can enforce
# "train slice must contain every class" without duplicating the magic
# number. If a future multiclass label adds a class, this constant moves
# in lockstep with the encoding map.
N_MULTICLASS_CLASSES: int = len(_MULTICLASS_LABEL_ENCODING)


def encode_multiclass(y: pd.Series) -> pd.Series:
    """Map triple-barrier label values (-1/0/+1) to LightGBM's 0..C-1
    indices. Any value outside the known set raises so a future label
    definition with a new class is caught at fit-time rather than
    silently mis-encoded.

    MG3 fix: detect both NaN and unknown integers in the float-space
    Series *before* coercing to int. ``.astype(int)`` on NaN raises
    ``IntCastingNaNError`` — a confusing trace for what is, in
    practice, "label column has NaN where the loader was supposed to
    have dropped it". This branch surfaces the same problem as a
    clean ValueError naming the offending values.
    """
    if y.isna().any():
        raise ValueError(
            f"multiclass label contains NaN ({int(y.isna().sum())} rows); "
            f"loader should have dropped these"
        )
    unique_raw = set(y.unique())
    # The encoding map is keyed by Python int; allow float values that
    # round-trip cleanly (e.g. -1.0 from feature_values' float64 column)
    # but reject true non-integers like 0.5.
    unknown: set[float] = set()
    for v in unique_raw:
        if int(v) != v or int(v) not in _MULTICLASS_LABEL_ENCODING:
            unknown.add(v)
    if unknown:
        raise ValueError(
            f"multiclass label contains unknown values {sorted(unknown)}; "
            f"expected subset of {sorted(_MULTICLASS_LABEL_ENCODING)}"
        )
    return y.astype(int).map(_MULTICLASS_LABEL_ENCODING).astype(int)


def decode_multiclass(y_idx: np.ndarray) -> np.ndarray:
    """Inverse of :func:`encode_multiclass` — predicted class index back
    to the original triple-barrier value."""
    return np.array([_MULTICLASS_LABEL_DECODING[int(i)] for i in y_idx], dtype=int)


def fit_and_evaluate(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    spec: ModelSpec,
) -> tuple["lgb.Booster | Ensemble", dict[str, float]]:
    """Fit one LightGBM (or a 3-model ensemble) under ``spec`` and
    compute its val metrics.

    Pure: no DB, no disk, no logging. Used by both :func:`train_one`
    (single fit) and ``search.grid_search_one`` (many fits over a
    hyperparam grid against a shared dataset).

    Return-type contract:

    * Single-model specs (``len(spec.base_models) == 1``) — return
      ``(lgb.Booster, metrics)``. Same shape as M5a.
    * Multi-model specs (``len(spec.base_models) > 1``) — return
      ``(Ensemble, metrics)`` where ``Ensemble`` carries all fitted
      base models (M5g.3 phase 2). The metrics dict carries per-base
      (``lgb_*``, ``xgb_*``, ``lr_*``) + ensemble (unprefixed multiclass
      keys) + ``mean_disagreement[_class_i]``.

    Callers (``train_with_dataset``, ``walk_forward.run_walk_forward``)
    pass the returned object through to :func:`build_payload`, which
    routes it into ``payload["booster"]`` or ``payload["ensemble"]``
    based on type.
    """
    model = _build_estimator(spec)
    if spec.objective == "binary":
        carve = _early_stopping_carve(X_tr, y_tr, spec)
        if carve is not None:
            X_in, y_in, X_es, y_es = carve
            y_in_int = y_in.astype(int)
            y_es_int = y_es.astype(int)
            # ES1 guard: LightGBM early stopping needs >= 2 classes in
            # the eval set to compute its binary log-loss callback. A
            # single-class inner slice would crash the fit, so fall back
            # to no-ES on the full train slice rather than skip the fold.
            if y_in_int.nunique() < 2 or y_es_int.nunique() < 2:
                logger.warning(
                    "early stopping fallback for spec=%s: single-class slice "
                    "(inner=%d, es_val=%d)",
                    spec.name, y_in_int.nunique(), y_es_int.nunique(),
                )
                model.fit(X_tr, y_tr.astype(int))
            else:
                model.fit(
                    X_in, y_in_int,
                    eval_set=[(X_es, y_es_int)],
                    callbacks=[
                        lgb.early_stopping(spec.early_stopping_rounds, verbose=False)
                    ],
                )
        else:
            model.fit(X_tr, y_tr.astype(int))
        y_val_pred = model.predict_proba(X_val)[:, 1]
    elif spec.objective == "multiclass":
        # Encode labels to 0..C-1 *inside* this function so the loader
        # stays storage-format-agnostic and other code paths still see
        # the original ±1/0 values. MS4 contract: callers
        # (train_with_dataset, walk_forward.run_walk_forward) filter
        # X_val / y_val to spec.interval rows BEFORE calling
        # fit_and_evaluate — see ``filter_eval_to_serving_interval``.
        # The train slice stays stacked; only eval is filtered.
        y_tr_enc = encode_multiclass(y_tr)
        y_val_enc = encode_multiclass(y_val)
        if len(spec.base_models) > 1:
            # Lazy import keeps ``train`` importable when xgboost isn't
            # installed (single-model path doesn't need it).
            from .ensemble import evaluate_ensemble, fit_ensemble
            meta_label = None
            meta_disabled_reason: str | None = None
            if spec.meta_label_enabled:
                from .meta_label import (
                    META_LABEL_TRAIN_FRACTION,
                    build_meta_features,
                    build_meta_target,
                    fit_meta_label,
                    gating_metrics,
                    per_base_model_probas_stack,
                    predict_meta_pwin,
                )
                # Chronological split: primary trains on the first 80%
                # of the fold's train slice, meta-label trains on the
                # ensemble's OUT-OF-SAMPLE predictions over the last
                # 20%. The 20% chunk doesn't leak into primary fitting
                # (chronological + non-overlapping) so the meta-label
                # sees primary's true generalisation behaviour, not its
                # in-sample memorisation.
                n_tr = len(X_tr)
                n_meta = max(1, int(round(n_tr * META_LABEL_TRAIN_FRACTION)))
                n_primary = n_tr - n_meta
                X_primary, X_meta_oos = X_tr.iloc[:n_primary], X_tr.iloc[n_primary:]
                y_primary, y_meta_oos = (
                    y_tr_enc.iloc[:n_primary], y_tr_enc.iloc[n_primary:],
                )
                # Guard: the inner 80/20 split can leave the primary
                # slice missing a class even when the outer fold-train
                # passes the WF1 guard (the manipulated rows from the
                # rolling-overlap pattern can cluster in the early part
                # of the train window). LightGBM auto-infers
                # ``num_class`` from y_train, so a 2-of-3-class primary
                # slice would crash multiclass fit. Fall back to fitting
                # the ensemble on the FULL fold-train slice and skip
                # meta-label for this fold — the ungated metrics still
                # describe a valid ensemble.
                primary_class_count = int(y_primary.astype(int).nunique())
                if primary_class_count < N_MULTICLASS_CLASSES:
                    meta_disabled_reason = (
                        f"primary slice missing class: {primary_class_count} of "
                        f"{N_MULTICLASS_CLASSES} classes present"
                    )
                    logger.warning(
                        "meta-label disabled for spec=%s: %s; "
                        "fitting ensemble on full fold-train slice",
                        spec.name, meta_disabled_reason,
                    )
                    ensemble = fit_ensemble(X_tr, y_tr_enc, spec)
                else:
                    ensemble = fit_ensemble(X_primary, y_primary, spec)
                    meta_train_stack = per_base_model_probas_stack(
                        ensemble, X_meta_oos, expected_n_classes=N_MULTICLASS_CLASSES,
                    )
                    X_meta_tr = build_meta_features(meta_train_stack)
                    y_meta_tr = build_meta_target(
                        meta_train_stack, y_meta_oos.to_numpy()
                    )
                    try:
                        meta_label = fit_meta_label(
                            X_meta_tr, y_meta_tr,
                            n_classes=N_MULTICLASS_CLASSES,
                            random_state=int(spec.hyperparams.get("random_state", 0) or 0),
                        )
                    except ValueError as exc:
                        # Meta-label couldn't fit (typically: every
                        # meta-train row was a win OR every row was a
                        # loss — primary collapsed on the slice). The
                        # ungated ensemble metrics still describe what
                        # we have; surface a numeric flag below.
                        meta_disabled_reason = str(exc)
                        logger.warning(
                            "meta-label disabled for spec=%s: %s", spec.name, exc,
                        )
            else:
                # No meta-label: ensemble fits on the full train slice.
                ensemble = fit_ensemble(X_tr, y_tr_enc, spec)
            ensemble_proba, ensemble_metrics = evaluate_ensemble(
                ensemble, X_val, y_val_enc, n_classes=N_MULTICLASS_CLASSES,
            )
            # Per-class precision/recall on the *ensemble's* averaged
            # proba — adds the gauntlet-relevant per-class detail that
            # ensemble._one_model_metrics deliberately skipped to keep
            # per-base entries terse. The keys log_loss/accuracy/
            # macro_auc_ovr are already in ensemble_metrics under the
            # ensemble (unprefixed) namespace; the merge is idempotent
            # for those keys, and the per-class keys are new.
            ensemble_metrics.update(_evaluate("multiclass", y_val_enc, ensemble_proba))
            # M5g.6: bootstrap CIs for the gauntlet's gate-3 primary
            # metric. Gate 3 reads ``macro_auc_ovr_ci_lower_5`` rather
            # than the point estimate so the threshold demands honest
            # confidence, not just a lucky mean.
            from .bootstrap import bootstrap_macro_auc_ovr
            ensemble_metrics.update(bootstrap_macro_auc_ovr(
                y_val_enc.to_numpy(), ensemble_proba,
                n_classes=N_MULTICLASS_CLASSES,
                random_state=int(spec.hyperparams.get("random_state", 0) or 0),
            ))
            # M5g.4: gated metrics. Only meaningful when meta_label was
            # fit successfully. When disabled (single-class meta-train
            # or spec opted out), gated_* keys are absent — downstream
            # consumers should treat missing as "gating not measured".
            if meta_label is not None:
                from .meta_label import per_base_model_probas_stack as _stack_val
                val_stack = _stack_val(
                    ensemble, X_val, expected_n_classes=N_MULTICLASS_CLASSES,
                )
                X_meta_val = build_meta_features(val_stack)
                val_pwin = predict_meta_pwin(meta_label, X_meta_val)
                primary_pred = ensemble_proba.argmax(axis=1)
                from .meta_label import META_LABEL_CONFIDENCE_THRESHOLD
                take_trade = val_pwin > META_LABEL_CONFIDENCE_THRESHOLD
                ensemble_metrics.update(gating_metrics(
                    primary_pred=primary_pred,
                    meta_pwin=val_pwin,
                    y_true=y_val_enc.to_numpy(),
                ))
                # M5g.8: cost-regime PnL simulation. Uses the
                # meta-label gate to decide which bars trade,
                # ensemble's argmax for direction, and the actual
                # triple-barrier outcome to score. Gauntlet gate 5
                # reads ``cost_realistic_profitable`` and
                # ``cost_conservative_profitable``.
                from .cost_model import simulate_cost_regime_metrics
                ensemble_metrics.update(simulate_cost_regime_metrics(
                    y_true_enc=y_val_enc.to_numpy(),
                    predicted_classes=primary_pred,
                    take_trade=take_trade,
                    interval=spec.interval,
                ))
                # M5g.9 + M5g.10: shared per-trade PnL computation, then
                # regime sub-cuts (gate 6) and DSR (gate 7). One
                # compute_per_trade_pnl_bps call feeds both — the cost
                # simulation above already does its own internal compute;
                # the duplicate is intentional and cheap (vectorized).
                from .cost_model import (
                    FEE_BPS_ROUND_TRIP,
                    FUNDING_BPS_ROUND_TRIP,
                    SLIPPAGE_BPS_BY_REGIME,
                    compute_per_trade_pnl_bps,
                )
                gross_bps_full, traded_full, _ = compute_per_trade_pnl_bps(
                    y_val_enc.to_numpy(), primary_pred, take_trade,
                    interval=spec.interval,
                )
                n_traded = int(traded_full.sum())
                if n_traded > 0:
                    gross_traded = gross_bps_full[traded_full]
                    realistic_cost = (
                        FEE_BPS_ROUND_TRIP
                        + SLIPPAGE_BPS_BY_REGIME["realistic"]
                        + FUNDING_BPS_ROUND_TRIP
                    )
                    net_per_trade = gross_traded - realistic_cost

                    # M5g.10: Deflated Sharpe Ratio (gauntlet gate 7).
                    # Trial-discount layer. ``n_trials=1`` and
                    # ``sr_variance=1.0`` are placeholders — the
                    # gauntlet aggregator (M5h) will override with
                    # values from a real trial registry. Until then,
                    # gate 7 with N=1 collapses to PSR(SR*=0) =
                    # "probability the strategy's true Sharpe > 0",
                    # which is still useful as a directional signal.
                    from .dsr import compute_dsr
                    ensemble_metrics.update(compute_dsr(
                        returns_per_trade=net_per_trade,
                        n_trials=1,
                        sr_variance_across_trials=1.0,
                    ))

                    # M5g.9: regime sub-cuts (gauntlet gate 6). One-sided
                    # t-test per regime on net PnL; gate fails if ≥2
                    # regimes are significantly negative.
                    #
                    # Phase 1 (vol regimes): high_vol / low_vol derived
                    # from train-set quantiles of ``btc_realized_vol_30d``.
                    # Phase 2 (2026-05-16): adds bull / bear / chop trend
                    # regimes from a momentum proxy. Trend column is
                    # picked by fallback chain so any spec with one of
                    # the momentum features gets the analysis for free.
                    # The combined 5-regime call uses the same "max 1
                    # failing" threshold — the blueprint's gate-6 false-
                    # positive analysis was sized for 5 regimes already
                    # (1 − 0.95⁵ ≈ 22% under H0).
                    from .regime_subcuts import (
                        classify_trend_regimes_from_train_quantiles,
                        classify_vol_regimes_from_train_quantiles,
                        regime_subcut_metrics,
                    )
                    regime_flags_full: dict[str, np.ndarray] = {}
                    if (
                        "btc_realized_vol_30d" in X_tr.columns
                        and "btc_realized_vol_30d" in X_val.columns
                    ):
                        vol_train = X_tr["btc_realized_vol_30d"].to_numpy()
                        vol_val = X_val["btc_realized_vol_30d"].to_numpy()
                        regime_flags_full.update(
                            classify_vol_regimes_from_train_quantiles(
                                vol_train, vol_val,
                            )
                        )
                    # Trend column fallback chain: prefer the dedicated
                    # 30d momentum if registered; fall back to shorter
                    # windows. The classifier doesn't care which —
                    # quantile-based tertiles work on any momentum-like
                    # series.
                    trend_col: str | None = None
                    for candidate in (
                        "btc_momentum_30d",
                        "btc_log_return_7d",
                        "btc_log_return_24h",
                    ):
                        if (
                            candidate in X_tr.columns
                            and candidate in X_val.columns
                        ):
                            trend_col = candidate
                            break
                    if trend_col is not None:
                        trend_train = X_tr[trend_col].to_numpy()
                        trend_val = X_val[trend_col].to_numpy()
                        regime_flags_full.update(
                            classify_trend_regimes_from_train_quantiles(
                                trend_train, trend_val,
                            )
                        )
                    if regime_flags_full:
                        # Sub-select to traded rows only — t-test is on
                        # trade returns, not on every val row.
                        regime_flags_traded = {
                            name: flag[traded_full]
                            for name, flag in regime_flags_full.items()
                        }
                        ensemble_metrics.update(regime_subcut_metrics(
                            gross_bps_per_trade=gross_traded,
                            cost_per_trade_bps=realistic_cost,
                            regime_flags=regime_flags_traded,
                        ))
            # MR2 note: an earlier version of this branch wrote
            # ``meta_label_disabled = 1.0`` into the metrics dict to
            # surface "this fold ran without gating". That produced a
            # misleading walk_forward aggregate: ``metric_means`` only
            # averages over folds where the key is present, so 3/6
            # folds disabled would report mean=1.0 (not 0.5). The log
            # warning above is the audit record; downstream consumers
            # can detect "no gating this fold" by the absence of
            # ``gated_*`` keys in the fold's metrics dict.
            # M5g.3 phase 2: the entire Ensemble is returned (was: only
            # the LightGBM booster). ``build_payload`` dispatches on
            # ``isinstance(.., Ensemble)`` to route this into
            # ``payload["ensemble"]`` and to hash all three base models
            # into ``content_sha256``. The fitted meta-label, when
            # present, still lives in this scope only — its persistence
            # is M5g.4 phase 2 and not part of this milestone.
            return ensemble, ensemble_metrics
        carve = _early_stopping_carve(X_tr, y_tr, spec)
        if carve is not None:
            X_in, y_in_raw, X_es, y_es_raw = carve
            y_in_enc = encode_multiclass(y_in_raw)
            y_es_enc = encode_multiclass(y_es_raw)
            # ES1 guard: LightGBM multiclass auto-infers num_class from
            # y_train. A 2-of-3-class inner slice produces a 2-class
            # booster whose .predict_proba won't align with the gauntlet
            # downstream — same WF1 failure mode. Fall back to no-ES on
            # the full train slice so all-classes are guaranteed (outer
            # walk-forward guard already enforced this for X_tr).
            if (
                y_in_enc.nunique() < N_MULTICLASS_CLASSES
                or y_es_enc.nunique() < 2
            ):
                logger.warning(
                    "early stopping fallback for spec=%s: multiclass slice "
                    "missing classes (inner=%d/%d, es_val=%d)",
                    spec.name, y_in_enc.nunique(), N_MULTICLASS_CLASSES,
                    y_es_enc.nunique(),
                )
                model.fit(X_tr, y_tr_enc)
            else:
                model.fit(
                    X_in, y_in_enc,
                    eval_set=[(X_es, y_es_enc)],
                    callbacks=[
                        lgb.early_stopping(spec.early_stopping_rounds, verbose=False)
                    ],
                )
        else:
            model.fit(X_tr, y_tr_enc)
        y_val_pred = model.predict_proba(X_val)
        metrics = _evaluate(spec.objective, y_val_enc, y_val_pred)
        return model.booster_, metrics
    else:
        carve = _early_stopping_carve(X_tr, y_tr, spec)
        if carve is not None:
            X_in, y_in, X_es, y_es = carve
            # ES1 guard: a constant inner-train label would yield a
            # zero-variance fit (LightGBM regressor returns the mean).
            # Skip ES if either slice is degenerate.
            if y_in.nunique() < 2 or y_es.nunique() < 2:
                logger.warning(
                    "early stopping fallback for spec=%s: constant regression "
                    "label in carved slice",
                    spec.name,
                )
                model.fit(X_tr, y_tr)
            else:
                model.fit(
                    X_in, y_in,
                    eval_set=[(X_es, y_es)],
                    callbacks=[
                        lgb.early_stopping(spec.early_stopping_rounds, verbose=False)
                    ],
                )
        else:
            model.fit(X_tr, y_tr)
        y_val_pred = model.predict(X_val)
    metrics = _evaluate(spec.objective, y_val, y_val_pred)
    return model.booster_, metrics


def run_integrity_or_raise(
    ds: LoadedDataset, spec: ModelSpec, *, allow_leakage: bool = False,
):
    """Run the integrity report, log it, and raise :class:`IntegrityError`
    on FAIL. Shared by every code path that wants to honour the gate
    before doing work.

    ``allow_leakage=True`` (CLI --allow-leakage) demotes a label-leakage
    FAIL to WARN so the run can continue — the check still records its
    findings in ``integrity.leakage_report``. Other FAIL checks are
    NOT affected; they still raise.
    """
    integrity = check_dataset(ds, spec)

    # R1.B: --allow-leakage demotes ONLY the label_leakage check. The
    # demotion mutates the CheckResult and recomputes the overall verdict
    # so downstream logging + the artifact payload reflect the operator's
    # explicit decision. The leakage_report dict is left intact for the
    # audit trail.
    if allow_leakage:
        for c in integrity.checks:
            if c.name == "label_leakage" and c.severity == "FAIL":
                c.severity = "WARN"
                c.message = "[--allow-leakage] " + c.message
        integrity.verdict = recompute_verdict(integrity.checks)

    fail_names = [c.name for c in integrity.checks if c.severity == "FAIL"]
    warn_names = [c.name for c in integrity.checks if c.severity == "WARN"]
    logger.info(
        "integrity | spec=%s verdict=%s warn=%s fail=%s fingerprint=%s",
        spec.name, integrity.verdict, warn_names, fail_names,
        integrity.data_fingerprint[:12],
    )
    for c in integrity.checks:
        if c.severity != "PASS":
            logger.warning("integrity %s | %s: %s", c.severity, c.name, c.message)
    if integrity.verdict == "FAIL":
        raise IntegrityError(
            f"data integrity FAIL for spec={spec.name} "
            f"(data_fingerprint={integrity.data_fingerprint}): "
            + "; ".join(f"{c.name}={c.message}" for c in integrity.checks if c.severity == "FAIL")
        )
    return integrity


PAYLOAD_VERSION = 2
"""Artifact payload schema version.

* v1 (M5a → M5g.3 phase 1): ``payload["booster"]`` holds a single
  ``lgb.Booster``. Ensemble specs persisted only LightGBM and landed
  with ``deployment_ready=False`` (EB1 blocker).
* v2 (M5g.3 phase 2, current): ``payload["booster"]`` is a single
  ``lgb.Booster`` for single-model specs OR ``None``;
  ``payload["ensemble"]`` is an :class:`ensemble.Ensemble` for
  multi-model specs OR ``None``. Exactly one of the two fields is
  populated. Content-sha hashes a deterministic signature of every
  persisted estimator.

Bump on any schema break that would make a v2 reader misinterpret a
future payload. v1 payloads are not produced by this codebase anymore;
``read_artifact`` accepts both but downstream consumers should branch
on this field if they care about the booster-vs-ensemble shape.
"""


def build_payload(
    ds: LoadedDataset,
    spec: ModelSpec,
    booster: "lgb.Booster | Ensemble",
    metrics: dict[str, float],
    integrity,
    *,
    n_train_rows: int,
    n_val_rows: int,
    eval_kind: str,
) -> dict[str, Any]:
    """Compose the artifact payload. Shared by the 80/20 path
    (:func:`train_with_dataset`) and the walk-forward path
    (``walk_forward.train_via_walk_forward``).

    The ``booster`` argument is a tagged union: a ``lgb.Booster`` for
    single-model specs (M5a path) or an
    :class:`ensemble.Ensemble` for multi-model specs (M5g.3 phase 2).
    The parameter name is kept as ``booster`` only for caller-kwarg
    back-compat; the dispatch is on ``isinstance(.., Ensemble)``.

    ``eval_kind`` documents what ``metrics`` describes:
    ``"holdout_80_20"`` for the chronological 80/20 split, or
    ``"walk_forward_last_fold"`` for the most recent walk-forward fold.
    Downstream consumers (M5d gauntlet, M5e registry) read this to know
    which metric semantics they're looking at. ``metrics`` always
    describes the estimator being saved (single booster or full
    ensemble).

    ``content_sha256`` is derived purely from model-identity fields. Two
    different estimators (e.g. 80/20 fit vs walk-forward last-fold fit on
    different windows) get different shas — addressing follows
    estimator identity, not just spec identity. For ensemble specs the
    signature hashes all three base models, so a partial-ensemble
    artifact cannot collide with a full-ensemble one.
    """
    # Local imports to avoid an import cycle at module load (ensemble.py
    # depends on specs.py only; train.py is loaded before ensemble.py).
    from .ensemble import Ensemble, ensemble_content_signature
    from .derived_features import DERIVED_LABELS

    spec_dict = asdict(spec)
    feature_names_list = list(ds.feature_names)

    is_ensemble = isinstance(booster, Ensemble)
    # Contract: the spec's base_models count must match the input shape.
    # An ensemble spec with a bare Booster would silently produce a
    # single-LightGBM artifact whose payload claims a 3-model ensemble
    # in its spec block — same metrics-vs-deployment divergence the EB1
    # audit closed. Fail fast.
    spec_is_ensemble = len(spec.base_models) > 1
    if spec_is_ensemble and not is_ensemble:
        raise ValueError(
            f"spec={spec.name} declares {len(spec.base_models)} base models "
            f"but build_payload received a single estimator (type "
            f"{type(booster).__name__}); pass an Ensemble"
        )
    if not spec_is_ensemble and is_ensemble:
        raise ValueError(
            f"spec={spec.name} is single-model but build_payload received "
            f"an Ensemble; pass the fitted Booster"
        )
    # Defense-in-depth: the type check above only verifies the union
    # tag (single vs ensemble). Also verify the ensemble's *contents*
    # match the spec — a partially-fit Ensemble (e.g. only LightGBM
    # populated for a 3-model spec) would otherwise produce an artifact
    # whose spec block lies about which models are persisted. Reproduces
    # the EB1 divergence one level deeper. ``fit_ensemble`` always
    # populates to match the spec, so this check is dormant in normal
    # operation; it guards against test code or future refactors that
    # construct an Ensemble by hand.
    if is_ensemble:
        ensemble_kinds = {fbm.kind for fbm in booster.models}
        spec_kinds = set(spec.base_models)
        if ensemble_kinds != spec_kinds:
            raise ValueError(
                f"spec={spec.name} declares base_models={spec.base_models!r} "
                f"but Ensemble carries kinds {sorted(ensemble_kinds)} — "
                f"refusing to write an artifact whose spec block claims "
                f"models the ensemble doesn't contain"
            )

    content_dict: dict[str, Any] = {
        "spec": spec_dict,
        "feature_names": feature_names_list,
        "objective": spec.objective,
        "label_feature": ds.label_feature,
        "label_version": ds.label_version,
    }
    if is_ensemble:
        # Ensemble signature spans all three base models. Distinct key
        # name from the single-model branch so the content_dict's shape
        # carries "this is a v2 ensemble artifact" intrinsically — a
        # consumer recomputing the sha can tell which branch produced
        # it from the input alone.
        content_dict["ensemble_signature"] = ensemble_content_signature(booster)
    else:
        # Single-model: same key the M5a path used, so single-model
        # content shas are unchanged by Phase 2 — existing artifacts on
        # disk keep their paths.
        content_dict["booster_model_str"] = booster.model_to_string()
    content_sha = compute_content_sha(content_dict)

    # Deployment readiness — three blockers, two surviving post-Phase-2:
    #
    # 1. Derived features without registry entries — the live inference
    #    worker can't compute them.
    # 2. Derived labels — same; inference can't know about a label that
    #    only lives in the train-time transformer registry.
    # 3. (Was EB1: ensemble persisted partially.) Lifted by M5g.3 phase 2
    #    — the artifact now carries the full Ensemble. SUPERSEDED BY:
    #    meta-label gating is fit at train time and adds ``gated_*``
    #    metrics to the payload, but the meta-label model itself is NOT
    #    yet persisted (M5g.4 phase 2). Promoting such an artifact would
    #    mean live inference runs the ensemble without the gate — every
    #    bar trades, instead of the (much rarer) gated-trade rate the
    #    metrics describe. Same EB1 hazard, one level up the stack.
    has_derived_inputs = bool(spec.derived_features)
    has_derived_label = ds.label_feature in DERIVED_LABELS
    meta_label_partially_persisted = bool(
        len(spec.base_models) > 1 and spec.meta_label_enabled
    )
    deployment_ready = not (
        has_derived_inputs or has_derived_label or meta_label_partially_persisted
    )
    unregistered_features = list(spec.derived_features)
    unregistered_label = ds.label_feature if has_derived_label else None

    return {
        "payload_version": PAYLOAD_VERSION,
        "content_sha256": content_sha,
        "data_fingerprint": integrity.data_fingerprint,
        "integrity": {
            "verdict": integrity.verdict,
            "checks": [
                {
                    "name": c.name,
                    "severity": c.severity,
                    "message": c.message,
                    "details": c.details,
                }
                for c in integrity.checks
            ],
        },
        # R1.B: detached snapshot of the label-leakage check's details so
        # the CLI can stamp it into experiment_run.leakage_report (V92)
        # without walking integrity.checks. None when the check was
        # skipped via check_dataset(skip_leakage=True).
        "leakage_report": integrity.leakage_report,
        "spec": spec_dict,
        "feature_names": ds.feature_names,
        "booster": None if is_ensemble else booster,
        "ensemble": booster if is_ensemble else None,
        "objective": spec.objective,
        "metrics": metrics,
        "eval_kind": eval_kind,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_train_rows": int(n_train_rows),
        "n_val_rows": int(n_val_rows),
        "n_features": int(len(feature_names_list)),
        "n_bar_slots_total": int(ds.n_bar_slots_total),
        "n_bar_slots_dropped_nan": int(ds.n_bar_slots_dropped_nan),
        "per_feature_non_null": dict(ds.per_feature_non_null),
        "per_feature_pct_non_null": dict(ds.per_feature_pct_non_null),
        "label_feature": ds.label_feature,
        "label_version": ds.label_version,
        "deployment_readiness": {
            "deployment_ready": deployment_ready,
            "unregistered_input_features": unregistered_features,
            "unregistered_label": unregistered_label,
        },
    }


def train_with_dataset(
    ds: LoadedDataset, spec: ModelSpec, *, allow_leakage: bool = False,
) -> dict[str, Any]:
    """Train using a chronological 80/20 split on an already-loaded dataset.

    The saved booster is the 80/20 fit; ``metrics`` is the holdout 20%.
    Used by the M5a/M5b path. The walk-forward path lives in
    ``walk_forward.train_via_walk_forward`` and saves the last fold's
    booster instead.

    ``allow_leakage`` (R1.B): when True, demotes the label-leakage FAIL
    to WARN. The check still runs + populates ``payload['leakage_report']``.
    """
    logger.info(
        "loaded | spec=%s bar_slots=%d dropped_nan=%d features=%d label=%s",
        spec.name, ds.n_bar_slots_total, ds.n_bar_slots_dropped_nan,
        len(ds.feature_names), spec.label_feature,
    )
    integrity = run_integrity_or_raise(ds, spec, allow_leakage=allow_leakage)

    X_tr, y_tr, X_val, y_val = split_chronological(ds.X, ds.y, spec.val_fraction)
    if len(X_tr) == 0 or len(X_val) == 0:
        raise ValueError(
            f"degenerate split: n_train={len(X_tr)}, n_val={len(X_val)} "
            f"(total rows={len(ds.X)}, val_fraction={spec.val_fraction})"
        )
    # MS4: when training is stacked, eval at serving interval only. Done
    # at the caller (not inside fit_and_evaluate) so the n_val_rows
    # recorded in the payload reflects the actually-evaluated slice.
    X_val, y_val = filter_eval_to_serving_interval(X_val, y_val, spec)
    # MS11: filter could leave the val slice empty for a degenerate
    # stacked spec (no serving-interval bars in the chronological 20%
    # tail). Surface this as a clear ValueError up here rather than
    # crashing sklearn deep in fit_and_evaluate.
    if len(X_val) == 0:
        raise ValueError(
            f"val slice empty after filtering to serving interval "
            f"{spec.interval}: stacked dataset has no {spec.interval} rows "
            f"in the last {spec.val_fraction:.0%} of the chronological window."
        )

    booster, metrics = fit_and_evaluate(X_tr, y_tr, X_val, y_val, spec)
    logger.info("trained | spec=%s metrics=%s", spec.name, metrics)

    payload = build_payload(
        ds, spec, booster, metrics, integrity,
        n_train_rows=len(X_tr),
        n_val_rows=len(X_val),
        eval_kind="holdout_80_20",
    )
    # Both training paths guarantee this key is present so downstream
    # consumers don't have to handle KeyError vs dict-with-content.
    payload["walk_forward"] = None
    return payload


def train_one(spec: ModelSpec) -> dict[str, Any]:
    """Load data, run integrity checks, fit LightGBM 80/20, return a
    payload ready for :func:`artifacts.write_artifact`.

    The returned dict carries ``content_sha256`` — the addressing key that
    :func:`artifacts.write_artifact` writes to disk under. The sha is
    derived purely from model-identity fields; metadata fields
    (``trained_at``, ``metrics``, ``n_train_rows``…) are present in the
    payload but do not affect addressing.
    """
    # M5g.5: ``load_stacked_dataset`` is the single-entry-point loader.
    # When ``spec.training_intervals`` is empty / single-element it
    # delegates to :func:`load_dataset` (Phase-2-identical behaviour);
    # when multi-element it stacks per-interval datasets with
    # ``interval_indicator``.
    ds: LoadedDataset = load_stacked_dataset(spec)
    return train_with_dataset(ds, spec)
