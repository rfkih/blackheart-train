"""Stacking meta-learner (R2 Session 3).

Take the top-k base models from a Bayesian sweep, generate their OOF
predictions via walk-forward, and train a meta-learner to blend them.

Why stacking on top of Bayesian search:

* The sweep finds the single best HP combo. Stacking exploits the fact
  that the 2nd-best combo often disagrees with the 1st on different
  market regimes — the meta-learner learns when each base voice is
  reliable. Classical Sharpe lift on the same alpha component.
* OOF predictions from ``run_walk_forward`` are out-of-sample by
  construction (the fold's test window was never seen by its model).
  Training the meta on OOF predictions avoids the leakage trap that
  retraining on the full set + blending would create.

Pipeline:

1. ``select_top_k(sweep, k)`` — pick the k best COMPLETE trials.
2. ``assemble_oof_matrix(ds, top_k_specs, n_folds)`` — refit each spec
   via walk_forward, gather OOF predictions across all folds, align by
   timestamp.
3. ``train_stacker_from_oof_matrix(matrix, y, objective)`` — fit a
   LogisticRegressionCV (binary) or RidgeCV (regression) on the OOF
   predictions as features.
4. ``predict_with_stacker(stacker, base_preds)`` — apply the meta to
   new base predictions.

Session 3 scope:

* Binary + regression meta-learners only.
* Single held-out alignment: intersect timestamps across all base
  models' OOF sets. A base spec that produced fewer OOF rows is
  ignored for rows where it's missing.
* No CLI wiring yet — that's session 4.

What's NOT in S3:

* Multiclass meta (codebase already uses ``ensemble.py`` for the
  multiclass path; integrating stacking with that is its own decision).
* OOF-from-test-fold caching. Every ``train_stacker`` call refits each
  base via walk_forward — k * n_folds * fit_time. Acceptable while we
  validate the methodology; cache when we move to production sweeps.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import numpy as np
import pandas as pd

from .bayesian_search import BayesianSearchResult, BayesianTrialResult
from .loader import LoadedDataset
from .specs import ModelSpec

logger = logging.getLogger(__name__)


# ── Result types ─────────────────────────────────────────────────────────


class StackingError(RuntimeError):
    """Raised when stacking can't produce a usable meta-learner.

    Causes:

    * No top-k specs survived (sweep had < k COMPLETE trials).
    * Base specs produced OOF predictions over disjoint windows — the
      timestamp intersection is empty.
    * Meta-learner couldn't fit (all-NaN target, single-class label).
    """


class _SupportsPredict(Protocol):
    """Duck-type for the meta-learner. Both sklearn's
    LogisticRegressionCV and RidgeCV expose ``predict`` and one of
    ``predict_proba`` / direct output."""

    def predict(self, X: np.ndarray) -> np.ndarray: ...


@dataclass
class Stacker:
    """Trained blender. ``meta_model`` is a sklearn estimator (Logistic
    or Ridge depending on the objective). ``base_spec_names`` is the
    canonical order — the meta's coefficients are in this order, so
    callers MUST pass base predictions in the same column order at
    prediction time.
    """

    objective: Literal["binary", "regression"]
    base_spec_names: tuple[str, ...]
    meta_model: _SupportsPredict
    # Training metrics — what the meta-learner achieved on the
    # OOF-prediction matrix it was fit on. Useful for "did stacking
    # improve over the best base?" comparisons in the CLI summary.
    train_metrics: dict[str, float] = field(default_factory=dict)
    # Sample count the meta was fit on (after timestamp intersection).
    n_meta_train_samples: int = 0


# ── Top-k selection ──────────────────────────────────────────────────────


def select_top_k(
    sweep: BayesianSearchResult, k: int,
) -> list[BayesianTrialResult]:
    """Pick the k best COMPLETE trials by score, descending. PRUNED and
    FAIL trials are excluded — they don't have usable HPs.

    Raises ``StackingError`` if fewer than k completed trials exist —
    the caller should reduce k or run a longer sweep before stacking.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}")
    completed = [r for r in sweep.runs if r.state == "COMPLETE"]
    if len(completed) < k:
        raise StackingError(
            f"only {len(completed)} COMPLETE trial(s) in sweep; "
            f"need at least {k} for top-{k} stacking. "
            f"Increase n_trials or lower k."
        )
    # R2 Bug-#8 fix: secondary sort by trial_number (ascending) so ties
    # on score are broken deterministically by the trial that ran first.
    # Two identical sweeps (same seed) thus produce identical top-k
    # picks even when scores collide.
    completed.sort(key=lambda r: (r.score, -r.trial_number), reverse=True)
    return completed[:k]


def top_k_to_specs(
    base_spec: ModelSpec, top_k: list[BayesianTrialResult],
) -> list[ModelSpec]:
    """Materialise the top-k overrides as tuned ModelSpecs ready to be
    handed to ``run_walk_forward``.
    """
    from dataclasses import replace
    out: list[ModelSpec] = []
    for trial in top_k:
        merged = {**base_spec.hyperparams, **trial.overrides}
        tuned = replace(base_spec, hyperparams=merged)
        # Tag the name so log lines distinguish multiple specs from the
        # same base in the same sweep.
        tuned = replace(tuned, name=f"{base_spec.name}#trial{trial.trial_number}")
        out.append(tuned)
    return out


# ── OOF-matrix assembly ──────────────────────────────────────────────────


# Type alias for the OOF-collection function injected by tests. Real
# code uses :func:`_collect_oof_default` which delegates to
# ``run_walk_forward``.
OOFCollectorFn = Any   # Callable[[LoadedDataset, ModelSpec, int], tuple[np.ndarray, np.ndarray, dict]]


def _collect_oof_default(
    ds: LoadedDataset, spec: ModelSpec, n_folds: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Default OOF collector — runs the full walk-forward and stitches
    the per-fold OOF predictions into a single (preds, timestamps)
    sequence. Late import of walk_forward keeps the module-import
    cycle shallow.

    Returns ``(oof_preds, oof_timestamps, info)`` where ``info`` carries
    metadata for the audit trail (n_valid_folds, primary_mean, etc).
    """
    from .walk_forward import run_walk_forward
    result = run_walk_forward(ds, spec, n_folds=n_folds)
    preds: list[float] = []
    timestamps: list[Any] = []
    for fold in result.folds:
        if fold.skipped_reason is not None:
            continue
        # Binary path emits a 1D probability per row; multiclass would
        # emit 2D (kept out of S3 scope). The np.asarray(...).reshape(-1)
        # flattens consistently.
        fold_preds = np.asarray(fold.oof_predictions).reshape(-1)
        fold_ts = list(fold.oof_timestamps)
        if len(fold_preds) != len(fold_ts):
            logger.warning(
                "stacker: fold %d has mismatched preds/ts lengths "
                "(%d/%d) for spec=%s — skipping fold",
                fold.fold, len(fold_preds), len(fold_ts), spec.name,
            )
            continue
        preds.extend(fold_preds.tolist())
        timestamps.extend(fold_ts)
    info = {
        "primary_mean": result.primary_mean,
        "n_folds_valid": result.n_folds_valid_metric,
    }
    return np.asarray(preds, dtype="float64"), np.asarray(timestamps), info


def assemble_oof_matrix(
    ds: LoadedDataset,
    specs: list[ModelSpec],
    *,
    n_folds: int = 6,
    collector: OOFCollectorFn | None = None,
) -> tuple[np.ndarray, np.ndarray, list[pd.Timestamp]]:
    """Refit each spec via walk_forward and align their OOF predictions
    into a single ``(n_intersect_samples, len(specs))`` matrix.

    Alignment uses the **intersection** of timestamps across all specs
    — a row is included only when every base spec emitted an OOF
    prediction for that timestamp. This keeps the meta-learner's
    feature matrix dense.

    Returns:
      * ``X_meta`` — float64 array shape ``(n_samples, len(specs))``.
        Column ``i`` is spec ``i``'s OOF prediction.
      * ``y_meta`` — float64 array shape ``(n_samples,)`` of the matched
        labels (from ``ds.y``).
      * ``ts`` — pd.Timestamp list of the intersection in order.

    Raises ``StackingError`` if the intersection is empty.
    """
    if not specs:
        raise StackingError("assemble_oof_matrix: no specs provided")
    # R2 Bug-#6 fix: explicit duplicate-name guard. Callers via
    # ``top_k_to_specs`` add a ``#trial{n}`` suffix so this can't trip,
    # but a future caller passing raw specs would otherwise produce a
    # silently-merged or cryptic pandas error from the concat below.
    spec_names = [s.name for s in specs]
    if len(set(spec_names)) != len(spec_names):
        from collections import Counter
        dupes = [n for n, c in Counter(spec_names).items() if c > 1]
        raise StackingError(
            f"assemble_oof_matrix: duplicate spec names {dupes!r}. "
            "Each base must have a unique name so OOF predictions can be "
            "aligned + addressed by column. Use top_k_to_specs() to add "
            "deterministic suffixes."
        )
    collector = collector or _collect_oof_default

    # Step 1: collect (preds, ts) per spec into a Series keyed by ts.
    series_list: list[pd.Series] = []
    for spec in specs:
        preds, timestamps, info = collector(ds, spec, n_folds)
        if len(preds) == 0:
            raise StackingError(
                f"assemble_oof_matrix: spec={spec.name} produced no OOF "
                f"predictions (n_folds_valid={info.get('n_folds_valid')}). "
                f"Cannot stack — refit failed or every fold was skipped."
            )
        idx = pd.DatetimeIndex(timestamps)
        # A spec's walk_forward may emit duplicates if folds overlap (it
        # shouldn't with proper embargo, but defend against it). Keep
        # the LAST value per timestamp — matches the "newest fold wins"
        # semantic of overlap during research iteration.
        s = pd.Series(preds, index=idx, name=spec.name)
        s = s[~s.index.duplicated(keep="last")]
        series_list.append(s)
        logger.info(
            "stacker: collected %d OOF predictions for spec=%s "
            "(primary_mean=%.4f)",
            len(s), spec.name, info.get("primary_mean", float("nan")),
        )

    # Step 2: align via DataFrame join (intersection = inner). The
    # concat path keeps memory tight on large indices.
    df = pd.concat(series_list, axis=1, join="inner")
    if df.empty:
        raise StackingError(
            "assemble_oof_matrix: timestamp intersection across "
            f"{len(specs)} spec(s) is empty. Likely cause: specs ran "
            "walk-forward over different windows / fold layouts."
        )

    # Step 3: match labels by joining against ds.y. Drop rows where the
    # label was NaN at the intersected timestamps (rare but possible if
    # a label was forward-filled at OOF capture time but later cleaned).
    label_idx = df.index.intersection(ds.y.index)
    df = df.loc[label_idx]
    y_meta = ds.y.loc[label_idx]
    mask = y_meta.notna()
    df = df.loc[mask]
    y_meta = y_meta.loc[mask]

    if df.empty:
        raise StackingError(
            "assemble_oof_matrix: no usable samples after label alignment. "
            "Check that ds.y covers the OOF timestamp range."
        )

    X_meta = df.to_numpy(dtype="float64", copy=False)
    y_arr = y_meta.to_numpy(dtype="float64", copy=False)
    ts = list(df.index)
    return X_meta, y_arr, ts


# ── Meta-learner fit ─────────────────────────────────────────────────────


def train_stacker_from_oof_matrix(
    X_meta: np.ndarray, y_meta: np.ndarray,
    *,
    objective: Literal["binary", "regression"],
    base_spec_names: tuple[str, ...],
    cv: int = 5,
    random_state: int = 0,
) -> Stacker:
    """Pure meta-fit on a pre-assembled OOF matrix.

    Binary → ``LogisticRegressionCV(penalty='l2', cv=cv)``. The CV
    picks the L2 strength; no external HP grid needed. Output:
    P(class=1).
    Regression → ``RidgeCV(alphas=...)`` with cross-validated alpha.

    Why CV-tuned: the meta-learner is fit on OOF predictions, which
    are noisier than raw features. A fixed regularisation strength
    risks under- or over-shrinking. CV picks the data-driven sweet
    spot.

    Args:
      X_meta: shape ``(n_samples, k)``. Column ``i`` is base spec
        ``i``'s OOF prediction.
      y_meta: shape ``(n_samples,)``. The true labels at the matched
        timestamps.
      objective: ``"binary"`` or ``"regression"``. Multiclass is out
        of S3 scope.
      base_spec_names: stored on the result so callers can verify
        the column order at predict time.
      cv: folds for the meta's CV.

    Raises ``StackingError`` on degenerate cases (single class, NaN
    target, etc).
    """
    n_samples, n_features = X_meta.shape
    if n_samples < 2 * cv:
        raise StackingError(
            f"train_stacker: only {n_samples} samples for cv={cv}; "
            f"need at least {2 * cv}. Increase walk-forward fold count or "
            "reduce cv."
        )
    if len(base_spec_names) != n_features:
        raise StackingError(
            f"train_stacker: base_spec_names has {len(base_spec_names)} "
            f"entries but X_meta has {n_features} columns. Mismatch."
        )
    if np.any(np.isnan(X_meta)) or np.any(np.isnan(y_meta)):
        raise StackingError(
            "train_stacker: X_meta / y_meta contain NaN. Run alignment "
            "again — assemble_oof_matrix should have stripped them."
        )

    if objective == "binary":
        # y must be binary (0/1). Reject floats not in {0, 1}.
        y_int = y_meta.astype("int64")
        unique = set(np.unique(y_int).tolist())
        if unique == {0} or unique == {1}:
            raise StackingError(
                f"train_stacker: binary objective but y has single class {unique}; "
                "meta-learner cannot fit."
            )
        if not unique.issubset({0, 1}):
            raise StackingError(
                f"train_stacker: binary objective requires labels in {{0, 1}}; "
                f"got {sorted(unique)}."
            )
        from sklearn.linear_model import LogisticRegressionCV
        from sklearn.metrics import log_loss, roc_auc_score

        # max_iter bumped — LogisticRegressionCV on small samples can
        # bounce against the default 100. Cs=10 (sklearn default) gives
        # a reasonable strength grid. ``l1_ratios=(0,)`` is the sklearn
        # 1.8+ replacement for the deprecated ``penalty='l2'`` — same
        # mathematical effect (pure L2), forward-compatible to 1.10+
        # where ``penalty`` is removed.
        meta = LogisticRegressionCV(
            cv=cv, l1_ratios=(0.0,), scoring="neg_log_loss",
            solver="saga", max_iter=2000, random_state=random_state,
            # sklearn 1.10 will simplify the fitted-attributes layout;
            # we explicitly opt into the legacy layout for now to keep
            # any downstream coef/lambda-inspection code stable. Switch
            # to False when the rest of the codebase audits sklearn 1.10
            # readiness.
            use_legacy_attributes=True,
        )
        meta.fit(X_meta, y_int)
        preds_proba = meta.predict_proba(X_meta)[:, 1]
        train_metrics = {
            "log_loss": float(log_loss(y_int, preds_proba)),
            "auc": float(roc_auc_score(y_int, preds_proba)) if len(unique) == 2 else float("nan"),
        }
    elif objective == "regression":
        from sklearn.linear_model import RidgeCV
        from sklearn.metrics import mean_absolute_error, mean_squared_error

        # R2 Bug-#11 fix: widened the alpha grid from (1e-3, 1e2) to
        # (1e-6, 1e6) — 25 log-spaced values. Real-world ridge alphas
        # commonly land outside the narrower range on highly correlated
        # base predictions (when sample size is small relative to
        # collinearity), and CV-picking the boundary value is a sign
        # the grid was wrong. The wider grid keeps the optimum interior
        # for typical stacking matrices.
        meta = RidgeCV(alphas=np.geomspace(1e-6, 1e6, num=25), cv=cv)
        meta.fit(X_meta, y_meta)
        preds = meta.predict(X_meta)
        # In-sample fit metrics — same caveat as the binary path: this
        # is training-set fit, NOT OOS performance. The actual OOS
        # generalisation lands when the artifact's gauntlet evaluates
        # the stacked model on a held-out window.
        mse = float(mean_squared_error(y_meta, preds))
        train_metrics = {
            "mse": mse,
            "rmse": float(math.sqrt(mse)),
            "mae": float(mean_absolute_error(y_meta, preds)),
            "pearson_r": float(np.corrcoef(y_meta, preds)[0, 1])
                if y_meta.std() > 0 else float("nan"),
        }
    else:
        raise StackingError(
            f"train_stacker: unsupported objective {objective!r}; "
            "S3 supports binary + regression only. Multiclass uses "
            "ensemble.py and is out of scope."
        )

    logger.info(
        "stacker fit | objective=%s n_samples=%d n_bases=%d metrics=%s",
        objective, n_samples, n_features, train_metrics,
    )
    return Stacker(
        objective=objective,
        base_spec_names=tuple(base_spec_names),
        meta_model=meta,
        train_metrics=train_metrics,
        n_meta_train_samples=int(n_samples),
    )


# ── Main entry point ─────────────────────────────────────────────────────


def train_stacker(
    ds: LoadedDataset,
    base_spec: ModelSpec,
    sweep: BayesianSearchResult,
    *,
    k: int = 5,
    n_folds: int = 6,
    cv: int = 5,
    random_state: int = 0,
    collector: OOFCollectorFn | None = None,
) -> Stacker:
    """End-to-end: top-k from sweep → refit walk-forward → assemble OOF
    matrix → fit meta-learner.

    Args:
      ds: the same LoadedDataset the sweep was run on (label alignment
        depends on ``ds.y``).
      base_spec: the spec that produced ``sweep``. Used to materialise
        top-k overrides into runnable specs.
      sweep: the search result. ``select_top_k`` picks from
        ``sweep.runs`` filtered on state=COMPLETE.
      k: how many base models to stack. 5 is a reasonable default —
        marginal gain falls off fast above 7.
      n_folds: walk-forward folds for each base spec.
      cv: meta-learner internal CV folds.
      collector: test seam — inject a stub OOF collector to avoid
        the cost of real walk_forward fits.
    """
    objective = base_spec.objective
    if objective not in ("binary", "regression"):
        raise StackingError(
            f"train_stacker: unsupported objective {objective!r}; "
            "S3 supports binary + regression."
        )
    top_k = select_top_k(sweep, k)
    specs = top_k_to_specs(base_spec, top_k)
    X_meta, y_meta, _ts = assemble_oof_matrix(
        ds, specs, n_folds=n_folds, collector=collector,
    )
    return train_stacker_from_oof_matrix(
        X_meta, y_meta,
        objective=objective,   # type: ignore[arg-type]
        base_spec_names=tuple(s.name for s in specs),
        cv=cv,
        random_state=random_state,
    )


# ── Prediction ───────────────────────────────────────────────────────────


def predict_with_stacker(
    stacker: Stacker,
    base_predictions: np.ndarray,
    *,
    column_names: tuple[str, ...] | list[str] | None = None,
) -> np.ndarray:
    """Apply the meta to a matrix of base predictions.

    ``base_predictions`` must have shape ``(n, len(stacker.base_spec_names))``
    with columns in the SAME ORDER as ``stacker.base_spec_names``.
    Returns shape ``(n,)``: blended probabilities (binary) or regression
    values.

    R2 Bug-#4 fix: ``column_names`` is the recommended way to call this
    function — when supplied, it MUST equal ``stacker.base_spec_names``
    exactly (same names, same order). A mismatch raises ValueError
    before the meta sees the wrong-order data. Without column_names the
    function still works (back-compat) but the caller is on their own
    to honour the documented order.
    """
    X = np.asarray(base_predictions, dtype="float64")
    if X.ndim != 2 or X.shape[1] != len(stacker.base_spec_names):
        raise ValueError(
            f"predict_with_stacker: base_predictions shape {X.shape} "
            f"incompatible with {len(stacker.base_spec_names)} base spec(s). "
            "Column order must match stacker.base_spec_names."
        )
    if column_names is not None:
        got = tuple(column_names)
        if got != stacker.base_spec_names:
            raise ValueError(
                f"predict_with_stacker: column_names mismatch. "
                f"Expected {stacker.base_spec_names}, got {got}. "
                "The meta-learner's coefficients are indexed by the "
                "training-time order; passing columns in a different "
                "order would silently produce wrong predictions."
            )
    if stacker.objective == "binary":
        # predict_proba returns (n, 2); slice the class-1 column.
        return stacker.meta_model.predict_proba(X)[:, 1]   # type: ignore[attr-defined]
    return stacker.meta_model.predict(X)


# ── Serialization ────────────────────────────────────────────────────────


def stacker_to_dict(stacker: Stacker) -> dict[str, Any]:
    """JSON-friendly view. Excludes the meta_model itself — it's a
    sklearn estimator, not JSON-safe. The artifact payload (future
    sessions) will pickle the meta separately and carry the dict here
    as audit metadata.
    """
    return {
        "objective": stacker.objective,
        "base_spec_names": list(stacker.base_spec_names),
        "train_metrics": dict(stacker.train_metrics),
        "n_meta_train_samples": stacker.n_meta_train_samples,
        "meta_model_class": type(stacker.meta_model).__name__,
    }
