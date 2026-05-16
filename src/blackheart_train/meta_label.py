"""Meta-label model for the directional ensemble (blueprint § 6.2).

The 3-model ensemble (M5g.3) emits a probability vector per row but
gives no answer to "should we *act* on this prediction?" — the
directional task's class 1 (horizon-end) collapse means many predictions
are noise even when the argmax looks confident.

The meta-label is a **binary** classifier that learns, retrospectively,
when the primary ensemble was right. At decision time, we condition on
``meta_label_P(win) > confidence_threshold`` (default 0.55): below
threshold = abstain (no trade); above = act on primary's call.

Phase-1 scope
-------------

* **Pure functions only.** Same shape as :mod:`ensemble` — fit, predict,
  evaluate, no I/O. The training-loop integration lives in
  :func:`train.fit_and_evaluate`.
* **Target**: ``primary_direction == y_true``. Simplest binary win
  definition; the trade-side semantics (long vs short on +1 / -1) are
  M7 inference-worker concerns.
* **Features** (6, blueprint § 6.2 minus the not-yet-built bits):

  - ``confidence`` — max(mean_proba) per row
  - ``spread`` — max(mean_proba) − min(mean_proba) per row
  - ``disagreement`` — mean across classes of per-class std across base
    models (per-row scalar)
  - ``direction_0``, ``direction_1``, ``direction_2`` — one-hot of argmax

* **Model**: ``LogisticRegressionCV(l1_ratios=(1.0,))`` — same
  L1-with-CV-tuned-strength interface as the ensemble's LogReg base
  model, on a StandardScaler'd input.
* **Train-side leakage discipline**: caller is responsible for not
  fitting the meta-label on the same rows the primary ensemble saw at
  fit time. :func:`train.fit_and_evaluate` splits the fold-train window
  into primary-train (first 80%) + meta-train (last 20%) chronologically
  so the meta-label fits on out-of-sample primary predictions.

Phase 2 (future) will persist the fitted meta-label in the artifact and
plumb it into the live inference path; for now it lives only in the
training loop and surfaces only via the gated metrics in
``payload["metrics"]``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# Default confidence threshold for acting on the primary's prediction.
# 0.55 per blueprint § 6.2 — well above the 0.5 random floor for a
# binary classifier but loose enough to admit a meaningful fraction of
# the primary's calls. M5g.4 phase 2 may tune this per spec via
# walk-forward on a tertiary metric (gated_accuracy vs selectivity).
META_LABEL_CONFIDENCE_THRESHOLD: float = 0.55

# Fraction of the fold-train window held out as the meta-label's
# training set. The primary ensemble fits on the first ``1 -
# META_LABEL_TRAIN_FRACTION`` of the fold-train rows; the remaining
# tail produces the out-of-sample predictions the meta-label trains on.
# Chronological split so future doesn't leak into past.
META_LABEL_TRAIN_FRACTION: float = 0.20

# Feature column order — pinned so the scaler + model see the same
# columns at predict time even if pandas reorders.
META_LABEL_FEATURE_COLUMNS: tuple[str, ...] = (
    "confidence", "spread", "disagreement",
    "direction_0", "direction_1", "direction_2",
)


# ── Dataclass ────────────────────────────────────────────────────────────


@dataclass
class MetaLabel:
    """A fitted meta-label model. Carries its scaler alongside so
    predict-time symmetry holds, plus the feature-column ordering it
    was fit on so :func:`predict_meta_pwin` can refuse a mis-ordered
    input. Pickle-clean.

    Why ``feature_columns`` (MR1 fix): :class:`StandardScaler` stores
    its means + stds by column INDEX, not by name. Passing a DataFrame
    whose columns are in a different order at predict time would scale
    the wrong columns and silently corrupt P(win) — detectable only by
    noticing degraded gating across folds. We pin the column tuple at
    fit time and check it at predict time so a future caller hitting
    this from outside :func:`build_meta_features` gets a loud error.
    """

    model: LogisticRegressionCV
    scaler: StandardScaler
    n_classes: int
    feature_columns: tuple[str, ...]


# ── Feature / target builders ────────────────────────────────────────────


def build_meta_features(probas_stack: np.ndarray) -> pd.DataFrame:
    """Per-row meta-label features derived from the primary ensemble's
    per-base-model proba stack.

    ``probas_stack`` shape: ``(n_models, n_rows, n_classes)``. Returns a
    DataFrame whose columns are :data:`META_LABEL_FEATURE_COLUMNS` in
    canonical order — important because :class:`MetaLabel` carries a
    scaler whose column ordering at fit time must match predict time.

    Direction is one-hot encoded across the n_classes the ensemble saw.
    Confidence + spread + disagreement are scalar per row.
    """
    if probas_stack.ndim != 3:
        raise ValueError(
            f"probas_stack must be (n_models, n_rows, n_classes); got shape {probas_stack.shape}"
        )
    _, n_rows, n_classes = probas_stack.shape
    mean_proba = probas_stack.mean(axis=0)                    # (n_rows, n_classes)
    direction = mean_proba.argmax(axis=1)                     # (n_rows,)
    confidence = mean_proba.max(axis=1)                       # (n_rows,)
    spread = mean_proba.max(axis=1) - mean_proba.min(axis=1)  # (n_rows,)
    # Per-row mean-across-classes of per-class std-across-models.
    disagreement = probas_stack.std(axis=0, ddof=0).mean(axis=1)
    # One-hot the direction so the linear model treats class membership
    # categorically — a linear feature on direction ∈ {0, 1, 2} would
    # imply an ordering that doesn't exist (SL < horizon < TP is
    # monotone but the linear model would treat the spacing as equal).
    direction_oh = np.zeros((n_rows, n_classes), dtype="float64")
    direction_oh[np.arange(n_rows), direction] = 1.0

    cols: dict[str, np.ndarray] = {
        "confidence": confidence.astype("float64"),
        "spread": spread.astype("float64"),
        "disagreement": disagreement.astype("float64"),
    }
    for i in range(n_classes):
        cols[f"direction_{i}"] = direction_oh[:, i]
    df = pd.DataFrame(cols)
    # Reorder to the canonical column tuple if the n_classes happens to
    # produce extra direction_i keys beyond what META_LABEL_FEATURE_COLUMNS
    # names; conservative for forward-compat with a hypothetical 4-class
    # label.
    expected = list(META_LABEL_FEATURE_COLUMNS) + [
        c for c in df.columns if c not in META_LABEL_FEATURE_COLUMNS
    ]
    return df[[c for c in expected if c in df.columns]]


def build_meta_target(
    probas_stack: np.ndarray, y_true_enc: np.ndarray
) -> np.ndarray:
    """Binary win target: 1 iff ``argmax(mean_proba) == y_true``.

    "Win" is defined as "primary's argmax matched the actual triple-
    barrier outcome class". Trade-side win semantics (i.e. would a long
    on direction=+1 prediction make money?) are M7 concerns; for the
    research-grade gauntlet, argmax-correctness is the cleanest binary
    target and avoids the ambiguity of class-1 (horizon-end) predictions.
    """
    mean_proba = probas_stack.mean(axis=0)
    pred = mean_proba.argmax(axis=1)
    return (pred == np.asarray(y_true_enc).astype(int)).astype(int)


# ── Fit / predict ────────────────────────────────────────────────────────


def fit_meta_label(
    X_meta: pd.DataFrame,
    y_win: np.ndarray,
    *,
    n_classes: int,
    random_state: int = 42,
) -> MetaLabel:
    """Fit Logistic-L1 with CV-tuned regularisation strength on the
    standard-scaled meta features.

    Edge case: when ``y_win`` is single-class (the primary was always
    right OR always wrong on the meta-train slice — typically because
    the slice was tiny or regime-locked), ``LogisticRegressionCV``
    raises. We surface that as a ``ValueError`` rather than letting it
    propagate so the training-loop integration can decide to disable
    gating for that fold gracefully.
    """
    y_win = np.asarray(y_win).astype(int)
    if len(set(y_win)) < 2:
        raise ValueError(
            f"meta-label target has only one class ({sorted(set(y_win))}); "
            "need both win and no-win examples to fit. Primary may be "
            "trivially right (or wrong) on the meta-train slice."
        )
    feature_columns = tuple(X_meta.columns)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_meta.to_numpy(dtype="float64"))
    model = LogisticRegressionCV(
        l1_ratios=(1.0,),
        solver="saga",
        max_iter=2000,
        cv=3,
        random_state=random_state,
        n_jobs=1,
        use_legacy_attributes=False,
    )
    model.fit(Xs, y_win)
    return MetaLabel(
        model=model, scaler=scaler,
        n_classes=n_classes, feature_columns=feature_columns,
    )


def predict_meta_pwin(meta: MetaLabel, X_meta: pd.DataFrame) -> np.ndarray:
    """Return P(win) per row, shape (n_rows,). Column order must match
    what :func:`fit_meta_label` saw — :func:`build_meta_features`
    guarantees the canonical ordering, but MR1 fix surfaces a clear
    error if a caller hands the model a different column tuple (vs
    silently scaling the wrong columns via the scaler's positional
    state)."""
    incoming = tuple(X_meta.columns)
    if incoming != meta.feature_columns:
        raise ValueError(
            f"X_meta columns {incoming} do not match meta-label's fit-time "
            f"columns {meta.feature_columns}. Reorder via "
            f"build_meta_features or rebuild X_meta from the per-base "
            f"probas stack."
        )
    Xs = meta.scaler.transform(X_meta.to_numpy(dtype="float64"))
    return meta.model.predict_proba(Xs)[:, 1]


# ── Gating metrics ───────────────────────────────────────────────────────


def gating_metrics(
    *,
    primary_pred: np.ndarray,
    meta_pwin: np.ndarray,
    y_true: np.ndarray,
    threshold: float = META_LABEL_CONFIDENCE_THRESHOLD,
) -> dict[str, float]:
    """Summarise meta-label-gated decision quality.

    Returns:

    * ``gated_selectivity`` — fraction of predictions kept (P(win) >
      threshold). Lower = more selective.
    * ``gated_accuracy`` — accuracy among kept predictions. ``nan`` when
      nothing was kept (the meta-label rejected every row).
    * ``gated_accuracy_uplift`` — ``gated_accuracy - accuracy`` (the
      ungated baseline lives under ``accuracy`` in the upstream metrics
      dict; we don't duplicate it here so the two numbers can't drift).
      Positive means gating helped, negative means the meta-label
      filtered out predictions that were actually correct. ``nan`` when
      no rows were kept.
    * ``gated_n_kept`` / ``gated_n_total`` — raw counts.

    MR3 note: an earlier version of this function returned
    ``ungated_accuracy`` as a separate key; it was bit-equal to
    ``_evaluate``'s ``accuracy`` for the same data. Dropped to keep one
    source of truth.
    """
    primary_pred = np.asarray(primary_pred).astype(int)
    meta_pwin = np.asarray(meta_pwin).astype("float64")
    y_true = np.asarray(y_true).astype(int)

    n_total = int(len(primary_pred))
    keep = meta_pwin > threshold
    n_kept = int(keep.sum())
    selectivity = n_kept / n_total if n_total else 0.0

    primary_correct = (primary_pred == y_true)
    ungated_accuracy = float(primary_correct.mean()) if n_total else float("nan")

    if n_kept == 0:
        gated_accuracy = float("nan")
        uplift = float("nan")
    else:
        gated_accuracy = float(primary_correct[keep].mean())
        uplift = gated_accuracy - ungated_accuracy

    return {
        "gated_selectivity": float(selectivity),
        "gated_accuracy": gated_accuracy,
        "gated_accuracy_uplift": uplift,
        "gated_n_kept": float(n_kept),
        "gated_n_total": float(n_total),
    }


# ── Convenience: collect per-base-model probas for the meta features ────


def per_base_model_probas_stack(
    ensemble: Any, X: pd.DataFrame, *, expected_n_classes: int,
) -> np.ndarray:
    """Stack each base model's ``predict_proba`` on ``X`` into a
    ``(n_models, n_rows, n_classes)`` array. The ensemble path's
    :func:`ensemble.predict_proba_ensemble` returns the mean only;
    meta-label features need the full stack so disagreement is computable.

    Centralised here so the train-loop integration doesn't reach into
    :mod:`ensemble`'s private ``_predict_one``.
    """
    from .ensemble import _predict_one
    probas = [_predict_one(fbm, X) for fbm in ensemble.models]
    shapes = {p.shape for p in probas}
    if len(shapes) > 1:
        raise ValueError(
            f"base models produced inconsistent proba shapes for meta-label: "
            f"{shapes}"
        )
    stack = np.stack(probas, axis=0)
    if stack.shape[-1] != expected_n_classes:
        raise ValueError(
            f"ensemble base models produced (n_models, n_rows, {stack.shape[-1]}) "
            f"but meta-label expected {expected_n_classes} classes — the train "
            f"guard should have skipped this fold"
        )
    return stack
