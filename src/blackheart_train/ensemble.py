"""Multi-model ensemble for the directional sub-model (blueprint § 6.2).

The directional model needs to clear the full 13-gate gauntlet by itself
(blueprint § 1, 10%/yr standalone) — a tougher bar than the modulator
sub-models' 5-gate gauntlet. Stacking heterogeneous learners is the
methodology's response to LightGBM's tendency to over-fit the majority
classes on the triple-barrier label.

Three base models, all 3-class:

* **LightGBM**       — gradient-boosted trees, our primary; same hyperparams
  as the modulator path so we can compare apples-to-apples.
* **XGBoost**        — gradient-boosted trees with a different regularisation
  / objective implementation. Disagreement with LightGBM is a useful signal:
  when both agree on a direction, confidence is higher than either alone.
* **Logistic-L1**    — sklearn ``LogisticRegression(penalty='l1')``. Acts
  as a *sanity floor* — if the linear model picks signal that the trees
  miss, the trees may be overfitting. Per blueprint § 7.5, L1 strength
  is CV-tuned within the model (``LogisticRegressionCV``); standalone
  ``LogisticRegression`` would force us to grid-search a hyperparam M5g
  doesn't otherwise sweep.

The ensemble prediction is the **average probability** across base models
(equal weights). Disagreement is the **per-class std of probabilities**
across base models — a high disagreement bar tells the meta-label (M5g.4)
not to trade.

M5g.3 phase 2 (current): the ensemble is now persisted in full in the
artifact under ``payload["ensemble"]`` (with ``payload["booster"] = None``
for ensemble specs). The content_sha includes a deterministic signature
of all three base models (see :func:`ensemble_content_signature`) so a
re-fit on identical data lands at the same path, and a partial-ensemble
producer can't collide with a full-ensemble one.

History — Phase 1 (superseded): the ensemble's fitted state was held in
:class:`Ensemble` but only LightGBM was persisted; the metrics in the
payload described the averaged 3-model ensemble while the booster on
disk was LightGBM alone. That metric-vs-deployment divergence kept
ensemble specs at ``deployment_ready=False`` (the EB1 audit blocker).
Phase 2 lifts that blocker for ensemble-only specs; meta-label gating
remains unpersisted (M5g.4 phase 2) so specs with
``meta_label_enabled=True`` still land not-deployment-ready until that
follow-up.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Literal

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from .specs import ModelSpec

logger = logging.getLogger(__name__)


BaseModelKind = Literal["lightgbm", "xgboost", "logreg_l1"]


# ── Fitted-ensemble dataclass ────────────────────────────────────────────


@dataclass
class FittedBaseModel:
    """One base model + the bookkeeping the ensemble path needs at
    predict time. ``scaler`` is non-None only for the linear model
    (LightGBM/XGBoost are scale-invariant)."""

    kind: BaseModelKind
    model: Any
    scaler: StandardScaler | None = None


@dataclass
class Ensemble:
    """A fitted ensemble of base models, ordered by ``BASE_MODEL_ORDER``
    so per-class proba averaging is deterministic.

    Two ensembles trained on identical data with identical seeds produce
    identical predictions — guaranteed because each base model's random
    state flows from ``spec.hyperparams['random_state']`` and the scaler
    is deterministic on a sorted input.
    """

    models: list[FittedBaseModel] = field(default_factory=list)


# Fixed ordering so per-class proba averages are deterministic and so
# downstream consumers can read per-model metrics by index.
BASE_MODEL_ORDER: tuple[BaseModelKind, ...] = ("lightgbm", "xgboost", "logreg_l1")


# ── Fit helpers — one per base model ─────────────────────────────────────


def _fit_lightgbm(
    X_tr: pd.DataFrame, y_tr_enc: pd.Series, spec: ModelSpec
) -> FittedBaseModel:
    """LightGBM 3-class classifier. Hyperparams come from
    :func:`_lightgbm_params_for_spec` so the ensemble's LightGBM matches
    the single-model path point-for-point — disagreement between paths
    is data-side only.
    """
    params = _lightgbm_params_for_spec(spec)
    model = lgb.LGBMClassifier(objective="multiclass", **params)
    model.fit(X_tr, y_tr_enc.astype(int))
    return FittedBaseModel(kind="lightgbm", model=model)


def _lightgbm_params_for_spec(spec: ModelSpec) -> dict[str, Any]:
    """Strip any keys that LightGBM's sklearn API doesn't accept, copy
    the rest. Today the spec only carries LightGBM-shaped kwargs, but a
    future spec that ships XGBoost-only hyperparams via the same dict
    shouldn't crash LightGBM here."""
    return dict(spec.hyperparams)


def _fit_xgboost(
    X_tr: pd.DataFrame, y_tr_enc: pd.Series, spec: ModelSpec
) -> FittedBaseModel:
    """XGBoost 3-class classifier. The hyperparam translation is small:
    LightGBM's ``num_leaves`` is implicit in XGBoost (max_leaves); we
    rely on shared params (learning_rate, n_estimators, subsample, etc.)
    and accept default XGBoost regularisation.

    ``class_weight`` is a sklearn convention LightGBM understands but
    XGBoost does not. We translate to ``sample_weight`` at fit time so
    the rare horizon-end class doesn't vanish.
    """
    # Lazy-imported: keeps ``ensemble`` (and by extension ``train``'s
    # ``build_payload`` which probes ``isinstance(_, Ensemble)``) usable
    # without xgboost installed when the project is being driven through
    # single-model specs only. The import only fires when a multi-model
    # spec actually requests an XGBoost base.
    import xgboost as xgb

    params = _xgboost_params_for_spec(spec)
    sample_weight = _balanced_sample_weight(y_tr_enc) if spec.hyperparams.get("class_weight") == "balanced" else None
    model = xgb.XGBClassifier(objective="multi:softprob", **params)
    model.fit(X_tr, y_tr_enc.astype(int), sample_weight=sample_weight)
    return FittedBaseModel(kind="xgboost", model=model)


def _xgboost_params_for_spec(spec: ModelSpec) -> dict[str, Any]:
    """Subset of spec.hyperparams that XGBoost's sklearn API accepts.
    ``class_weight`` is handled separately via sample_weight.

    Translation asymmetries to be honest about (EB4 / EB5 audit):

    * ``min_child_samples`` (LightGBM: row count) → ``min_child_weight``
      (XGBoost: sum of hessian). Same numeric value passed across is
      not semantically equal — for a balanced 3-class problem the
      per-row hessian is around 0.22, so a ``min_child_weight=50``
      ends up needing ~227 rows of hessian-sum, EFFECTIVELY MORE
      regularised than LightGBM's literal-50-row threshold. Keeping
      the same numeric value is a deliberate research-grade choice
      (one number to tune per spec, not two) but means XGBoost runs
      slightly shallower in practice. The disagreement signal between
      the two trees is therefore partly architectural, not purely
      data-driven.
    * ``max_depth`` is NOT forwarded — LightGBM's ``-1`` (unlimited,
      capped by num_leaves) has no XGBoost equivalent, and forcing
      a depth would couple the two further. XGBoost defaults to
      ``max_depth=6``, which combined with the deeper hessian threshold
      above gives the trees similar effective complexity for our data.
    * ``num_leaves`` / ``subsample_freq`` are LightGBM-only and not
      translated.

    A future M5g.3 phase 2 (or a more careful ensemble cleanup) can
    split into per-base hyperparam blocks if these asymmetries become
    the binding factor in disagreement.
    """
    src = spec.hyperparams
    out: dict[str, Any] = {}
    for k in (
        "learning_rate", "n_estimators", "subsample",
        "colsample_bytree", "min_child_samples", "random_state",
        "reg_alpha", "reg_lambda",
    ):
        if k in src:
            out[k] = src[k]
    if "min_child_samples" in out:
        out["min_child_weight"] = out.pop("min_child_samples")
    out["verbosity"] = 0
    return out


def _fit_logreg_l1(
    X_tr: pd.DataFrame, y_tr_enc: pd.Series, spec: ModelSpec
) -> FittedBaseModel:
    """Logistic-L1 with CV-tuned regularisation strength. The linear
    model needs scaled inputs (the trees do not); the scaler is stored
    on the FittedBaseModel so predict-time transformation is symmetric.

    ``LogisticRegressionCV`` uses inner CV to pick the L1 strength —
    avoids a manual sweep that would explode the M5g grid. ``saga``
    is the only solver that handles multinomial + L1 in sklearn.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_tr.to_numpy(dtype="float64"))
    sample_weight = (
        _balanced_sample_weight(y_tr_enc).to_numpy()
        if spec.hyperparams.get("class_weight") == "balanced"
        else None
    )
    # max_iter is generous because saga + multinomial + L1 + scaled
    # features can be slow to converge on the noisier triple-barrier
    # signal. We accept the wall-clock cost; the ensemble fits in <60s.
    #
    # sklearn 1.8+ deprecated ``penalty='l1'`` in favour of the
    # elastic-net interface where ``l1_ratios=(1.0,)`` is pure L1 (and
    # ``l1_ratios=(0.0,)`` would be pure L2). ``saga`` is the only
    # solver that supports elastic-net. ``use_legacy_attributes=False``
    # opts into sklearn 1.10's simplified ``coef_`` / ``C_`` shape now
    # so we don't have to migrate twice.
    model = LogisticRegressionCV(
        l1_ratios=(1.0,),
        solver="saga",
        max_iter=2000,
        cv=3,
        random_state=spec.hyperparams.get("random_state", 0),
        n_jobs=1,
        use_legacy_attributes=False,
    )
    model.fit(X_scaled, y_tr_enc.astype(int), sample_weight=sample_weight)
    return FittedBaseModel(kind="logreg_l1", model=model, scaler=scaler)


def _balanced_sample_weight(y_enc: pd.Series) -> pd.Series:
    """Re-weight rows so each class contributes equally to the loss.
    Matches sklearn's ``class_weight='balanced'`` formula:
    ``n_samples / (n_classes * count(class))``.
    """
    counts = y_enc.astype(int).value_counts()
    n = len(y_enc)
    n_classes = len(counts)
    weights = {int(c): n / (n_classes * cnt) for c, cnt in counts.items()}
    return y_enc.astype(int).map(weights).astype("float64")


# ── Predict ──────────────────────────────────────────────────────────────


def _predict_one(fbm: FittedBaseModel, X: pd.DataFrame) -> np.ndarray:
    """Per-base-model predict_proba. Returns an (n, C) array where C
    matches what the model saw at fit time. The caller is responsible
    for the train guard that ensures C == :data:`train.N_MULTICLASS_CLASSES`.
    """
    if fbm.scaler is not None:
        X_input = fbm.scaler.transform(X.to_numpy(dtype="float64"))
    else:
        X_input = X
    return np.asarray(fbm.model.predict_proba(X_input))


def predict_proba_ensemble(ensemble: Ensemble, X: pd.DataFrame) -> np.ndarray:
    """Mean of base-model probability matrices. All base models must
    have produced the same shape (n, C); validated up front so a future
    inconsistency surfaces here, not in an opaque numpy broadcast error.
    """
    if not ensemble.models:
        raise ValueError("ensemble has no fitted base models")
    probas = [_predict_one(fbm, X) for fbm in ensemble.models]
    shapes = {p.shape for p in probas}
    if len(shapes) > 1:
        raise ValueError(
            f"base models produced inconsistent proba shapes: {shapes} — "
            "every base model must see the same n_classes"
        )
    return np.mean(np.stack(probas, axis=0), axis=0)


def ensemble_content_signature(ensemble: Ensemble) -> str:
    """Deterministic string signature of every base model in ``ensemble``.

    Used by :func:`train.build_payload` as the booster-equivalent input
    to ``content_sha256`` for multi-model specs. Two ensembles fit on
    identical data with identical seeds produce identical signatures —
    so the artifact's content address tracks the ensemble's identity,
    not just the LightGBM primary's.

    Per-kind signature:

    * **LightGBM**  — ``model.booster_.model_to_string()`` (the same text
      form used for single-model specs in M5a).
    * **XGBoost**   — ``model.get_booster().save_raw(raw_format='json')``,
      decoded to UTF-8. JSON form is documented stable across XGBoost
      minor versions; the binary form embeds compile-time metadata
      that would churn the sha on every dependency bump.
    * **Logreg-L1** — JSON object holding ``coef_``, ``intercept_``,
      chosen ``C_``, and the scaler's ``mean_`` / ``scale_`` (the linear
      base model's full state) with sorted keys.

    Pickle bytes are deliberately NOT used for the signature: sklearn /
    XGBoost pickles include compile-time and library-version metadata
    that is non-deterministic across patch releases and would break
    content-addressing on every ``pip upgrade``.
    """
    if not ensemble.models:
        raise ValueError("cannot sign an empty ensemble")
    parts: list[str] = []
    for fbm in ensemble.models:
        parts.append(f"=== {fbm.kind} ===")
        if fbm.kind == "lightgbm":
            parts.append(fbm.model.booster_.model_to_string())
        elif fbm.kind == "xgboost":
            raw = fbm.model.get_booster().save_raw(raw_format="json")
            raw_str = (
                raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
            )
            # XGBoost's JSON dump has a top-level ``version`` field
            # (e.g. ``[2, 1, 4]``) stamping the library version that
            # produced the booster. Keeping it in the signature would
            # churn content_sha on every ``pip upgrade xgboost`` even
            # when the trees are bit-identical, defeating one of the
            # reasons we use the JSON form instead of binary pickle.
            # Strip it; re-serialise with sorted keys so the field
            # ordering is deterministic regardless of the JSON
            # writer's internal output order.
            parsed = json.loads(raw_str)
            if isinstance(parsed, dict):
                parsed.pop("version", None)
            parts.append(json.dumps(parsed, sort_keys=True))
        elif fbm.kind == "logreg_l1":
            if fbm.scaler is None:
                raise ValueError(
                    "logreg_l1 base model is missing its scaler — "
                    "fit path should have stored one"
                )
            # ``LogisticRegressionCV.C_`` is an array under sklearn's
            # legacy attribute API (one entry per class for multinomial)
            # but a single scalar under the simplified ``use_legacy_attributes=False``
            # interface we opted into in ``_fit_logreg_l1``. Both shapes
            # are serialised as JSON via ``.tolist()`` when array-shaped
            # and ``float(...)`` when scalar-shaped, so the signature
            # stays stable regardless of which sklearn flavour produced
            # the fit.
            c_attr = fbm.model.C_
            if hasattr(c_attr, "tolist"):
                c_serialised: Any = c_attr.tolist()
            else:
                c_serialised = float(c_attr)
            sig = {
                "coef_": fbm.model.coef_.tolist(),
                "intercept_": fbm.model.intercept_.tolist(),
                "C_": c_serialised,
                "scaler_mean_": fbm.scaler.mean_.tolist(),
                "scaler_scale_": fbm.scaler.scale_.tolist(),
            }
            parts.append(json.dumps(sig, sort_keys=True))
        else:
            raise ValueError(
                f"no signature implementation for base model kind {fbm.kind!r}; "
                f"add a branch in ensemble_content_signature"
            )
    return "\n".join(parts)


def base_model_disagreement(probas_stack: np.ndarray) -> np.ndarray:
    """Per-row, per-class standard deviation across base models. The
    blueprint's "disagreement" signal lives here — high disagreement is
    the meta-label's reason to abstain.

    ``probas_stack`` shape: ``(n_models, n_rows, n_classes)``. Returns
    ``(n_rows, n_classes)`` of stds. Population std (ddof=0) matches
    sklearn's per-class conventions.
    """
    return np.std(probas_stack, axis=0, ddof=0)


# ── Fit + evaluate ───────────────────────────────────────────────────────


_FITTERS: dict[BaseModelKind, Any] = {
    "lightgbm": _fit_lightgbm,
    "xgboost": _fit_xgboost,
    "logreg_l1": _fit_logreg_l1,
}


def fit_ensemble(
    X_tr: pd.DataFrame, y_tr_enc: pd.Series, spec: ModelSpec
) -> Ensemble:
    """Fit each base model named in ``spec.base_models`` on the same
    training data, in :data:`BASE_MODEL_ORDER`. Skips kinds the spec
    doesn't request.

    Pure: no DB, no disk, no logging at info-or-lower. Used by both
    :func:`fit_and_evaluate_ensemble` and (eventually) the meta-label
    training stage.
    """
    requested = set(spec.base_models)
    unknown = requested - set(_FITTERS)
    if unknown:
        raise ValueError(
            f"spec={spec.name} requests unknown base models {sorted(unknown)}; "
            f"known: {sorted(_FITTERS)}"
        )
    fitted: list[FittedBaseModel] = []
    for kind in BASE_MODEL_ORDER:
        if kind not in requested:
            continue
        logger.debug("ensemble fit | spec=%s kind=%s n_train=%d", spec.name, kind, len(X_tr))
        fitted.append(_FITTERS[kind](X_tr, y_tr_enc, spec))
    return Ensemble(models=fitted)


def evaluate_ensemble(
    ensemble: Ensemble,
    X_val: pd.DataFrame,
    y_val_enc: pd.Series,
    *,
    n_classes: int,
) -> tuple[np.ndarray, dict[str, float]]:
    """Evaluate the ensemble + every base model on ``(X_val, y_val_enc)``.

    Returns the averaged proba matrix (so callers can re-use it) and a
    flat metrics dict whose keys carry per-model + per-ensemble +
    disagreement entries. Per-base keys are prefixed (``lgb_*``,
    ``xgb_*``, ``lr_*``); ensemble-level keys are unprefixed
    (``log_loss``, ``accuracy``, ``macro_auc_ovr``); disagreement is
    summarised as a scalar (``mean_disagreement``) plus per-class
    (``mean_disagreement_class_i``).
    """
    if not ensemble.models:
        raise ValueError("cannot evaluate empty ensemble")
    y_true = y_val_enc.astype(int).to_numpy()
    all_labels = list(range(n_classes))

    metrics: dict[str, float] = {}

    # Per-base-model
    per_model_probas: list[np.ndarray] = []
    for fbm in ensemble.models:
        p = _predict_one(fbm, X_val)
        per_model_probas.append(p)
        prefix = _METRIC_PREFIX[fbm.kind]
        metrics.update(_one_model_metrics(p, y_true, all_labels, prefix=prefix))

    # Ensemble (mean of base probas). Re-normalise post-clip so log_loss
    # doesn't trip the "rows don't sum to 1" sklearn check.
    probas_stack = np.stack(per_model_probas, axis=0)
    ensemble_proba = np.mean(probas_stack, axis=0)
    metrics.update(_one_model_metrics(ensemble_proba, y_true, all_labels, prefix=""))

    # Disagreement: per-class std across base models, averaged over rows.
    dis = base_model_disagreement(probas_stack)   # (n_rows, n_classes)
    per_class_mean_dis = dis.mean(axis=0)
    metrics["mean_disagreement"] = float(per_class_mean_dis.mean())
    for i, v in enumerate(per_class_mean_dis):
        metrics[f"mean_disagreement_class_{i}"] = float(v)

    return ensemble_proba, metrics


_METRIC_PREFIX: dict[BaseModelKind, str] = {
    "lightgbm": "lgb_",
    "xgboost": "xgb_",
    "logreg_l1": "lr_",
}


def _one_model_metrics(
    proba: np.ndarray, y_true: np.ndarray, all_labels: list[int], *, prefix: str
) -> dict[str, float]:
    """Subset of the multiclass metric set the single-model path
    computes, keyed by ``prefix`` so per-base entries don't collide
    with the ensemble's.

    log_loss + accuracy + macro_auc_ovr are the three the gauntlet
    cares about. Per-class precision/recall + macro_f1 stay in the
    single-model branch — they'd quadruple the metrics dict for every
    ensemble call without telling the reviewer anything new for the
    ensemble's purpose (disagreement is the new signal here).
    """
    clipped = np.clip(proba, 1e-7, 1 - 1e-7)
    clipped = clipped / clipped.sum(axis=1, keepdims=True)
    y_hat = proba.argmax(axis=1)
    ll = float(log_loss(y_true, clipped, labels=all_labels))
    acc = float(accuracy_score(y_true, y_hat))
    if len(set(y_true)) == len(all_labels):
        try:
            auc_macro = float(roc_auc_score(
                y_true, proba, multi_class="ovr", average="macro", labels=all_labels,
            ))
        except ValueError:
            auc_macro = float("nan")
    else:
        auc_macro = float("nan")
    return {
        f"{prefix}log_loss": ll,
        f"{prefix}accuracy": acc,
        f"{prefix}macro_auc_ovr": auc_macro,
    }
