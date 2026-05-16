"""Adversarial validation per fold (blueprint § 7.3, gauntlet gate 4).

Per-fold sanity check on covariate shift: can a classifier
distinguish train-set bars from test-set bars using features alone?
If yes (AUC > 0.6), the fold straddles a regime change — the model's
metric on that fold is not trustworthy because the underlying feature
distribution differs from what the model saw at fit time.

The classifier is intentionally simple — LightGBM on a fast 5-fold CV
— because we're asking "is there ANY signal differentiating the two
sets?" not "fit the best possible discriminator". Speed matters: this
runs per walk-forward fold.

Implementation contract:

* Inputs are the fold's X_train + X_test (the y the actual model
  consumes is intentionally NOT passed — adversarial validation looks
  at FEATURES only).
* Returns one scalar AUC. If 5-fold inner-CV produces no valid AUC
  (degenerate input — too few rows, single-class folds), returns NaN.
* The blueprint's 0.6 threshold isn't enforced here; the gauntlet
  consumes the per-fold value and applies the threshold. Keeping the
  threshold at the gauntlet boundary lets us tighten/loosen one
  knob (M5h) without touching the per-fold compute.
"""
from __future__ import annotations

import logging

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


# Inner CV folds for the adversarial classifier. 5 is the blueprint's
# spec for stat-rigor checks; fewer would let one bad split dominate
# the AUC, more would slow the walk-forward measurably.
_ADV_INNER_FOLDS: int = 5

# Inner LightGBM is deliberately shallow + small — we want a fast
# signal, not the strongest possible discriminator. A deeper model
# can find a coincidental difference that doesn't reflect true
# covariate shift.
_ADV_LGBM_PARAMS: dict[str, object] = {
    "n_estimators": 50,
    "num_leaves": 15,
    "learning_rate": 0.1,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "verbosity": -1,
}


def adversarial_auc(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    *,
    random_state: int = 42,
) -> float:
    """Train a LightGBM to distinguish train rows from test rows using
    FEATURES only. Returns the mean of 5-fold inner-CV AUC scores.

    AUC interpretation:

    * ``≈ 0.5`` — train and test are indistinguishable feature-wise;
      no covariate shift detected; fold is trustworthy.
    * ``> 0.6`` — features carry enough signal to label-which-set;
      covariate shift exists; gauntlet gate 4 rejects this fold.
    * ``NaN`` — degenerate input (too few rows, every inner fold
      single-class). Caller should treat NaN as "test undefined" not
      "test passed".

    The function does not take the target y — adversarial validation
    is intentionally label-free. Tests for "did features change?" not
    "did labels change?".
    """
    if len(X_train) == 0 or len(X_test) == 0:
        return float("nan")
    # Don't include columns the adversarial classifier could trivially
    # use to identify the set (e.g. interval_indicator if both sets
    # came from different intervals). Drop interval_indicator if
    # present — the adversarial check should be on real features, not
    # the stacked-interval bookkeeping.
    cols_to_drop = [c for c in ("interval_indicator",) if c in X_train.columns]
    X_train_use = X_train.drop(columns=cols_to_drop) if cols_to_drop else X_train
    X_test_use = X_test.drop(columns=cols_to_drop) if cols_to_drop else X_test

    if len(X_train_use.columns) == 0:
        # Nothing to discriminate on — adversarial check is undefined.
        return float("nan")

    combined = pd.concat([X_train_use, X_test_use], axis=0, ignore_index=True)
    y = np.concatenate([
        np.zeros(len(X_train_use), dtype=int),
        np.ones(len(X_test_use), dtype=int),
    ])
    if len(set(y)) < 2:
        # Defensive: empty-array guard above should already prevent
        # this, but if a caller hands us a degenerate split we
        # surface NaN rather than crash StratifiedKFold.
        return float("nan")

    cv = StratifiedKFold(
        n_splits=_ADV_INNER_FOLDS, shuffle=True, random_state=random_state,
    )
    aucs: list[float] = []
    for tr_idx, te_idx in cv.split(combined, y):
        y_tr = y[tr_idx]
        y_te = y[te_idx]
        # An inner fold with a single class in the test slice would
        # produce undefined AUC. Skip.
        if len(set(y_te)) < 2 or len(set(y_tr)) < 2:
            continue
        model = lgb.LGBMClassifier(**_ADV_LGBM_PARAMS)
        model.fit(combined.iloc[tr_idx], y_tr)
        try:
            p = model.predict_proba(combined.iloc[te_idx])[:, 1]
            auc = float(roc_auc_score(y_te, p))
            aucs.append(auc)
        except ValueError:
            continue

    if not aucs:
        return float("nan")
    return float(np.mean(aucs))
