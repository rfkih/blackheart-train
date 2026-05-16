"""Per-fold feature selection (blueprint § 7.5 Fix 5).

Tree models (LightGBM, XGBoost) tolerate collinear features fine —
each tree split picks one. The logistic meta-label DOES NOT: it spreads
coefficient mass across collinear features unstably, hurting the gating
decision. The blueprint's response: per fold, prune correlated features
using mutual-information as the tiebreaker, then cap the feature count.

Why per fold not global:

Selecting features once on the full dataset leaks future-fold
information into earlier folds. Per-fold selection uses ONLY the fold's
training window — same PIT discipline as the model itself.

What we actually compute, per fold:

1. **Correlation pruning.** For each pair with ``|corr| > 0.7``, drop
   the one with lower mutual information with the label.
2. **Top-K by MI.** If more than ``max_features`` survive, keep the
   top-K by MI score.
3. **Always-keep columns** (e.g. ``interval_indicator``) bypass both
   stages — they're structural bookkeeping the model needs to
   differentiate stacked-interval rows.

The returned column list is stored per-fold in ``FoldMetrics`` so a
reviewer can ask "which features survived in ≥5 of 6 folds?"
(consistently informative) vs "this only survived in fold 2" (noise).

Note on the meta-label L1 stage:

Blueprint § 7.5 also mentions L1 pruning for the meta-label
specifically. ``meta_label.fit_meta_label`` already uses
``LogisticRegressionCV(l1_ratios=(1.0,))`` which performs that L1 prune
implicitly — no separate stage needed in this module.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

logger = logging.getLogger(__name__)


# Blueprint § 7.5: drop one of any pair with |corr| > 0.7.
CORR_THRESHOLD_DEFAULT: float = 0.7

# Blueprint § 7.5: cap at 8 features post-correlation-prune. Small
# enough to keep the meta-label's L1 stage tractable; large enough to
# let the model use real signal where it exists.
MAX_FEATURES_DEFAULT: int = 8

# Always-keep columns — structural bookkeeping the model needs even if
# it has low MI with the label. ``interval_indicator`` lives here so
# stacked-interval specs don't accidentally lose the column that lets
# the model differentiate 1h from 15m rows.
ALWAYS_KEEP: tuple[str, ...] = ("interval_indicator",)


def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    max_features: int = MAX_FEATURES_DEFAULT,
    corr_threshold: float = CORR_THRESHOLD_DEFAULT,
    random_state: int = 42,
) -> list[str]:
    """Return a subset of X's columns chosen via correlation pruning +
    MI top-K. Always-keep columns are returned first, in their input
    order; the remaining slots are filled by the MI-ranked survivors.

    The returned list preserves column ordering from X for the
    surviving candidates, which keeps the trained model's feature
    matrix layout stable across runs on the same fold.
    """
    if max_features < 1:
        raise ValueError(f"max_features must be >= 1; got {max_features}")

    all_cols = list(X.columns)
    if not all_cols:
        return []

    keep_always = [c for c in all_cols if c in ALWAYS_KEEP]
    candidates = [c for c in all_cols if c not in ALWAYS_KEEP]
    if not candidates:
        return keep_always

    X_cand = X[candidates]
    y_int = y.astype(int)

    # MI scores per candidate. mutual_info_classif handles
    # multiclass + binary natively; for regression a future caller can
    # swap to mutual_info_regression.
    try:
        mi_arr = mutual_info_classif(
            X_cand.to_numpy(dtype="float64"),
            y_int.to_numpy(),
            random_state=random_state,
        )
    except Exception as exc:
        # Defensive: a degenerate input (all-constant column, NaN
        # somewhere) shouldn't crash walk-forward. Fall back to
        # keeping the first ``max_features - len(keep_always)``
        # candidates so the model still trains.
        logger.warning(
            "feature_selection MI failed (%s); falling back to first-N candidates", exc,
        )
        budget = max(1, max_features - len(keep_always))
        return keep_always + candidates[:budget]

    mi = pd.Series(mi_arr, index=candidates)

    # Correlation matrix on candidates. ``.corr()`` produces NaN for
    # constant columns; the NaN > threshold check is False, so
    # constants don't trigger pruning here (they survive into the
    # MI top-K stage where they get ranked last and likely dropped).
    corr = X_cand.corr().abs()
    np.fill_diagonal(corr.values, 0.0)

    dropped: set[str] = set()
    # Deterministic iteration order — sorted lexicographically so a
    # corr-tie produces the same dropped column on every run.
    sorted_cols = sorted(candidates)
    for i_idx, i in enumerate(sorted_cols):
        if i in dropped:
            continue
        for j in sorted_cols[i_idx + 1:]:
            if j in dropped:
                continue
            if not np.isfinite(corr.loc[i, j]):
                continue
            if corr.loc[i, j] > corr_threshold:
                # Drop the one with lower MI. Tiebreak: drop the
                # alphabetically-later one (deterministic).
                if mi[i] < mi[j]:
                    dropped.add(i)
                    break    # i is gone, move to next i
                else:
                    dropped.add(j)

    surviving = [c for c in candidates if c not in dropped]

    # Cap at max_features (minus what always-keep already consumed).
    budget = max(1, max_features - len(keep_always))
    if len(surviving) > budget:
        # Stable sort so MI-ties don't shuffle across runs.
        sorted_by_mi = mi.loc[surviving].sort_values(
            ascending=False, kind="stable",
        )
        kept_candidates = list(sorted_by_mi.head(budget).index)
        # Preserve input column order for the kept set.
        surviving = [c for c in candidates if c in kept_candidates]

    return keep_always + surviving
