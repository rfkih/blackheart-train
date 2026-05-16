"""Bootstrap confidence intervals for per-fold metrics (blueprint § 6.4).

The walk-forward pipeline reports a primary metric per fold (e.g.
``macro_auc_ovr`` for multiclass). A point estimate can look fine while
hiding wide uncertainty — the gauntlet's gate 3 explicitly uses
``CI-lower-5%``, not the mean, to require honest confidence.

Implementation: bootstrap-resample the ``(y_true, y_proba)`` pairs with
replacement N times (default 1000), compute the metric on each resample,
report mean + 5th + 95th percentile. Resamples where the metric is
undefined (e.g. multiclass AUC OVR with a class missing from the
resample) are dropped — the surviving count is surfaced so the reviewer
can tell "5 of 1000 dropped" apart from "990 of 1000 dropped".

Why pure functions + scope-limited: the bootstrap is computed inside
``train.fit_and_evaluate`` so the resulting CIs flow through the metrics
dict alongside the point metric. Each walk-forward fold gets its own
bootstrap; the aggregator's ``metric_means`` then yields the mean of
CI-lowers across folds, which is what M5h's gauntlet will read.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


# Default resamples. 1000 is standard for stable 5th/95th percentiles
# without blowing the wall-clock budget — at ~1600 val rows × 1000
# resamples × ~3 ms per AUC OVR call, the bootstrap is ~3-5 s per
# fold, ~20-30 s per walk-forward.
N_BOOTSTRAP_DEFAULT: int = 1000


def bootstrap_macro_auc_ovr(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    n_classes: int,
    n_bootstrap: int = N_BOOTSTRAP_DEFAULT,
    random_state: int = 42,
) -> dict[str, float]:
    """Bootstrap mean / CI-lower-5% / CI-upper-95% for macro AUC OVR.

    ``y_true`` is the encoded integer class index per row; ``y_proba``
    is the (n_rows, n_classes) probability matrix. Resamples with
    replacement and recomputes ``roc_auc_score(..., multi_class='ovr',
    average='macro', labels=range(n_classes))`` each time.

    Resamples that don't contain every class are dropped — AUC OVR
    is undefined without all classes present. The surviving count is
    in the returned ``n_valid_resamples`` key so the reviewer can
    distinguish "narrow CI" from "wide CI with most resamples dropped".

    Returns NaN values when no resample produced a valid metric, so
    downstream metric_means aggregation handles missing data the same
    way as for the point estimate.
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba)
    n_rows = len(y_true)
    if n_rows == 0:
        return _nan_result(n_bootstrap)

    rng = np.random.default_rng(random_state)
    all_labels = list(range(n_classes))
    samples: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_rows, size=n_rows)
        y_t = y_true[idx]
        y_p = y_proba[idx]
        if len(set(y_t)) < n_classes:
            continue
        try:
            auc = float(roc_auc_score(
                y_t, y_p, multi_class="ovr", average="macro", labels=all_labels,
            ))
        except ValueError:
            # roc_auc_score raises on degenerate resamples (e.g. all
            # probas identical for some class). Drop and move on.
            continue
        samples.append(auc)

    if not samples:
        return _nan_result(n_bootstrap)

    arr = np.array(samples)
    return {
        "macro_auc_ovr_bootstrap_mean": float(arr.mean()),
        "macro_auc_ovr_ci_lower_5": float(np.quantile(arr, 0.05)),
        "macro_auc_ovr_ci_upper_95": float(np.quantile(arr, 0.95)),
        "macro_auc_ovr_bootstrap_std": float(arr.std(ddof=0)),
        "n_valid_resamples": float(len(samples)),
        "n_bootstrap": float(n_bootstrap),
    }


def _nan_result(n_bootstrap: int) -> dict[str, float]:
    return {
        "macro_auc_ovr_bootstrap_mean": float("nan"),
        "macro_auc_ovr_ci_lower_5": float("nan"),
        "macro_auc_ovr_ci_upper_95": float("nan"),
        "macro_auc_ovr_bootstrap_std": float("nan"),
        "n_valid_resamples": 0.0,
        "n_bootstrap": float(n_bootstrap),
    }
