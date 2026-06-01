"""Conditional invariance test per fold (replaces adversarial AUC as
the transferability gate, blueprint follow-up to ``project_v2_adversarial_auc.md``).

What this measures
------------------

The adversarial-AUC test asks "are P(features)_train and P(features)_test
distinguishable?" — that's *covariate shift*. On hourly bars with macro
features this is structurally near 1.0 (each rolling window has a
different macro density) but the **conditional** relationship
``P(y | features)`` can still be invariant — which is what matters for
generalisation.

Conditional invariance asks the right question: given a feature value,
does the label-distribution shift between train and test? If
``mean(y | feature ≈ v)`` is similar across train and test for every
feature value ``v``, the model trained on train transfers honestly to
test, regardless of how different the marginal feature density was.

Implementation
--------------

For each key feature ``f``:

1. Compute Q-quantile cut points from TRAIN data only (point-in-time
   correct; test never informs the binning).
2. Assign each train and test row to a bin.
3. For each bin with ``≥ min_bin_samples`` rows on BOTH sides, compute
   ``mean(y_train | bin)`` and ``mean(y_test | bin)``.
4. Per-bin divergence = ``|train_mean − test_mean|``.

Aggregates:

* ``max_abs_diff`` — worst per-(feature, bin) divergence.
* ``mean_abs_diff`` — average across (feature, bin) pairs.
* ``n_pairs_evaluated`` — count of (feature, bin) pairs that survived
  the min-bin-samples filter.

Pass thresholds (gauntlet boundary):

* Binary objective — ``max_abs_diff < 0.15`` (a 15-percentage-point
  shift in P(y=1) per bin is acceptable label noise; bigger means the
  conditional relationship is genuinely shifting).
* Regression — ``max_abs_diff < 0.5 * std(y_train)`` (half a
  train-side standard deviation per bin).

The threshold lives at the gauntlet boundary (``gauntlet_directional``,
gate 4) — not in this module — so we can tighten/loosen one knob
without touching the per-fold compute.

Non-goals
---------

* Multiclass: for each bin, divergence = max over classes k of
  |P(y=k|bin,train) − P(y=k|bin,test)|. Same aggregation structure as
  binary. Probabilities are already in [0,1] so no std-scaling is
  applied (same as binary). Pass threshold: 0.15 per class-probability
  shift (same conservative bound as binary — any single class drifting
  15+ pp per bin is a genuine conditional shift worth flagging).
* This is a per-fold diagnostic; the gauntlet aggregates across folds
  via mean over per-fold ``max_abs_diff``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Defaults ───────────────────────────────────────────────────────────────


DEFAULT_N_BINS: int = 5
DEFAULT_MIN_BIN_SAMPLES: int = 20


# ── Records ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ConditionalInvarianceResult:
    """Per-fold conditional-invariance summary.

    Fields:

    * ``max_abs_diff`` — worst per-(feature, bin) shift in
      ``mean(y|bin)`` between train and test.
    * ``mean_abs_diff`` — average across all evaluated (feature, bin)
      pairs.
    * ``n_pairs_evaluated`` — how many (feature, bin) pairs survived
      the min-bin-samples filter on BOTH sides.
    * ``per_feature_max_diff`` — by-feature worst shift; useful for
      operator debugging ("which feature broke?").
    * ``skipped_reason`` — non-None when the fold was unrunnable
      (no eligible features, no rows, etc.); the gauntlet treats
      missing as FAIL rather than silently passing.
    """

    max_abs_diff: float
    mean_abs_diff: float
    n_pairs_evaluated: int
    per_feature_max_diff: dict[str, float]
    skipped_reason: str | None = None


# ── Public API ─────────────────────────────────────────────────────────────


def conditional_invariance(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    objective: Literal["binary", "regression", "multiclass"],
    features: Iterable[str] | None = None,
    n_bins: int = DEFAULT_N_BINS,
    min_bin_samples: int = DEFAULT_MIN_BIN_SAMPLES,
) -> ConditionalInvarianceResult:
    """Compute conditional-invariance shift between train and test.

    See module docstring for the methodology rationale.

    Args:
        X_train, y_train: training fold features and label.
        X_test, y_test: test fold features and label.
        objective: ``"binary"`` (label ∈ {0, 1}) or ``"regression"``
            (label ∈ ℝ). Binary uses raw means (P(y=1) per bin);
            regression uses z-score-normalised means so the same
            threshold applies across scale.
        features: which columns to test. Defaults to every numeric
            column shared between X_train and X_test.
        n_bins: quantile bins per feature. Default 5 (quintiles).
        min_bin_samples: minimum rows per bin on EACH side (train and
            test) for the bin to count. Sub-min bins are skipped to
            avoid 1-sample-fluke divergence.

    Returns:
        ConditionalInvarianceResult — see dataclass docstring.
    """
    if objective not in ("binary", "regression", "multiclass"):
        raise ValueError(
            f"objective must be 'binary', 'regression', or 'multiclass', "
            f"got {objective!r}"
        )
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins!r}")
    if min_bin_samples < 1:
        raise ValueError(f"min_bin_samples must be >= 1, got {min_bin_samples!r}")

    if len(X_train) == 0 or len(X_test) == 0:
        return ConditionalInvarianceResult(
            max_abs_diff=float("nan"),
            mean_abs_diff=float("nan"),
            n_pairs_evaluated=0,
            per_feature_max_diff={},
            skipped_reason="empty_train_or_test",
        )

    # Pick features. Default to every numeric column the two sides share.
    if features is None:
        shared = [c for c in X_train.columns if c in X_test.columns]
        features = [
            c for c in shared
            if pd.api.types.is_numeric_dtype(X_train[c])
            and pd.api.types.is_numeric_dtype(X_test[c])
        ]
    else:
        features = list(features)

    if not features:
        return ConditionalInvarianceResult(
            max_abs_diff=float("nan"),
            mean_abs_diff=float("nan"),
            n_pairs_evaluated=0,
            per_feature_max_diff={},
            skipped_reason="no_numeric_features",
        )

    # Regression: pre-compute y_train std for normalised divergence.
    # Binary / multiclass: raw probabilities already in [0, 1] — no scaling.
    y_train_arr = y_train.astype(float).to_numpy()
    y_test_arr = y_test.astype(float).to_numpy()
    if objective == "regression":
        std = float(np.std(y_train_arr, ddof=0))
        if not np.isfinite(std) or std == 0.0:
            # Degenerate label: constant y_train means every bin's mean
            # is identical — divergence is undefined as a transferability
            # signal. Skip the fold rather than emit a meaningless 0.
            return ConditionalInvarianceResult(
                max_abs_diff=float("nan"),
                mean_abs_diff=float("nan"),
                n_pairs_evaluated=0,
                per_feature_max_diff={},
                skipped_reason="degenerate_train_label",
            )
    else:
        std = 1.0  # binary / multiclass: probability shift in [0,1]

    # For multiclass, pre-compute the integer class labels once.
    mc_classes: np.ndarray | None = None
    if objective == "multiclass":
        y_tr_int = y_train_arr.astype(int)
        y_te_int = y_test_arr.astype(int)
        mc_classes = np.unique(np.concatenate([y_tr_int, y_te_int]))

    per_feature_max: dict[str, float] = {}
    all_diffs: list[float] = []

    for feat in features:
        x_tr = X_train[feat].to_numpy()
        x_te = X_test[feat].to_numpy()
        # Quantile cut points from TRAIN only — point-in-time correct.
        # Drop NaN rows from train when computing edges; bin assignment
        # uses np.digitize which treats NaN as the right-most bin and
        # we filter them via the count threshold.
        x_tr_clean = x_tr[~np.isnan(x_tr)]
        if len(x_tr_clean) < n_bins * min_bin_samples:
            # Not enough data to populate every bin even nominally; skip.
            continue
        # ``np.quantile`` with ``q = [1/n, 2/n, ..., (n-1)/n]`` gives the
        # n-1 internal cut points. ``np.digitize`` then maps each row to
        # bin index in {0, ..., n-1}. Identical quantile values collapse
        # to fewer-than-n bins; we keep going — under-represented bins
        # are filtered by min_bin_samples below.
        cuts = np.quantile(x_tr_clean, np.linspace(1 / n_bins, 1 - 1 / n_bins, n_bins - 1))
        # Deduplicate; if every cut is equal, the feature is degenerate
        # (e.g. interval_indicator with one value in this fold). With
        # ``n_bins >= 2`` validated up front, ``cuts`` always has at
        # least one element, so ``cuts_unique`` cannot be empty —
        # ``np.digitize`` is safe to call.
        cuts_unique = np.unique(cuts)
        train_bins = np.digitize(x_tr, cuts_unique)
        test_bins = np.digitize(x_te, cuts_unique)
        feat_max = 0.0
        for b in range(len(cuts_unique) + 1):
            tr_mask = (train_bins == b) & ~np.isnan(x_tr)
            te_mask = (test_bins == b) & ~np.isnan(x_te)
            n_tr = int(tr_mask.sum())
            n_te = int(te_mask.sum())
            if n_tr < min_bin_samples or n_te < min_bin_samples:
                continue
            if objective == "multiclass":
                # Per-class P(y=k|bin) divergence: max over classes.
                # Probabilities in [0,1] — no std scaling needed.
                assert mc_classes is not None
                y_tr_bin = y_train_arr[tr_mask].astype(int)
                y_te_bin = y_test_arr[te_mask].astype(int)
                bin_diff = 0.0
                for k in mc_classes:
                    p_tr_k = float((y_tr_bin == k).mean())
                    p_te_k = float((y_te_bin == k).mean())
                    bin_diff = max(bin_diff, abs(p_tr_k - p_te_k))
                diff = bin_diff
            else:
                tr_mean = float(y_train_arr[tr_mask].mean())
                te_mean = float(y_test_arr[te_mask].mean())
                diff = abs(tr_mean - te_mean) / std
            all_diffs.append(diff)
            if diff > feat_max:
                feat_max = diff
        # Record every feature whose loop produced at least one
        # eligible bin (signalled by feat_max > 0.0 in a non-degenerate
        # fold). A feature whose every bin was filtered out by
        # min_bin_samples won't appear in per_feature_max_diff — the
        # operator can detect "skipped features" by their absence.
        if feat_max > 0.0:
            per_feature_max[feat] = feat_max

    if not all_diffs:
        return ConditionalInvarianceResult(
            max_abs_diff=float("nan"),
            mean_abs_diff=float("nan"),
            n_pairs_evaluated=0,
            per_feature_max_diff=per_feature_max,
            skipped_reason="no_eligible_bins",
        )

    return ConditionalInvarianceResult(
        max_abs_diff=float(max(all_diffs)),
        mean_abs_diff=float(np.mean(all_diffs)),
        n_pairs_evaluated=len(all_diffs),
        per_feature_max_diff=per_feature_max,
        skipped_reason=None,
    )


# ── Gauntlet-facing thresholds ─────────────────────────────────────────────


# Pass threshold for ``max_abs_diff`` per objective. Lives here (not in
# gauntlet_directional) so a future objective addition only needs to
# extend this map.
PASS_THRESHOLD: dict[str, float] = {
    "binary": 0.15,
    "regression": 0.5,
    # Multiclass: per-class P(y=k|bin) shift — same 15-pp bound as binary.
    # A single class drifting 15+ pp per feature-bin is a genuine
    # conditional shift worth blocking.
    "multiclass": 0.15,
}


def passes(result: ConditionalInvarianceResult, objective: str) -> bool:
    """Whether the per-fold result clears the gauntlet's threshold for
    its objective. Returns False on a skipped/degenerate fold — the
    gauntlet caller decides whether that's FAIL or SKIP.
    """
    if result.skipped_reason is not None:
        return False
    if not np.isfinite(result.max_abs_diff):
        return False
    threshold = PASS_THRESHOLD.get(objective)
    if threshold is None:
        return False
    return result.max_abs_diff < threshold
