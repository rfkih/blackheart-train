"""Training-data integrity checks.

Runs after ``loader.load_dataset`` and before LightGBM fit. The point is
to catch degenerate or surprising input *before* burning compute, and to
record a fingerprint so re-runs on identical data are provably
identical. This is the foundation that gauntlet gate 1 (``labels_pit_safe``)
will build on in M5d.

Verdict model
-------------

Each individual check returns :class:`CheckResult` with severity in
``{PASS, WARN, FAIL}``. :func:`check_dataset` aggregates with worst-of
semantics:

* **FAIL** — training should refuse. The training pipeline raises
  :class:`IntegrityError` so no LightGBM time is spent on degenerate data.
* **WARN** — training proceeds; the report is logged + embedded in the
  artifact payload so a reviewer (M5d) can audit it.
* **PASS** — quiet success.

Checks performed
----------------

1. **min_rows** — refuse if the post-NaN-drop matrix has fewer than
   ``min_rows`` examples (default 1,000). Below ~1,000 rows, walk-forward
   with 6 folds and 7-day embargo collapses.
2. **binary_class_balance** — for binary objectives, WARN if the minority
   class is below 15% of rows. 50/50 ideal; anything more skewed makes
   AUC interpretation harder and may need class weighting.
3. **train_val_class_balance** — for binary objectives, WARN if the
   train and val class proportions diverge by more than 10pp. A val set
   from a different regime is the silent killer of "great-looking" model.
4. **regression_label_distribution** — for regression, WARN if any
   label sits more than 6 standard deviations from the mean (likely
   data anomaly, not signal).
5. **constant_features** — WARN if any feature has zero or near-zero
   variance (std < 1e-12). LightGBM tolerates them but they're usually
   ingestion bugs.
6. **stale_tail** — WARN per-feature if the most recent non-null value
   sits more than 168 hours (7 days) before the training window's end.
   Surfaces "feature stopped publishing" mid-training-window.
7. **label_leakage** (R1.B, 2026-05-17) — refuses if any feature has
   |Pearson ρ| (binary/regression) or normalized MI (multiclass) against
   the label at or above 0.95. Catches the textbook leakage pattern where
   a derived feature accidentally carries the label or future bars. The
   Phase 4 regime_btc_v3 lifecycle landed only after manual detection of
   this kind of bug — automating the gate makes future lifecycles cheaper.
   Tunable via ``allow_leakage=True`` (demotes FAIL → WARN).
8. **data_fingerprint** — sha256 over the contiguous bytes of
   ``X.values``, ``y.values``, and the timestamp index. Stable across
   re-runs on identical data; changes the moment any value or timestamp
   changes. Stored in the artifact payload alongside ``content_sha256``
   (which is model-identity; fingerprint is *data*-identity).
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Literal

import numpy as np
import pandas as pd

from .loader import LoadedDataset
from .specs import ModelSpec

logger = logging.getLogger(__name__)


Severity = Literal["PASS", "WARN", "FAIL"]


class IntegrityError(RuntimeError):
    """Raised by :func:`check_dataset` callers when the overall verdict
    is FAIL. The training pipeline catches this at the right boundary
    and converts it into a clean CLI error rather than a stack trace.
    """


@dataclass
class CheckResult:
    name: str
    severity: Severity
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class IntegrityReport:
    verdict: Severity
    checks: list[CheckResult]
    data_fingerprint: str
    # R1.B: detached copy of the label-leakage check's details so the CLI
    # can stamp it into experiment_run.leakage_report (V92 column) without
    # re-walking ``checks``. None when the check didn't run (e.g.
    # check_dataset called with skip_leakage=True for test fixtures).
    leakage_report: dict[str, Any] | None = None


# ── Individual checks ──────────────────────────────────────────────────────


_MIN_ROWS_DEFAULT = 1_000
_BINARY_MIN_CLASS_PCT = 0.15
_TRAIN_VAL_BALANCE_TOLERANCE_PCT = 0.10
_REGRESSION_OUTLIER_STD = 6.0
_CONSTANT_FEATURE_STD = 1e-12
_STALE_TAIL_HOURS = 168   # 7 days

# R1.B label-leakage detection. 0.95 catches the "feature is the label
# plus noise" pattern. Real leakage typically registers ≥ 0.99; the buffer
# absorbs subtler cases (e.g. a feature derived from a one-bar-shifted
# label). Tighter than 0.95 risks false-positive on legitimate strong
# predictors (regime score vs regime label can correlate 0.6–0.8); 0.95+
# is unambiguous.
_LABEL_LEAKAGE_CORR_THRESHOLD = 0.95
# Top-K offending features surfaced in the report. The CLI logs the full
# list; the leakage_report JSONB (V92) stores them so a post-mortem can
# trace which feature was the culprit.
_LABEL_LEAKAGE_TOP_K = 5
# For multiclass labels Pearson is meaningless (label is categorical).
# We fall back to ``mutual_info_classif`` on a downsample for speed —
# MI on 100k+ rows × 50 features takes ~30s, and we'd rather pay <2s
# every train than burn that compute on every invocation.
_LABEL_LEAKAGE_MI_SAMPLE = 5_000
# Multiclass thresholds (directional model / triple-barrier — 3 classes).
# Minority pct threshold is lower than binary's 15% because triple-barrier
# data is naturally skewed: with TP and SL barriers that converge on the
# horizon, the "neither hit" (class 0) bucket is rare by design. Below 1%
# we FAIL though — at that point the class is too sparse to fit even with
# class_weight=balanced.
_MULTICLASS_MIN_CLASS_PCT_FAIL = 0.01
_MULTICLASS_MIN_CLASS_PCT_WARN = 0.05
_MULTICLASS_MIN_PER_CLASS_ROWS = 50


def _check_min_rows(ds: LoadedDataset, min_rows: int) -> CheckResult:
    n = len(ds.X)
    if n < min_rows:
        return CheckResult(
            name="min_rows",
            severity="FAIL",
            message=f"only {n} rows after NaN drop (threshold {min_rows})",
            details={"n_rows": n, "threshold": min_rows},
        )
    return CheckResult(
        name="min_rows",
        severity="PASS",
        message=f"{n} rows >= {min_rows}",
        details={"n_rows": n, "threshold": min_rows},
    )


_LABEL_STATIONARITY_DRIFT_WARN = 0.15


def _check_label_stationarity(ds: LoadedDataset) -> CheckResult:
    """Compare the binary label base-rate between the first and second
    chronological half of the dataset.

    A large base-rate drift means the label's meaning shifts over time —
    e.g. a triple-barrier win-rate that tracks the prevailing market trend
    regime (high in 2024-25 bull, low in 2022 chop). Downstream this
    manifests as ``adversarial_auc ≈ 1.0`` (the model/adversary separates
    epochs from the label alone), which is only discovered after the full
    walk-forward + gauntlet (30-90 min). This cheap pre-check surfaces the
    risk in the artifact *before* that spend.

    WARN-only — it never blocks an exploratory run, it just flags it.
    """
    n = len(ds.y)
    if n < 200:
        return CheckResult(
            name="label_stationarity", severity="PASS",
            message="too few rows to assess", details={"n_rows": n},
        )
    y = ds.y.astype("float64")
    p_first = float(y.iloc[: n // 2].mean())
    p_second = float(y.iloc[n // 2 :].mean())
    drift = abs(p_first - p_second)
    details = {
        "first_half_mean": round(p_first, 4),
        "second_half_mean": round(p_second, 4),
        "drift": round(drift, 4),
    }
    if drift > _LABEL_STATIONARITY_DRIFT_WARN:
        return CheckResult(
            name="label_stationarity", severity="WARN",
            message=(
                f"label base-rate drifts {drift:.2f} between first/second half "
                f"({p_first:.2f} -> {p_second:.2f}) — non-stationary label, "
                f"expect elevated adversarial_auc / weak transferability"
            ),
            details=details,
        )
    return CheckResult(
        name="label_stationarity", severity="PASS",
        message=f"label base-rate drift {drift:.2f} within tolerance",
        details=details,
    )


def _check_binary_class_balance(ds: LoadedDataset) -> CheckResult:
    counts = ds.y.astype(int).value_counts().to_dict()
    total = sum(counts.values())
    if total == 0:
        return CheckResult(
            name="binary_class_balance", severity="FAIL",
            message="empty label", details=counts,
        )
    if len(counts) < 2:
        return CheckResult(
            name="binary_class_balance", severity="FAIL",
            message=f"only one class present: {counts}",
            details=counts,
        )
    min_pct = min(counts.values()) / total
    pretty = {int(k): int(v) for k, v in counts.items()}
    if min_pct < _BINARY_MIN_CLASS_PCT:
        return CheckResult(
            name="binary_class_balance", severity="WARN",
            message=f"minority class is {min_pct:.1%} (threshold {_BINARY_MIN_CLASS_PCT:.0%})",
            details={"counts": pretty, "min_pct": round(min_pct, 4)},
        )
    return CheckResult(
        name="binary_class_balance", severity="PASS",
        message=f"min class {min_pct:.1%}",
        details={"counts": pretty, "min_pct": round(min_pct, 4)},
    )


def _check_train_val_class_balance(ds: LoadedDataset, spec: ModelSpec) -> CheckResult:
    """Split the dataset the same way ``train.py`` will and compare class
    proportions. A val set from a different regime is the silent killer
    of credible AUC numbers.
    """
    n = len(ds.X)
    n_val = max(1, int(round(n * spec.val_fraction)))
    n_train = n - n_val
    y_tr = ds.y.iloc[:n_train].astype(int)
    y_val = ds.y.iloc[n_train:].astype(int)
    if y_tr.empty or y_val.empty:
        return CheckResult(
            name="train_val_class_balance", severity="FAIL",
            message=f"degenerate split: n_train={len(y_tr)}, n_val={len(y_val)}",
            details={"n_train": int(len(y_tr)), "n_val": int(len(y_val))},
        )
    p_tr = float(y_tr.mean())
    p_val = float(y_val.mean())
    drift = abs(p_tr - p_val)
    details = {
        "train_class1_pct": round(p_tr, 4),
        "val_class1_pct": round(p_val, 4),
        "drift_pp": round(drift, 4),
    }
    if drift > _TRAIN_VAL_BALANCE_TOLERANCE_PCT:
        return CheckResult(
            name="train_val_class_balance", severity="WARN",
            message=(
                f"class-1 mix drifts {drift:.1%} between train ({p_tr:.1%}) and val "
                f"({p_val:.1%}) — val accuracy may be regime-dependent"
            ),
            details=details,
        )
    return CheckResult(
        name="train_val_class_balance", severity="PASS",
        message=f"class-1 drift {drift:.1%} within tolerance",
        details=details,
    )


def _check_multiclass_class_balance(ds: LoadedDataset) -> CheckResult:
    """Class-balance check for multiclass (3-class triple-barrier) labels.

    Two thresholds, not one:
      * FAIL if any class has fewer than ``_MULTICLASS_MIN_PER_CLASS_ROWS``
        examples — the model can't learn a boundary with so few points
        regardless of weighting.
      * WARN if minority pct is below ``_MULTICLASS_MIN_CLASS_PCT_WARN``
        (5%) — triple-barrier data is *expected* to be skewed because TP/SL
        barriers converge before the horizon-end "no barrier hit" class
        accumulates. ``class_weight=balanced`` in the spec handles it; we
        just want the reviewer to know.
    """
    counts = ds.y.astype(int).value_counts().to_dict()
    total = sum(counts.values())
    pretty = {int(k): int(v) for k, v in counts.items()}
    if total == 0:
        return CheckResult(
            name="multiclass_class_balance", severity="FAIL",
            message="empty label", details=pretty,
        )
    if len(counts) < 2:
        return CheckResult(
            name="multiclass_class_balance", severity="FAIL",
            message=f"only one class present: {pretty}",
            details=pretty,
        )
    min_count = min(counts.values())
    min_pct = min_count / total
    details = {"counts": pretty, "min_pct": round(min_pct, 4), "min_count": int(min_count)}
    if min_count < _MULTICLASS_MIN_PER_CLASS_ROWS or min_pct < _MULTICLASS_MIN_CLASS_PCT_FAIL:
        return CheckResult(
            name="multiclass_class_balance", severity="FAIL",
            message=(
                f"minority class has {min_count} rows ({min_pct:.2%}); thresholds "
                f"{_MULTICLASS_MIN_PER_CLASS_ROWS} rows / {_MULTICLASS_MIN_CLASS_PCT_FAIL:.0%}"
            ),
            details=details,
        )
    if min_pct < _MULTICLASS_MIN_CLASS_PCT_WARN:
        return CheckResult(
            name="multiclass_class_balance", severity="WARN",
            message=(
                f"minority class is {min_pct:.2%} ({min_count} rows) — "
                f"triple-barrier skew is expected, ensure class_weight=balanced is on"
            ),
            details=details,
        )
    return CheckResult(
        name="multiclass_class_balance", severity="PASS",
        message=f"min class {min_pct:.2%} ({min_count} rows)",
        details=details,
    )


def _check_regression_label_distribution(ds: LoadedDataset) -> CheckResult:
    y = ds.y.astype("float64")
    if y.std(ddof=0) == 0.0:
        return CheckResult(
            name="regression_label_distribution", severity="FAIL",
            message="label has zero variance (constant target)",
            details={"min": float(y.min()), "max": float(y.max())},
        )
    z = (y - y.mean()) / y.std(ddof=0)
    extreme = int((z.abs() > _REGRESSION_OUTLIER_STD).sum())
    details = {
        "min": float(y.min()), "max": float(y.max()),
        "mean": float(y.mean()), "std": float(y.std(ddof=0)),
        "extreme_count": extreme,
        "extreme_threshold_std": _REGRESSION_OUTLIER_STD,
    }
    if extreme > 0:
        return CheckResult(
            name="regression_label_distribution", severity="WARN",
            message=f"{extreme} labels beyond ±{_REGRESSION_OUTLIER_STD}σ — possible data anomaly",
            details=details,
        )
    return CheckResult(
        name="regression_label_distribution", severity="PASS",
        message="no extreme outliers", details=details,
    )


def _check_constant_features(ds: LoadedDataset) -> CheckResult:
    stds = ds.X.std(ddof=0)
    constant = [name for name, s in stds.items() if s < _CONSTANT_FEATURE_STD]
    if constant:
        return CheckResult(
            name="constant_features", severity="WARN",
            message=f"{len(constant)} feature(s) with near-zero variance: {constant}",
            details={"features": constant, "threshold_std": _CONSTANT_FEATURE_STD},
        )
    return CheckResult(
        name="constant_features", severity="PASS",
        message=f"all {len(stds)} features have variance > {_CONSTANT_FEATURE_STD:g}",
        details={"n_features": int(len(stds))},
    )


def _check_stale_tail(ds: LoadedDataset, spec: ModelSpec) -> CheckResult:
    """For each feature, find the last non-null row in the loaded matrix
    and measure its distance to ``spec.train_end``. A large gap means
    that feature stopped publishing mid-window.

    Note: this runs against the *cleaned* matrix (post-NaN-drop), so a
    feature's "last non-null" here is effectively the last row that
    survived the join. That's the right semantic — it tells you "how
    fresh was the data when the model last saw it."
    """
    threshold = timedelta(hours=_STALE_TAIL_HOURS)
    end = pd.Timestamp(spec.train_end)
    stale: dict[str, int] = {}
    for col in ds.feature_names:
        col_non_null = ds.X[col].dropna()
        if col_non_null.empty:
            continue
        last_ts = pd.Timestamp(col_non_null.index.max())
        gap = end - last_ts
        if gap > threshold:
            stale[col] = int(gap.total_seconds() // 3600)
    if stale:
        return CheckResult(
            name="stale_tail", severity="WARN",
            message=f"{len(stale)} feature(s) have stale-tail > {_STALE_TAIL_HOURS}h: {stale}",
            details={"stale_hours_by_feature": stale, "threshold_hours": _STALE_TAIL_HOURS},
        )
    return CheckResult(
        name="stale_tail", severity="PASS",
        message=f"all features fresh within {_STALE_TAIL_HOURS}h of train_end",
        details={"threshold_hours": _STALE_TAIL_HOURS},
    )


# ── Label-leakage detection (R1.B) ─────────────────────────────────────────


def _pearson_corr_vs_label(X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """|Pearson ρ| between each column of ``X`` and ``y``.

    Returns a dict ``{feature: |corr|}``. Constant features (std=0) are
    omitted — Pearson is undefined and they'd otherwise yield NaN. The
    constant-features check already surfaces them as WARN; double-flagging
    here adds noise.

    Vectorised — avoids Python-loop overhead even with 200 features × 50k
    rows. Uses float64 to keep precision stable across pandas backends.
    """
    y_arr = y.astype("float64").to_numpy()
    y_centered = y_arr - y_arr.mean()
    y_norm = np.linalg.norm(y_centered)
    if y_norm == 0.0:
        return {}
    X_arr = X.astype("float64").to_numpy()
    X_centered = X_arr - X_arr.mean(axis=0)
    X_norms = np.linalg.norm(X_centered, axis=0)
    out: dict[str, float] = {}
    for i, name in enumerate(X.columns):
        if X_norms[i] == 0.0:
            continue
        rho = float((X_centered[:, i] @ y_centered) / (X_norms[i] * y_norm))
        out[str(name)] = float(abs(rho))
    return out


def _normalized_mi_vs_label(X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Mutual information per feature, normalized to ``[0, 1]`` by H(y).

    For multiclass labels Pearson is meaningless (the label codes 0/1/2
    are categorical, not ordinal). MI handles arbitrary discrete labels.
    Normalizing by H(y) puts MI on the same 0..1 scale as |Pearson| so
    the same _LABEL_LEAKAGE_CORR_THRESHOLD applies uniformly.

    Downsamples to ``_LABEL_LEAKAGE_MI_SAMPLE`` rows to keep wall-clock
    under ~2 seconds. The downsample is chronologically stratified
    (every N-th row) so the slice still spans the training window — a
    random downsample on autocorrelated time-series would inflate MI.
    """
    from sklearn.feature_selection import mutual_info_classif  # type: ignore[import-untyped]

    n = len(X)
    if n > _LABEL_LEAKAGE_MI_SAMPLE:
        stride = n // _LABEL_LEAKAGE_MI_SAMPLE
        idx = np.arange(0, n, stride)[:_LABEL_LEAKAGE_MI_SAMPLE]
        X_s = X.iloc[idx]
        y_s = y.iloc[idx]
    else:
        X_s, y_s = X, y

    # Bug-#4 fix (2026-05-17): factorize the label before passing it to
    # mutual_info_classif. Sparse / negative codes (e.g. {-1, 0, 1} or
    # {0, 2, 5}) bias sklearn's MI estimator — it doesn't reindex
    # internally. pd.factorize returns 0..num_classes-1 unique codes so
    # the MI computation is on the canonical form. H(y) is computed from
    # the SAME factorized array — using bincount on the original codes
    # with `- y_arr.min()` worked for contiguous-with-offset labels but
    # broke on sparse codes (bincount would allocate huge zero-filled
    # buckets).
    raw_y = y_s.astype("int64").to_numpy()
    factorized, _ = pd.factorize(raw_y)   # → 0..k-1, dense
    counts = np.bincount(factorized)
    probs = counts[counts > 0] / counts.sum()
    h_y = float(-(probs * np.log(probs)).sum())
    if h_y <= 0.0:
        return {}

    mi = mutual_info_classif(
        X_s.astype("float64").to_numpy(),
        factorized,
        discrete_features=False,
        random_state=0,
    )
    return {str(name): float(mi[i] / h_y) for i, name in enumerate(X.columns)}


def _check_label_leakage(ds: LoadedDataset, spec: ModelSpec) -> CheckResult:
    """Flag features that carry near-perfect information about the label.

    Binary + regression → |Pearson ρ| against ``y``.
    Multiclass → normalized MI against ``y`` (Pearson is meaningless on
    categorical labels).

    Threshold is :data:`_LABEL_LEAKAGE_CORR_THRESHOLD` (0.95). Above that
    the feature is almost certainly the label, a one-bar-shifted derivative,
    or a future-bar leak. Severity FAIL with the offending feature(s) in
    details. ``run_integrity_or_raise`` honours ``allow_leakage`` to demote
    FAIL → WARN when an operator explicitly opts in.

    Always populates ``details`` with the full ranking (top-K) so the
    CLI can stamp it into ``experiment_run.leakage_report`` regardless of
    verdict — useful for trend-tracking the highest non-leaking
    correlations across runs.
    """
    if spec.objective == "multiclass":
        corrs = _normalized_mi_vs_label(ds.X, ds.y)
        method = "mutual_info_norm"
    else:
        corrs = _pearson_corr_vs_label(ds.X, ds.y)
        method = "pearson_abs"

    if not corrs:
        # Empty corr map = constant label or no usable features (the
        # constant-features / class-balance checks have already flagged
        # this; nothing useful to add here).
        return CheckResult(
            name="label_leakage", severity="PASS",
            message="leakage check skipped (no usable features)",
            details={"method": method, "threshold": _LABEL_LEAKAGE_CORR_THRESHOLD},
        )

    ranked = sorted(corrs.items(), key=lambda kv: kv[1], reverse=True)
    top_offenders = [{"feature": name, "score": round(score, 6)} for name, score in ranked[:_LABEL_LEAKAGE_TOP_K]]
    max_feature, max_score = ranked[0]
    details: dict[str, Any] = {
        "method": method,
        "threshold": _LABEL_LEAKAGE_CORR_THRESHOLD,
        "max_score": round(max_score, 6),
        "max_score_feature": max_feature,
        "top_offenders": top_offenders,
    }

    leaking = [(n, s) for n, s in ranked if s >= _LABEL_LEAKAGE_CORR_THRESHOLD]
    if leaking:
        leaking_names = [n for n, _ in leaking]
        details["leaking_features"] = leaking_names
        return CheckResult(
            name="label_leakage", severity="FAIL",
            message=(
                f"{len(leaking)} feature(s) at or above {method} threshold "
                f"{_LABEL_LEAKAGE_CORR_THRESHOLD}: {leaking_names[:3]}"
                f"{'…' if len(leaking) > 3 else ''} (max {max_feature}={max_score:.3f})"
            ),
            details=details,
        )
    return CheckResult(
        name="label_leakage", severity="PASS",
        message=f"max {method} {max_feature}={max_score:.3f} below threshold {_LABEL_LEAKAGE_CORR_THRESHOLD}",
        details=details,
    )


# ── Fingerprint ────────────────────────────────────────────────────────────


def compute_dataset_sha(ds: LoadedDataset, spec: ModelSpec) -> str:
    """Coarse schema-and-range fingerprint of a loaded dataset.

    Sibling to :func:`compute_data_fingerprint`, but deliberately coarse.
    Hashes only ``(symbol, interval, label_feature, label_version, X.shape,
    y.shape, X.index.min(), X.index.max(), feature_names sorted)``.

    Same shape + same window + same feature set → same sha, even if the
    cell values differ (e.g. a backfill rerun produced slightly different
    fundamentals). data_fingerprint changes whenever any byte changes;
    dataset_sha is stable under that. Use them together:

      * Two runs with same dataset_sha + same data_fingerprint = bit-exact
        re-train.
      * Two runs with same dataset_sha but different data_fingerprint =
        same dataset *schema*, refreshed underlying values.
      * Different dataset_sha = comparing apples to oranges.

    The orchestrator's :class:`experiment_run.dataset_sha` column carries
    this so leaderboard filtering can answer "all runs against this dataset
    shape" cleanly.
    """
    h = hashlib.sha256()
    h.update(b"symbol|")
    h.update((spec.symbol or "").encode("utf-8"))
    h.update(b"|interval|")
    h.update((spec.interval or "").encode("utf-8"))
    h.update(b"|label|")
    h.update(ds.label_feature.encode("utf-8"))
    h.update(b"|label_version|")
    h.update(str(ds.label_version).encode("utf-8"))
    h.update(b"|X.shape|")
    h.update(f"{ds.X.shape[0]},{ds.X.shape[1]}".encode("utf-8"))
    h.update(b"|y.shape|")
    h.update(str(ds.y.shape[0]).encode("utf-8"))
    # Bug-#3 fix (2026-05-17): floor to seconds before isoformat. The
    # naive isoformat preserves microseconds, which means two load paths
    # reading the same data at different precisions (feature_store at
    # µs vs a CSV reload at sec) would churn the dataset_sha for the
    # same semantic dataset. Hourly bars never need sub-second precision
    # — the floor is a no-op on real bar data and absorbs precision
    # drift on the edge cases.
    h.update(b"|ts.min|")
    h.update(pd.Timestamp(ds.X.index.min()).floor("s").isoformat().encode("utf-8"))
    h.update(b"|ts.max|")
    h.update(pd.Timestamp(ds.X.index.max()).floor("s").isoformat().encode("utf-8"))
    h.update(b"|features|")
    # Sorted so column-order changes (e.g. a new ingest cron reorders the
    # registry query) don't churn the sha. Real feature changes — adds /
    # removes — DO change it.
    for name in sorted(ds.feature_names):
        h.update(name.encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()


def compute_data_fingerprint(ds: LoadedDataset) -> str:
    """sha256 over X.values, y.values, and the timestamp index.

    Stable across runs on identical data. Two different datasets — even
    one row different, one timestamp different — produce different
    fingerprints. Stored next to ``content_sha256`` so the question
    "was this artifact trained on the same data?" has a one-line answer.

    Implementation note: we explicitly materialise float64 + int64
    representations so the bytes are stable across pandas versions
    (which sometimes change default int widths on Windows).
    """
    h = hashlib.sha256()
    h.update(b"X|")
    # contiguous column-major: column order is fixed by feature_names
    for name in ds.feature_names:
        h.update(name.encode("utf-8"))
        h.update(b"|")
        h.update(ds.X[name].to_numpy(dtype="float64", copy=False).tobytes())
        h.update(b"|")
    h.update(b"y|")
    h.update(ds.y.to_numpy(dtype="float64", copy=False).tobytes())
    h.update(b"|ts|")
    h.update(ds.X.index.astype("datetime64[ns]").asi8.tobytes())
    return h.hexdigest()


# ── Aggregator ─────────────────────────────────────────────────────────────


_SEVERITY_RANK: dict[Severity, int] = {"PASS": 0, "WARN": 1, "FAIL": 2}


def _worst(severities: list[Severity]) -> Severity:
    if not severities:
        return "PASS"
    return max(severities, key=lambda s: _SEVERITY_RANK[s])


def recompute_verdict(checks: list[CheckResult]) -> Severity:
    """Public wrapper over :func:`_worst` for callers that mutate a check's
    severity (e.g. ``run_integrity_or_raise(allow_leakage=True)`` demoting
    leakage FAIL → WARN) and need to refresh ``IntegrityReport.verdict``.
    """
    return _worst([c.severity for c in checks])


def check_dataset(
    ds: LoadedDataset,
    spec: ModelSpec,
    *,
    min_rows: int = _MIN_ROWS_DEFAULT,
    skip_leakage: bool = False,
) -> IntegrityReport:
    """Run all integrity checks and return an aggregated report.

    Caller is expected to inspect ``report.verdict``:
      * ``FAIL`` → raise :class:`IntegrityError` (training would be
        meaningless on this data)
      * ``WARN`` → log + persist in the artifact payload
      * ``PASS`` → proceed quietly

    ``skip_leakage`` is a test-only escape hatch. The ``--allow-leakage``
    CLI flag does NOT use it — that flag wants the check to RUN (to record
    the leakage_report) but demote FAIL→WARN. ``run_integrity_or_raise``
    handles the demotion downstream.
    """
    checks: list[CheckResult] = [
        _check_min_rows(ds, min_rows),
        _check_constant_features(ds),
        _check_stale_tail(ds, spec),
    ]
    if spec.objective == "binary":
        checks.append(_check_binary_class_balance(ds))
        checks.append(_check_train_val_class_balance(ds, spec))
        checks.append(_check_label_stationarity(ds))
    elif spec.objective == "multiclass":
        checks.append(_check_multiclass_class_balance(ds))
    else:
        checks.append(_check_regression_label_distribution(ds))

    leakage_details: dict[str, Any] | None = None
    if not skip_leakage:
        leakage = _check_label_leakage(ds, spec)
        checks.append(leakage)
        # Always surface the leakage details — even on PASS — so the
        # experiment_run.leakage_report column carries an audit trail of
        # "max correlation we saw" per run.
        leakage_details = dict(leakage.details)
        leakage_details["severity"] = leakage.severity
        leakage_details["message"] = leakage.message

    fingerprint = compute_data_fingerprint(ds)
    verdict = _worst([c.severity for c in checks])
    return IntegrityReport(
        verdict=verdict, checks=checks, data_fingerprint=fingerprint,
        leakage_report=leakage_details,
    )
