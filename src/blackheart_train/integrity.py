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
7. **data_fingerprint** — sha256 over the contiguous bytes of
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


# ── Individual checks ──────────────────────────────────────────────────────


_MIN_ROWS_DEFAULT = 1_000
_BINARY_MIN_CLASS_PCT = 0.15
_TRAIN_VAL_BALANCE_TOLERANCE_PCT = 0.10
_REGRESSION_OUTLIER_STD = 6.0
_CONSTANT_FEATURE_STD = 1e-12
_STALE_TAIL_HOURS = 168   # 7 days
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


# ── Fingerprint ────────────────────────────────────────────────────────────


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


def check_dataset(
    ds: LoadedDataset,
    spec: ModelSpec,
    *,
    min_rows: int = _MIN_ROWS_DEFAULT,
) -> IntegrityReport:
    """Run all integrity checks and return an aggregated report.

    Caller is expected to inspect ``report.verdict``:
      * ``FAIL`` → raise :class:`IntegrityError` (training would be
        meaningless on this data)
      * ``WARN`` → log + persist in the artifact payload
      * ``PASS`` → proceed quietly
    """
    checks: list[CheckResult] = [
        _check_min_rows(ds, min_rows),
        _check_constant_features(ds),
        _check_stale_tail(ds, spec),
    ]
    if spec.objective == "binary":
        checks.append(_check_binary_class_balance(ds))
        checks.append(_check_train_val_class_balance(ds, spec))
    elif spec.objective == "multiclass":
        checks.append(_check_multiclass_class_balance(ds))
    else:
        checks.append(_check_regression_label_distribution(ds))

    fingerprint = compute_data_fingerprint(ds)
    verdict = _worst([c.severity for c in checks])
    return IntegrityReport(verdict=verdict, checks=checks, data_fingerprint=fingerprint)
