"""Phase 4 / Phase E — shadow-prediction analytics.

Joins ``signal_history`` rows to the corresponding label in
``feature_values`` (each model's signal_definition links to the binary
label via the trained spec), then reports:

* Overall metrics — AUC, Brier, log-loss, accuracy@0.5
* In-sample vs out-of-sample subset comparison (training-set rows
  vs the last walk-forward fold's val rows)
* Calibration table — predicted-probability deciles vs actual rate
  of class=1
* Per-month performance drift

The CLI saves a markdown report to ``blackheart-trading-engine/research/`` so the
operator can review verbatim. Stdout shows the same tables for quick
inspection.

Why this is its own module (not part of train.py):

* Operates on already-trained-and-backfilled artifacts. It's
  research-time analytics, not training-time.
* Reuses blackheart-train's deps (sklearn for metrics, psycopg for
  the join) without bloating the training pipeline.
* Output is markdown (operator preference for tables + plain text).

Limitations:

* "val set" is approximated as ``signal_history``'s last
  ``n_val_rows`` (from the artifact). For walk-forward fold 6 this is
  exact; for other walk-forward fold structures it's an approximation.
* Forward-only predictions (Phase D shadow logs from real-time
  inference) aren't yet produced — this module reports on backfilled
  predictions only.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
import pandas as pd
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from .artifacts import read_artifact
from .db import get_connection
from .settings import get_settings

logger = logging.getLogger(__name__)


def _load_predictions_with_labels(
    conn,
    signal_id: UUID,
    label_feature: str,
    label_version: int,
    symbol: str,
) -> pd.DataFrame:
    """Join signal_history -> feature_values on (symbol, ts).

    Returns columns ``[ts, predicted, actual]`` over the intersection
    of timestamps where BOTH a prediction and a realised label exist.
    """
    sql = """
        SELECT sh.ts, sh.value AS predicted, fv.value AS actual
        FROM signal_history sh
        JOIN feature_values fv
          ON  fv.feature_name = %(label_feature)s
          AND fv.version      = %(label_version)s
          AND fv.symbol       = %(symbol)s
          AND fv.ts           = sh.ts
        WHERE sh.signal_id = %(signal_id)s
          AND sh.symbol    = %(symbol)s
        ORDER BY sh.ts ASC
    """
    with conn.cursor() as cur:
        cur.execute(
            sql,
            {
                "signal_id": signal_id,
                "label_feature": label_feature,
                "label_version": label_version,
                "symbol": symbol,
            },
        )
        rows = cur.fetchall()
    if not rows:
        return pd.DataFrame(columns=["ts", "predicted", "actual"])
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(None)
    df["predicted"] = df["predicted"].astype("float64")
    df["actual"] = df["actual"].astype("float64")
    return df


def _binary_metrics(df: pd.DataFrame, threshold: float = 0.5) -> dict[str, float]:
    """AUC + Brier + log-loss + accuracy on a (predicted, actual) frame."""
    if df.empty:
        return {"n": 0}
    y_true = df["actual"].to_numpy()
    y_pred = df["predicted"].to_numpy()
    # Guard against degenerate single-class subsets (AUC undefined).
    unique = np.unique(y_true)
    metrics: dict[str, float] = {
        "n": int(len(df)),
        "actual_class1_rate": float(y_true.mean()),
        "predicted_mean": float(y_pred.mean()),
        "predicted_std": float(y_pred.std()),
        "brier": float(brier_score_loss(y_true, y_pred)),
        "log_loss": float(
            log_loss(y_true, np.clip(y_pred, 1e-15, 1 - 1e-15))
        ),
        "accuracy@0.5": float(((y_pred >= threshold) == (y_true >= 0.5)).mean()),
    }
    if len(unique) == 2:
        metrics["auc"] = float(roc_auc_score(y_true, y_pred))
    else:
        metrics["auc"] = float("nan")
    return metrics


def _calibration_bins(df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """Bin predictions into ``n_bins`` equal-width bins on [0, 1].

    For each bin: count, mean predicted prob, mean actual rate (=
    fraction of actual=1 rows), absolute gap |pred - actual|. A
    well-calibrated model has gaps near zero across all bins.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # right=False so values exactly at 1.0 get assigned to bin n_bins-1
    # rather than overflowing past the last edge.
    df = df.copy()
    df["bin"] = pd.cut(
        df["predicted"], bins=edges, labels=False, include_lowest=True
    )
    grouped = df.groupby("bin", observed=True).agg(
        count=("predicted", "size"),
        mean_pred=("predicted", "mean"),
        actual_rate=("actual", "mean"),
    )
    grouped["abs_gap"] = (grouped["mean_pred"] - grouped["actual_rate"]).abs()
    grouped["bin_range"] = [
        f"[{edges[int(b)]:.2f}, {edges[int(b)+1]:.2f}{'  ]' if int(b)+1==n_bins else ')'}"
        for b in grouped.index
    ]
    return grouped.reset_index()[
        ["bin", "bin_range", "count", "mean_pred", "actual_rate", "abs_gap"]
    ]


def _expected_calibration_error(df: pd.DataFrame, n_bins: int = 10) -> float:
    """ECE — weighted-mean absolute gap across bins, weighted by bin
    population. The standard one-number summary of calibration quality.
    0.0 = perfectly calibrated; higher = worse.
    """
    bins = _calibration_bins(df, n_bins=n_bins)
    if bins.empty or bins["count"].sum() == 0:
        return float("nan")
    weighted = (bins["abs_gap"] * bins["count"]).sum() / bins["count"].sum()
    return float(weighted)


def _per_month_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Performance by calendar month — surfaces signal drift over time."""
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["month"] = df["ts"].dt.to_period("M").astype(str)
    rows = []
    for month, sub in df.groupby("month", observed=True):
        m = _binary_metrics(sub)
        m["month"] = month
        rows.append(m)
    out = pd.DataFrame(rows)
    return out[
        [
            "month", "n", "actual_class1_rate",
            "predicted_mean", "auc", "brier", "log_loss", "accuracy@0.5",
        ]
    ]


def _format_metric_table(metrics: dict[str, Any]) -> str:
    """One-row metric dict -> markdown table."""
    lines = ["| metric | value |", "|---|---|"]
    for k in [
        "n", "actual_class1_rate", "predicted_mean", "predicted_std",
        "auc", "brier", "log_loss", "accuracy@0.5",
    ]:
        v = metrics.get(k)
        if v is None:
            continue
        if isinstance(v, float):
            lines.append(f"| {k} | {v:.6f} |")
        else:
            lines.append(f"| {k} | {v} |")
    return "\n".join(lines)


def _format_df_md(df: pd.DataFrame, *, float_cols: tuple[str, ...] = ()) -> str:
    if df.empty:
        return "_(empty)_"
    df = df.copy()
    for c in float_cols:
        if c in df.columns:
            df[c] = df[c].apply(
                lambda x: f"{x:.4f}" if isinstance(x, (int, float)) and not pd.isna(x) else x
            )
    return df.to_markdown(index=False)


def _parse_iso(s):
    """Parse an iso-ish datetime string from artifact storage."""
    if isinstance(s, datetime):
        return s
    return datetime.fromisoformat(s.replace("Z", "+00:00").split("+")[0])


def analyze_signal(
    signal_name: str,
    *,
    val_rows: int | None = None,  # kept for API stability; unused after fold-aware rewrite
) -> str:
    """Top-level — produce a markdown report for one signal_definition.

    Walk-forward-aware: uses the artifact's stored fold structure
    (artifact['walk_forward']['folds']) to identify exactly which time
    windows each fold's booster validated on. The SAVED booster (= last
    fold's booster) is then evaluated against signal_history for two
    distinct purposes:

    1. **On every fold's test window** — exposes that earlier folds'
       test windows are INSIDE the saved booster's training window
       (so its AUC there is training-set-biased / memorized). Only the
       LAST fold's test window is honestly out-of-sample for the saved
       booster.
    2. **On the post-walk-forward zone** — the rows in
       ``[fold-N.test_end, train_end)`` that the gauntlet leaves
       unvalidated. Surfaces continued signal decay past the last
       gauntlet fold.

    The report contrasts the gauntlet's per-fold AUC (each fold's own
    fresh booster, stored in the artifact) with the saved booster's
    AUC on the same windows (computed here from signal_history). The
    delta tells the operator how much each fold's AUC was inflated by
    memorization when computed from a single saved booster.
    """
    settings = get_settings()

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT sd.signal_id, sd.name AS signal_name, sd.status AS signal_status,
                       sd.horizon, sd.model_id,
                       mr.symbol, mr.interval, mr.purpose, mr.status AS model_status,
                       mr.artifact_sha256
                FROM signal_definition sd
                JOIN model_registry mr ON mr.id = sd.model_id
                WHERE sd.name = %s
                """,
                (signal_name,),
            )
            row = cur.fetchone()
        if row is None:
            raise ValueError(
                f"signal_definition.name={signal_name!r} not found "
                f"(or its model_registry row is missing)."
            )

        content_sha = row["artifact_sha256"]
        artifact = read_artifact(content_sha, settings.artifact_dir)
        label_feature = artifact["label_feature"]
        label_version = artifact["label_version"]

        df = _load_predictions_with_labels(
            conn,
            signal_id=row["signal_id"],
            label_feature=label_feature,
            label_version=label_version,
            symbol=row["symbol"],
        )

    if df.empty:
        return (
            f"# {signal_name} — Phase E shadow analytics\n\n"
            f"_No matched (prediction, label) rows. Run inference backfill "
            f"first, or verify the label feature exists in feature_values._\n"
        )

    df_sorted = df.sort_values("ts").reset_index(drop=True)

    # ── Pull fold structure from the artifact (the source of truth for
    # how the gauntlet validated this model).
    wf = artifact.get("walk_forward") or {}
    folds_info = wf.get("folds") or []
    spec = artifact.get("spec") or {}
    spec_train_start = _parse_iso(spec.get("train_start"))
    spec_train_end = _parse_iso(spec.get("train_end"))

    # The SAVED booster is the last fold's booster. Its training window
    # is the LAST fold's train window — used here to label which
    # historical bars are "memorized" vs "new" for inference purposes.
    if folds_info:
        last_fold = folds_info[-1]
        saved_train_start = _parse_iso(last_fold["train_start"])
        saved_train_end = _parse_iso(last_fold["train_end"])
        last_test_end = _parse_iso(last_fold["test_end"])
    else:
        saved_train_start = spec_train_start
        saved_train_end = spec_train_end
        last_test_end = spec_train_end

    # ── Per-fold breakdown table ──────────────────────────────────
    # For each fold:
    #   - gauntlet_auc: from artifact (each fold's OWN booster)
    #   - saved_booster_auc: from my analytics (the LAST fold's
    #     booster, predicting on this window)
    # Where gauntlet_auc << saved_booster_auc -> the window is
    # INSIDE the saved booster's training set (memorized).
    fold_rows = []
    for f in folds_info:
        ts_start = _parse_iso(f["test_start"])
        ts_end = _parse_iso(f["test_end"])
        sub = df_sorted[
            (df_sorted["ts"] >= ts_start) & (df_sorted["ts"] < ts_end)
        ]
        m = _binary_metrics(sub) if not sub.empty else {"n": 0}
        gauntlet_auc = f.get("metrics", {}).get("auc")
        fold_rows.append(
            {
                "fold": f["fold"],
                "test_window": f"[{ts_start.date()}, {ts_end.date()})",
                "n_test_in_signal_history": m.get("n", 0),
                "gauntlet_auc": gauntlet_auc,
                "saved_booster_auc": m.get("auc"),
                "note": (
                    "INSIDE saved-booster train (memorized)"
                    if ts_end <= saved_train_end
                    else (
                        "LAST FOLD — out-of-sample"
                        if f is folds_info[-1]
                        else "PARTIAL overlap with saved-booster train"
                    )
                ),
            }
        )
    folds_df = pd.DataFrame(fold_rows)

    # ── Post-walk-forward zone ────────────────────────────────────
    post_wf_df = df_sorted[df_sorted["ts"] >= last_test_end]
    post_wf_metrics = (
        _binary_metrics(post_wf_df) if not post_wf_df.empty else {"n": 0}
    )

    # ── Gauntlet val (last fold test) — the honest figure ─────────
    if folds_info:
        last_test_start = _parse_iso(folds_info[-1]["test_start"])
        last_fold_df = df_sorted[
            (df_sorted["ts"] >= last_test_start)
            & (df_sorted["ts"] < last_test_end)
        ]
    else:
        last_fold_df = df_sorted.iloc[0:0]
    last_fold_metrics = (
        _binary_metrics(last_fold_df) if not last_fold_df.empty else {"n": 0}
    )
    last_fold_ece = (
        _expected_calibration_error(last_fold_df)
        if not last_fold_df.empty
        else float("nan")
    )
    post_wf_ece = (
        _expected_calibration_error(post_wf_df)
        if not post_wf_df.empty
        else float("nan")
    )
    last_fold_bins = (
        _calibration_bins(last_fold_df) if not last_fold_df.empty else pd.DataFrame()
    )
    post_wf_bins = (
        _calibration_bins(post_wf_df) if not post_wf_df.empty else pd.DataFrame()
    )

    overall = _binary_metrics(df_sorted)
    ece_overall = _expected_calibration_error(df_sorted)
    monthly = _per_month_metrics(df_sorted)

    sections = [
        f"# {signal_name} — Phase E shadow analytics (v2, fold-aware)",
        "",
        f"- Anchored: {datetime.now().date()}",
        f"- model_id: `{row['model_id']}`",
        f"- content_sha: `{content_sha}`",
        f"- symbol/interval: `{row['symbol']}` / `{row['interval']}`",
        f"- label: `{label_feature}` v{label_version}",
        f"- signal_history rows with realised label: **{len(df_sorted):,}**",
        f"- ts range in signal_history: `{df_sorted['ts'].min()}` -> `{df_sorted['ts'].max()}`",
        "",
        "## Walk-forward fold structure (from artifact)",
        "",
        f"- Spec training window: **{spec_train_start.date()} -> {spec_train_end.date()}**",
        f"- **Saved booster's actual training window** (= last fold's train window): "
        f"**{saved_train_start.date()} -> {saved_train_end.date()}**",
        f"- Gauntlet validated through: **{last_test_end.date()}** "
        f"(last fold's test_end)",
        f"- **Post-walk-forward zone** "
        f"({last_test_end.date()} -> {spec_train_end.date()}, "
        f"~{(spec_train_end - last_test_end).days} days): never validated by gauntlet",
        f"- Gauntlet's reported walk-forward mean AUC: "
        f"**{wf.get('primary_mean', float('nan')):.4f}** "
        f"(median {wf.get('primary_median', float('nan')):.4f}, "
        f"std {wf.get('primary_std', float('nan')):.4f})",
        "",
        "## Per-fold comparison: gauntlet vs saved-booster predictions",
        "",
        "Each fold's gauntlet metric comes from a FRESH booster trained "
        "on that fold's train window. The saved-booster column re-predicts "
        "with the LAST fold's booster on each window — so windows inside "
        "the saved booster's train get inflated AUC (memorization).",
        "",
        _format_df_md(
            folds_df,
            float_cols=("gauntlet_auc", "saved_booster_auc"),
        ),
        "",
        "## Gauntlet val (last fold's test window — the honest OOS figure)",
        "",
        _format_metric_table(last_fold_metrics),
        f"\nECE (10 bins): **{last_fold_ece:.6f}**",
        "",
        "## Post-walk-forward zone (never validated by gauntlet)",
        "",
        _format_metric_table(post_wf_metrics),
        f"\nECE (10 bins): **{post_wf_ece:.6f}**",
        "",
        "Compare to the gauntlet-val metrics above — the post-WF zone "
        "shows how the saved booster performs past the last validated "
        "window. Larger AUC degradation = sharper signal decay = more "
        "urgent need for rolling retrain.",
        "",
        "## Overall metrics (entire signal_history — mixed in-sample / OOS)",
        "",
        _format_metric_table(overall),
        f"\nECE (10 bins): **{ece_overall:.6f}**",
        "",
        "_Caveat_: most of the signal_history (folds 0-4 test windows + "
        "saved-booster train data) is INSIDE the saved booster's training "
        "window, so the overall AUC is heavily inflated by memorization. "
        "Use the per-fold table and gauntlet-val section, not this row, "
        "to judge whether the model has a real edge.",
        "",
        "## Calibration — gauntlet val (last fold)",
        "",
        _format_df_md(
            last_fold_bins,
            float_cols=("mean_pred", "actual_rate", "abs_gap"),
        )
        if not last_fold_bins.empty
        else "_(empty)_",
        "",
        "## Calibration — post-walk-forward zone",
        "",
        _format_df_md(
            post_wf_bins,
            float_cols=("mean_pred", "actual_rate", "abs_gap"),
        )
        if not post_wf_bins.empty
        else "_(empty)_",
        "",
        "## Per-month drift (saved booster predicting through history)",
        "",
        "AUC near 1.0 in early months = booster predicting on its own "
        "training data (memorization, not skill). AUC dropping toward "
        "0.5 in later months = approaching / passing the saved booster's "
        "train boundary; this is where real signal lives.",
        "",
        _format_df_md(
            monthly,
            float_cols=(
                "actual_class1_rate", "predicted_mean",
                "auc", "brier", "log_loss", "accuracy@0.5",
            ),
        ),
        "",
        "---",
        "",
        "## How to read this report",
        "",
        "1. **Use the per-fold table to spot memorization.** A row where "
        "`saved_booster_auc >> gauntlet_auc` is a window inside the "
        "saved booster's training data. The gauntlet column is the "
        "honest figure for that fold; the saved-booster column is "
        "inflated.",
        "2. **The gauntlet's walk-forward mean AUC** in the header is the "
        "honest cross-validated metric. Each fold's AUC contributes one "
        "vote; the mean is the headline figure.",
        "3. **The post-walk-forward zone** is data the gauntlet did NOT "
        "evaluate. Its AUC tells you how the saved booster degrades "
        "past its training boundary. Larger AUC drop = shorter shelf "
        "life of a trained model = more urgent re-train cadence.",
        "4. **The gauntlet has a blind spot**: the ~31 days between the "
        "last fold's test_end and train_end are unvalidated. A future "
        "gauntlet upgrade should add a post-walk-forward holdout gate "
        "to surface decay earlier.",
        "5. **Calibration** (ECE) tells you whether predicted probabilities "
        "match realised rates. 0.0 = perfect; 0.05+ = systematic "
        "miscalibration; affects how a strategy should size on signal.",
        "",
        "## Verdict (interpret in light of the headline number)",
        "",
        f"- Gauntlet WF mean AUC = **{wf.get('primary_mean', float('nan')):.4f}** "
        f"(median {wf.get('primary_median', float('nan')):.4f}, "
        f"std {wf.get('primary_std', float('nan')):.4f})",
        f"- Gauntlet last-fold AUC = **{last_fold_metrics.get('auc', float('nan')):.4f}**",
        f"- Saved-booster post-WF AUC = **{post_wf_metrics.get('auc', float('nan')):.4f}**",
        "",
        "If WF mean is in the 0.55-0.60 range, signal is real but small "
        "(\"useful for modulation, not a standalone strategy\"). Re-train "
        "cadence: monthly or per-fold so live serving always uses a model "
        "whose decay zone hasn't started yet.",
    ]

    return "\n".join(sections)


# ── CLI ──────────────────────────────────────────────────────────────────


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="blackheart-train-analyze-shadow",
        description="Phase E shadow-prediction analytics for a signal_definition.",
    )
    p.add_argument(
        "--signal", type=str, required=True,
        help="signal_definition.name (e.g. regime_btc_v3)",
    )
    p.add_argument(
        "--output", type=Path, default=None,
        help="Write the markdown report to this path. If omitted, the "
        "report only goes to stdout.",
    )
    p.add_argument(
        "--val-rows", type=int, default=None,
        help="Override the last-N-rows count treated as val set. Default: "
        "n_val_rows from the artifact.",
    )
    return p


def main() -> int:
    args = _build_arg_parser().parse_args()
    report = analyze_signal(args.signal, val_rows=args.val_rows)
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"\n[wrote report to {args.output}]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
