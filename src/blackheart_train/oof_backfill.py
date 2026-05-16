"""Phase 4 / Phase D-execution — OOF prediction backfill into ``signal_history``.

Reads an artifact pickle produced by ``blackheart-train --walk-forward``
(after V79 stage A wired in OOF capture), iterates each fold's
``oof_predictions`` + ``oof_timestamps``, and UPSERTs them into
``signal_history``. The Java backtest path then reads them point-in-time
via ``mlSignalService.getAt(name, symbol, ts)``.

Why this is its own module (not bolted onto blackheart-ingest's
inference worker):

* The inference worker is a forward-only stream — it predicts one bar at
  a time as bars close. Backfilling 6 months of historical bars by
  replaying that loop would require time-travelling the feature store
  and is operationally pointless when the walk-forward training already
  computed the predictions point-in-time-correctly.
* Walk-forward CV's out-of-fold predictions ARE the honest historical
  signal: fold ``k``'s predictions come from a model trained only on
  data up to ``fold[k].train_end``. Replaying them into signal_history
  preserves that point-in-time discipline.
* Keeping this in blackheart-train means we reuse :func:`read_artifact`
  and the registry without depending on blackheart-ingest's runtime.

Point-in-time contract:

* ``oof_timestamps[i]`` is the bar's close time.
* ``oof_predictions[i]`` was produced by a model that had seen NO data
  past ``fold.train_end`` (≤ ``test_start - embargo_days``).
* When the Java gate later asks "what did the model say at bar T?", a
  row with ``ts = T`` returns the correct value.

Limitations:

* **Binary specs only.** Multiclass predictions are 2D arrays of
  per-class probabilities — backfill emits an error rather than guessing
  which class to store as ``value``. Regime / flow models are binary
  today; revisit when a multiclass signal lands.
* **Skipped folds are skipped here too.** A fold with empty
  ``oof_predictions`` (skipped at training time) gets no row in
  signal_history — there's no honest prediction to store.
* **Source = ``'historical_replay'``** per the V66 CHECK constraint.
  Distinguishable in audit queries from forward-stream rows.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from .artifacts import read_artifact
from .db import get_connection
from .settings import get_settings

logger = logging.getLogger(__name__)


_DEFAULT_BY = "blackheart-train:oof_backfill"
_SOURCE = "historical_replay"


def _parse_iso(s: str) -> datetime:
    """ISO-8601 string -> naive UTC datetime. Walk-forward writes ISO
    strings without timezones; signal_history.ts is TIMESTAMPTZ and
    psycopg will coerce a naive datetime to the session's UTC zone
    (we SET TIME ZONE 'UTC' in :func:`db.get_connection`).
    """
    return datetime.fromisoformat(s)


def _lookup_signal_id(conn: Any, signal_name: str) -> UUID:
    """signal_definition.name -> signal_definition.signal_id (UUID).

    Raises if the signal_definition row doesn't exist — the operator
    must register the signal before backfilling predictions for it.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT signal_id FROM signal_definition WHERE name = %s",
            (signal_name,),
        )
        row = cur.fetchone()
    if row is None:
        raise ValueError(
            f"signal_definition.name={signal_name!r} not found — "
            f"register the signal first via /signals/register, then re-run backfill"
        )
    # dict_row factory → row is a dict
    return row["signal_id"]


def _validate_binary_predictions(folds: list[dict[str, Any]]) -> None:
    """Refuse 2D prediction arrays (multiclass) — we don't know which
    class to write to ``value``. Raised before any DB writes so we
    don't leave partial data.
    """
    for f in folds:
        preds = f.get("oof_predictions") or []
        if not preds:
            continue
        first = preds[0]
        if isinstance(first, list):
            raise ValueError(
                f"fold {f.get('fold')} has 2D predictions "
                f"(shape per row = {len(first)}). This script backfills "
                f"BINARY signals only; multiclass backfill is not yet "
                f"implemented. Fix: extend oof_backfill.py to pick a "
                f"specific class column per the signal's contract."
            )


def _build_rows(
    folds: list[dict[str, Any]],
    *,
    signal_id: UUID,
    symbol: str,
    artifact_sha: str,
) -> list[dict[str, Any]]:
    """Flatten per-fold OOF arrays into one row list. Skips NaN preds
    and folds without OOF (training-time skipped folds).
    """
    rows: list[dict[str, Any]] = []
    for f in folds:
        preds = f.get("oof_predictions") or []
        tss = f.get("oof_timestamps") or []
        if not preds:
            continue
        if len(preds) != len(tss):
            raise ValueError(
                f"fold {f.get('fold')} oof shape mismatch: "
                f"{len(preds)} preds vs {len(tss)} timestamps"
            )
        fold_meta = {
            "fold": f.get("fold"),
            "test_start": f.get("test_start"),
            "test_end": f.get("test_end"),
            "train_end": f.get("train_end"),
            "artifact_content_sha256": artifact_sha,
            "source_run": "oof_backfill",
        }
        for value, ts_iso in zip(preds, tss, strict=True):
            if value is None or (isinstance(value, float) and math.isnan(value)):
                continue
            rows.append({
                "signal_id": signal_id,
                "symbol": symbol,
                "ts": _parse_iso(ts_iso),
                "value": float(value),
                # Binary single-model: confidence == value (probability of
                # class=1). For the Java gate's threshold checks the value
                # IS the confidence.
                "confidence": float(value),
                "meta": fold_meta,
            })
    return rows


def _upsert(rows: list[dict[str, Any]], conn: Any) -> int:
    """Batched UPSERT into signal_history. Mirrors blackheart-ingest's
    persist.persist_predictions SQL so backfill and forward inference
    produce row-format-identical entries.
    """
    if not rows:
        return 0
    produced_at = datetime.now(timezone.utc)
    params = []
    for r in rows:
        params.append({
            "signal_id": r["signal_id"],
            "symbol": r["symbol"],
            "ts": r["ts"],
            "value": r["value"],
            "confidence": r["confidence"],
            "produced_at": produced_at,
            "source": _SOURCE,
            "meta": json.dumps(r["meta"] or {}, default=str),
            "by": _DEFAULT_BY,
        })
    sql = """
        INSERT INTO signal_history (
            signal_id, symbol, ts, value, confidence,
            produced_at, source, meta, created_by, updated_by
        ) VALUES (
            %(signal_id)s, %(symbol)s, %(ts)s, %(value)s, %(confidence)s,
            %(produced_at)s, %(source)s, %(meta)s::jsonb,
            %(by)s, %(by)s
        )
        ON CONFLICT (signal_id, symbol, ts) DO UPDATE SET
            value = EXCLUDED.value,
            confidence = EXCLUDED.confidence,
            produced_at = EXCLUDED.produced_at,
            source = EXCLUDED.source,
            meta = EXCLUDED.meta,
            updated_time = NOW(),
            updated_by = EXCLUDED.updated_by
    """
    with conn.cursor() as cur:
        cur.executemany(sql, params)
    conn.commit()
    return len(params)


def backfill_from_artifact(
    artifact_sha: str,
    *,
    signal_name: str,
    symbol: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run the backfill end-to-end. Returns a summary dict (per-fold
    counts + total).

    Designed for both CLI + future API use; never prints. Caller
    formats output.
    """
    settings = get_settings()
    artifact = read_artifact(artifact_sha, settings.artifact_dir)
    wf = artifact.get("walk_forward") or {}
    folds = wf.get("folds") or []
    if not folds:
        raise ValueError(
            f"artifact {artifact_sha[:12]} has no walk_forward.folds — "
            f"re-train with --walk-forward and ensure stage A (OOF capture) is wired"
        )

    spec = artifact.get("spec") or {}
    spec_objective = spec.get("objective")
    if spec_objective != "binary":
        raise ValueError(
            f"spec.objective={spec_objective!r}; this script handles binary only"
        )

    _validate_binary_predictions(folds)

    summary: dict[str, Any] = {
        "artifact_content_sha256": artifact_sha,
        "spec_name": wf.get("spec_name"),
        "signal_name": signal_name,
        "symbol": symbol,
        "dry_run": dry_run,
        "per_fold": [],
        "total_rows": 0,
        "total_skipped_nan": 0,
    }

    with get_connection() as conn:
        signal_id = _lookup_signal_id(conn, signal_name)
        summary["signal_id"] = str(signal_id)

        rows = _build_rows(
            folds,
            signal_id=signal_id,
            symbol=symbol,
            artifact_sha=artifact_sha,
        )

        # Per-fold breakdown — what we'd write.
        for f in folds:
            preds = f.get("oof_predictions") or []
            tss = f.get("oof_timestamps") or []
            nan_count = sum(
                1 for v in preds
                if v is None or (isinstance(v, float) and math.isnan(v))
            )
            usable = len(preds) - nan_count
            summary["per_fold"].append({
                "fold": f.get("fold"),
                "test_start": f.get("test_start"),
                "test_end": f.get("test_end"),
                "n_test": f.get("n_test"),
                "n_preds": len(preds),
                "n_timestamps": len(tss),
                "n_nan_skipped": nan_count,
                "n_to_upsert": usable,
                "skipped_reason": f.get("skipped_reason"),
            })
            summary["total_skipped_nan"] += nan_count
        summary["total_rows"] = len(rows)

        if dry_run:
            logger.info(
                "DRY RUN | signal=%s symbol=%s would upsert %d rows; no DB writes",
                signal_name, symbol, len(rows),
            )
            return summary

        written = _upsert(rows, conn)
        logger.info(
            "backfill complete | signal=%s symbol=%s rows_upserted=%d source=%s",
            signal_name, symbol, written, _SOURCE,
        )
        summary["rows_upserted"] = written

    return summary


# ── CLI ──────────────────────────────────────────────────────────────────


def _format_summary(summary: dict[str, Any]) -> str:
    """Human-readable summary table. Markdown so it copy-pastes cleanly
    into reports (matches the operator's preference for tables).
    """
    lines: list[str] = []
    lines.append(f"# OOF backfill — {summary['signal_name']} @ {summary['symbol']}")
    lines.append("")
    lines.append(f"- artifact_content_sha256: `{summary['artifact_content_sha256']}`")
    lines.append(f"- spec_name: `{summary['spec_name']}`")
    lines.append(f"- signal_id: `{summary.get('signal_id', '?')}`")
    lines.append(f"- dry_run: **{summary['dry_run']}**")
    lines.append("")
    lines.append("| fold | test_window | n_test | n_preds | n_nan | n_to_upsert | skipped |")
    lines.append("|-----:|:------------|-------:|--------:|------:|------------:|:--------|")
    for r in summary["per_fold"]:
        window = f"[{r.get('test_start')[:10]}, {r.get('test_end')[:10]})"
        lines.append(
            f"| {r['fold']} | {window} | {r['n_test']} | "
            f"{r['n_preds']} | {r['n_nan_skipped']} | {r['n_to_upsert']} | "
            f"{r.get('skipped_reason') or '-'} |"
        )
    lines.append("")
    lines.append(f"**Total rows {'(would be) ' if summary['dry_run'] else ''}upserted: "
                 f"{summary.get('rows_upserted', summary['total_rows'])}**")
    if summary["total_skipped_nan"] > 0:
        lines.append(f"**Total NaN preds skipped: {summary['total_skipped_nan']}**")
    return "\n".join(lines)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="blackheart-train-backfill-oof",
        description=(
            "Backfill walk-forward OOF predictions from an artifact into "
            "signal_history. Point-in-time-correct by construction — each "
            "fold's predictions come from a model trained only on data up "
            "to that fold's train_end. Phase 4 / D-execution backtest "
            "validation prerequisite."
        ),
    )
    p.add_argument(
        "--artifact-sha", type=str, required=True,
        help="content_sha256 of the artifact to read. Find via `ls "
        "blackheart-train/artifacts/<sha[:2]>/`.",
    )
    p.add_argument(
        "--signal", type=str, required=True,
        help="signal_definition.name to write under (e.g. regime_btc_v3). "
        "Must already exist in signal_definition.",
    )
    p.add_argument(
        "--symbol", type=str, default="BTCUSDT",
        help="Trading symbol the predictions apply to. Default BTCUSDT.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Report what would be upserted without touching the DB.",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Enable INFO-level logging to stderr.",
    )
    return p


def main() -> int:
    args = _build_arg_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        stream=sys.stderr,
    )
    try:
        summary = backfill_from_artifact(
            args.artifact_sha,
            signal_name=args.signal,
            symbol=args.symbol,
            dry_run=args.dry_run,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}), file=sys.stderr)
        return 1
    print(_format_summary(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
