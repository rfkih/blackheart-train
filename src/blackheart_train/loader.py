"""Load a (features, label) matrix for one ModelSpec.

The DB has features in long form in ``feature_values`` (key:
``(feature_name, version, symbol, interval, ts)``). A bar-level model
needs them wide: one row per ``ts``, one column per feature, plus the
label vector.

PIT discipline (blueprint § 4 + features/compute.py convention):

* Per-bar features and labels are stamped at ``ts = bar.start_time``.
  The label at ``ts=T`` is computed from future bars ``T+1..T+horizon``.
  Input features at ``ts=T`` reflect data complete at bar T's close,
  which is known by ``T+interval``. Aligning label@T with inputs@T is
  therefore safe at the trading JVM's decide-at-bar-close convention:
  at decision moment T+interval, both feature@T and label@T's reference
  price (close[T]) are observable.

* Global features (macro, sentiment composites) are stamped at their
  publisher's event_time and arrive at lower cadence (daily, monthly).
  We forward-fill them onto the bar grid via :func:`pandas.merge_asof`
  with a per-feature time-based cap (``feature_registry.max_ffill_age_hours``)
  so a stale value can't propagate indefinitely. ``merge_asof`` caps by
  clock time, not by row count — robust to off-grid publish timestamps.

This module returns a ``LoadedDataset`` whose ``X`` and ``y`` are aligned
by ``ts`` and free of NaN. Rows where any input or the label is NaN are
dropped (LightGBM handles NaN, but for the M5a smoke we want a clean
matrix so row-count metrics are interpretable). Per-feature non-null
counts are surfaced so callers can decide whether to drop low-coverage
features in a follow-up training run.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd
import psycopg

from .db import get_connection
from .derived_features import (
    DERIVED_FEATURES,
    DERIVED_LABELS,
    compute_derived_input,
    compute_derived_label,
    fetch_market_data_bundle,
    required_symbols_for,
)
from .specs import EXCLUDED_FROM_INPUTS, ModelSpec

logger = logging.getLogger(__name__)


@dataclass
class _FeatureMatrixCache:
    """Internal cache record for the input-feature side of a dataset.

    All three M5 sub-models share the same training window and the same
    eligible feature set — only their label differs. By keying the cache
    on ``(symbol, interval, train_start, train_end)`` we let
    ``cli --model all`` fetch the feature matrix once and reuse it
    across the three specs, paying DB cost once instead of three times.
    The cache is opt-in: pass ``feature_cache=`` to ``load_dataset`` to
    enable. Tests and one-shot CLI calls leave it unset.
    """

    cols: dict[str, pd.Series]
    feature_names: tuple[str, ...]
    per_feature_non_null: dict[str, int]
    bar_index: pd.DatetimeIndex


@dataclass
class LoadedDataset:
    """Result of :func:`load_dataset`.

    ``X`` and ``y`` are aligned by index (the bar's ``ts``) and contain
    no NaN. ``feature_names`` is the deterministic column order of ``X``
    — preserved across train / predict so the trained model's feature
    order matches at inference.

    Bar-slot accounting:
      ``n_bar_slots_total`` counts every position on the canonical bar
      grid for the spec's window (e.g. 17 months × 1h ≈ 12,696 slots).
      ``n_bar_slots_dropped_nan`` is the count removed by the NaN filter
      — driven mostly by short-history features whose ffill cap expires
      in the older part of the window. Per-feature non-null fractions
      identify the dragger.
    """

    X: pd.DataFrame
    y: pd.Series
    feature_names: tuple[str, ...]
    n_bar_slots_total: int
    n_bar_slots_dropped_nan: int
    per_feature_non_null: dict[str, int] = field(default_factory=dict)
    per_feature_pct_non_null: dict[str, float] = field(default_factory=dict)
    label_feature: str = ""
    label_version: int = 0


# ── Registry queries ─────────────────────────────────────────────────────────


def _list_input_features(
    conn: psycopg.Connection,
    extra_excluded: tuple[str, ...] | frozenset[str] = (),
) -> list[dict[str, Any]]:
    """Return all registered features that are eligible inputs.

    Eligible = ``status='registered'`` AND not in
    :data:`EXCLUDED_FROM_INPUTS` AND not in ``extra_excluded`` AND
    ``label_direction <> 'forward'``.

    Two filters with non-overlapping responsibilities:

    * ``label_direction != 'forward'`` is the **primary** exclusion —
      schema-based, automatically catches every label (current and
      future) by reading the registry's own metadata. Originally added
      2026-05-16 to close the V77 foot-gun where ``label_regime_risk_on_24h``
      was graduated to the registry without the loader knowing it was a
      label (regime_btc_v3 trained on its own label → AUC=1.0).
    * ``EXCLUDED_FROM_INPUTS`` covers **non-label** features that are
      deferred for non-leakage reasons (Tier-C source-capped features
      with <30 days of history). After the H9 cleanup (2026-05-16)
      this set no longer contains labels — schema-based filter is
      sufficient. See :data:`EXCLUDED_FROM_INPUTS` for the residual
      entries and their motivation.
    * ``extra_excluded`` is the spec-scoped exclusion (2026-05-21 Path C).
      Used by directional_eth_1h_v1 to opt out of macro features
      (fear_greed_value, stablecoin_supply_*, eth_btc_ratio_*) so the
      trained artifact's feature_set stays sidecar-servable — those
      features are persisted at symbol='', interval='' (daily global
      cadence) which blackheart-inference's exact-match
      fetch_per_bar_values_at_ts cannot resolve at bar-cadence ts.
    """
    combined_excluded = list(set(EXCLUDED_FROM_INPUTS) | set(extra_excluded))
    sql = """
        SELECT feature_name, version, family, max_ffill_age_hours,
               symbols, intervals, ffill_policy
        FROM feature_registry
        WHERE status = 'registered'
          AND feature_name <> ALL(%(excluded)s)
          AND (label_direction IS NULL OR label_direction <> 'forward')
        ORDER BY feature_name, version
    """
    with conn.cursor() as cur:
        cur.execute(sql, {"excluded": combined_excluded})
        return list(cur.fetchall())


# ── Value queries ────────────────────────────────────────────────────────────


def _read_per_bar_feature(
    conn: psycopg.Connection,
    feature_name: str,
    version: int,
    symbol: str,
    interval: str,
    start: datetime,
    end: datetime,
) -> pd.Series:
    """Read one per-bar feature for the (symbol, interval) the spec wants.

    Returns a Series indexed by ts with ``name=feature_name``. Empty if
    no rows in the window.
    """
    sql = """
        SELECT ts, value
        FROM feature_values
        WHERE feature_name = %(name)s
          AND version = %(version)s
          AND symbol = %(symbol)s
          AND interval = %(interval)s
          AND ts >= %(start)s
          AND ts < %(end)s
        ORDER BY ts ASC
    """
    with conn.cursor() as cur:
        cur.execute(sql, {
            "name": feature_name, "version": version,
            "symbol": symbol, "interval": interval,
            "start": start, "end": end,
        })
        rows = cur.fetchall()
    if not rows:
        return pd.Series(dtype="float64", name=feature_name)
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(None)
    return pd.Series(df["value"].astype("float64").values, index=df["ts"], name=feature_name)


def _read_per_bar_features_batch(
    conn: psycopg.Connection,
    metas: list[dict[str, Any]],
    symbol: str,
    interval: str,
    start: datetime,
    end: datetime,
) -> dict[tuple[str, int], pd.Series]:
    """Bulk-read every per-bar feature in ``metas`` for one (symbol, interval).

    One DB round-trip regardless of feature count — replaces the per-feature
    SELECT loop that used to fan out to ~50 round-trips per spec load.
    """
    if not metas:
        return {}
    names = [m["feature_name"] for m in metas]
    sql = """
        SELECT feature_name, version, ts, value
        FROM feature_values
        WHERE feature_name = ANY(%(names)s)
          AND symbol = %(symbol)s
          AND interval = %(interval)s
          AND ts >= %(start)s
          AND ts < %(end)s
        ORDER BY feature_name, version, ts ASC
    """
    with conn.cursor() as cur:
        cur.execute(sql, {
            "names": names,
            "symbol": symbol, "interval": interval,
            "start": start, "end": end,
        })
        rows = cur.fetchall()
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(None)
    out: dict[tuple[str, int], pd.Series] = {}
    for (name, version), group in df.groupby(["feature_name", "version"], sort=False):
        out[(name, int(version))] = pd.Series(
            group["value"].astype("float64").values, index=group["ts"], name=name
        )
    return out


def _read_global_features_batch(
    conn: psycopg.Connection,
    metas: list[dict[str, Any]],
    start: datetime,
    end: datetime,
) -> dict[tuple[str, int], pd.Series]:
    """Bulk-read every global (symbol='', interval='') feature in ``metas``."""
    if not metas:
        return {}
    names = [m["feature_name"] for m in metas]
    sql = """
        SELECT feature_name, version, ts, value
        FROM feature_values
        WHERE feature_name = ANY(%(names)s)
          AND symbol = ''
          AND interval = ''
          AND ts >= %(start)s
          AND ts < %(end)s
        ORDER BY feature_name, version, ts ASC
    """
    with conn.cursor() as cur:
        cur.execute(sql, {
            "names": names,
            "start": start, "end": end,
        })
        rows = cur.fetchall()
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(None)
    out: dict[tuple[str, int], pd.Series] = {}
    for (name, version), group in df.groupby(["feature_name", "version"], sort=False):
        out[(name, int(version))] = pd.Series(
            group["value"].astype("float64").values, index=group["ts"], name=name
        )
    return out


def _project_series_to_grid(
    series: pd.Series,
    bar_index: pd.DatetimeIndex,
    cap_hours: int | None,
    feature_name: str,
) -> pd.Series:
    """Project a sparser source Series onto ``bar_index`` via backward
    merge_asof with a time-based cap. Used both by the cross-interval ffill
    path (1h source → 15m grid) and conceptually by the global ffill path
    (lower-cadence publish → bar grid). Extracted so the batched feature
    loader can hand off after a single DB read.
    """
    if series.empty:
        return pd.Series(index=bar_index, dtype="float64", name=feature_name)
    df_grid = pd.DataFrame({"ts": bar_index})
    df_src = series.to_frame("value").rename_axis("ts").reset_index().sort_values("ts")
    tolerance = pd.Timedelta(hours=cap_hours) if cap_hours else None
    merged = pd.merge_asof(
        df_grid, df_src, on="ts", direction="backward", tolerance=tolerance,
    )
    result = merged.set_index("ts")["value"]
    result.name = feature_name
    return result


# ── Alignment ───────────────────────────────────────────────────────────────


def _bar_grid(start: datetime, end: datetime, interval: str) -> pd.DatetimeIndex:
    """The canonical hourly (or N-minute) grid the model trains on."""
    freq_map = {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h"}
    if interval not in freq_map:
        raise ValueError(f"Unsupported interval '{interval}'. Add to freq_map in loader.py.")
    return pd.date_range(start=start, end=end, freq=freq_map[interval], inclusive="left")


def _ffill_global_to_grid(
    series: pd.Series,
    bar_index: pd.DatetimeIndex,
    cap_hours: int | None,
) -> pd.Series:
    """Forward-fill a global feature onto ``bar_index`` with a time-based cap.

    Uses ``pd.merge_asof`` with ``direction='backward'`` and an explicit
    ``tolerance``. For each bar slot we take the most recent series
    value whose ts is ``<= slot`` and within ``cap_hours``. Older values
    become NaN — no silent propagation of stale data.

    Why merge_asof over reindex+ffill(limit=N): ``ffill(limit=N)`` counts
    rows, not hours. Works for daily-on-hourly when timestamps coincide,
    but breaks the moment a global feature publishes mid-hour. merge_asof
    is time-aware and survives that case.
    """
    if series.empty:
        return pd.Series(index=bar_index, dtype="float64", name=series.name)

    df_series = (
        series.to_frame("value")
        .rename_axis("ts")
        .reset_index()
        .sort_values("ts")
    )
    df_grid = pd.DataFrame({"ts": bar_index})
    tolerance = pd.Timedelta(hours=cap_hours) if cap_hours else None

    merged = pd.merge_asof(
        df_grid,
        df_series,
        on="ts",
        direction="backward",
        tolerance=tolerance,
    )
    result = merged.set_index("ts")["value"]
    result.name = series.name
    return result


# ── Assembly ────────────────────────────────────────────────────────────────


def _fetch_feature_matrix(conn: psycopg.Connection, spec: ModelSpec) -> _FeatureMatrixCache:
    """Read every eligible input feature for ``spec``'s window into a
    cache record. No label fetch and no NaN drop — those are the parts
    that vary per sub-model, so they stay in :func:`load_dataset`.

    DB cost: two round-trips (per-bar + global) regardless of feature count.
    """
    inputs_meta = _list_input_features(conn, spec.extra_excluded_features)
    bar_index = _bar_grid(spec.train_start, spec.train_end, spec.interval)

    per_bar_metas: list[dict[str, Any]] = []
    global_metas: list[dict[str, Any]] = []
    for meta in inputs_meta:
        symbols = meta["symbols"] or []
        intervals = meta["intervals"] or []
        is_per_bar = bool(symbols) and bool(intervals)
        if is_per_bar:
            if spec.symbol in symbols and spec.interval in intervals:
                per_bar_metas.append(meta)
        else:
            global_metas.append(meta)

    per_bar_data = _read_per_bar_features_batch(
        conn, per_bar_metas, spec.symbol, spec.interval,
        spec.train_start, spec.train_end,
    )
    global_data = _read_global_features_batch(
        conn, global_metas, spec.train_start, spec.train_end,
    )

    feature_names: list[str] = []
    cols: dict[str, pd.Series] = {}
    per_feature_non_null: dict[str, int] = {}

    for meta in per_bar_metas:
        name = meta["feature_name"]
        version = meta["version"]
        series = per_bar_data.get((name, version), pd.Series(dtype="float64", name=name))
        aligned = series.reindex(bar_index)
        non_null = int(aligned.notna().sum())
        if non_null == 0:
            logger.warning("Feature %s v%d is all-NaN on the bar grid — skipping", name, version)
            continue
        feature_names.append(name)
        cols[name] = aligned
        per_feature_non_null[name] = non_null

    for meta in global_metas:
        name = meta["feature_name"]
        version = meta["version"]
        max_age_h = meta["max_ffill_age_hours"]
        ffill_policy = meta["ffill_policy"]
        series = global_data.get((name, version), pd.Series(dtype="float64", name=name))
        if series.empty:
            logger.warning("Global feature %s v%d has no rows in window — skipping", name, version)
            continue
        if ffill_policy == "last_value":
            cap = max_age_h if max_age_h else 24 * 7
            aligned = _ffill_global_to_grid(series, bar_index, cap_hours=cap)
        else:
            aligned = series.reindex(bar_index)
        non_null = int(aligned.notna().sum())
        if non_null == 0:
            logger.warning("Feature %s v%d is all-NaN on the bar grid — skipping", name, version)
            continue
        feature_names.append(name)
        cols[name] = aligned
        per_feature_non_null[name] = non_null

    return _FeatureMatrixCache(
        cols=cols,
        feature_names=tuple(feature_names),
        per_feature_non_null=per_feature_non_null,
        bar_index=bar_index,
    )


@dataclass
class FeatureCache:
    """Process-local cache for repeated ``load_dataset`` calls.

    Two cache dicts, each opt-in and only consulted/populated when the
    cache is threaded into the loader:

    * ``feature_matrices`` — the wide input matrix keyed by
      ``(symbol, interval, train_start, train_end)``. All three M5
      sub-models share that key, so ``--model all`` pays the DB cost
      once across the three labels.
    * ``market_bundles`` — the OHLCV bundle for derived features/labels,
      keyed by ``(required_symbols, interval, train_start, train_end)``.
      Same idea: shared across specs whose derived inputs need the same
      symbols at the same interval.
    """

    feature_matrices: dict[tuple[str, str, datetime, datetime], _FeatureMatrixCache] = field(default_factory=dict)
    market_bundles: dict[tuple[tuple[str, ...], str, datetime, datetime], dict[str, pd.DataFrame]] = field(default_factory=dict)


def _get_or_fetch_market_bundle(
    conn: psycopg.Connection,
    required: tuple[str, ...],
    interval: str,
    start: datetime,
    end: datetime,
    cache: FeatureCache | None,
) -> dict[str, pd.DataFrame]:
    key = (required, interval, start, end)
    if cache is not None and key in cache.market_bundles:
        logger.info("market bundle cache HIT | key=%s", key)
        return cache.market_bundles[key]
    bundle = fetch_market_data_bundle(conn, required, interval, start, end)
    if cache is not None:
        cache.market_bundles[key] = bundle
        logger.info("market bundle cache MISS, stored | key=%s", key)
    return bundle


def load_dataset(
    spec: ModelSpec,
    *,
    feature_cache: FeatureCache | None = None,
) -> LoadedDataset:
    """Load X, y for ``spec``. See module docstring for PIT discipline.

    Drops any row where any input is NaN OR the label is NaN. Records
    per-feature non-null counts (pre-NaN-drop) so the caller can see
    which feature is dragging coverage. Raises ``ValueError`` if the
    resulting matrix is empty.

    Optional ``feature_cache`` — when not None, both the input feature
    matrix AND the market_data bundle are memoised. The label is always
    fetched fresh (per-spec). This lets ``--model all`` pay the input-
    side DB cost once and reuse across the three labels.
    """
    cache_key = (spec.symbol, spec.interval, spec.train_start, spec.train_end)
    needs_market_data = bool(spec.derived_features) or spec.label_feature in DERIVED_LABELS
    with get_connection() as conn:
        if feature_cache is not None and cache_key in feature_cache.feature_matrices:
            fmc = feature_cache.feature_matrices[cache_key]
            logger.info("feature matrix cache HIT | key=%s", cache_key)
        else:
            fmc = _fetch_feature_matrix(conn, spec)
            if feature_cache is not None:
                feature_cache.feature_matrices[cache_key] = fmc
                logger.info("feature matrix cache MISS, stored | key=%s", cache_key)

        if not fmc.cols:
            raise ValueError("No input features produced any rows on the bar grid.")

        market_bundle: dict[str, pd.DataFrame] | None = None
        if needs_market_data:
            required = required_symbols_for(spec)
            market_bundle = _get_or_fetch_market_bundle(
                conn, required, spec.interval,
                spec.train_start, spec.train_end, feature_cache,
            )

        if spec.label_feature in DERIVED_LABELS:
            if market_bundle is None:
                raise RuntimeError("derived label requested but market_bundle missing")
            y_series_full = compute_derived_label(spec.label_feature, market_bundle)
            y_aligned = y_series_full.reindex(fmc.bar_index)
        else:
            y_series = _read_per_bar_feature(
                conn, spec.label_feature, spec.label_version,
                spec.symbol, spec.interval,
                spec.train_start, spec.train_end,
            )
            y_aligned = y_series.reindex(fmc.bar_index)

    # Copy the cached per-feature counts so adding entries can't mutate
    # the cache by accident.
    per_feature_non_null = dict(fmc.per_feature_non_null)
    cols = dict(fmc.cols)   # avoid mutating cache
    feature_names_list = list(fmc.feature_names)

    # Add derived input features after registry-side ones. They appear
    # at the end of feature_names so the registry order is preserved
    # for existing v1 specs.
    if spec.derived_features:
        if market_bundle is None:
            raise RuntimeError("derived features requested but market_bundle missing")
        # Defensive: derived names must not collide with registry names
        # (would produce duplicate columns and silent confusion in
        # downstream consumers reading by column name).
        registry_names = set(feature_names_list)
        collisions = [
            n for n in spec.derived_features if n in registry_names
        ]
        if collisions:
            raise ValueError(
                f"derived feature names collide with registry feature names: "
                f"{collisions}. Rename the derived feature or remove the "
                f"registry entry before training."
            )
        for feat_name in spec.derived_features:
            if feat_name not in DERIVED_FEATURES:
                raise KeyError(f"unknown derived feature: {feat_name!r}")
            series = compute_derived_input(feat_name, market_bundle)
            aligned = series.reindex(fmc.bar_index)
            non_null = int(aligned.notna().sum())
            if non_null == 0:
                logger.warning(
                    "derived feature %s is all-NaN on the bar grid — skipping",
                    feat_name,
                )
                continue
            cols[feat_name] = aligned
            feature_names_list.append(feat_name)
            per_feature_non_null[feat_name] = non_null

    per_feature_non_null[spec.label_feature] = int(y_aligned.notna().sum())

    bar_slots_total = len(fmc.bar_index)
    X = pd.DataFrame(cols)[feature_names_list]
    joined = X.assign(__y__=y_aligned)
    cleaned = joined.dropna()
    n_dropped = bar_slots_total - len(cleaned)
    if cleaned.empty:
        raise ValueError(
            f"All {bar_slots_total} bar slots dropped after NaN filter "
            f"(features={len(feature_names_list)}, label={spec.label_feature}). "
            f"Per-feature non-null counts: {per_feature_non_null}"
        )

    X_out = cleaned[feature_names_list]
    y_out = cleaned["__y__"].rename("y")
    per_feature_pct = {
        k: round(v / bar_slots_total, 4) for k, v in per_feature_non_null.items()
    }
    return LoadedDataset(
        X=X_out,
        y=y_out,
        feature_names=tuple(feature_names_list),
        n_bar_slots_total=bar_slots_total,
        n_bar_slots_dropped_nan=n_dropped,
        per_feature_non_null=per_feature_non_null,
        per_feature_pct_non_null=per_feature_pct,
        label_feature=spec.label_feature,
        label_version=spec.label_version,
    )


# ── Stacked-interval loading (M5g.5, blueprint § 6.3 Fix 1a) ─────────────


# Fixed encoding for the interval_indicator categorical column. Pinned
# so a trained model's per-row interval lookup is stable across runs;
# the model uses this as a categorical feature.
_INTERVAL_INDICATOR_ENCODING: dict[str, int] = {
    "5m": 0, "15m": 1, "1h": 2, "4h": 3,
}


# Source interval → hours, used by MS1's cross-interval ffill cap.
_INTERVAL_HOURS: dict[str, float] = {
    "5m": 5 / 60, "15m": 0.25, "1h": 1.0, "4h": 4.0,
}


def _interval_to_hours(interval: str) -> int:
    """Return the integer number of hours one bar at ``interval``
    spans, rounded up so a 15m bar gives a 1-hour cap (the cross-
    interval ffill needs to at least cover the next bar). Used as the
    default ``cap_hours`` when a per-bar feature has no
    ``max_ffill_age_hours``.
    """
    if interval not in _INTERVAL_HOURS:
        raise KeyError(
            f"unknown interval {interval!r}; add to _INTERVAL_HOURS "
            f"before stacking against it"
        )
    hours = _INTERVAL_HOURS[interval]
    # Floor to whole hours but clamp at >= 1 so sub-hour sources still
    # get a non-zero cap (which would otherwise mean "no projection").
    return max(1, int(hours))


def _fetch_aux_feature_matrix(
    conn: psycopg.Connection,
    spec: ModelSpec,
    *,
    aux_interval: str,
) -> _FeatureMatrixCache:
    """Build a feature matrix at ``aux_interval`` using ``spec.interval``'s
    feature values, cross-projected onto the aux bar grid.

    Mirrors :func:`_fetch_feature_matrix` but:

    * Per-bar features are read from ``spec.interval`` (e.g. 1h) and
      forward-filled onto the aux bar grid (e.g. 15m) via merge_asof.
      Cap = the feature's ``max_ffill_age_hours`` (same policy as
      load_dataset's global-feature path); else falls back to
      ``_interval_to_hours(spec.interval)`` so a 1h feature projecting
      onto 15m bars propagates at most one source-bar period (the MS1
      fix preserved from the per-feature path).
    * Global features (symbol='', interval='') project onto the aux bar
      grid the same way they project onto the serving grid.

    The serving interval's per-feature semantics are preserved — a 1h
    feature's value at 14:00 applies to all 15m bars in [14:00, 15:00).

    DB cost: two round-trips (per-bar + global) regardless of feature count.
    """
    inputs_meta = _list_input_features(conn, spec.extra_excluded_features)
    bar_index = _bar_grid(spec.train_start, spec.train_end, aux_interval)

    per_bar_metas: list[dict[str, Any]] = []
    global_metas: list[dict[str, Any]] = []
    for meta in inputs_meta:
        symbols = meta["symbols"] or []
        intervals = meta["intervals"] or []
        is_per_bar = bool(symbols) and bool(intervals)
        if is_per_bar:
            if spec.symbol in symbols and spec.interval in intervals:
                per_bar_metas.append(meta)
        else:
            global_metas.append(meta)

    per_bar_data = _read_per_bar_features_batch(
        conn, per_bar_metas, spec.symbol, spec.interval,
        spec.train_start, spec.train_end,
    )
    global_data = _read_global_features_batch(
        conn, global_metas, spec.train_start, spec.train_end,
    )

    feature_names: list[str] = []
    cols: dict[str, pd.Series] = {}
    per_feature_non_null: dict[str, int] = {}

    for meta in per_bar_metas:
        name = meta["feature_name"]
        version = meta["version"]
        max_age_h = meta["max_ffill_age_hours"]
        cap = int(max_age_h) if max_age_h else _interval_to_hours(spec.interval)
        src = per_bar_data.get((name, version), pd.Series(dtype="float64", name=name))
        aligned = _project_series_to_grid(src, bar_index, cap_hours=cap, feature_name=name)
        non_null = int(aligned.notna().sum())
        if non_null == 0:
            logger.warning(
                "Aux feature %s v%d is all-NaN on the %s bar grid — skipping",
                name, version, aux_interval,
            )
            continue
        feature_names.append(name)
        cols[name] = aligned
        per_feature_non_null[name] = non_null

    for meta in global_metas:
        name = meta["feature_name"]
        version = meta["version"]
        max_age_h = meta["max_ffill_age_hours"]
        ffill_policy = meta["ffill_policy"]
        series = global_data.get((name, version), pd.Series(dtype="float64", name=name))
        if series.empty:
            logger.warning(
                "Global feature %s v%d has no rows in window — skipping",
                name, version,
            )
            continue
        if ffill_policy == "last_value":
            cap_global = max_age_h if max_age_h else 24 * 7
            aligned = _ffill_global_to_grid(series, bar_index, cap_hours=cap_global)
        else:
            aligned = series.reindex(bar_index)
        non_null = int(aligned.notna().sum())
        if non_null == 0:
            logger.warning(
                "Aux feature %s v%d is all-NaN on the %s bar grid — skipping",
                name, version, aux_interval,
            )
            continue
        feature_names.append(name)
        cols[name] = aligned
        per_feature_non_null[name] = non_null

    return _FeatureMatrixCache(
        cols=cols,
        feature_names=tuple(feature_names),
        per_feature_non_null=per_feature_non_null,
        bar_index=bar_index,
    )


def _load_aux_interval_dataset(
    spec: ModelSpec,
    aux_interval: str,
    *,
    feature_cache: FeatureCache | None = None,
) -> LoadedDataset:
    """Build a LoadedDataset at ``aux_interval`` with the spec's serving-
    interval registry features forward-filled onto the aux bar grid,
    plus the spec's derived features and label recomputed at the aux
    cadence. Used by :func:`load_stacked_dataset`.
    """
    from dataclasses import replace
    aux_spec = replace(spec, interval=aux_interval, training_intervals=())
    needs_market_data = bool(spec.derived_features) or spec.label_feature in DERIVED_LABELS
    if not needs_market_data:
        # We deliberately route the directional spec (and any future
        # stacked-interval spec) through DERIVED_LABELS — the registry's
        # label_triple_barrier is 1h-only, so the aux 15m label must be
        # computed in-train. Refuse a stacked spec that doesn't ask for
        # market_data: it has no way to label aux bars.
        raise ValueError(
            f"spec={spec.name}: stacked-interval training requires either "
            f"derived_features or a derived label so aux interval "
            f"{aux_interval} can be labeled. Move the label to DERIVED_LABELS."
        )
    with get_connection() as conn:
        fmc = _fetch_aux_feature_matrix(conn, spec, aux_interval=aux_interval)
        required = required_symbols_for(spec)
        market_bundle = _get_or_fetch_market_bundle(
            conn, required, aux_interval,
            spec.train_start, spec.train_end, feature_cache,
        )
    # Label (derived path only — stacked specs route here)
    y_series_full = compute_derived_label(spec.label_feature, market_bundle)
    y_aligned = y_series_full.reindex(fmc.bar_index)

    per_feature_non_null = dict(fmc.per_feature_non_null)
    cols = dict(fmc.cols)
    feature_names_list = list(fmc.feature_names)

    if spec.derived_features:
        registry_names = set(feature_names_list)
        collisions = [n for n in spec.derived_features if n in registry_names]
        if collisions:
            raise ValueError(
                f"derived feature names collide with registry feature names: "
                f"{collisions}."
            )
        for feat_name in spec.derived_features:
            if feat_name not in DERIVED_FEATURES:
                raise KeyError(f"unknown derived feature: {feat_name!r}")
            series = compute_derived_input(feat_name, market_bundle)
            aligned = series.reindex(fmc.bar_index)
            non_null = int(aligned.notna().sum())
            if non_null == 0:
                logger.warning(
                    "derived feature %s is all-NaN on the %s bar grid — skipping",
                    feat_name, aux_interval,
                )
                continue
            cols[feat_name] = aligned
            feature_names_list.append(feat_name)
            per_feature_non_null[feat_name] = non_null

    per_feature_non_null[spec.label_feature] = int(y_aligned.notna().sum())
    bar_slots_total = len(fmc.bar_index)
    X = pd.DataFrame(cols)[feature_names_list]
    joined = X.assign(__y__=y_aligned)
    cleaned = joined.dropna()
    n_dropped = bar_slots_total - len(cleaned)
    if cleaned.empty:
        raise ValueError(
            f"All {bar_slots_total} aux bar slots dropped after NaN filter "
            f"(interval={aux_interval})."
        )
    X_out = cleaned[feature_names_list]
    y_out = cleaned["__y__"].rename("y")
    per_feature_pct = {
        k: round(v / bar_slots_total, 4) for k, v in per_feature_non_null.items()
    }
    return LoadedDataset(
        X=X_out, y=y_out,
        feature_names=tuple(feature_names_list),
        n_bar_slots_total=bar_slots_total,
        n_bar_slots_dropped_nan=n_dropped,
        per_feature_non_null=per_feature_non_null,
        per_feature_pct_non_null=per_feature_pct,
        label_feature=spec.label_feature,
        label_version=spec.label_version,
    )


def load_stacked_dataset(
    spec: ModelSpec,
    *,
    feature_cache: FeatureCache | None = None,
) -> LoadedDataset:
    """Multi-interval training loader (M5g.5, blueprint § 6.3 Fix 1a).

    When ``spec.training_intervals`` is empty or a single-element tuple,
    behaviour is identical to :func:`load_dataset` — backward-compatible
    with every Phase-2 spec. When it lists multiple intervals, load each
    interval's dataset (serving via :func:`load_dataset`, auxiliaries via
    :func:`_load_aux_interval_dataset` which cross-fills the serving
    registry features onto the aux bar grid), concat the rows, and add
    an ``interval_indicator`` integer-encoded column so the model can
    condition on the source cadence.

    The label MUST live in ``DERIVED_LABELS`` so it can be computed at
    every interval — the registry's per-bar labels are stamped at the
    serving interval only.

    Returned ``feature_names`` order: registry features (same as the
    serving load), then derived features, then ``interval_indicator``
    last. Pinned ordering keeps the trained model's column lookup
    stable; the categorical at the end lets ``LightGBM``'s
    ``categorical_feature`` parameter be specified by index without
    surprises.
    """
    if not spec.training_intervals or len(spec.training_intervals) == 1:
        return load_dataset(spec, feature_cache=feature_cache)

    if spec.label_feature not in DERIVED_LABELS:
        raise ValueError(
            f"spec={spec.name}: stacked-interval training requires the label "
            f"({spec.label_feature!r}) to be in DERIVED_LABELS so it can be "
            f"computed at every interval. Registry-stored labels exist only "
            f"at the serving interval."
        )

    pieces: list[tuple[str, LoadedDataset]] = []
    for itv in spec.training_intervals:
        if itv == spec.interval:
            from dataclasses import replace
            serving_spec = replace(spec, training_intervals=())
            piece = load_dataset(serving_spec, feature_cache=feature_cache)
        else:
            piece = _load_aux_interval_dataset(spec, itv, feature_cache=feature_cache)
        pieces.append((itv, piece))

    # Sanity: every piece should have identical feature_names (since we
    # use the same spec.derived_features and the same registry input set
    # via cross-fill). If they diverge, the union of features would
    # produce sparse columns — surface loudly.
    base_features = pieces[0][1].feature_names
    for itv, piece in pieces[1:]:
        if piece.feature_names != base_features:
            extra = set(piece.feature_names) - set(base_features)
            missing = set(base_features) - set(piece.feature_names)
            raise ValueError(
                f"stacked-interval feature mismatch: interval={itv} differs "
                f"from serving={spec.interval}. extra={extra!r} missing={missing!r}. "
                f"Check that the same registry features land at both intervals."
            )

    # Concat — add interval_indicator as the final column. Encoded so
    # the trained model has a stable categorical lookup across runs.
    parts_X: list[pd.DataFrame] = []
    parts_y: list[pd.Series] = []
    total_bar_slots = 0
    total_dropped = 0
    per_feature_non_null: dict[str, int] = {n: 0 for n in base_features}
    for itv, piece in pieces:
        df = piece.X.copy()
        if itv not in _INTERVAL_INDICATOR_ENCODING:
            raise KeyError(
                f"unknown interval {itv!r}; add to _INTERVAL_INDICATOR_ENCODING "
                f"to make it usable in stacked-interval training."
            )
        df["interval_indicator"] = _INTERVAL_INDICATOR_ENCODING[itv]
        parts_X.append(df)
        parts_y.append(piece.y)
        total_bar_slots += piece.n_bar_slots_total
        total_dropped += piece.n_bar_slots_dropped_nan
        for k, v in piece.per_feature_non_null.items():
            per_feature_non_null[k] = per_feature_non_null.get(k, 0) + v

    X_stacked = pd.concat(parts_X, axis=0)
    y_stacked = pd.concat(parts_y, axis=0)
    # Stable sort by index — equal-timestamp rows keep their concat
    # order (serving first, then auxiliaries) so a 1h-and-15m collision
    # at HH:00 reads serving-first when downstream tools iterate.
    sort_order = X_stacked.index.argsort(kind="stable")
    X_stacked = X_stacked.iloc[sort_order]
    y_stacked = y_stacked.iloc[sort_order]

    feature_names_stacked = list(base_features) + ["interval_indicator"]
    per_feature_non_null["interval_indicator"] = len(X_stacked)
    per_feature_pct = {
        k: (round(v / total_bar_slots, 4) if total_bar_slots else 0.0)
        for k, v in per_feature_non_null.items()
    }
    return LoadedDataset(
        X=X_stacked[feature_names_stacked],
        y=y_stacked,
        feature_names=tuple(feature_names_stacked),
        n_bar_slots_total=total_bar_slots,
        n_bar_slots_dropped_nan=total_dropped,
        per_feature_non_null=per_feature_non_null,
        per_feature_pct_non_null=per_feature_pct,
        label_feature=spec.label_feature,
        label_version=spec.label_version,
    )
