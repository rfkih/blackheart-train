"""In-train derived features and labels.

The registry path (``feature_values`` + ``feature_registry``) is the
canonical way to make features available to all training runs and live
inference. But adding a feature to the registry requires a Flyway
migration, an ingest-pipeline definition update, and a backfill — a
multi-day round trip across packages.

For M5d-followup we need to iterate quickly: M5d's gauntlet showed all
three v1 sub-models fail, and the operator asked us to combine "more
features" with "shorter labels" before promoting any HYBRID model. The
fast path is to compute extra features inside the training loop,
directly from ``market_data``. They never land in ``feature_values``;
they're computed fresh on every training run.

Trade-off: these derived features won't be available to a live
inference worker until they're graduated into the registry. That's
fine for research-grade iteration — once a v2 sub-model passes the
gauntlet, we promote the winning features into the registry.

Each :class:`DerivedFeature` declares which symbols of ``market_data``
its transformer needs. The loader fetches all required symbols once
and passes a ``{symbol: DataFrame}`` dict to the transformer. This
keeps cross-asset features (e.g. ``eth_btc_corr_24h``) first-class.

PIT discipline:

* Backward transformers (rolling stats, momentum) are ``pit_safe=True``.
  Aligned with the registry convention: a value stamped at ts=T was
  observable at T+interval (bar T's close), which is the moment the
  trading-JVM treats as "decision time" for the next bar.
* Forward transformers (labels) are ``pit_safe=False`` — they read
  future bars by design. Walk-forward embargo blocks leakage on the
  train/test boundary.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import psycopg
from numpy.lib.stride_tricks import sliding_window_view

from .specs import ModelSpec

logger = logging.getLogger(__name__)


# ── DerivedFeature record ─────────────────────────────────────────────────


# Transformers take ``{symbol: market_data_df}`` and return a Series
# indexed by ts (bar start_time). The loader's downstream pipeline
# reindexes onto the canonical bar_grid before joining with the rest of
# the feature matrix.
DerivedTransformer = Callable[[dict[str, pd.DataFrame]], pd.Series]


@dataclass(frozen=True)
class DerivedFeature:
    name: str
    family: str
    required_symbols: tuple[str, ...]
    transformer: DerivedTransformer
    pit_safe: bool = True


# ── Transformers — input features (backward-looking) ──────────────────────


def _t_btc_log_return_24h(md: dict[str, pd.DataFrame]) -> pd.Series:
    close = md["BTCUSDT"]["close_price"].astype("float64")
    return np.log(close / close.shift(24)).rename("btc_log_return_24h")


def _t_btc_realized_vol_7d(md: dict[str, pd.DataFrame]) -> pd.Series:
    """7-day realized volatility from 1h log returns. Shorter window
    than the registry's btc_realized_vol_30d — captures regime shifts
    that the 30d window smears over."""
    close = md["BTCUSDT"]["close_price"].astype("float64")
    log_ret = np.log(close / close.shift(1))
    # 168 = 7 days * 24h. Annualisation factor not applied: the model
    # only sees this feature as a relative magnitude.
    return log_ret.rolling(168).std().rename("btc_realized_vol_7d")


def _t_btc_volume_zscore_24h(md: dict[str, pd.DataFrame]) -> pd.Series:
    """24h rolling z-score of bar volume. Captures volume regimes —
    risk-off events often pair with elevated volume z-scores."""
    vol = md["BTCUSDT"]["volume"].astype("float64")
    mean = vol.rolling(24).mean()
    std = vol.rolling(24).std()
    # Threshold std > 1e-12 (not > 0) — float underflow can produce
    # microscopic positive std values that divide into huge spurious
    # z-scores. The threshold floor is well below any plausible real
    # volume variance.
    z = (vol - mean) / std.where(std > 1e-12)
    return z.rename("btc_volume_zscore_24h")


def _t_eth_btc_corr_24h(md: dict[str, pd.DataFrame]) -> pd.Series:
    """Rolling 24h correlation between ETH and BTC 1h log returns.

    Cross-asset: when correlation breaks down (decouples), crypto-wide
    regime is often shifting. A pure-BTC model can't see this; combining
    ETH's behaviour into a BTC-trained model gives the model a signal
    it otherwise misses.

    Robustness: explicit inner-join on the timestamp index before the
    rolling correlation. Pandas' ``rolling(N).corr(other)`` aligns by
    index implicitly — if BTC and ETH have a gap in either series,
    misaligned bars get NaN in the window pair-count and the
    correlation silently uses fewer points than expected. We control
    the join up front so the window always has 24 paired observations
    when finite, or NaN otherwise.
    """
    btc_close = md["BTCUSDT"]["close_price"].astype("float64")
    eth_close = md["ETHUSDT"]["close_price"].astype("float64")
    btc_ret = np.log(btc_close / btc_close.shift(1))
    eth_ret = np.log(eth_close / eth_close.shift(1))
    # Inner-align: rows where either return is NaN drop out so the
    # rolling window sees only paired observations.
    paired = pd.DataFrame({"btc": btc_ret, "eth": eth_ret}).dropna()
    rho = paired["btc"].rolling(24).corr(paired["eth"])
    # Reindex back onto BTC's original index so the loader's
    # reindex(bar_index) step has the full index to project from. Bars
    # without paired observations remain NaN (the loader's per-feature
    # non-null tracking will surface this honestly).
    return rho.reindex(btc_close.index).rename("eth_btc_corr_24h")


# ── Transformers — labels (forward-looking) ───────────────────────────────


def _t_label_return_24h(md: dict[str, pd.DataFrame]) -> pd.Series:
    """24h forward return. Shorter than the registry's
    label_return_7d (168h) — sub-models often pick up tighter signal at
    shorter horizons before the noise floor swamps it."""
    close = md["BTCUSDT"]["close_price"].astype("float64")
    return ((close.shift(-24) - close) / close).rename("label_return_24h")


def _t_label_regime_risk_on_24h(md: dict[str, pd.DataFrame]) -> pd.Series:
    """24h forward Sharpe sign. Mirrors the registry's
    label_regime_risk_on_48h but with a 24h horizon — same as the
    blueprint's `label_meanrev_24h` window so all three v2 sub-models
    share a common evaluation horizon."""
    close = md["BTCUSDT"]["close_price"].astype("float64")
    log_ret = np.log(close / close.shift(1))
    fwd_ret = (close.shift(-24) - close) / close
    fwd_std = log_ret.shift(-24).rolling(24).std() * np.sqrt(24)
    sharpe = fwd_ret / fwd_std.where(fwd_std > 0)
    binary = (sharpe > 0).astype("float64")
    # Where sharpe is NaN (insufficient forward bars near the end),
    # propagate NaN so the loader's dropna filters out those rows.
    binary[sharpe.isna()] = np.nan
    return binary.rename("label_regime_risk_on_24h")


def _atr(md_btc: pd.DataFrame, n: int = 14) -> pd.Series:
    """Average True Range over ``n`` bars (SMA, matches the
    blackheart-ingest registry transformer's choice — Wilder's EMA
    would change the label numerics so we don't).

    Ported from ``blackheart_ingest.features.definitions._atr``. The
    interval at which it's evaluated is whatever the caller's
    ``md_btc`` DataFrame is at — that's how this same code computes a
    24-bar = 24h ATR at 1h cadence vs 24-bar = 6h ATR at 15m cadence.
    """
    high = md_btc["high_price"].astype("float64")
    low = md_btc["low_price"].astype("float64")
    close_prev = md_btc["close_price"].astype("float64").shift(1)
    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def _t_label_long_win_tb_loose(md: dict[str, pd.DataFrame]) -> pd.Series:
    """Loose-threshold short-horizon label (k_tp=1.0, k_sl=0.5, horizon=6).
    Asks 'TP at +1 ATR or SL at -0.5 ATR within 6 bars?'. More positive
    labels => looser model => more gate-on trades.

    Path C late-session: v5 with k_tp=2.0/k_sl=0.75/6-bar produced
    POSITIVE_DELTA but gate-on n<100. v6 trades that mechanic against
    higher conviction by lowering thresholds; expect lower-AUC model
    but more permissive gate (more trades through).
    """
    horizon_bars = 6
    k_tp = 1.0
    k_sl = 0.5
    atr_window = 14
    df_btc = md["BTCUSDT"]
    c = df_btc["close_price"].astype("float64").to_numpy()
    h = df_btc["high_price"].astype("float64").to_numpy()
    lo = df_btc["low_price"].astype("float64").to_numpy()
    atr = _atr(df_btc, atr_window).to_numpy()
    n = len(c)
    out = np.full(n, np.nan, dtype="float64")
    last_t = n - horizon_bars
    if last_t <= 0:
        return pd.Series(out, index=df_btc.index, name="label_long_win_tb_loose_v1")
    h_win = sliding_window_view(h, horizon_bars)[1 : 1 + last_t]
    lo_win = sliding_window_view(lo, horizon_bars)[1 : 1 + last_t]
    entry = c[:last_t]
    atr_v = atr[:last_t]
    tp_lvl = entry + k_tp * atr_v
    sl_lvl = entry - k_sl * atr_v
    sl_hit = lo_win <= sl_lvl[:, None]
    tp_hit = h_win >= tp_lvl[:, None]
    sl_any = sl_hit.any(axis=1)
    tp_any = tp_hit.any(axis=1)
    sl_idx = np.where(sl_any, sl_hit.argmax(axis=1), horizon_bars)
    tp_idx = np.where(tp_any, tp_hit.argmax(axis=1), horizon_bars)
    labels = (tp_idx < sl_idx).astype("float64")
    labels[~(np.isfinite(atr_v) & (atr_v > 0))] = np.nan
    out[:last_t] = labels
    return pd.Series(out, index=df_btc.index, name="label_long_win_tb_loose_v1")


def _t_label_long_win_tb_short(md: dict[str, pd.DataFrame]) -> pd.Series:
    """Short-horizon variant of long-win triple-barrier (6 bars instead of
    24, k_tp=2.0 instead of 1.5, k_sl=0.75 instead of 1.0). Asks "does
    TP at +2 ATR hit before SL at -0.75 ATR within 6 bars?" — a tighter,
    higher-conviction entry-timing question than the 24-bar smoothed
    label.

    Authored 2026-05-21 Path C continuation: 24-bar label_long_win_tb_1h_v1
    on directional_btc_1h_v4 produced AUC=0.5575 but paired-delta still
    -7 pp ag90 on DCB-BTC-1h. Mechanistic hypothesis: the 24h horizon
    matches the 24h-lookback features but not the 1h-decision cadence
    where DCB actually opens trades. A 6h horizon + asymmetric stops
    aligns the label's time-scope with the strategy's holding period
    AND with the new bar-level features' lookback (4h-and-shorter).
    """
    horizon_bars = 6
    k_tp = 2.0
    k_sl = 0.75
    atr_window = 14
    df_btc = md["BTCUSDT"]
    c = df_btc["close_price"].astype("float64").to_numpy()
    h = df_btc["high_price"].astype("float64").to_numpy()
    lo = df_btc["low_price"].astype("float64").to_numpy()
    atr = _atr(df_btc, atr_window).to_numpy()
    n = len(c)
    out = np.full(n, np.nan, dtype="float64")
    last_t = n - horizon_bars
    if last_t <= 0:
        return pd.Series(out, index=df_btc.index, name="label_long_win_tb_short_v1")
    h_win = sliding_window_view(h, horizon_bars)[1 : 1 + last_t]
    lo_win = sliding_window_view(lo, horizon_bars)[1 : 1 + last_t]
    entry = c[:last_t]
    atr_v = atr[:last_t]
    tp_lvl = entry + k_tp * atr_v
    sl_lvl = entry - k_sl * atr_v
    sl_hit = lo_win <= sl_lvl[:, None]
    tp_hit = h_win >= tp_lvl[:, None]
    sl_any = sl_hit.any(axis=1)
    tp_any = tp_hit.any(axis=1)
    sl_idx = np.where(sl_any, sl_hit.argmax(axis=1), horizon_bars)
    tp_idx = np.where(tp_any, tp_hit.argmax(axis=1), horizon_bars)
    labels = (tp_idx < sl_idx).astype("float64")
    labels[~(np.isfinite(atr_v) & (atr_v > 0))] = np.nan
    out[:last_t] = labels
    return pd.Series(out, index=df_btc.index, name="label_long_win_tb_short_v1")


def _t_label_short_win_tb(md: dict[str, pd.DataFrame]) -> pd.Series:
    """Binary triple-barrier label for SHORT entries: 1 if TP-first within
    horizon for a short entry at the bar, 0 otherwise.

    Mirror of _t_label_long_win_tb with inverted barrier levels:
      TP = entry - k_tp * ATR  (price falls to this level)
      SL = entry + k_sl * ATR  (price rises to this level)
    TP-hit checks the LOW series; SL-hit checks the HIGH series.

    Same params as the long-win label (k_tp=1.5, k_sl=1.0, horizon=24,
    atr_window=14) so results are directly comparable.

    Motivation: the funding-crowding hypothesis. When BTC 8h funding is
    persistently positive, longs are crowded and squeeze risk is elevated
    — implying short entries have elevated TP probability. Training funding
    features on label_long_win_tb_1h_v1 (a LONG win label) is the wrong
    direction. This label lets the model answer the correct question.
    """
    horizon_bars = 24
    k_tp = 1.5
    k_sl = 1.0
    atr_window = 14
    df_btc = md["BTCUSDT"]
    c = df_btc["close_price"].astype("float64").to_numpy()
    h = df_btc["high_price"].astype("float64").to_numpy()
    lo = df_btc["low_price"].astype("float64").to_numpy()
    atr = _atr(df_btc, atr_window).to_numpy()
    n = len(c)
    out = np.full(n, np.nan, dtype="float64")
    last_t = n - horizon_bars
    if last_t <= 0:
        return pd.Series(out, index=df_btc.index, name="label_short_win_tb_1h_v1")
    h_win = sliding_window_view(h, horizon_bars)[1 : 1 + last_t]
    lo_win = sliding_window_view(lo, horizon_bars)[1 : 1 + last_t]
    entry = c[:last_t]
    atr_v = atr[:last_t]
    # Short: TP is below entry (price falls), SL is above entry (price rises).
    tp_lvl = entry - k_tp * atr_v
    sl_lvl = entry + k_sl * atr_v
    # TP hit when LOW drops to or below the TP level.
    tp_hit = lo_win <= tp_lvl[:, None]
    # SL hit when HIGH rises to or above the SL level.
    sl_hit = h_win >= sl_lvl[:, None]
    sl_any = sl_hit.any(axis=1)
    tp_any = tp_hit.any(axis=1)
    sl_idx = np.where(sl_any, sl_hit.argmax(axis=1), horizon_bars)
    tp_idx = np.where(tp_any, tp_hit.argmax(axis=1), horizon_bars)
    # Short-win = TP hit STRICTLY before SL (tie goes to SL, conservative).
    labels = (tp_idx < sl_idx).astype("float64")
    labels[~(np.isfinite(atr_v) & (atr_v > 0))] = np.nan
    out[:last_t] = labels
    return pd.Series(out, index=df_btc.index, name="label_short_win_tb_1h_v1")


def _t_label_long_win_tb(md: dict[str, pd.DataFrame]) -> pd.Series:
    """Binary triple-barrier label: 1 if TP-first within horizon for a
    long entry at the bar, 0 otherwise (SL-first OR neutral expiry).

    Same machinery as :func:`_t_label_triple_barrier` (k_tp=1.5,
    k_sl=1.0, horizon=24 bars, atr_window=14, conservative tie rule)
    but collapses the three-class output {-1, 0, +1} into a binary
    "long-side wins" indicator. The orchestrator's model_registry
    validator only accepts ``objective ∈ {binary, regression}`` — this
    label makes the triple-barrier signal usable on the binary path.

    Authored 2026-05-21 in response to three HYBRID falsifications
    using ``label_regime_risk_on_24h`` (forward-Sharpe-sign): the
    forward-Sharpe-sign label asks "is the next 24h on average
    bullish?" which is a smoothed aggregate; entry gates need the
    point-in-time question "if I enter long now, will TP hit before
    SL?". This transformer answers exactly that.

    PIT-safe: uses only future bars from t+1 onward. Same lookahead
    boundary as the parent triple-barrier transformer; the last
    ``horizon_bars`` rows are NaN.
    """
    horizon_bars = 24
    k_tp = 1.5
    k_sl = 1.0
    atr_window = 14
    df_btc = md["BTCUSDT"]
    c = df_btc["close_price"].astype("float64").to_numpy()
    h = df_btc["high_price"].astype("float64").to_numpy()
    lo = df_btc["low_price"].astype("float64").to_numpy()
    atr = _atr(df_btc, atr_window).to_numpy()
    n = len(c)
    out = np.full(n, np.nan, dtype="float64")
    last_t = n - horizon_bars
    if last_t <= 0:
        return pd.Series(out, index=df_btc.index, name="label_long_win_tb_1h_v1")
    h_win = sliding_window_view(h, horizon_bars)[1 : 1 + last_t]
    lo_win = sliding_window_view(lo, horizon_bars)[1 : 1 + last_t]
    entry = c[:last_t]
    atr_v = atr[:last_t]
    tp_lvl = entry + k_tp * atr_v
    sl_lvl = entry - k_sl * atr_v
    sl_hit = lo_win <= sl_lvl[:, None]
    tp_hit = h_win >= tp_lvl[:, None]
    sl_any = sl_hit.any(axis=1)
    tp_any = tp_hit.any(axis=1)
    sl_idx = np.where(sl_any, sl_hit.argmax(axis=1), horizon_bars)
    tp_idx = np.where(tp_any, tp_hit.argmax(axis=1), horizon_bars)
    # Long-win = TP hit STRICTLY before SL (intra-bar tie goes to SL,
    # matching the parent transformer's conservative rule).
    labels = (tp_idx < sl_idx).astype("float64")
    # ATR-invalid rows become NaN so the loader drops them.
    labels[~(np.isfinite(atr_v) & (atr_v > 0))] = np.nan
    out[:last_t] = labels
    return pd.Series(out, index=df_btc.index, name="label_long_win_tb_1h_v1")


def _t_label_triple_barrier(md: dict[str, pd.DataFrame]) -> pd.Series:
    """López de Prado triple-barrier labeler, ported from
    ``blackheart_ingest.features.definitions._forward_triple_barrier``.

    Reproduces blueprint § 5.6 semantics — k_tp=1.5, k_sl=1.0,
    horizon_bars=24, atr_window=14, conservative intra-bar tie rule
    (SL wins) — so the values this derived label computes at 1h are
    bit-equal to the registry's ``label_triple_barrier`` rows.

    The transformer is interval-agnostic: it runs on whatever
    ``md["BTCUSDT"]`` is. At 1h that's 24-bar forward = 24h horizon;
    at 15m that's 24-bar forward = 6h horizon. The semantics match the
    blueprint § 6.5 "horizon class → bar multiplier" intent — same
    bar-count, different absolute time, which is the M5g.5 stacked-
    interval training's whole point.

    Why port here when the registry version already exists at 1h:
    we need this at 15m too for stacked-interval training (M5g.5).
    The registry-side path would need a Flyway + ingest backfill
    multi-day round trip; computing in-train mirrors the M5d-followup
    derived-features pattern and lets us iterate at research speed.
    """
    horizon_bars = 24
    k_tp = 1.5
    k_sl = 1.0
    atr_window = 14
    df_btc = md["BTCUSDT"]
    c = df_btc["close_price"].astype("float64").to_numpy()
    h = df_btc["high_price"].astype("float64").to_numpy()
    lo = df_btc["low_price"].astype("float64").to_numpy()
    atr = _atr(df_btc, atr_window).to_numpy()
    n = len(c)
    out = np.full(n, np.nan, dtype="float64")
    last_t = n - horizon_bars
    if last_t <= 0:
        return pd.Series(out, index=df_btc.index, name="label_triple_barrier")
    # Future-bar windows: h_win[t] = h[t+1 .. t+horizon_bars] (size horizon_bars).
    h_win = sliding_window_view(h, horizon_bars)[1 : 1 + last_t]
    lo_win = sliding_window_view(lo, horizon_bars)[1 : 1 + last_t]
    entry = c[:last_t]
    atr_v = atr[:last_t]
    tp_lvl = entry + k_tp * atr_v
    sl_lvl = entry - k_sl * atr_v
    sl_hit = lo_win <= sl_lvl[:, None]
    tp_hit = h_win >= tp_lvl[:, None]
    sl_any = sl_hit.any(axis=1)
    tp_any = tp_hit.any(axis=1)
    sl_idx = np.where(sl_any, sl_hit.argmax(axis=1), horizon_bars)
    tp_idx = np.where(tp_any, tp_hit.argmax(axis=1), horizon_bars)
    labels = np.zeros(last_t, dtype="float64")
    labels[sl_idx < tp_idx] = -1.0
    labels[tp_idx < sl_idx] = 1.0
    # Intra-bar tie: SL wins (matches the original loop's order-of-checks).
    labels[(sl_idx == tp_idx) & sl_any] = -1.0
    labels[~(np.isfinite(atr_v) & (atr_v > 0))] = np.nan
    out[:last_t] = labels
    return pd.Series(out, index=df_btc.index, name="label_triple_barrier")


# ── Registry ─────────────────────────────────────────────────────────────


DERIVED_FEATURES: dict[str, DerivedFeature] = {
    "btc_log_return_24h": DerivedFeature(
        name="btc_log_return_24h",
        family="technical",
        required_symbols=("BTCUSDT",),
        transformer=_t_btc_log_return_24h,
    ),
    "btc_realized_vol_7d": DerivedFeature(
        name="btc_realized_vol_7d",
        family="technical",
        required_symbols=("BTCUSDT",),
        transformer=_t_btc_realized_vol_7d,
    ),
    "btc_volume_zscore_24h": DerivedFeature(
        name="btc_volume_zscore_24h",
        family="technical",
        required_symbols=("BTCUSDT",),
        transformer=_t_btc_volume_zscore_24h,
    ),
    "eth_btc_corr_24h": DerivedFeature(
        name="eth_btc_corr_24h",
        family="cross_asset",
        required_symbols=("BTCUSDT", "ETHUSDT"),
        transformer=_t_eth_btc_corr_24h,
    ),
}


def _t_label_long_win_tb_eth(md: dict[str, pd.DataFrame]) -> pd.Series:
    """ETH-native long-win triple-barrier label (2026-06-02).

    Identical mechanics to :func:`_t_label_long_win_tb` (k_tp=1.5,
    k_sl=1.0, horizon=24, atr_window=14) but uses ETHUSDT price data.
    Required so OFI microstructure features for ETHUSDT can be trained
    against a correct same-symbol entry-quality label.
    """
    horizon_bars = 24
    k_tp = 1.5
    k_sl = 1.0
    atr_window = 14
    df = md["ETHUSDT"]
    c = df["close_price"].astype("float64").to_numpy()
    h = df["high_price"].astype("float64").to_numpy()
    lo = df["low_price"].astype("float64").to_numpy()
    atr = _atr(df, atr_window).to_numpy()
    n = len(c)
    out = np.full(n, np.nan, dtype="float64")
    last_t = n - horizon_bars
    if last_t <= 0:
        return pd.Series(out, index=df.index, name="label_long_win_tb_eth_1h_v1")
    h_win = sliding_window_view(h, horizon_bars)[1 : 1 + last_t]
    lo_win = sliding_window_view(lo, horizon_bars)[1 : 1 + last_t]
    entry = c[:last_t]
    atr_v = atr[:last_t]
    tp_lvl = entry + k_tp * atr_v
    sl_lvl = entry - k_sl * atr_v
    sl_hit = lo_win <= sl_lvl[:, None]
    tp_hit = h_win >= tp_lvl[:, None]
    sl_any = sl_hit.any(axis=1)
    tp_any = tp_hit.any(axis=1)
    sl_idx = np.where(sl_any, sl_hit.argmax(axis=1), horizon_bars)
    tp_idx = np.where(tp_any, tp_hit.argmax(axis=1), horizon_bars)
    labels = (tp_idx < sl_idx).astype("float64")
    labels[~(np.isfinite(atr_v) & (atr_v > 0))] = np.nan
    out[:last_t] = labels
    return pd.Series(out, index=df.index, name="label_long_win_tb_eth_1h_v1")


DERIVED_LABELS: dict[str, DerivedFeature] = {
    "label_return_24h": DerivedFeature(
        name="label_return_24h",
        family="label",
        required_symbols=("BTCUSDT",),
        transformer=_t_label_return_24h,
        pit_safe=False,
    ),
    # Phase 4 / Session 2 (2026-05-16): ``label_regime_risk_on_24h`` was
    # graduated into ``feature_registry`` via Flyway V77 and backfilled
    # into ``feature_values``. Removing it from DERIVED_LABELS routes the
    # loader through ``_read_per_bar_feature`` (registry path) instead of
    # computing in-train. The transformer function
    # ``_t_label_regime_risk_on_24h`` above is retained — the equivalence
    # tests in blackheart-ingest import it to confirm bit-equivalence
    # against the registry version (_forward_sharpe_binary_sign_train_
    # compat). Resolution change is numerically bit-equivalent for both
    # v2 and the new v3; v2's deployment_ready stays False due to its
    # ``derived_features`` tuple, v3's flips True (registry-only).
    # M5g.5: triple-barrier label computed in-train so it can run at
    # both 1h and 15m for stacked-interval training. At 1h the
    # numerics match the registry's ``label_triple_barrier`` rows
    # (same algorithm, same params); at 15m the registry has no
    # equivalent. Loader's interval-aware path picks the derived
    # implementation only when the loader can't find a registry row
    # for the (label_feature, version, symbol, interval) tuple.
    "label_triple_barrier": DerivedFeature(
        name="label_triple_barrier",
        family="label",
        required_symbols=("BTCUSDT",),
        transformer=_t_label_triple_barrier,
        pit_safe=False,
    ),
    # 2026-05-21: binary long-win triple-barrier label. Resolves the
    # label-misalignment failure mode documented in journal entries
    # c936710c (DCB-BTC), c34ceb35 (MMR-BTC), 3a4451f4 (DCB-ETH) where
    # forward-Sharpe-sign smoothed-aggregate predictions destroyed value
    # as an entry gate. Computed in-train; not in feature_registry so
    # the loader picks this derived path automatically.
    "label_long_win_tb_1h_v1": DerivedFeature(
        name="label_long_win_tb_1h_v1",
        family="label",
        required_symbols=("BTCUSDT",),
        transformer=_t_label_long_win_tb,
        pit_safe=False,
    ),
    # 2026-05-21 Path C continuation: short-horizon (6-bar) triple-barrier
    # binary label with asymmetric stops (k_tp=2.0, k_sl=0.75). Aligns
    # label time-scope with strategy holding period and bar-level features'
    # lookbacks. Consumed by directional_btc_1h_v5.
    "label_long_win_tb_short_v1": DerivedFeature(
        name="label_long_win_tb_short_v1",
        family="label",
        required_symbols=("BTCUSDT",),
        transformer=_t_label_long_win_tb_short,
        pit_safe=False,
    ),
    # 2026-05-21 Path C late: loose-threshold variant of short-horizon
    # label. k_tp=1.0, k_sl=0.5, horizon=6. Tests whether a less-
    # discriminating model produces a less-restrictive gate (more
    # gate-on trades through, clearing V11 n>=100).
    "label_long_win_tb_loose_v1": DerivedFeature(
        name="label_long_win_tb_loose_v1",
        family="label",
        required_symbols=("BTCUSDT",),
        transformer=_t_label_long_win_tb_loose,
        pit_safe=False,
    ),
    # 2026-05-27: short-entry mirror of label_long_win_tb_1h_v1.
    # Same params (k_tp=1.5, k_sl=1.0, horizon=24) but barrier levels
    # are inverted: TP below entry (LOW must drop), SL above entry (HIGH
    # must rise). Needed for the funding-crowding → squeeze hypothesis:
    # funding features predict crowded-long → squeeze → SHORT wins, not
    # LONG wins. Training on label_long_win_tb_1h_v1 was the wrong
    # direction. Consumed by funding_short_v1.
    "label_short_win_tb_1h_v1": DerivedFeature(
        name="label_short_win_tb_1h_v1",
        family="label",
        required_symbols=("BTCUSDT",),
        transformer=_t_label_short_win_tb,
        pit_safe=False,
    ),
    # 2026-06-02: ETH-native long-win triple-barrier label.
    # Identical mechanics to label_long_win_tb_1h_v1 (k_tp=1.5, k_sl=1.0,
    # horizon=24, atr_window=14) but computed from ETHUSDT price data.
    # Required by directional_eth_ofi_1h_v1 so OFI features can be trained
    # against a correct same-symbol entry-quality label rather than the
    # regime/forward-Sharpe label (which showed adversarial_auc≈1.0 on OFI).
    "label_long_win_tb_eth_1h_v1": DerivedFeature(
        name="label_long_win_tb_eth_1h_v1",
        family="label",
        required_symbols=("ETHUSDT",),
        transformer=_t_label_long_win_tb_eth,
        pit_safe=False,
    ),
}


# ── Market data fetcher ──────────────────────────────────────────────────


def _market_data_interval_filter(interval: str) -> str:
    """Map a ModelSpec interval ('1h') to the market_data ``interval``
    column value. They happen to match today but the registry pattern
    is to keep them distinct in case live and backtest use different
    bar conventions."""
    return interval


def _read_market_data(
    conn: psycopg.Connection,
    symbol: str,
    interval: str,
    start,
    end,
) -> pd.DataFrame:
    """Read BTCUSDT / ETHUSDT bars from ``market_data``. Returns a
    DataFrame indexed by ``start_time`` (naive UTC) with the standard
    OHLCV columns.
    """
    sql = """
        SELECT start_time, open_price, high_price, low_price, close_price,
               volume, quote_asset_volume, taker_buy_base_volume,
               taker_buy_quote_volume, trade_count
        FROM market_data
        WHERE symbol = %(symbol)s
          AND interval = %(interval)s
          AND start_time >= %(start)s
          AND start_time < %(end)s
        ORDER BY start_time ASC
    """
    with conn.cursor() as cur:
        cur.execute(sql, {
            "symbol": symbol,
            "interval": _market_data_interval_filter(interval),
            "start": start,
            "end": end,
        })
        rows = cur.fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["start_time"] = pd.to_datetime(df["start_time"])
    df = df.set_index("start_time").sort_index()
    return df


class MissingMarketDataError(RuntimeError):
    """Raised by :func:`fetch_market_data_bundle` when a required
    symbol has no rows in the requested window. Surfaces a clear
    operational error instead of letting the transformer crash on an
    empty DataFrame with a confusing pandas KeyError.
    """


def fetch_market_data_bundle(
    conn: psycopg.Connection,
    symbols: tuple[str, ...],
    interval: str,
    start,
    end,
) -> dict[str, pd.DataFrame]:
    """Read each symbol's market_data once, return a dict keyed by symbol.

    A derived-feature backward window (e.g. 168h for realized_vol_7d)
    and forward window (e.g. 24h for label_return_24h) both read
    outside the ``[start, end)`` slice. For correctness we extend the
    fetch by a pessimistic buffer on both sides. The transformer's
    output gets reindexed onto the bar grid by the loader, so
    out-of-window intermediate values don't leak in — they're just
    temporarily there for rolling computations.

    Raises :class:`MissingMarketDataError` when a required symbol has
    zero rows in the extended window. The transformers downstream
    cannot recover from an empty DataFrame so we fail loudly here
    rather than silently produce all-NaN derived features.
    """
    # 8 days back, 8 days forward — covers our longest current windows
    # (168h backward for realized vol, 24h forward for labels) with margin.
    buffer = pd.Timedelta(days=8)
    extended_start = start - buffer
    extended_end = end + buffer
    bundle: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = _read_market_data(conn, sym, interval, extended_start, extended_end)
        if df.empty:
            raise MissingMarketDataError(
                f"market_data has no rows for symbol={sym} interval={interval} "
                f"in window [{extended_start}, {extended_end}). Backfill required "
                f"before training a spec that needs this symbol."
            )
        bundle[sym] = df
        logger.info(
            "market_data fetched | symbol=%s rows=%d extended_window=[%s, %s)",
            sym, len(df), extended_start, extended_end,
        )
    return bundle


# ── Public API for the loader ────────────────────────────────────────────


def required_symbols_for(spec: ModelSpec) -> tuple[str, ...]:
    """Union of every required symbol across the spec's derived
    features + derived label (if any). Always includes spec.symbol so
    the loader can compute the label even when the spec uses no derived
    features.
    """
    symbols: set[str] = {spec.symbol}
    for feat_name in spec.derived_features:
        feat = DERIVED_FEATURES.get(feat_name)
        if feat is None:
            raise KeyError(f"unknown derived feature: {feat_name!r}")
        symbols.update(feat.required_symbols)
    if spec.label_feature in DERIVED_LABELS:
        symbols.update(DERIVED_LABELS[spec.label_feature].required_symbols)
    return tuple(sorted(symbols))


def compute_derived_input(
    feat_name: str, bundle: dict[str, pd.DataFrame]
) -> pd.Series:
    feat = DERIVED_FEATURES[feat_name]
    return feat.transformer(bundle)


def compute_derived_label(
    label_name: str, bundle: dict[str, pd.DataFrame]
) -> pd.Series:
    label = DERIVED_LABELS[label_name]
    return label.transformer(bundle)
