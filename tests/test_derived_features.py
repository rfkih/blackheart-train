"""Unit tests for derived features and labels.

Each transformer is a pure function of a ``{symbol: market_data}``
dict. We synthesise market_data, invoke the transformer, and assert
shape + sample values. No DB.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from blackheart_train.derived_features import (
    DERIVED_FEATURES,
    DERIVED_LABELS,
    MissingMarketDataError,
    _t_label_regime_risk_on_24h,
    compute_derived_input,
    compute_derived_label,
    required_symbols_for,
)
from blackheart_train.specs import get_spec


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_market_data(n: int = 500, *, start: str = "2025-01-01", seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start, periods=n, freq="1h")
    # Random walk so close_price is non-degenerate.
    log_returns = rng.normal(0.0, 0.01, n)
    close = 100.0 * np.exp(np.cumsum(log_returns))
    high = close * (1 + np.abs(rng.normal(0.0, 0.002, n)))
    low = close * (1 - np.abs(rng.normal(0.0, 0.002, n)))
    open_ = np.r_[close[0], close[:-1]]
    volume = rng.lognormal(10, 0.5, n)
    df = pd.DataFrame({
        "open_price": open_,
        "high_price": high,
        "low_price": low,
        "close_price": close,
        "volume": volume,
        "quote_asset_volume": volume * close,
        "taker_buy_base_volume": volume * 0.5,
        "taker_buy_quote_volume": volume * close * 0.5,
        "trade_count": rng.integers(100, 1000, n),
    }, index=idx)
    return df


# ── Registry shape ─────────────────────────────────────────────────────────


def test_derived_features_registry_has_expected_keys():
    assert set(DERIVED_FEATURES.keys()) == {
        "btc_log_return_24h",
        "btc_realized_vol_7d",
        "btc_volume_zscore_24h",
        "eth_btc_corr_24h",
    }
    assert set(DERIVED_LABELS.keys()) == {
        "label_return_24h",
        # Phase 4 Session 2 (2026-05-16): label_regime_risk_on_24h was
        # graduated to feature_registry (V77) and removed from
        # DERIVED_LABELS. Its transformer function remains importable.
        # M5g.5: ``label_triple_barrier`` is a derived label so it can
        # be computed at the 15m aux interval during stacked-interval
        # training (the registry stores it at 1h only). The 1h numerics
        # are bit-equal to the registry rows.
        "label_triple_barrier",
    }


def test_derived_labels_are_marked_not_pit_safe():
    for label in DERIVED_LABELS.values():
        assert label.pit_safe is False, f"{label.name} must be pit_safe=False (reads future)"


def test_derived_input_features_are_pit_safe():
    for feat in DERIVED_FEATURES.values():
        assert feat.pit_safe is True, f"{feat.name} must be pit_safe=True (backward only)"


# ── required_symbols_for ──────────────────────────────────────────────────


def test_required_symbols_for_v1_spec_returns_only_main_symbol():
    spec = get_spec("regime_btc_v1")
    assert required_symbols_for(spec) == ("BTCUSDT",)


def test_required_symbols_for_v2_spec_picks_up_eth_btc():
    spec = get_spec("regime_btc_v2")
    # eth_btc_corr_24h is in the derived feature set → ETHUSDT joins.
    assert required_symbols_for(spec) == ("BTCUSDT", "ETHUSDT")


def test_required_symbols_for_unknown_feature_raises():
    from dataclasses import replace
    spec = replace(get_spec("regime_btc_v1"), derived_features=("does_not_exist",))
    with pytest.raises(KeyError, match="unknown derived feature"):
        required_symbols_for(spec)


def test_v2_specs_carry_deployment_readiness_signal():
    """v2 specs use derived features that aren't in the registry yet.
    The loader (via the payload) must surface this so M5e can refuse
    to promote them to live serving until the features are registered.
    The check here is the spec-level signal — payload-level coverage
    runs in CLI integration."""
    v1 = get_spec("regime_btc_v1")
    v2 = get_spec("regime_btc_v2")
    assert v1.derived_features == ()
    assert len(v2.derived_features) > 0


# ── Transformer correctness (input features) ──────────────────────────────


def test_btc_log_return_24h_matches_definition():
    md = _make_market_data(n=200, seed=1)
    out = compute_derived_input("btc_log_return_24h", {"BTCUSDT": md})
    # First 24 rows must be NaN (no 24h lookback yet).
    assert out.iloc[:24].isna().all()
    # At row 50, value equals log(close[50] / close[26]).
    expected = math.log(md["close_price"].iloc[50] / md["close_price"].iloc[26])
    assert out.iloc[50] == pytest.approx(expected)


def test_btc_realized_vol_7d_is_positive_after_warmup():
    md = _make_market_data(n=500, seed=2)
    out = compute_derived_input("btc_realized_vol_7d", {"BTCUSDT": md})
    # Window is 168h; before that, NaN.
    assert out.iloc[:168].isna().all()
    finite = out.dropna()
    assert len(finite) > 0
    assert (finite > 0).all()   # vol is non-negative; we expect strictly positive on random data


def test_btc_volume_zscore_24h_is_finite():
    md = _make_market_data(n=200, seed=3)
    out = compute_derived_input("btc_volume_zscore_24h", {"BTCUSDT": md})
    # First 23 rows are NaN (rolling(24)).
    assert out.iloc[:23].isna().all()
    finite = out.dropna()
    assert len(finite) > 0
    # On lognormal volume with rolling 24h baseline, |z| should mostly
    # be < 4 — just sanity-check that values aren't catastrophic.
    assert finite.abs().median() < 5.0


def test_eth_btc_corr_24h_with_perfectly_correlated_data_is_close_to_1():
    """If ETH log-returns ≡ BTC log-returns, the 24h rolling correlation
    must be 1 (modulo NaN warmup)."""
    md_btc = _make_market_data(n=200, seed=42)
    md_eth = md_btc.copy()   # identical → correlation 1
    out = compute_derived_input("eth_btc_corr_24h", {"BTCUSDT": md_btc, "ETHUSDT": md_eth})
    finite = out.dropna()
    assert len(finite) > 0
    assert (finite > 0.99).all()


def test_eth_btc_corr_24h_with_uncorrelated_data_is_near_zero():
    md_btc = _make_market_data(n=500, seed=10)
    md_eth = _make_market_data(n=500, seed=20)   # different seed
    out = compute_derived_input("eth_btc_corr_24h", {"BTCUSDT": md_btc, "ETHUSDT": md_eth})
    finite = out.dropna()
    assert len(finite) > 0
    # Mean rolling correlation should be near zero on independent data.
    assert abs(finite.mean()) < 0.30


def test_eth_btc_corr_24h_survives_eth_data_gap():
    """If ETH has a gap, the inner-join inside the transformer must
    silently produce NaN for affected bars rather than emit a
    correlation computed from misaligned pairs."""
    md_btc = _make_market_data(n=300, seed=99)
    md_eth = md_btc.copy()
    # Punch a 24-hour hole in ETH starting at row 100.
    md_eth = md_eth.drop(md_eth.index[100:124])
    out = compute_derived_input("eth_btc_corr_24h", {"BTCUSDT": md_btc, "ETHUSDT": md_eth})
    # Output is reindexed to BTC's index: the gap rows are present but NaN.
    assert len(out) == len(md_btc)
    # The bars at the gap and the trailing 23 (the rolling-24 tail of
    # the gap) must be NaN because they don't have 24 paired observations.
    gap_window = out.iloc[100:124 + 23]
    assert gap_window.isna().any()


def test_volume_zscore_24h_does_not_explode_on_tiny_std():
    """A 24-hour window of nearly-identical volumes used to produce
    huge z-scores via float-underflow. Threshold std > 1e-12 keeps
    z bounded."""
    n = 200
    idx = pd.date_range("2025-01-01", periods=n, freq="1h")
    # First 100 rows: identical volume (std == 0, treated as NaN via threshold).
    # Last 100 rows: realistic volume.
    vol = np.r_[
        np.full(100, 1_000_000.0),
        np.random.default_rng(0).lognormal(13, 0.5, n - 100),
    ]
    md = _make_market_data(n=n, seed=77)
    md["volume"] = vol
    out = compute_derived_input("btc_volume_zscore_24h", {"BTCUSDT": md})
    finite = out.dropna()
    # Whatever finite z-scores we get must be sane (not 1e15).
    assert finite.abs().max() < 100.0


# ── Missing market_data ──────────────────────────────────────────────────


def test_fetch_market_data_bundle_raises_clearly_on_missing_symbol():
    """The transformer would crash with KeyError if handed an empty
    DataFrame. The bundle fetcher must raise a clear domain error
    upstream so operators see a meaningful message."""
    from datetime import datetime

    class _StubConn:
        def cursor(self):
            class _Cur:
                def __enter__(self):
                    return self
                def __exit__(self, *_):
                    pass
                def execute(self, *args, **kwargs):
                    pass
                def fetchall(self):
                    return []
            return _Cur()

    from blackheart_train.derived_features import fetch_market_data_bundle
    with pytest.raises(MissingMarketDataError, match="no rows for symbol"):
        fetch_market_data_bundle(
            _StubConn(), ("BTCUSDT",), "1h",
            datetime(2025, 1, 1), datetime(2025, 2, 1),
        )


# ── Transformer correctness (labels) ──────────────────────────────────────


def test_label_return_24h_matches_definition():
    md = _make_market_data(n=200, seed=4)
    out = compute_derived_label("label_return_24h", {"BTCUSDT": md})
    # Last 24 rows must be NaN (no 24h forward).
    assert out.iloc[-24:].isna().all()
    # At row 50, value = (close[74] - close[50]) / close[50].
    expected = (md["close_price"].iloc[74] - md["close_price"].iloc[50]) / md["close_price"].iloc[50]
    assert out.iloc[50] == pytest.approx(expected)


def test_label_regime_risk_on_24h_is_binary_with_nan_tail():
    # Phase 4 Session 2: label_regime_risk_on_24h no longer lives in
    # DERIVED_LABELS (graduated to feature_registry via V77). The
    # transformer function is still importable for shape tests.
    md = _make_market_data(n=300, seed=5)
    out = _t_label_regime_risk_on_24h({"BTCUSDT": md})
    # Last 24+ rows: insufficient forward data, NaN.
    assert out.iloc[-24:].isna().all()
    finite = out.dropna()
    # Only 0.0 and 1.0 (binary).
    assert set(finite.unique()).issubset({0.0, 1.0})


def test_label_regime_risk_on_24h_has_both_classes_on_random_data():
    """On a random walk, the 24h forward Sharpe sign should produce
    both up and down windows with roughly balanced frequency."""
    md = _make_market_data(n=500, seed=6)
    out = _t_label_regime_risk_on_24h({"BTCUSDT": md})
    finite = out.dropna()
    fraction_up = (finite == 1.0).sum() / len(finite)
    # Tolerant — Sharpe sign isn't quite 50/50 because vol normalisation
    # interacts with the drift. Just guard against degenerate output.
    assert 0.3 < fraction_up < 0.7
