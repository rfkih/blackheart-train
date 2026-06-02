"""Locked sub-model specs per blueprint § 6.1.

Three HYBRID modulator sub-models on BTC 1h bars, training window
2024-12-01 → 2026-05-14. Input feature set is derived from the registry
*excluding* the four labels and the four source-capped features deferred
to v2 (see project_ml_blueprint.md, Tier C in the M5 kickoff audit).

A ModelSpec is a pure data record. The training pipeline consumes it,
the artifact metadata serialises it, and M5e's registry write will
materialise it into a model_registry row.

Immutability:

* The dataclass is frozen — direct field replacement is blocked.
* ``hyperparams`` is a plain dict. The training pipeline copies it via
  ``dict(spec.hyperparams)`` before forwarding to LightGBM, so the
  shared default factory's output is not mutated by the model layer.
  Don't mutate ``spec.hyperparams`` in place — there is no enforcement
  (a MappingProxyType view was tried but breaks ``dataclasses.asdict``
  on Python 3.14, which can't deepcopy mappingproxy). If you want to
  experiment with hyperparams, build a new ``ModelSpec`` via ``replace``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

Objective = Literal["binary", "regression", "multiclass"]
# multiclass: 3-class directional model (label_triple_barrier; classes
# -1/0/+1 mapped to 0/1/2 inside train.py before LightGBM sees them).
# Used by the Phase 3 directional model (blueprint § 6.2), not by the
# Phase 2 modulator sub-models.


# Features in feature_registry that are NOT inputs to the sub-models.
#
# Originally this set held BOTH labels (the four ``label_*`` rows) AND
# the four Tier-C features whose source-side history is capped <30 days
# (Binance public futures-data endpoints + CoinGecko free-tier
# dominance). H9 audit (2026-05-16): labels are now filtered exclusively
# by ``loader._list_input_features``'s schema-based predicate
# ``(label_direction IS NULL OR label_direction <> 'forward')`` — the
# seed migrations V73/V74/V77 set ``label_direction='forward'`` on every
# label row. Keeping labels in this set as well meant TWO lists had to
# stay in sync; the morning of 2026-05-16 a label leaked into a train
# matrix because V77 added a label but the EXCLUDED set wasn't updated
# (regime_btc_v3 trivial AUC=1.0 leak — see Session 1+2 memo).
#
# Now: this set covers only the Tier-C source-capped features (NOT
# labels) — those cannot be filtered by ``label_direction`` because
# they aren't labels. Adding a NEW non-label feature that should be
# excluded still requires editing this set.
#
# A future label added without ``label_direction='forward'`` would
# bypass the schema filter. Operators adding label features in a
# future migration must set ``label_direction='forward'`` in the
# INSERT. ``test_excluded_from_inputs_has_no_labels`` pins the
# invariant on the Python side; a DB-level CHECK constraint on
# ``feature_name LIKE 'label_%' => label_direction IS NOT NULL`` would
# be the next hardening step.
EXCLUDED_FROM_INPUTS: frozenset[str] = frozenset({
    "btc_oi_change_24h_pct",
    "taker_buy_ratio_4h",
    "topls_ratio_change_24h",
    "btc_dominance_change_7d",
})


# Intervals usable in stacked-interval training. Must stay in sync with
# ``loader._INTERVAL_INDICATOR_ENCODING`` and ``loader._INTERVAL_HOURS``.
# A spec asking for an interval outside this set fails at construction
# rather than at load time.
_STACKABLE_INTERVALS: frozenset[str] = frozenset({"5m", "15m", "1h", "4h"})


# Default hyperparams shared across the three M5a specs. The factory
# returns a fresh dict per ``ModelSpec`` instance so two specs do not
# alias each other.
def _default_hyperparams() -> dict[str, object]:
    return {
        "num_leaves": 31,
        "max_depth": -1,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "min_child_samples": 50,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbosity": -1,
    }


@dataclass(frozen=True)
class ModelSpec:
    """A trainable sub-model. Frozen + read-only hyperparams so the spec
    hashes deterministically into artifact metadata.
    """

    name: str
    purpose: Literal["regime", "positioning", "flow", "directional"]
    label_feature: str
    label_version: int
    objective: Objective
    symbol: str
    interval: str

    # Training window. Inclusive of start, exclusive of end (standard half-open).
    train_start: datetime
    train_end: datetime

    # Fraction of the (chronologically-ordered) window held out for
    # validation. Walk-forward replaces this in M5c.
    val_fraction: float = 0.2

    # Convention: do not mutate in place. See module docstring.
    hyperparams: dict[str, object] = field(default_factory=_default_hyperparams)

    # M5d-followup: derived features computed in-train from market_data.
    # An empty tuple means "registry features only" (v1 behaviour).
    # Names must exist in ``derived_features.DERIVED_FEATURES``. The
    # loader fetches the necessary market_data symbols and computes the
    # series at runtime — they never land in feature_values until
    # promoted into the registry. Labels are still resolved by
    # ``label_feature``: the loader checks ``DERIVED_LABELS`` first,
    # falling back to the registry's ``feature_values`` table.
    derived_features: tuple[str, ...] = ()

    # M5g.3: base models for the ensemble. Single-element ``("lightgbm",)``
    # is the Phase 2 modulator default and produces a single-model
    # training path identical to M5a/b/c behaviour. Multi-element values
    # — e.g. ``("lightgbm", "xgboost", "logreg_l1")`` — route training
    # through the ensemble path in :mod:`blackheart_train.ensemble`,
    # adding per-base + ensemble + disagreement metrics to the artifact.
    # Only meaningful for ``objective == "multiclass"`` today (binary
    # and regression paths ignore this field).
    #
    # Invariant (EB2/EB3 fixes): "lightgbm" MUST be the first element.
    # Originally this guarded the M5g.3 phase 1 booster-extraction path
    # which only persisted LightGBM. Post-Phase-2 the full ensemble is
    # persisted, but the constraint stays: ``BASE_MODEL_ORDER`` puts
    # LightGBM first so per-class proba averaging is deterministic;
    # ``lgb_*`` is treated as the primary metric prefix; the M5d gauntlet
    # gate names assume a LightGBM primary. Allowing a non-LightGBM-led
    # tuple would silently shift those semantics with no spec-visible
    # warning. Cheaper to refuse the spec.
    base_models: tuple[str, ...] = ("lightgbm",)

    # M5g.4: meta-label gating. When True (and ``len(base_models) > 1``),
    # the training loop fits a Logistic-L1 meta-label on out-of-sample
    # primary predictions and adds ``gated_*`` metrics to the artifact
    # (gated_selectivity / gated_accuracy / gated_accuracy_uplift /
    # ungated_accuracy). The meta-label itself isn't yet persisted —
    # M5g.4 phase 2 wires that in along with the inference plumbing.
    # Ignored for single-base-model specs.
    meta_label_enabled: bool = False

    # M5g.5: stacked-interval training (blueprint § 6.3 Fix 1a).
    # Empty tuple or ``(interval,)`` → single-interval training (default,
    # identical to M5g.1-4 behaviour). Multi-element — e.g.
    # ``("1h", "15m")`` — routes through ``loader.load_stacked_dataset``
    # which concats per-interval datasets with an ``interval_indicator``
    # categorical column. The serving interval (``spec.interval``) MUST
    # appear in the tuple; auxiliary intervals get the registry's 1h
    # features forward-filled onto their bar grid, plus the derived
    # features and triple-barrier label recomputed at the new cadence.
    #
    # Used only by the directional spec today — the modulator sub-models
    # have plenty of 1h data and don't need the 4× sample-size uplift
    # that stacking 15m gives.
    training_intervals: tuple[str, ...] = ()

    # M5g.7: per-fold feature selection (blueprint § 7.5 Fix 5).
    # When True, ``walk_forward.run_walk_forward`` runs correlation
    # pruning + MI top-K on each fold's X_tr BEFORE fit_and_evaluate,
    # capping at ``feature_selection.MAX_FEATURES_DEFAULT`` (=8).
    # Per-fold selected column lists land in ``FoldMetrics.features_selected``
    # so a reviewer can audit "which features survived in ≥5 of 6 folds?"
    # (consistent signal) vs single-fold survivors (noise).
    # Off by default to preserve M5g.1-6 behaviour. Only the directional
    # spec opts in today; modulators have fewer collinearity concerns at
    # their feature counts.
    feature_selection_enabled: bool = False

    # 2026-05-21 Path C: spec-scoped extra feature exclusions, additive
    # to the module-level :data:`EXCLUDED_FROM_INPUTS` allowlist.
    # Motivation: macro features (fear_greed_value, stablecoin_supply_*,
    # eth_btc_ratio_*) live in feature_registry with symbols=[],
    # intervals=[] (global cadence) and are persisted in feature_values
    # at symbol='', interval='' (daily). Every prior model picked them
    # up automatically via the loader's global-feature path. That works
    # for training but is unservable by blackheart-inference's sidecar
    # which exact-matches (symbol, interval, ts) on feature_values —
    # macro features then 409 feature_value_missing at inference time.
    #
    # A spec that wants a SIDECAR-SERVABLE feature stack adds the
    # macro feature names here so the loader skips them at training
    # time and the trained artifact's feature_set excludes them.
    #
    # Order-of-application: the spec-level set is unioned with the
    # module-level set before the registry SQL is built. Names that
    # appear in either are excluded. Mismatched names (typos, defunct
    # features) are silently no-ops — the loader logs nothing because
    # an excluded name that doesn't appear in feature_registry is
    # already a no-op.
    extra_excluded_features: tuple[str, ...] = ()

    # ES1: LightGBM early stopping for single-model paths.
    # When > 0, ``fit_and_evaluate`` carves the chronological tail of
    # X_tr (size = early_stopping_val_fraction) as an inner validation
    # slice, applies a label-horizon embargo gap between the inner-train
    # tail and the inner-val head, and fits with
    # ``callbacks=[lgb.early_stopping(rounds, verbose=False)]``. Stops
    # when no improvement in ``rounds`` consecutive boosting rounds; the
    # spec's ``hyperparams['n_estimators']`` becomes a hard cap, not a
    # fixed iteration count.
    #
    # Off by default (= 0) so existing locked specs (regime_btc_v3,
    # flow_btc_v2, etc.) hash to a different content_sha only if they
    # explicitly opt in. The inner-val slice is held out from the fit
    # so it does NOT leak into the booster — but it DOES leak its row
    # count into hyperparameter choice (early stop point). That is the
    # standard early-stopping tradeoff; the OOF fold's test slice is
    # still genuinely out-of-sample.
    #
    # Skipped silently for ensemble specs (len(base_models) > 1). M5g.3
    # ensemble's three base models would each need their own ES wiring;
    # deferred to a separate change.
    early_stopping_rounds: int = 0
    # Chronological tail fraction of X_tr used as the early-stopping
    # inner validation slice. Default 0.15 keeps 85% of the fold's
    # training data for fitting. Only consulted when
    # early_stopping_rounds > 0.
    early_stopping_val_fraction: float = 0.15

    def __post_init__(self) -> None:
        if not self.base_models:
            raise ValueError(
                f"spec={self.name}: base_models cannot be empty"
            )
        if self.base_models[0] != "lightgbm":
            raise ValueError(
                f"spec={self.name}: base_models must start with 'lightgbm' "
                f"(got {self.base_models!r}). LightGBM is the primary base "
                f"model — per-class proba averaging, the lgb_* metric "
                f"prefix, and the M5d gauntlet's primary-metric semantics "
                f"all assume LightGBM is first. Reorder or remove the "
                f"non-LightGBM kinds if you need a non-LightGBM primary."
            )
        # M5g.5: if training_intervals is set, serving interval must
        # appear in it. Otherwise the model would be trained on auxiliary
        # data only and serve on a cadence it never saw — meaningless.
        if self.training_intervals and self.interval not in self.training_intervals:
            raise ValueError(
                f"spec={self.name}: serving interval {self.interval!r} must "
                f"appear in training_intervals {self.training_intervals!r}. "
                f"Stacked-interval training expands the training set; the "
                f"model still serves on spec.interval."
            )
        # MS3: training_intervals must not contain duplicates — the
        # loader assumes a unique mapping per interval. Two identical
        # entries would double-count rows and corrupt the
        # interval_indicator semantics.
        if self.training_intervals and len(set(self.training_intervals)) != len(self.training_intervals):
            raise ValueError(
                f"spec={self.name}: training_intervals contains duplicates "
                f"({self.training_intervals!r}); each interval must appear once."
            )
        # MS3: every interval must be loader-stackable. Catches typos
        # ('1H' vs '1h') and intervals we haven't added support for yet
        # at construction time rather than at load time.
        unknown = set(self.training_intervals) - _STACKABLE_INTERVALS
        if unknown:
            raise ValueError(
                f"spec={self.name}: training_intervals contains "
                f"unstackable values {sorted(unknown)!r}; loader supports "
                f"{sorted(_STACKABLE_INTERVALS)!r}"
            )
        # ES1: early-stopping field validation. Sentinel 0 disables;
        # negative rounds and out-of-range fractions are user errors.
        if self.early_stopping_rounds < 0:
            raise ValueError(
                f"spec={self.name}: early_stopping_rounds must be >= 0 "
                f"(got {self.early_stopping_rounds!r}); 0 disables, "
                f"positive enables LightGBM early stopping"
            )
        if not (0.0 < self.early_stopping_val_fraction < 1.0):
            raise ValueError(
                f"spec={self.name}: early_stopping_val_fraction must be in "
                f"(0, 1) (got {self.early_stopping_val_fraction!r})"
            )


# Locked training window for M5 (blueprint § 19): the 17-month gate window
# we just finished backfilling. Walk-forward folds in M5c subdivide this.
_TRAIN_START = datetime(2024, 12, 1)
_TRAIN_END = datetime(2026, 5, 14)


SPECS: dict[str, ModelSpec] = {
    "regime_btc_v1": ModelSpec(
        name="regime_btc_v1",
        purpose="regime",
        label_feature="label_regime_risk_on_48h",
        label_version=1,
        objective="binary",
        symbol="BTCUSDT",
        interval="1h",
        train_start=_TRAIN_START,
        train_end=_TRAIN_END,
    ),
    "positioning_btc_v1": ModelSpec(
        name="positioning_btc_v1",
        purpose="positioning",
        label_feature="label_meanrev_24h",
        label_version=1,
        objective="regression",
        symbol="BTCUSDT",
        interval="1h",
        train_start=_TRAIN_START,
        train_end=_TRAIN_END,
    ),
    "flow_btc_v1": ModelSpec(
        name="flow_btc_v1",
        purpose="flow",
        label_feature="label_return_7d",
        label_version=1,
        objective="regression",
        symbol="BTCUSDT",
        interval="1h",
        train_start=_TRAIN_START,
        train_end=_TRAIN_END,
    ),
    # M5d-followup v2 variants: registry features + 4 derived features,
    # and shorter forward horizons on the labels. Each v2 spec isolates
    # one or two changes vs its v1 sibling so the gauntlet's verdict
    # tells us which lever moved the needle (or whether none did).
    "regime_btc_v2": ModelSpec(
        name="regime_btc_v2",
        purpose="regime",
        # 24h forward Sharpe sign instead of 48h — shorter horizon is
        # less smeared by intraday noise integration over 2 days.
        label_feature="label_regime_risk_on_24h",
        label_version=1,
        objective="binary",
        symbol="BTCUSDT",
        interval="1h",
        train_start=_TRAIN_START,
        train_end=_TRAIN_END,
        derived_features=(
            "btc_log_return_24h",
            "btc_realized_vol_7d",
            "btc_volume_zscore_24h",
            "eth_btc_corr_24h",
        ),
    ),
    # Phase 4 / Session 2 — registry-only twin of regime_btc_v2. Same
    # label, same input set, but derived_features=() and the label is
    # resolved via feature_registry (V77 seeded the rows; train-side
    # DERIVED_LABELS no longer carries label_regime_risk_on_24h). This
    # flips deployment_readiness.deployment_ready=True so the artifact
    # lands as status=trained instead of awaiting_operator_review.
    #
    # Bit-equivalence to v2 was verified pre-registration: the V77 seed
    # transformer for label_regime_risk_on_24h is the train-compat twin
    # (_forward_sharpe_binary_sign_train_compat) which exactly mirrors
    # blackheart_train.derived_features._t_label_regime_risk_on_24h. The
    # four input features are also bit-equivalent (verified by
    # blackheart-ingest/tests/test_train_ingest_equivalence.py). So v3's
    # walk-forward AUC should match v2 within rounding.
    "regime_btc_v3": ModelSpec(
        name="regime_btc_v3",
        purpose="regime",
        label_feature="label_regime_risk_on_24h",
        label_version=1,
        objective="binary",
        symbol="BTCUSDT",
        interval="1h",
        train_start=_TRAIN_START,
        train_end=_TRAIN_END,
        # Empty: all inputs now come from feature_registry via V77.
        derived_features=(),
        # Cross-symbol OB exclusions: ob_*_eth features are stored
        # with symbols=NULL (global) so they land in every training
        # matrix. ETH order-book state is not a valid input for a
        # BTC model — exclude to prevent cross-symbol leakage.
        extra_excluded_features=(
            "ob_spread_bps_eth",
            "ob_imbalance_eth",
            "ob_imbalance_momentum_8h_eth",
        ),
    ),
    "positioning_btc_v2": ModelSpec(
        name="positioning_btc_v2",
        purpose="positioning",
        # Same label as v1 (label_meanrev_24h is already 24h forward) —
        # the only variable changing here is the feature set.
        label_feature="label_meanrev_24h",
        label_version=1,
        objective="regression",
        symbol="BTCUSDT",
        interval="1h",
        train_start=_TRAIN_START,
        train_end=_TRAIN_END,
        derived_features=(
            "btc_log_return_24h",
            "btc_realized_vol_7d",
            "btc_volume_zscore_24h",
            "eth_btc_corr_24h",
        ),
    ),
    "flow_btc_v2": ModelSpec(
        name="flow_btc_v2",
        purpose="flow",
        # 24h forward return instead of 168h — 7-day flow is too
        # macro-driven for hourly bars; 24h pulls signal into a window
        # the features can plausibly explain.
        label_feature="label_return_24h",
        label_version=1,
        objective="regression",
        symbol="BTCUSDT",
        interval="1h",
        train_start=_TRAIN_START,
        train_end=_TRAIN_END,
        derived_features=(
            "btc_log_return_24h",
            "btc_realized_vol_7d",
            "btc_volume_zscore_24h",
            "eth_btc_corr_24h",
        ),
    ),
    # 2026-05-20 — first per-symbol ML spec on ETHUSDT. Built after the
    # bar-OHLCV parametric search hit lifecycle exhaustion across
    # BTC+ETH × {5m,15m,1h,4h}; ETH ML was previously blocked because
    # all 10 derived features were symbols=('BTCUSDT',) only. V110
    # migration + the matching FeatureDef edits expanded 5 features
    # (btc_log_return_24h, btc_realized_vol_7d/30d, btc_volume_zscore_24h,
    # label_regime_risk_on_24h) to symbols=('BTCUSDT','ETHUSDT'). The
    # "btc_*" name prefix is now a historical scope artifact — the
    # transformers are symbol-agnostic and the values for symbol=ETHUSDT
    # are computed from ETH close_price / volume.
    #
    # This spec is the BTC-v3 twin with symbol swapped: same label
    # (label_regime_risk_on_24h), same input set (registry features
    # filtered by label_direction <> 'forward' and EXCLUDED_FROM_INPUTS),
    # same hyperparams. If the regime structure is comparable between
    # the two assets, the model should produce comparable AUC (~0.55-
    # 0.62 range from v2/v3 history). If ETH has a structurally
    # different regime distribution, AUC will diverge — and that's the
    # falsifying observation. Walk-forward expected to spot it.
    "regime_eth_v1": ModelSpec(
        name="regime_eth_v1",
        purpose="regime",
        label_feature="label_regime_risk_on_24h",
        label_version=1,
        objective="binary",
        symbol="ETHUSDT",
        interval="1h",
        train_start=_TRAIN_START,
        train_end=_TRAIN_END,
        # Empty — all inputs come from feature_registry rows whose
        # symbols array contains 'ETHUSDT'. V110 expansion gives us 4
        # input features + 1 label scoped for ETH. The cross-asset
        # eth_btc_corr_24h is BTC-stamped only by design (V77 note)
        # and is NOT in the ETH spec's input set — the model relies on
        # the 4 per-bar features alone for v1. If AUC > 0.55, a v2 can
        # add a cross-asset feature with ETH-stamped output as a
        # follow-up FeatureDef.
        derived_features=(),
    ),
    # 2026-05-21 — binary directional twin of directional_btc_1h_v1.
    # Resolves ANTI_PATTERN 20fa437f: orchestrator model_registry
    # validator only accepts objective ∈ {'binary', 'regression'} and
    # rejects 'multiclass' with HTTP 422. The v1 multiclass spec is
    # therefore structurally unregisterable and cannot be wired into a
    # HYBRID _ml_signal_name sweep regardless of gauntlet outcome. This
    # v2 is the minimum-surface fix: registerable peer that the
    # researcher loop can drive into HYBRID experiments.
    #
    # Reuses ``label_regime_risk_on_24h`` (V77 / V110 seed; binary
    # forward-Sharpe-sign label scoped to BTC+ETH × 1h, already proven
    # via regime_btc_v3 / regime_eth_v1). Forward-Sharpe sign is a
    # directional question — "did the next-24h risk-adjusted move
    # favour a long?" — so framing this as ``purpose="directional"``
    # rather than "regime" is honest and matches the HYBRID intended
    # use (entry-direction gate, not regime filter).
    #
    # Kept deliberately minimal vs v1: single base model (LightGBM),
    # no stacked-interval training (label only exists at 1h in the
    # registry), no meta-label gating, no per-fold feature selection,
    # default hyperparams. The point of v2 is to UNBLOCK the ML
    # hypothesis class. If walk-forward produces signal, a v3 can layer
    # ensemble + stacking back on for apples-to-apples comparison.
    "directional_btc_1h_v2": ModelSpec(
        name="directional_btc_1h_v2",
        purpose="directional",
        label_feature="label_regime_risk_on_24h",
        label_version=1,
        objective="binary",
        symbol="BTCUSDT",
        interval="1h",
        train_start=_TRAIN_START,
        train_end=_TRAIN_END,
        derived_features=(),
        # Cross-symbol OB exclusions: ETH order-book state is not a
        # valid input for a BTC model.
        extra_excluded_features=(
            "ob_spread_bps_eth",
            "ob_imbalance_eth",
            "ob_imbalance_momentum_8h_eth",
        ),
    ),
    # 2026-05-21: bar-boundary triple-barrier binary directional spec.
    # Addresses the label-misalignment failure mode that falsified the
    # v2 label_regime_risk_on_24h gate on three host strategies
    # (DCB-BTC, MMR-BTC, DCB-ETH). Forward-Sharpe-sign is a smoothed
    # aggregate; triple-barrier-binary asks the exact entry-gate
    # question "if I enter long here, does TP hit before SL within K
    # bars?". Same feature stack as v2 (registry-only). Single-base
    # LightGBM, no stacking, no meta-label.
    "directional_btc_1h_v3": ModelSpec(
        name="directional_btc_1h_v3",
        purpose="directional",
        label_feature="label_long_win_tb_1h_v1",
        label_version=1,
        objective="binary",
        symbol="BTCUSDT",
        interval="1h",
        train_start=_TRAIN_START,
        train_end=_TRAIN_END,
        derived_features=(),
        # Cross-symbol OB exclusions: ETH order-book state is not a
        # valid input for a BTC model.
        extra_excluded_features=(
            "ob_spread_bps_eth",
            "ob_imbalance_eth",
            "ob_imbalance_momentum_8h_eth",
        ),
    ),
    # 2026-05-21 (Path B): same triple-barrier binary label as v3, but
    # the registry-resolved input feature set is now AUGMENTED with 5
    # bar-level entry-timing features added via V111 migration:
    #   btc_rsi_14_1h        v1
    #   btc_atr_ratio_14_24  v1
    #   btc_log_return_1h    v1
    #   btc_log_return_4h    v1
    #   btc_volume_zscore_4h v1
    # The loader's _fetch_feature_matrix() pulls every per-bar feature
    # whose (symbols, intervals) match the spec (BTCUSDT, 1h), so adding
    # the rows to feature_registry + backfilling feature_values is all
    # that's needed for v4 to consume them — no derived_features = ()
    # change.
    # Rationale: prior-session RUN_SUMMARY a957d7cf — gate harm is
    # structural to a 24h-aggregation feature stack regardless of label
    # choice. v4 keeps the v3 label (label_long_win_tb_1h_v1) and adds
    # bar-level entry-timing signal so the model can locate "where in
    # the 1h cycle to enter", not just "what regime is the market in".
    "directional_btc_1h_v4": ModelSpec(
        name="directional_btc_1h_v4",
        purpose="directional",
        label_feature="label_long_win_tb_1h_v1",
        label_version=1,
        objective="binary",
        symbol="BTCUSDT",
        interval="1h",
        train_start=_TRAIN_START,
        train_end=_TRAIN_END,
        derived_features=(),
        # Cross-symbol OB exclusions: ETH order-book state is not a
        # valid input for a BTC model.
        extra_excluded_features=(
            "ob_spread_bps_eth",
            "ob_imbalance_eth",
            "ob_imbalance_momentum_8h_eth",
        ),
    ),
    # 2026-05-21 (Path C continuation): same registry feature stack as v4
    # (14 features including the 5 V111 bar-level entries) but consumes
    # the new short-horizon asymmetric-stops label label_long_win_tb_
    # short_v1. Hypothesis: the 24-bar v4 label's time-scope mismatched
    # the 1h decision cadence; a 6-bar label aligns the question "TP-first
    # in next 6h?" with the strategy's holding period and the bar-level
    # features' lookbacks (1h, 4h, 14-bar RSI).
    "directional_btc_1h_v5": ModelSpec(
        name="directional_btc_1h_v5",
        purpose="directional",
        label_feature="label_long_win_tb_short_v1",
        label_version=1,
        objective="binary",
        symbol="BTCUSDT",
        interval="1h",
        train_start=_TRAIN_START,
        train_end=_TRAIN_END,
        derived_features=(),
        # Cross-symbol OB exclusions: ETH order-book state is not a
        # valid input for a BTC model.
        extra_excluded_features=(
            "ob_spread_bps_eth",
            "ob_imbalance_eth",
            "ob_imbalance_momentum_8h_eth",
        ),
    ),
    # 2026-05-21 Path C late-session: same V111 feature stack + 6-bar
    # horizon as v5 but k_tp=1.0/k_sl=0.5 (looser). More positive labels
    # in training -> less restrictive gate -> may unblock V11 n>=100
    # while still preserving v5's POSITIVE_DELTA magnitude.
    "directional_btc_1h_v6": ModelSpec(
        name="directional_btc_1h_v6",
        purpose="directional",
        label_feature="label_long_win_tb_loose_v1",
        label_version=1,
        objective="binary",
        symbol="BTCUSDT",
        interval="1h",
        train_start=_TRAIN_START,
        train_end=_TRAIN_END,
        derived_features=(),
        # Cross-symbol OB exclusions: ETH order-book state is not a
        # valid input for a BTC model.
        extra_excluded_features=(
            "ob_spread_bps_eth",
            "ob_imbalance_eth",
            "ob_imbalance_momentum_8h_eth",
        ),
    ),
    # 2026-05-27: first SOLUSDT ML regime spec. Mirrors regime_eth_v2 with
    # three differences: symbol=SOLUSDT, interval=4h, train_start=2024-01-01.
    #
    # Background: DCB-SOLUSDT-4h is the highest-EV candidate from the SOLUSDT
    # archetype sweep (ag90=30.36%/yr, n=148) but blocked by PF CI [0.867,1.921]
    # spanning 1.0 and DSR=0.822. The regime gate on DCB-ETH-1h x regime_eth_v2
    # narrowed variance and produced the 4th strategy graduation (ag90=+94.99%/yr
    # ROBUST) -- same mechanism expected here.
    #
    # Features: 9 registry per-bar features (btc_log_return_1h/_4h/_24h,
    # btc_realized_vol_7d/_30d, btc_rsi_14_1h, btc_atr_ratio_14_24,
    # btc_volume_zscore_24h/_4h) now available for SOLUSDT x 4h via V123
    # migration + compute_features run (52,060 rows).
    #
    # Label: label_regime_risk_on_24h at 4h cadence -- horizon_bars=24 covers
    # 96h forward Sharpe (4-day regime). Class balance: 49.4%/50.6%.
    #
    # Train window: 2024-01-01 to 2026-05-14 (full 2-year SOL history;
    # NOT _TRAIN_START=2024-12-01 which would waste 11 months of SOL data).
    # Walk-forward needs ~3,200 bars for 6 folds; we have ~5,214 -- adequate.
    "regime_sol_v1": ModelSpec(
        name="regime_sol_v1",
        purpose="regime",
        label_feature="label_regime_risk_on_24h",
        label_version=1,
        objective="binary",
        symbol="SOLUSDT",
        interval="4h",
        train_start=datetime(2024, 1, 1),
        train_end=_TRAIN_END,
        derived_features=(),
        extra_excluded_features=(
            "fear_greed_value",
            "stablecoin_supply_change_7d",
            "stablecoin_supply_change_30d",
            "eth_btc_ratio_momentum_20d",
        ),
    ),
    # 2026-05-21 Path C continuation — sidecar-servable ETH regime spec.
    # Direct response to directional_eth_1h_v1's 13-gate gauntlet FAIL
    # (walk-forward AUC mean 0.534-0.538 across 3 hyperparam variants,
    # all below the 0.55 directional bar). The 5-gate modulator gauntlet
    # has a looser 0.52 AUC bar that this signal-feature combination
    # plausibly clears (regime_eth_v1 with the macro-augmented stack hit
    # AUC 0.545 and PASSed, so the bar-only stack at ~0.534 should still
    # PASS unless the macro features were doing all the work).
    #
    # Identical feature stack and label to directional_eth_1h_v1, only
    # difference is purpose='regime' so:
    #   (a) gauntlet dispatch routes through the 5-gate modulator gauntlet
    #       (the same gate set regime_eth_v1 cleared).
    #   (b) JVM HYBRID consumers can wire a signal_definition pointing
    #       at this model_id and use it via mlGateEnabled / mlGateShadowMode
    #       through the MLRegimeGateGuard.
    #   (c) content_sha256 differs from both regime_eth_v1 AND
    #       directional_eth_1h_v1 so the registry's forward-only
    #       lifecycle accepts this as a fresh row.
    "regime_eth_v2": ModelSpec(
        name="regime_eth_v2",
        purpose="regime",
        label_feature="label_regime_risk_on_24h",
        label_version=1,
        objective="binary",
        symbol="ETHUSDT",
        interval="1h",
        train_start=_TRAIN_START,
        train_end=_TRAIN_END,
        derived_features=(),
        # Macro feature exclusions — same as directional_eth_1h_v1.
        # These are the symbol=''/interval='' features the sidecar
        # cannot resolve. Without them the loader picks them up via
        # the global path and the trained artifact's feature_set is
        # unservable.
        # Cross-symbol OB exclusions: ob_*_btc features are stored
        # with symbols=NULL (global) so they land in every training
        # matrix. BTC order-book state is not a valid input for an
        # ETH regime model — exclude to prevent cross-symbol leakage.
        extra_excluded_features=(
            "fear_greed_value",
            "stablecoin_supply_change_7d",
            "stablecoin_supply_change_30d",
            "eth_btc_ratio_momentum_20d",
            "ob_spread_bps_btc",
            "ob_imbalance_btc",
            "ob_imbalance_momentum_8h_btc",
        ),
    ),
    "regime_eth_sharpe_v1": ModelSpec(
        name="regime_eth_sharpe_v1",
        purpose="regime",
        label_feature="label_forward_sharpe_24h",
        label_version=1,
        objective="regression",
        symbol="ETHUSDT",
        interval="1h",
        train_start=_TRAIN_START,
        train_end=_TRAIN_END,
        derived_features=(),
        extra_excluded_features=(
            "fear_greed_value",
            "stablecoin_supply_change_7d",
            "stablecoin_supply_change_30d",
            "eth_btc_ratio_momentum_20d",
            # Cross-symbol OB: BTC order-book state is not a valid
            # input for an ETH model.
            "ob_spread_bps_btc",
            "ob_imbalance_btc",
            "ob_imbalance_momentum_8h_btc",
        ),
    ),
    # 2026-06-01 Microstructure-enhanced ETH regime model.
    # Extends regime_eth_v2 with OFI (Order Flow Imbalance) features:
    # ofi_ratio, ofi_zscore_24h, ofi_momentum_8h, cvd_proxy_zscore_24h.
    # These are bar-level ETHUSDT/1h features — the sidecar CAN serve them
    # (symbol='ETHUSDT', interval='1h' rows in feature_values).
    # OFI features are deliberately NOT excluded so they participate in training.
    # Same macro exclusions as regime_eth_v2 (sidecar cannot resolve global
    # symbol=''/interval='' macro rows).
    "regime_eth_ofi_v1": ModelSpec(
        name="regime_eth_ofi_v1",
        purpose="regime",
        label_feature="label_regime_risk_on_24h",
        label_version=1,
        objective="binary",
        symbol="ETHUSDT",
        interval="1h",
        train_start=_TRAIN_START,
        train_end=_TRAIN_END,
        derived_features=(),
        extra_excluded_features=(
            "fear_greed_value",
            "stablecoin_supply_change_7d",
            "stablecoin_supply_change_30d",
            "eth_btc_ratio_momentum_20d",
            # Cross-symbol OB: BTC order-book state is not a valid
            # input for an ETH model.
            "ob_spread_bps_btc",
            "ob_imbalance_btc",
            "ob_imbalance_momentum_8h_btc",
        ),
    ),
    # 2026-05-21 Path C — first sidecar-servable ETH directional spec.
    # Resolves HARD_RULE_BLOCK_INFERENCE_STAMPING_2026-05-21 (RUN_SUMMARY
    # 21f4e296): blackheart-inference/repo/features.py exact-matches
    # (symbol, interval, ts) on feature_values, but the macro features
    # used by regime_eth_v1 / regime_btc_v3 are persisted symbol=''
    # interval='' at daily cadence, so the sidecar's
    # fetch_per_bar_values_at_ts cannot resolve them and returns
    # 409 feature_value_missing on every inference attempt.
    #
    # This spec consumes ONLY bar-level features that already have
    # symbol='ETHUSDT', interval='1h' rows in feature_values (V110
    # expanded the symbols array on 10 derived features; all 9 non-label
    # rows here verified present at 12,800+ rows each across 2024-12 to
    # 2026-05). The 'btc_' name prefix is a historical scope artifact —
    # the transformers are symbol-agnostic and compute from the symbol's
    # own close_price / volume.
    #
    # SAME LABEL as regime_eth_v1 (label_regime_risk_on_24h, the only
    # ETH-1h-stamped label in feature_registry) but purpose='directional'
    # so:
    #   (a) gauntlet dispatch routes through gauntlet_directional (13
    #       gates), matching how v4/v5/v6 are evaluated.
    #   (b) the HYBRID strategy's _ml_signal_name wiring sees this as a
    #       directional gate (entry-direction filter) rather than a
    #       regime modulator.
    #   (c) the content_sha256 differs from regime_eth_v1, so the
    #       model_registry forward-only lifecycle does not reject this
    #       registration even though regime_eth_v1 also exists.
    "directional_eth_1h_v1": ModelSpec(
        name="directional_eth_1h_v1",
        purpose="directional",
        label_feature="label_regime_risk_on_24h",
        label_version=1,
        objective="binary",
        symbol="ETHUSDT",
        interval="1h",
        train_start=_TRAIN_START,
        train_end=_TRAIN_END,
        # Empty — all 9 inputs come from feature_registry rows whose
        # symbols array contains 'ETHUSDT' AND intervals array contains
        # '1h'. The loader's _list_input_features filters on these and
        # automatically picks up the 9 sidecar-servable features:
        #   btc_atr_ratio_14_24, btc_log_return_1h, btc_log_return_4h,
        #   btc_log_return_24h, btc_realized_vol_7d, btc_realized_vol_30d,
        #   btc_rsi_14_1h, btc_volume_zscore_4h, btc_volume_zscore_24h.
        # No macro / cross-asset features (intentional — those are the
        # ones the sidecar cannot resolve today).
        derived_features=(),
        # Macro feature exclusions: these 4 features are stored at
        # symbol='', interval='' (daily global cadence) and are
        # unservable by the blackheart-inference sidecar's exact-match
        # fetch. Without this exclusion the loader picks them up via
        # its global-feature path and the trained artifact gets a
        # feature_set the sidecar cannot resolve at 1h ts.
        # Cross-symbol OB exclusions: ob_*_btc features are stored
        # with symbols=NULL (global) so they land in every training
        # matrix. BTC order-book state is not a valid input for an
        # ETH model — exclude to prevent cross-symbol leakage.
        extra_excluded_features=(
            "fear_greed_value",
            "stablecoin_supply_change_7d",
            "stablecoin_supply_change_30d",
            "eth_btc_ratio_momentum_20d",
            "ob_spread_bps_btc",
            "ob_imbalance_btc",
            "ob_imbalance_momentum_8h_btc",
        ),
    ),
    # Phase 3 / M5g foundation — directional model (blueprint § 6.2).
    # Targets triple-barrier outcomes (TP-hit / SL-hit / horizon-end).
    # Used standalone by the ML_DIRECTIONAL strategy if it clears the
    # full 13-gate gauntlet. M5g.1 shipped the multiclass training path;
    # M5g.2 added walk-forward support; M5g.3 enables the 3-model
    # ensemble via ``base_models`` (LightGBM + XGBoost + Logistic-L1
    # per blueprint § 6.2). Meta-label gating is M5g.4. Hyperparams
    # override ``n_estimators`` upward (3-class problem needs more trees
    # to separate three boundaries) and set ``class_weight=balanced``
    # so the 3%-frequency neutral class isn't ignored. The non-LightGBM
    # base models honour the same ``class_weight`` (XGBoost via
    # sample_weight translation, LogReg natively).
    #
    # NOTE 2026-05-21: this multiclass spec is currently UNREGISTERABLE
    # via the orchestrator model_registry endpoint (ANTI_PATTERN
    # 20fa437f — validator rejects objective='multiclass'). It still
    # trains fine and produces walk-forward metrics for research, but
    # the trained artifact cannot land in model_registry and so cannot
    # be wired into a HYBRID _ml_signal_name sweep. Use
    # directional_btc_1h_v2 (binary peer above) for HYBRID experiments
    # until the validator accepts a third literal.
    "directional_btc_1h_v1": ModelSpec(
        name="directional_btc_1h_v1",
        purpose="directional",
        label_feature="label_triple_barrier",
        label_version=1,
        objective="multiclass",
        symbol="BTCUSDT",
        interval="1h",
        train_start=_TRAIN_START,
        train_end=_TRAIN_END,
        hyperparams={
            "num_leaves": 31,
            "max_depth": -1,
            "learning_rate": 0.05,
            "n_estimators": 800,
            "min_child_samples": 50,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "verbosity": -1,
            "class_weight": "balanced",
        },
        base_models=("lightgbm", "xgboost", "logreg_l1"),
        meta_label_enabled=True,
        # M5g.5: stack 1h (serving) + 15m for 4× sample-size uplift.
        # The 15m branch uses the same triple-barrier algorithm
        # (k_tp=1.5, k_sl=1.0, horizon=24 bars) but at 15m cadence, so
        # the per-interval semantics differ — 1h horizon=24h, 15m
        # horizon=6h. The model conditions on ``interval_indicator``
        # to differentiate. Inference uses the 1h-only path.
        training_intervals=("1h", "15m"),
        # M5g.7: per-fold correlation + MI feature selection.
        feature_selection_enabled=True,
    ),
    # 2026-05-27: Funding-rate regime detector — orthogonal to price-action.
    # Mechanism hypothesis: BTC 8h funding rate encodes levered-market
    # crowding. Persistently positive = longs crowded (squeeze-vulnerable);
    # persistently negative = shorts crowded. The sign_streak and
    # percentile_30d features capture PERSISTENCE and RELATIVE ELEVATION
    # of funding, which price-action features cannot see.
    #
    # Feature set: btc_funding_8h, btc_funding_zscore_30d,
    # btc_funding_sign_streak, btc_funding_percentile_30d (V121).
    # All 4 are global (symbol='', interval='') in feature_values and are
    # forward-filled onto the 1h bar grid by the loader. ALL other registry
    # features are excluded via extra_excluded_features to keep the
    # funding signal isolated. If this model shows AUC > 0.55 in
    # walk-forward, it is genuinely orthogonal to regime_btc_v3 and can
    # be tested as a second modulator in the HYBRID strategy.
    #
    # purpose="positioning" routes through the 5-gate modulator gauntlet
    # (same as regime_*); the label is label_regime_risk_on_24h (binary
    # forward-Sharpe-sign, same as regime_btc_v3) so results are directly
    # comparable. The feature set is the only variable changing vs v3.
    "funding_regime_v1": ModelSpec(
        name="funding_regime_v1",
        purpose="positioning",
        label_feature="label_regime_risk_on_24h",
        label_version=1,
        objective="binary",
        symbol="BTCUSDT",
        interval="1h",
        train_start=_TRAIN_START,
        train_end=_TRAIN_END,
        derived_features=(),
        extra_excluded_features=(
            # Technical / price-action per-bar features (BTCUSDT 1h)
            "btc_log_return_1h",
            "btc_log_return_4h",
            "btc_log_return_24h",
            "btc_realized_vol_7d",
            "btc_realized_vol_30d",
            "btc_volume_zscore_24h",
            "btc_volume_zscore_4h",
            "btc_rsi_14_1h",
            "btc_atr_ratio_14_24",
            # Cross-asset (per-bar, BTCUSDT 1h)
            "eth_btc_corr_24h",
            # Global macro (FRED, daily)
            "vix_close",
            "dxy_close",
            "dxy_zscore_30d",
            "real_yield_10y_level",
            "real_yield_10y_change_20d",
            "dxy_zscore_252d",
            "dxy_momentum_20d",
            "vix_percentile_252d",
            "term_spread_2s10s",
            # Global flow / market_structure / sentiment
            "stablecoin_supply_change_7d",
            "stablecoin_supply_change_30d",
            "btc_dominance_change_7d",
            "eth_btc_ratio_momentum_20d",
            "fear_greed_value",
            # Cross-symbol OB: ETH order-book state is not a valid
            # input for a BTC model.
            "ob_spread_bps_eth",
            "ob_imbalance_eth",
            "ob_imbalance_momentum_8h_eth",
        ),
    ),
    # 2026-05-27: funding features with the CORRECT label.
    # funding_regime_v1 failed (AUC=0.50) because label_regime_risk_on_24h
    # is a price-action label — funding features cannot predict price
    # direction. This spec keeps the identical 4-feature funding set but
    # changes the label to label_long_win_tb_1h_v1, which asks "if I enter
    # long here, does TP hit before SL in 24 bars?" — a question funding
    # context can plausibly inform (crowded longs = elevated SL risk,
    # so the model should learn to suppress long entries when funding is
    # high). purpose="directional" routes through the 13-gate gauntlet.
    "funding_entry_v1": ModelSpec(
        name="funding_entry_v1",
        purpose="directional",
        label_feature="label_long_win_tb_1h_v1",
        label_version=1,
        objective="binary",
        symbol="BTCUSDT",
        interval="1h",
        train_start=_TRAIN_START,
        train_end=_TRAIN_END,
        derived_features=(),
        extra_excluded_features=(
            # Technical / price-action per-bar features (BTCUSDT 1h)
            "btc_log_return_1h",
            "btc_log_return_4h",
            "btc_log_return_24h",
            "btc_realized_vol_7d",
            "btc_realized_vol_30d",
            "btc_volume_zscore_24h",
            "btc_volume_zscore_4h",
            "btc_rsi_14_1h",
            "btc_atr_ratio_14_24",
            # Cross-asset (per-bar, BTCUSDT 1h)
            "eth_btc_corr_24h",
            # Global macro (FRED, daily)
            "vix_close",
            "dxy_close",
            "dxy_zscore_30d",
            "real_yield_10y_level",
            "real_yield_10y_change_20d",
            "dxy_zscore_252d",
            "dxy_momentum_20d",
            "vix_percentile_252d",
            "term_spread_2s10s",
            # Global flow / market_structure / sentiment
            "stablecoin_supply_change_7d",
            "stablecoin_supply_change_30d",
            "btc_dominance_change_7d",
            "eth_btc_ratio_momentum_20d",
            "fear_greed_value",
            # Cross-symbol OB: ETH order-book state is not a valid
            # input for a BTC model.
            "ob_spread_bps_eth",
            "ob_imbalance_eth",
            "ob_imbalance_momentum_8h_eth",
        ),
    ),
    # 2026-05-27: combined bar-level + funding features with the TB label.
    # directional_btc_1h_v4 used only bar-level features (RSI, ATR ratio,
    # log returns, volume) and still produced negative paired-delta despite
    # the correct label (label_long_win_tb_1h_v1). Hypothesis: bar-level
    # momentum alone isn't sufficient to predict entry quality — funding
    # context encodes levered-market crowding that momentum cannot see.
    # This spec adds the V121 funding features (btc_funding_8h,
    # btc_funding_zscore_30d, btc_funding_sign_streak,
    # btc_funding_percentile_30d) to v4's bar-level stack. Macro and
    # sentiment features are excluded to limit covariate-shift noise
    # and keep the feature set servable-in-principle by the sidecar once
    # funding features are promoted to per-bar cadence.
    "directional_btc_1h_v7": ModelSpec(
        name="directional_btc_1h_v7",
        purpose="directional",
        label_feature="label_long_win_tb_1h_v1",
        label_version=1,
        objective="binary",
        symbol="BTCUSDT",
        interval="1h",
        train_start=_TRAIN_START,
        train_end=_TRAIN_END,
        derived_features=(),
        # Exclude macro (FRED/daily) and sentiment features. Keep bar-level
        # BTCUSDT 1h features AND the V121 funding features (global cadence).
        extra_excluded_features=(
            "vix_close",
            "dxy_close",
            "dxy_zscore_30d",
            "real_yield_10y_level",
            "real_yield_10y_change_20d",
            "dxy_zscore_252d",
            "dxy_momentum_20d",
            "vix_percentile_252d",
            "term_spread_2s10s",
            "stablecoin_supply_change_7d",
            "stablecoin_supply_change_30d",
            "btc_dominance_change_7d",
            "eth_btc_ratio_momentum_20d",
            "fear_greed_value",
            # Cross-symbol OB: ETH order-book state is not a valid
            # input for a BTC model.
            "ob_spread_bps_eth",
            "ob_imbalance_eth",
            "ob_imbalance_momentum_8h_eth",
        ),
    ),
    # 2026-05-27: funding-crowding → squeeze hypothesis, SHORT side.
    # funding_entry_v1 tests whether funding predicts LONG entry quality.
    # This spec tests the other direction: persistently positive funding
    # → longs crowded → squeeze imminent → SHORT entries win. Uses the
    # new label_short_win_tb_1h_v1 (mirror of the long-win label with
    # inverted barriers). Same isolated 4-feature funding set as
    # funding_regime_v1 / funding_entry_v1.
    "funding_short_v1": ModelSpec(
        name="funding_short_v1",
        purpose="directional",
        label_feature="label_short_win_tb_1h_v1",
        label_version=1,
        objective="binary",
        symbol="BTCUSDT",
        interval="1h",
        train_start=_TRAIN_START,
        train_end=_TRAIN_END,
        derived_features=(),
        extra_excluded_features=(
            "btc_log_return_1h",
            "btc_log_return_4h",
            "btc_log_return_24h",
            "btc_realized_vol_7d",
            "btc_realized_vol_30d",
            "btc_volume_zscore_24h",
            "btc_volume_zscore_4h",
            "btc_rsi_14_1h",
            "btc_atr_ratio_14_24",
            "eth_btc_corr_24h",
            "vix_close",
            "dxy_close",
            "dxy_zscore_30d",
            "real_yield_10y_level",
            "real_yield_10y_change_20d",
            "dxy_zscore_252d",
            "dxy_momentum_20d",
            "vix_percentile_252d",
            "term_spread_2s10s",
            "stablecoin_supply_change_7d",
            "stablecoin_supply_change_30d",
            "btc_dominance_change_7d",
            "eth_btc_ratio_momentum_20d",
            "fear_greed_value",
            # Cross-symbol OB: ETH order-book state is not a valid
            # input for a BTC model.
            "ob_spread_bps_eth",
            "ob_imbalance_eth",
            "ob_imbalance_momentum_8h_eth",
        ),
    ),
    # 2026-05-31 — VBO HYBRID gate candidates.
    # regime_btc_v3 was retired; VBO_RESEARCH BTCUSDT HYBRID needs a fresh
    # BTC regime signal. regime_vbo_btc_v1 is a clean re-train of the v3
    # architecture under a new signal name so its lifecycle is independent
    # of the retired model.
    "regime_vbo_btc_v1": ModelSpec(
        name="regime_vbo_btc_v1",
        purpose="regime",
        label_feature="label_regime_risk_on_24h",
        label_version=1,
        objective="binary",
        symbol="BTCUSDT",
        interval="1h",
        train_start=_TRAIN_START,
        train_end=_TRAIN_END,
        derived_features=(),
        # Cross-symbol OB exclusions: ETH order-book state is not a
        # valid input for a BTC model.
        extra_excluded_features=(
            "ob_spread_bps_eth",
            "ob_imbalance_eth",
            "ob_imbalance_momentum_8h_eth",
        ),
    ),
    # v2 — excludes the 4 macro/global features (fear_greed, stablecoin,
    # eth_btc_ratio) that caused only 1 of 6 walk-forward folds to complete
    # in v1 (sparse global rows in some fold windows). Same exclusion set as
    # regime_vbo_eth_v1 so all 6 folds run successfully.
    "regime_vbo_btc_v2": ModelSpec(
        name="regime_vbo_btc_v2",
        purpose="regime",
        label_feature="label_regime_risk_on_24h",
        label_version=1,
        objective="binary",
        symbol="BTCUSDT",
        interval="1h",
        train_start=_TRAIN_START,
        train_end=_TRAIN_END,
        derived_features=(),
        extra_excluded_features=(
            "fear_greed_value",
            "stablecoin_supply_change_7d",
            "stablecoin_supply_change_30d",
            "eth_btc_ratio_momentum_20d",
            # Cross-symbol OB: ETH order-book state is not a valid
            # input for a BTC model.
            "ob_spread_bps_eth",
            "ob_imbalance_eth",
            "ob_imbalance_momentum_8h_eth",
        ),
    ),
    # ETH variant — allows VBO_RESEARCH ETHUSDT HYBRID sweeps to gate on
    # an ETH-specific regime signal independent of regime_eth_v2's lifecycle.
    "regime_vbo_eth_v1": ModelSpec(
        name="regime_vbo_eth_v1",
        purpose="regime",
        label_feature="label_regime_risk_on_24h",
        label_version=1,
        objective="binary",
        symbol="ETHUSDT",
        interval="1h",
        train_start=_TRAIN_START,
        train_end=_TRAIN_END,
        derived_features=(),
        extra_excluded_features=(
            "fear_greed_value",
            "stablecoin_supply_change_7d",
            "stablecoin_supply_change_30d",
            "eth_btc_ratio_momentum_20d",
            # Cross-symbol OB: BTC order-book state is not a valid
            # input for an ETH model.
            "ob_spread_bps_btc",
            "ob_imbalance_btc",
            "ob_imbalance_momentum_8h_btc",
        ),
    ),
    # 2026-06-02: OFI microstructure directional model for ETHUSDT.
    # regime_eth_ofi_v1 was falsified (adversarial_auc=0.9983) because
    # regime classification via the forward-Sharpe label suffers severe
    # covariate shift on OFI features. Reframed as a directional (entry-
    # quality) model using the ETH triple-barrier label: the question
    # changes from "is the market regime risk-on?" to "if I enter long
    # here, does TP hit before SL in 24 bars?" — a question OFI context
    # can plausibly answer (high buy-side imbalance → elevated TP probability).
    # Uses label_long_win_tb_eth_1h_v1 (new ETH-native derived label),
    # purpose="directional" → routes through 13-gate gauntlet_directional.
    # Feature set: all sidecar-servable ETHUSDT/1h features PLUS the 4 OFI
    # features (ofi_ratio, ofi_zscore_24h, ofi_momentum_8h,
    # cvd_proxy_zscore_24h) which are in feature_values at ~406k rows 2022-26.
    # Excludes macro/global features (sidecar cannot resolve daily cadence)
    # and cross-symbol BTC OB (not valid ETH inputs).
    "directional_eth_ofi_1h_v1": ModelSpec(
        name="directional_eth_ofi_1h_v1",
        purpose="directional",
        label_feature="label_long_win_tb_eth_1h_v1",
        label_version=1,
        objective="binary",
        symbol="ETHUSDT",
        interval="1h",
        train_start=_TRAIN_START,
        train_end=_TRAIN_END,
        derived_features=(),
        extra_excluded_features=(
            # Macro / global daily features — sidecar cannot resolve
            "fear_greed_value",
            "stablecoin_supply_change_7d",
            "stablecoin_supply_change_30d",
            "eth_btc_ratio_momentum_20d",
            "vix_close",
            "dxy_close",
            "dxy_zscore_30d",
            "real_yield_10y_level",
            "real_yield_10y_change_20d",
            "dxy_zscore_252d",
            "dxy_momentum_20d",
            "vix_percentile_252d",
            "term_spread_2s10s",
            "btc_dominance_change_7d",
            # Cross-symbol OB: BTC order-book state is not valid for ETH
            "ob_spread_bps_btc",
            "ob_imbalance_btc",
            "ob_imbalance_momentum_8h_btc",
        ),
    ),
    # 2026-06-02: BTC OFI directional — pivot from falsified ETH variants.
    # directional_eth_ofi_1h_v1 and regime_eth_ofi_v1 both showed
    # adversarial_auc ≈ 1.0 (structural OFI covariate shift on ETH 1h).
    # BTC OFI may behave differently: deeper liquidity, more institutional
    # participation, and larger absolute volume may produce more stationary
    # OFI distributions. Uses the existing label_long_win_tb_1h_v1 (BTC TB).
    # OFI features for BTCUSDT verified present at ~406k rows 2022-2026.
    "directional_btc_ofi_1h_v1": ModelSpec(
        name="directional_btc_ofi_1h_v1",
        purpose="directional",
        label_feature="label_long_win_tb_1h_v1",
        label_version=1,
        objective="binary",
        symbol="BTCUSDT",
        interval="1h",
        train_start=_TRAIN_START,
        train_end=_TRAIN_END,
        derived_features=(),
        extra_excluded_features=(
            # Macro / global daily features
            "fear_greed_value",
            "stablecoin_supply_change_7d",
            "stablecoin_supply_change_30d",
            "eth_btc_ratio_momentum_20d",
            "vix_close",
            "dxy_close",
            "dxy_zscore_30d",
            "real_yield_10y_level",
            "real_yield_10y_change_20d",
            "dxy_zscore_252d",
            "dxy_momentum_20d",
            "vix_percentile_252d",
            "term_spread_2s10s",
            "btc_dominance_change_7d",
            # Cross-symbol OB: ETH order-book state not valid for BTC
            "ob_spread_bps_eth",
            "ob_imbalance_eth",
            "ob_imbalance_momentum_8h_eth",
        ),
    ),
    # 2026-06-02: ETH OFI at 4h interval.
    # ETH 1h OFI showed adversarial_auc ≈ 1.0 (covariate shift in the
    # 1h feature distribution). Longer horizon (4h aggregation) reduces
    # noise and may produce more stationary OFI distributions. The 4h
    # ETH OFI features (~406k rows 2022-2026) represent different temporal
    # aggregation of the same microstructure signal.
    # Uses ETH triple-barrier label at 4h — label_long_win_tb_eth_1h_v1
    # is available (computed from ETHUSDT 1h bars; the horizon_bars=24
    # maps to 24h at 1h and 96h at 4h — a different holding period).
    "directional_eth_ofi_4h_v1": ModelSpec(
        name="directional_eth_ofi_4h_v1",
        purpose="directional",
        label_feature="label_long_win_tb_eth_1h_v1",
        label_version=1,
        objective="binary",
        symbol="ETHUSDT",
        interval="4h",
        train_start=_TRAIN_START,
        train_end=_TRAIN_END,
        derived_features=(),
        extra_excluded_features=(
            "fear_greed_value",
            "stablecoin_supply_change_7d",
            "stablecoin_supply_change_30d",
            "eth_btc_ratio_momentum_20d",
            "vix_close",
            "dxy_close",
            "dxy_zscore_30d",
            "real_yield_10y_level",
            "real_yield_10y_change_20d",
            "dxy_zscore_252d",
            "dxy_momentum_20d",
            "vix_percentile_252d",
            "term_spread_2s10s",
            "btc_dominance_change_7d",
            # Cross-symbol OB: BTC order-book state not valid for ETH
            "ob_spread_bps_btc",
            "ob_imbalance_btc",
            "ob_imbalance_momentum_8h_btc",
        ),
    ),
}


def get_spec(name: str) -> ModelSpec:
    if name not in SPECS:
        known = ", ".join(sorted(SPECS.keys()))
        raise KeyError(f"Unknown model spec '{name}'. Known: {known}")
    return SPECS[name]
