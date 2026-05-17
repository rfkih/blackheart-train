"""Bayesian hyperparameter search via Optuna (R2 Session 1).

Replaces the M5b 8-point grid (``search.py``) with a TPE-sampled
search over a configurable LightGBM hyperparameter space. Each trial
runs a full walk-forward and scores on the mean primary metric across
folds — the same thing the gauntlet's gate 2 grades.

Why walk-forward as the objective (not the M5b 80/20 split):

* The 80/20 split optimises for a *single* validation window. With
  hourly-bar data and regime drift, that window's score is noisy +
  regime-dependent; the optimum on it doesn't necessarily transfer.
* The 13-gate gauntlet binds on the walk-forward result anyway. If
  we pick HPs against the 80/20 split, half the post-search gauntlet
  gates are evaluating a configuration that wasn't optimised for them.
* The cost is real — ~6 fits per trial (six folds). A 30-trial sweep
  on real data takes 20-60 min depending on spec. Acceptable for the
  "one alpha lifecycle at a time" cadence.

What's intentionally NOT in session 1 (saved for later sessions):

* **DSR objective.** OOF DSR is methodologically purer than mean AUC
  but requires the OOF predictions → per-trade returns translation
  (the Phase 4 backtest path). Session 1 stays library-only; the
  ``objective_fn`` parameter is the seam where a DSR-based scorer can
  drop in later without changing the sweep machinery.
* **CLI flag + tracking integration.** Session 4 wires ``--bayesian``
  and pipes per-trial metrics into ``experiment_metric.step``.

Session 2 additions (2026-05-17):

* **MedianPruner.** Installs Optuna's median-rule pruner so a trial
  that's tracking below the median at fold k gets stopped early.
  Caveat: pruning only fires when the scorer reports per-fold
  intermediates via ``trial.report(value, step)``. The default
  ``_walk_forward_score`` does NOT report intermediates today (it
  delegates to ``run_walk_forward`` which is atomic over folds). The
  mechanism is in place for a future walk_forward refactor; tests use
  a stub scorer that exercises the pruning path end-to-end.
* **timeout_s wall-clock cap.** Optuna's native ``study.optimize(
  timeout=...)`` honours this hard — works today, no scorer
  cooperation needed. Use it for autonomous-loop sweeps where "stop
  after N minutes regardless of progress" is the right safety net.
* **Sweep telemetry.** ``BayesianSearchResult`` now carries
  ``n_trials_completed``, ``n_trials_pruned``, ``n_trials_failed``,
  ``wall_seconds`` so the tracking client (session 4) can log them.

Return shape mirrors :class:`search.SearchResult` so the existing CLI
path (``cli.py:tuned_spec(spec, best_overrides)``) accepts our result
without branching. The two dataclasses are not interchangeable at type-
check level (different names) but the field names align — see
:func:`bayesian_search_result_to_dict` for the wire format.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, replace
from typing import Any, Callable

import time

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from .loader import LoadedDataset
from .search import primary_metric
from .specs import ModelSpec
from .walk_forward import run_walk_forward

logger = logging.getLogger(__name__)


# ── Defaults ─────────────────────────────────────────────────────────────


#: Default trial count. TPE typically converges within 20-50 trials on
#: 5-10 dimensional spaces; 30 is a middle-ground default that runs in
#: ~30-60 min on real data and gives the sampler enough exploration
#: budget. Caller can override via ``n_trials``.
DEFAULT_N_TRIALS: int = 30

#: Seed for both Optuna's TPESampler and the per-trial LightGBM
#: ``random_state``. Fixing this makes sweeps deterministic: two runs
#: with the same dataset + spec + seed produce identical search
#: trajectories. The seed is also recorded in the result so the CLI's
#: artifact payload carries the search's reproducibility key.
DEFAULT_SEED: int = 7

#: Number of folds in the walk-forward used per trial. Matches
#: ``walk_forward.DEFAULT_N_FOLDS`` — set explicitly here so a future
#: walk_forward refactor that changes the default doesn't silently
#: change the sweep's accuracy/runtime tradeoff.
DEFAULT_N_FOLDS: int = 6

#: MedianPruner parameters (R2.S2). ``n_startup_trials`` keeps the
#: first N trials unprunable so the sampler has enough data to compute
#: a meaningful median. ``n_warmup_steps`` lets the first N intermediate
#: reports through unconditionally — a trial can look weak after one
#: fold and recover; we want to see at least two fold reports before
#: judging.
DEFAULT_PRUNER_STARTUP_TRIALS: int = 5
DEFAULT_PRUNER_WARMUP_STEPS: int = 2


# ── Result dataclasses (mirror search.SearchResult) ──────────────────────


@dataclass
class BayesianTrialResult:
    """One trial's record. Mirrors :class:`search.SearchRunResult` with
    one extra field — ``trial_number`` — so callers can correlate with
    Optuna's study log if they want to dig deeper.
    """

    trial_number: int
    overrides: dict[str, object]
    metrics: dict[str, float]
    score: float
    state: str   # "COMPLETE" / "PRUNED" / "FAIL"


@dataclass
class BayesianSearchResult:
    """Top-level result. Field names match
    :class:`search.SearchResult` for drop-in CLI compatibility — the
    CLI's ``tuned_spec(spec, result.best_overrides)`` works identically.
    Additional fields:

    * ``n_trials_run`` — total trials attempted (completed + pruned + failed).
    * ``n_trials_completed`` — trials that ran to the final fold (R2.S2).
    * ``n_trials_pruned`` — trials Optuna's pruner cut short (R2.S2).
    * ``n_trials_failed`` — trials where the scorer raised (R2.S2).
    * ``wall_seconds`` — total study wall-clock (R2.S2). Useful for
      sizing the next sweep's ``timeout_s``.
    * ``sampler`` / ``seed`` — reproducibility metadata.
    """

    spec_name: str
    primary_metric: str
    best_overrides: dict[str, object]
    best_metric: float
    baseline_metric: float | None
    runs: list[BayesianTrialResult] = field(default_factory=list)
    n_trials_run: int = 0
    n_trials_completed: int = 0
    n_trials_pruned: int = 0
    n_trials_failed: int = 0
    wall_seconds: float = 0.0
    sampler: str = "TPESampler"
    pruner: str = "MedianPruner"
    seed: int = DEFAULT_SEED


class BayesianSearchError(RuntimeError):
    """Raised when the sweep cannot produce a usable result —
    typically every trial's primary metric came back NaN. Caller (CLI
    or test) converts to a clean error rather than picking the first
    trial.
    """


# ── Search space ─────────────────────────────────────────────────────────


def _suggest_lightgbm_hps(trial: optuna.Trial, objective: str) -> dict[str, object]:
    """Define the LightGBM search space for a trial.

    Choices reflect what's actually controllable in the gauntlet's
    regime — the existing 80/20 grid covers ``num_leaves``,
    ``learning_rate``, ``min_child_samples``; we expand to also tune
    bagging + feature subsampling + L1/L2 regularisation, which are
    the standard LightGBM levers for handling regime drift.

    The objective branch sets the multiclass path's slightly different
    ranges — multiclass needs deeper trees and lower LR to escape the
    triple-barrier majority-class collapse (same intuition as the M5g.1
    multiclass grid in search.py).
    """
    if objective == "multiclass":
        num_leaves_range = (31, 255)
        lr_range = (1e-3, 0.1)
    else:
        # binary + regression share the same ranges — the underlying
        # LightGBM kernel is the same, only the objective + loss differ.
        num_leaves_range = (15, 127)
        lr_range = (5e-3, 0.2)

    return {
        "num_leaves": trial.suggest_int("num_leaves", *num_leaves_range, log=True),
        "learning_rate": trial.suggest_float("learning_rate", *lr_range, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 200, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        # bagging_freq=0 disables bagging; positive values enable per-N-iteration
        # re-sampling. Keep 0 in the space so the sampler can find "bagging off
        # is best" without us removing it post-hoc.
        "bagging_freq": trial.suggest_int("bagging_freq", 0, 7),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
    }


# ── Objective ─────────────────────────────────────────────────────────────


# Type signature for the per-trial scorer. Takes a tuned spec + dataset
# + a fold-count, returns ``(score, metrics_dict)``. ``trial`` is passed
# as a kwarg so scorers that want to report intermediate fold scores for
# pruning can call ``trial.report(value, step)`` + ``trial.should_prune()``.
# Default scorer is :func:`_walk_forward_score`; tests inject a stub.
#
# R2.S2 note: the default scorer doesn't yet report intermediates —
# ``run_walk_forward`` is atomic over folds and would need a callback
# refactor to surface them mid-flight. That's a separate change. The
# trial-kwarg seam is in place so a future fold-by-fold scorer drops in
# without touching the search machinery.
ObjectiveFn = Callable[..., tuple[float, dict[str, float]]]


def _walk_forward_score(
    spec: ModelSpec, ds: LoadedDataset, n_folds: int, *,
    trial: optuna.Trial | None = None,
) -> tuple[float, dict[str, float]]:
    """Default objective — mean of the primary metric across walk-forward
    folds. Returns ``(score, extra_metrics)`` where ``extra_metrics`` is
    the full ``metric_means`` dict for audit logging.

    NaN handling: a NaN ``primary_mean`` (every fold degenerate) returns
    score ``-inf`` so it never wins. The trial's state stays COMPLETE —
    Optuna will simply not select this region.

    R2 Bug-#5 fix: ``+inf`` is also clamped to ``-inf``. A perfect-
    separation fit (or any degeneracy that produces an infinite metric)
    would otherwise dominate Optuna's maximize loop and converge the
    sampler on a clearly broken region.

    ``trial`` is accepted but unused — see the module-level note about
    why pruning isn't active for the default scorer. Stubs in the test
    suite use ``trial.report(...)`` to exercise the pruner path.
    """
    result = run_walk_forward(ds, spec, n_folds=n_folds)
    score = result.primary_mean
    if score is None or math.isnan(score) or math.isinf(score):
        return float("-inf"), dict(result.metric_means or {})
    return float(score), dict(result.metric_means or {})


def _make_optuna_objective(
    ds: LoadedDataset, spec: ModelSpec, *,
    n_folds: int,
    objective_fn: ObjectiveFn,
    runs: list[BayesianTrialResult],
    on_trial_complete: Callable[[BayesianTrialResult], None] | None = None,
) -> Callable[[optuna.Trial], float]:
    """Curry the Optuna objective. ``runs`` is the mutable list each
    trial appends to — avoids leaking Optuna's Study object back to the
    caller while preserving the audit trail.

    R2.S2: passes ``trial`` to the scorer as a keyword argument so
    pruning-aware scorers can call ``trial.report`` / ``trial.should_prune``.
    Catches ``optuna.TrialPruned`` from the scorer separately from generic
    exceptions so the result record reflects the right state.

    R2 Bug-#3 fix: ``on_trial_complete`` is called with each appended
    BayesianTrialResult (whether COMPLETE / PRUNED / FAIL) so the CLI
    can stream per-trial telemetry to the tracking client AS trials
    finish, not after the whole sweep completes. A failure that aborts
    the sweep no longer loses the trajectory of trials that did finish.
    Callback exceptions are caught + logged so a misbehaving callback
    can't crash the sweep.
    """
    def _emit(record: BayesianTrialResult) -> None:
        runs.append(record)
        if on_trial_complete is not None:
            try:
                on_trial_complete(record)
            except Exception:
                logger.warning(
                    "bayesian-search: on_trial_complete callback raised "
                    "for trial %d — telemetry may be incomplete",
                    record.trial_number, exc_info=True,
                )

    def _trial_fn(trial: optuna.Trial) -> float:
        overrides = _suggest_lightgbm_hps(trial, spec.objective)
        merged = {**spec.hyperparams, **overrides}
        tuned = replace(spec, hyperparams=merged)
        try:
            score, metric_means = objective_fn(tuned, ds, n_folds, trial=trial)
        except optuna.TrialPruned:
            # Pruner cut this trial short. Optuna's in-flight Trial
            # object doesn't expose intermediate_values directly (only
            # the post-completion FrozenTrial does), so we record -inf
            # as the trial's effective score — the state=PRUNED tag is
            # the audit signal. If a future caller needs the exact last
            # intermediate, query `study.trials[trial.number]` after
            # `study.optimize` returns.
            _emit(BayesianTrialResult(
                trial_number=trial.number,
                overrides=dict(overrides),
                metrics={},
                score=float("-inf"),
                state="PRUNED",
            ))
            logger.info(
                "bayesian-search trial %d pruned | spec=%s",
                trial.number, spec.name,
            )
            raise   # re-raise: Optuna needs to see TrialPruned to update its state
        except Exception:
            # R2 Bug-#1 fix (2026-05-17): re-raise. Previously this
            # returned float("-inf") to Optuna, which records the trial
            # as COMPLETE with value -inf — TPE then treats the crashed
            # point as a legitimate -inf observation and biases its
            # belief about the search space. With ``catch=(Exception,)``
            # on the outer ``study.optimize`` call, re-raising makes
            # Optuna mark the trial as state=FAIL (excluded from TPE's
            # samples) while still keeping the sweep alive.
            logger.exception(
                "bayesian-search trial %d failed | spec=%s overrides=%s",
                trial.number, spec.name, overrides,
            )
            _emit(BayesianTrialResult(
                trial_number=trial.number,
                overrides=dict(overrides),
                metrics={},
                score=float("-inf"),
                state="FAIL",
            ))
            raise

        _emit(BayesianTrialResult(
            trial_number=trial.number,
            overrides=dict(overrides),
            metrics=dict(metric_means),
            score=score,
            state="COMPLETE",
        ))
        logger.info(
            "bayesian-search trial %d | spec=%s score=%.4f overrides=%s",
            trial.number, spec.name, score, overrides,
        )
        return score
    return _trial_fn


# ── Entry point ──────────────────────────────────────────────────────────


def bayesian_search_one(
    ds: LoadedDataset,
    spec: ModelSpec,
    *,
    n_trials: int = DEFAULT_N_TRIALS,
    seed: int = DEFAULT_SEED,
    n_folds: int = DEFAULT_N_FOLDS,
    timeout_s: float | None = None,
    objective_fn: ObjectiveFn | None = None,
    pruner_startup_trials: int = DEFAULT_PRUNER_STARTUP_TRIALS,
    pruner_warmup_steps: int = DEFAULT_PRUNER_WARMUP_STEPS,
    on_trial_complete: Callable[[BayesianTrialResult], None] | None = None,
) -> BayesianSearchResult:
    """Bayesian search via Optuna TPESampler with MedianPruner.

    Drop-in replacement for :func:`search.grid_search_one`. Returns a
    :class:`BayesianSearchResult` with ``best_overrides`` compatible
    with :func:`search.tuned_spec`.

    Parameters
    ----------
    n_trials
        Hard cap on trials. The actual number run may be less when
        ``timeout_s`` fires first.
    timeout_s
        R2.S2: optional wall-clock cap in seconds. Optuna stops kicking
        off new trials when this fires; the in-flight trial is allowed
        to finish (no cooperative cancellation). Use for the autonomous
        loop's "stop after N minutes" safety net.
    n_folds
        Walk-forward folds per trial. Same default as
        :data:`DEFAULT_N_FOLDS`.
    objective_fn
        Seam for tests / future scorers. Production code leaves None
        to use the default walk-forward scorer. The signature is
        ``(spec, ds, n_folds, *, trial=None) -> (score, extra_metrics)``;
        scorers that want pruning call ``trial.report(value, step)``
        and ``trial.should_prune()``.
    pruner_startup_trials / pruner_warmup_steps
        MedianPruner knobs. Keep first N trials unprunable so the
        sampler has data to compute medians; let the first M
        intermediate reports through unconditionally so a slow-starting
        trial isn't killed prematurely.

    Raises
    ------
    BayesianSearchError
        Every non-pruned, non-failed trial returned ``-inf`` (all
        regions of the search space are degenerate). Surfaces rather
        than silently picking the first trial.
    ValueError
        ``n_trials < 1`` or ``timeout_s <= 0``.
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1; got {n_trials}")
    if timeout_s is not None and timeout_s <= 0:
        raise ValueError(f"timeout_s must be > 0 when set; got {timeout_s}")
    metric_name = primary_metric(spec.objective)
    objective_fn = objective_fn or _walk_forward_score

    runs: list[BayesianTrialResult] = []

    # Silence Optuna's INFO-level chatter (one line per trial) — we log
    # our own per-trial summary at INFO and the duplicate is noise.
    # R2 Bug-#7 fix: save the prior verbosity and restore it on exit so
    # this function doesn't permanently mutate global Optuna logging
    # state for concurrent / subsequent sweeps in the same process.
    _prior_verbosity = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = TPESampler(seed=seed)
    # R2.S2: MedianPruner. n_startup_trials keeps early trials safe so
    # the sampler has signal; n_warmup_steps gives each trial a grace
    # period before pruning kicks in.
    pruner = MedianPruner(
        n_startup_trials=pruner_startup_trials,
        n_warmup_steps=pruner_warmup_steps,
    )
    study = optuna.create_study(
        direction="maximize", sampler=sampler, pruner=pruner,
    )
    objective = _make_optuna_objective(
        ds, spec, n_folds=n_folds, objective_fn=objective_fn, runs=runs,
        on_trial_complete=on_trial_complete,
    )

    logger.info(
        "bayesian-search start | spec=%s n_trials=%d seed=%d n_folds=%d "
        "metric=%s timeout_s=%s",
        spec.name, n_trials, seed, n_folds, metric_name,
        f"{timeout_s:.1f}" if timeout_s is not None else "off",
    )
    wall_start = time.monotonic()
    # ``catch=(Exception,)`` lets a single failing trial be marked FAIL
    # in Optuna without aborting the whole sweep. Coupled with R2 Bug-#1
    # fix in the trial fn — re-raising exceptions instead of returning
    # -inf — this keeps TPE's belief clean of crashed-trial contamination.
    # ``optuna.TrialPruned`` is handled by Optuna's framework regardless
    # of the catch tuple, so pruning still marks state=PRUNED.
    try:
        study.optimize(
            objective, n_trials=n_trials, timeout=timeout_s,
            show_progress_bar=False, catch=(Exception,),
        )
    finally:
        # R2 Bug-#7 fix: restore Optuna's global log verbosity even if
        # optimize() raises (catch=(Exception,) makes that unlikely, but
        # KeyboardInterrupt and TimeoutError can still escape).
        optuna.logging.set_verbosity(_prior_verbosity)
    wall_seconds = time.monotonic() - wall_start

    # R2.S2 telemetry: count terminal states. The scorer's PRUNED/FAIL
    # bookkeeping lives in ``runs``; we tally from there so the numbers
    # match what callers see, not Optuna's internal trial enumeration
    # (which can race when timeout_s cuts a trial mid-flight).
    n_completed = sum(1 for r in runs if r.state == "COMPLETE")
    n_pruned = sum(1 for r in runs if r.state == "PRUNED")
    n_failed = sum(1 for r in runs if r.state == "FAIL")

    # All-degenerate guard. Only counts COMPLETE trials — PRUNED and
    # FAIL are expected attrition. If every COMPLETE trial returned
    # -inf the sweep has no signal.
    complete_scores = [r.score for r in runs if r.state == "COMPLETE"]
    if complete_scores and all(s == float("-inf") for s in complete_scores):
        raise BayesianSearchError(
            f"bayesian search degenerate for spec={spec.name}: "
            f"every one of {len(complete_scores)} completed trial(s) returned "
            f"-inf on primary metric '{metric_name}'. Check the dataset / spec "
            f"/ search space."
        )
    if not complete_scores:
        raise BayesianSearchError(
            f"bayesian search produced no completed trials for spec={spec.name}: "
            f"{n_pruned} pruned, {n_failed} failed in {wall_seconds:.1f}s. "
            f"Lower the pruner aggressiveness or extend timeout_s."
        )

    best_trial = study.best_trial
    best_overrides = dict(best_trial.params)
    best_score = float(study.best_value)

    # Baseline: a single walk-forward fit with the spec's default HPs.
    # We rerun even if the defaults happen to be the best — keeps the
    # baseline computation deterministic and uncoupled from sampler
    # exploration. ~one extra walk-forward; acceptable cost for the
    # cleaner audit semantic. Note: baseline doesn't go through the
    # pruner (trial=None) so it always runs to completion.
    try:
        baseline_score, _ = objective_fn(spec, ds, n_folds, trial=None)
    except Exception:
        logger.exception("bayesian-search baseline fit failed | spec=%s", spec.name)
        baseline_score = float("nan")

    baseline_metric = (
        float(baseline_score) if not math.isnan(baseline_score) and baseline_score != float("-inf")
        else None
    )

    logger.info(
        "bayesian-search best | spec=%s %s=%.4f overrides=%s baseline=%s "
        "| completed=%d pruned=%d failed=%d wall=%.1fs",
        spec.name, metric_name, best_score, best_overrides,
        f"{baseline_metric:.4f}" if baseline_metric is not None else "N/A",
        n_completed, n_pruned, n_failed, wall_seconds,
    )

    return BayesianSearchResult(
        spec_name=spec.name,
        primary_metric=metric_name,
        best_overrides=best_overrides,
        best_metric=best_score,
        baseline_metric=baseline_metric,
        runs=runs,
        n_trials_run=len(runs),
        n_trials_completed=n_completed,
        n_trials_pruned=n_pruned,
        n_trials_failed=n_failed,
        wall_seconds=wall_seconds,
        seed=seed,
    )


# ── Serialization ────────────────────────────────────────────────────────


def bayesian_search_result_to_dict(result: BayesianSearchResult) -> dict[str, Any]:
    """JSON-friendly view — used by the CLI summary and any future
    artifact payload that records the sweep trajectory. Field names
    match :func:`search.search_result_to_dict` plus the bayesian-only
    extras (``n_trials_run``, ``sampler``, ``seed``).
    """
    return {
        "spec_name": result.spec_name,
        "primary_metric": result.primary_metric,
        "best_overrides": result.best_overrides,
        "best_metric": result.best_metric,
        "baseline_metric": result.baseline_metric,
        "n_trials_run": result.n_trials_run,
        "n_trials_completed": result.n_trials_completed,
        "n_trials_pruned": result.n_trials_pruned,
        "n_trials_failed": result.n_trials_failed,
        "wall_seconds": result.wall_seconds,
        "sampler": result.sampler,
        "pruner": result.pruner,
        "seed": result.seed,
        "trials": [
            {
                "trial_number": t.trial_number,
                "overrides": t.overrides,
                "metrics": t.metrics,
                "score": t.score,
                "state": t.state,
            }
            for t in result.runs
        ],
    }
