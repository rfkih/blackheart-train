"""CLI entry point.

Usage::

    # Train one model with the spec's default hyperparams (M5a behaviour)
    python -m blackheart_train.cli --model regime_btc_v1

    # Train one model after a small hyperparam grid search (M5b)
    python -m blackheart_train.cli --model regime_btc_v1 --search

    # Train all three sub-models (M5b deliverable)
    python -m blackheart_train.cli --model all --search

    # Smoke / no persistence
    python -m blackheart_train.cli --model regime_btc_v1 --no-write

    # Diagnostics: embed traceback in error JSON for offline post-mortem
    python -m blackheart_train.cli --model all --search --verbose

Prints a JSON summary on stdout. Logs go to stderr. Exit code 0 on
success, 1 if any model failed.

JSON discipline: this module produces strict RFC-7159 JSON. NaN and
Infinity are replaced with ``null`` before serialisation so downstream
parsers (jq, the orchestrator's pydantic models) never see invalid
tokens. The strict path is anchored by ``allow_nan=False`` in
``json.dump`` — if sanitisation misses a value, the dump raises rather
than silently emitting bad JSON.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import math
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .artifacts import write_artifact
from .bayesian_search import (
    BayesianSearchError,
    bayesian_search_one,
    bayesian_search_result_to_dict,
)
from .gauntlet import GauntletError, gauntlet_to_dict, run_gauntlet
from .gauntlet_directional import (
    DirectionalGauntletError,
    directional_gauntlet_to_dict,
    run_directional_gauntlet,
)
from .integrity import compute_dataset_sha
from .loader import FeatureCache, load_dataset, load_stacked_dataset
from .register_client import RegisterError, register_with_orchestrator
from .search import grid_search_one, search_result_to_dict, tuned_spec
from .settings import get_settings
from .specs import SPECS, get_spec
from .stacking import StackingError, stacker_to_dict, train_stacker
from .tracking import ExperimentClient, derive_summary_tags, extract_run_metrics
from .train import train_with_dataset
from .walk_forward import train_via_walk_forward


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _json_default(obj: Any) -> Any:
    """Strict JSON encoder fallback. Handles known non-JSON-native types
    explicitly and raises on anything else, so a future payload type
    can't be silently ``str()``-coerced and slipped into the output.
    """
    if isinstance(obj, _dt.datetime):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively replace NaN/Infinity with ``None`` so the output is
    RFC-7159 JSON. Downstream consumers (jq, browser fetch, pydantic)
    reject the bare ``NaN`` token Python's json module emits by default.

    Walks dicts, lists, and tuples. Leaves other types untouched —
    handled either natively by json or by :func:`_json_default`.
    """
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(x) for x in obj]
    return obj


def _run_one_model(
    spec_name: str,
    *,
    do_search: bool,
    do_walk_forward: bool,
    do_gauntlet: bool,
    do_write: bool,
    do_register: bool,
    artifact_dir: Path,
    orchestrator_url: str,
    orchestrator_token: str,
    orchestrator_timeout_s: float,
    agent_name: str,
    tracking_client: ExperimentClient | None = None,
    feature_cache: FeatureCache | None = None,
    last_n_folds: int | None = None,
    allow_leakage: bool = False,
    do_bayesian: bool = False,
    bayesian_trials: int = 30,
    bayesian_timeout_s: float | None = None,
    stack_top_k: int = 0,
) -> dict[str, Any]:
    """Load → (search) → fit → (walk-forward) → (write) one sub-model.
    Returns a JSON-safe summary dict. Raises on any pipeline failure;
    the caller (``main``) converts that into an error entry per model.

    Order of operations matters: hyperparam search picks the best
    hyperparams against the simple 80/20 split (M5b), then the chosen
    spec is used both for the final 80/20 fit (the artifact's booster)
    and for the walk-forward validation. Walk-forward metrics are
    metadata; they do not change the booster identity or content_sha.

    ``feature_cache`` is forwarded to :func:`load_dataset`. When ``main``
    is running ``--model all`` it passes a shared dict so the input
    feature matrix is fetched once and reused across the three labels.
    """
    spec = get_spec(spec_name)

    # Bug-#2 fix (2026-05-17): pre-declare every local that the failure-
    # path finish_run wants to surface so they're in scope inside the
    # except handler. Without this, a load-time or train-time exception
    # would land a status='failed' row missing dataset_sha + leakage_report
    # — the very metadata that points at the root cause.
    dataset_sha: str | None = None
    leakage_report: dict[str, Any] | None = None
    # R2 Bug-#2/#3/#9 fix: lift the sweep + stacker summaries to outer
    # scope. The except handler reads them to surface "stacking was
    # requested but sweep failed" in the summary, and to know whether
    # any partial sweep telemetry was already streamed.
    bayesian_result_obj: Any = None
    bayesian_summary: dict[str, Any] | None = None
    stacker_summary: dict[str, Any] | None = None

    try:
        # Start tracking before the heavy lifting so a failure during load /
        # search still leaves an audit row — finish_run is called from the
        # except handler below with whatever locals were populated by the
        # moment the exception fired. spec_horizon_bars is not on ModelSpec
        # — fall back to None.
        if tracking_client is not None:
            tracking_client.start_run(
                spec_name=spec.name,
                spec_symbol=getattr(spec, "symbol", None),
                spec_interval=getattr(spec, "interval", None),
                spec_horizon_bars=getattr(spec, "horizon_bars", None),
                params=getattr(spec, "hyperparams", None),
                tags={"agent": agent_name, "lifecycle": spec_name},
            )

        # M5g.5: ``load_stacked_dataset`` dispatches single- vs multi-interval
        # — single-interval specs see identical behaviour to load_dataset.
        ds = load_stacked_dataset(spec, feature_cache=feature_cache)

        # R1 close-out: coarse schema/range fingerprint, computed post-load.
        # Lands on experiment_run.dataset_sha at finish_run time. Two runs
        # against the same dataset shape get the same value; comparing
        # across different shapes is a flag in the leaderboard.
        dataset_sha = compute_dataset_sha(ds, spec)
        if tracking_client is not None:
            tracking_client.set_tags({"dataset_sha": dataset_sha}, merge=True)

        # search_summary stays local — only used on the grid-search
        # path which doesn't share state with the except handler.
        search_summary: dict[str, Any] | None = None
        if do_bayesian:
            # R2.S4: Bayesian search via TPE + MedianPruner replaces the
            # M5b 8-point grid. Each trial runs a full walk-forward, so
            # n_trials × n_folds × fit_time is the real wall-clock — use
            # bayesian_timeout_s as the safety net.
            #
            # R2 Bug-#3 fix: per-trial telemetry is streamed via
            # on_trial_complete callback so each trial lands in
            # experiment_metric AS it finishes — a sweep that crashes
            # at minute 24 still leaves a partial trajectory in the DB.
            def _log_trial(t: Any) -> None:
                if tracking_client is None or t.score == float("-inf"):
                    return
                tracking_client.log_metric(
                    "bayesian_trial_score",
                    float(t.score),
                    step=int(t.trial_number),
                )

            bayesian_result_obj = bayesian_search_one(
                ds, spec, n_trials=bayesian_trials,
                timeout_s=bayesian_timeout_s,
                on_trial_complete=_log_trial,
            )
            bayesian_summary = bayesian_search_result_to_dict(bayesian_result_obj)
            final_spec = tuned_spec(spec, bayesian_result_obj.best_overrides)
            if tracking_client is not None:
                tracking_client.set_params(getattr(final_spec, "hyperparams", None) or {})
                tracking_client.set_tags({
                    "bayesian.n_trials_run": bayesian_result_obj.n_trials_run,
                    "bayesian.n_trials_completed": bayesian_result_obj.n_trials_completed,
                    "bayesian.n_trials_pruned": bayesian_result_obj.n_trials_pruned,
                    "bayesian.n_trials_failed": bayesian_result_obj.n_trials_failed,
                    "bayesian.wall_seconds": round(bayesian_result_obj.wall_seconds, 2),
                    "bayesian.best_metric": round(bayesian_result_obj.best_metric, 6),
                    "bayesian.baseline_metric": (
                        round(bayesian_result_obj.baseline_metric, 6)
                        if bayesian_result_obj.baseline_metric is not None else None
                    ),
                }, merge=True)
        elif do_search:
            result = grid_search_one(ds, spec)
            search_summary = search_result_to_dict(result)
            final_spec = tuned_spec(spec, result.best_overrides)
            # Snapshot the post-search hyperparams so the leaderboard
            # reflects what was actually trained, not the spec defaults.
            if tracking_client is not None:
                tracking_client.set_params(getattr(final_spec, "hyperparams", None) or {})
        else:
            final_spec = spec

        # WF1 fix: when --walk-forward, the saved booster is the last
        # fold's model (validated by the prior folds). When off, the 80/20
        # fit. ``eval_kind`` in the payload documents which split produced
        # the metrics so M5d/M5e can read them unambiguously.
        if do_walk_forward:
            payload = train_via_walk_forward(
                ds, final_spec, last_n_folds=last_n_folds, allow_leakage=allow_leakage,
            )
        else:
            payload = train_with_dataset(ds, final_spec, allow_leakage=allow_leakage)

        # Capture leakage_report into the outer scope as soon as it's
        # available — the except handler reads it. Train_with_dataset /
        # train_via_walk_forward populate this even when the run later
        # fails (e.g. during gauntlet).
        leakage_report = payload.get("leakage_report")

        # Always-set keys (even when None) so downstream consumers don't
        # have to handle KeyError vs dict-with-content as two distinct
        # shapes.
        payload["search"] = search_summary
        payload["bayesian_search"] = bayesian_summary
        walk_forward_summary = payload["walk_forward"]

        # R2.S4 stacker — runs AFTER training so the artifact's
        # content_sha is unaffected (stacker is post-hoc analysis until
        # S5 wires persistence). Skipped on multiclass (S3 limitation),
        # skipped when --stack-top-k=0, and skipped without a Bayesian
        # sweep (the stacker needs the sweep's top-k for diversity).
        # stacker_summary is declared at outer scope (Bug #2/#9 fix).
        if stack_top_k > 0 and bayesian_result_obj is not None:
            try:
                # R2 Bug-#13 fix: pass the original ``spec`` instead of
                # ``final_spec``. Functionally identical today (every
                # Bayesian trial sets all 8 search-space keys, so the
                # merge ``{**spec.hyperparams, **trial.overrides}`` and
                # ``{**final_spec.hyperparams, **trial.overrides}``
                # produce the same dict). Switching to ``spec`` makes
                # the intent clearer: the stacker rebuilds each of the
                # top-k from the ORIGINAL spec + that trial's overrides,
                # not from "best-spec-then-overlaid-with-this-trial."
                stacker = train_stacker(
                    ds, spec, bayesian_result_obj,
                    k=stack_top_k,
                )
                stacker_summary = stacker_to_dict(stacker)
                if tracking_client is not None:
                    # Stacker train metrics land as run-level scalars so
                    # the leaderboard can sort by e.g. stacker_auc.
                    tracking_client.log_metrics([
                        {"name": f"stacker_{k}", "value": float(v)}
                        for k, v in (stacker.train_metrics or {}).items()
                        if isinstance(v, (int, float)) and math.isfinite(float(v))
                    ])
                    tracking_client.set_tags({
                        "stacker.k": stack_top_k,
                        "stacker.n_meta_train_samples": stacker.n_meta_train_samples,
                        "stacker.meta_model_class": type(stacker.meta_model).__name__,
                    }, merge=True)
            except StackingError as exc:
                # Stacking failure shouldn't kill the run — the base
                # model on disk is still good. Record + continue.
                logging.warning("stacker training failed for %s: %s", spec_name, exc)
                stacker_summary = {"status": "error", "error": str(exc)}
        payload["stacker"] = stacker_summary

        gauntlet_summary: dict[str, Any] | None = None
        if do_gauntlet:
            # M5h dispatch: directional / multiclass specs run the 13-gate
            # gauntlet; binary / regression sub-models run the 5-gate
            # modulator gauntlet. The 5-gate path already raises for
            # multiclass with a pointer to M5h, so this branch must
            # mirror its check to avoid the error.
            spec_block = payload.get("spec") or {}
            is_directional = (
                spec_block.get("objective") == "multiclass"
                or spec_block.get("purpose") == "directional"
            )
            if is_directional:
                d_report = run_directional_gauntlet(payload)
                gauntlet_summary = directional_gauntlet_to_dict(d_report)
            else:
                report = run_gauntlet(payload)
                gauntlet_summary = gauntlet_to_dict(report)
        payload["gauntlet"] = gauntlet_summary

        artifact_info: dict[str, Any] | None
        if do_write:
            content_sha = payload["content_sha256"]
            path, size = write_artifact(payload, content_sha, artifact_dir)
            artifact_info = {
                "content_sha256": content_sha,
                "path": path,
                "size_bytes": size,
            }
        else:
            artifact_info = None

        register_info: dict[str, Any] | None = None
        if do_register:
            try:
                register_info = register_with_orchestrator(
                    payload,
                    artifact_info,
                    orchestrator_url=orchestrator_url,
                    auth_token=orchestrator_token,
                    agent_name=agent_name,
                    timeout_s=orchestrator_timeout_s,
                )
            except RegisterError as exc:
                # Don't fail the whole training run on a registration error
                # — the artifact is already on disk; the operator can retry
                # registration manually. Surface the failure in the summary.
                logging.warning("registration failed for %s: %s", spec_name, exc)
                register_info = {"status": "error", "error": str(exc)}

        # Tracking lifecycle close on success. Order matters: log metrics +
        # tags BEFORE finish_run so the row's summary_metrics + tags JSONB
        # are populated by the time the row goes terminal. content_sha256
        # only attaches when an artifact was actually written (the V92 FK
        # enforces existence). leakage_report carries the per-feature
        # correlation snapshot for the audit trail.
        if tracking_client is not None:
            tracking_client.log_metrics(extract_run_metrics(payload))
            tracking_client.set_tags(derive_summary_tags(payload), merge=True)
            tracking_client.finish_run(
                "completed",
                content_sha256=(artifact_info or {}).get("content_sha256"),
                leakage_report=leakage_report,
                dataset_sha=dataset_sha,
            )
    except BaseException as exc:
        # R1 Bug-#2 fix: finish the run RIGHT HERE with the locals that
        # exist at the moment of failure. main()'s outer except still
        # catches the re-raised exception for its own bookkeeping; that
        # outer finish_run call becomes a client-side no-op via the
        # _finished guard.
        #
        # R2 Bug-#2 fix: if --stack-top-k was requested but the sweep
        # never produced a result, log a clear warning + populate the
        # stacker_summary so the operator sees the silent skip. The
        # surrounding context (cli main's except handler) will surface
        # stacker_summary in the error JSON if it threads it through.
        if stack_top_k > 0 and bayesian_result_obj is None:
            logging.warning(
                "stacker skipped: --stack-top-k=%d requested but the "
                "Bayesian sweep produced no result (%s). The stacker "
                "needs the sweep's top-k for diversity — re-run after "
                "addressing the sweep failure.",
                stack_top_k, f"{type(exc).__name__}: {exc}",
            )
            # Only set if no earlier stacker outcome was recorded; the
            # success path always assigns before this point.
            if stacker_summary is None:
                stacker_summary = {
                    "status": "skipped",
                    "reason": "bayesian_sweep_failed",
                }
        if tracking_client is not None:
            tracking_client.finish_run(
                "failed",
                error_message=f"{type(exc).__name__}: {exc}",
                leakage_report=leakage_report,
                dataset_sha=dataset_sha,
            )
        raise

    return {
        "status": "ok",
        "model": spec_name,
        "spec": asdict(final_spec),
        "feature_names": list(payload["feature_names"]),
        "n_train_rows": payload["n_train_rows"],
        "n_val_rows": payload["n_val_rows"],
        "n_features": payload["n_features"],
        "n_bar_slots_total": payload["n_bar_slots_total"],
        "n_bar_slots_dropped_nan": payload["n_bar_slots_dropped_nan"],
        "per_feature_pct_non_null": payload["per_feature_pct_non_null"],
        "integrity": payload["integrity"],
        "leakage_report": payload.get("leakage_report"),
        "data_fingerprint": payload["data_fingerprint"],
        "bayesian_search": bayesian_summary,
        "stacker": stacker_summary,
        "search": search_summary,
        "walk_forward": walk_forward_summary,
        "gauntlet": gauntlet_summary,
        "deployment_readiness": payload["deployment_readiness"],
        "eval_kind": payload["eval_kind"],
        "metrics": payload["metrics"],
        "artifact": artifact_info,
        "registration": register_info,
    }


def _spec_choices() -> list[str]:
    # Preserve specs.py declaration order so --help and --model all
    # show models in the same order the blueprint discusses them
    # (regime → positioning → flow), not alphabetical.
    return [*SPECS.keys(), "all"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="blackheart-train")
    parser.add_argument(
        "--model",
        required=True,
        choices=_spec_choices(),
        help="ModelSpec name to train, or 'all' for every locked spec.",
    )
    parser.add_argument(
        "--search",
        action="store_true",
        help="Run a small hyperparam grid search and use the best for the final fit.",
    )
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help=(
            "Train via walk-forward CV (rolling refit with embargo). The "
            "saved booster is the last fold's model; earlier folds are "
            "validation evidence. Per-fold + aggregate metrics are added "
            "to the artifact payload and JSON summary."
        ),
    )
    parser.add_argument(
        "--gauntlet",
        action="store_true",
        help=(
            "Run the M5d 5-gate sub-model gauntlet on the trained artifact. "
            "Implies --walk-forward (gauntlet gates 2-5 read the walk_forward "
            "block). Gauntlet verdict + per-gate detail land in the payload."
        ),
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Dev-velocity knob: when set with --walk-forward, run only the "
            "last N folds (most recent windows). Fold k indices are preserved "
            "from the full 6-fold sequence so per-fold output stays "
            "interpretable. Aggregates describe only the executed subset — do "
            "NOT publish gate verdicts from a --folds run, since fewer folds "
            "violate the binding-gate evidence contract."
        ),
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Train but skip artifact write. Useful for smoke checks.",
    )
    parser.add_argument(
        "--register",
        action="store_true",
        help=(
            "After training + writing the artifact, POST metadata to the "
            "orchestrator at ${TRAIN_ORCHESTRATOR_URL}/models/register. "
            "Endpoint is idempotent on content_sha256, so re-running with "
            "--register against an unchanged artifact is a no-op replay."
        ),
    )
    parser.add_argument(
        "--allow-leakage",
        action="store_true",
        help=(
            "Demote the label-leakage integrity check (R1.B) from FAIL to "
            "WARN. The check still runs and its full report still lands in "
            "the artifact payload + experiment_run.leakage_report. Use only "
            "when you've manually inspected the leakage_report and confirmed "
            "the high correlation is intentional (e.g. a regime-conditional "
            "feature designed to track the regime label by construction)."
        ),
    )
    parser.add_argument(
        "--bayesian",
        action="store_true",
        help=(
            "R2: replace the M5b 8-point grid (--search) with Optuna's "
            "TPESampler + MedianPruner over an 8-dim LightGBM search space. "
            "Each trial runs a full walk-forward and scores on mean primary "
            "metric across folds. Mutually exclusive with --search. "
            "Combine with --bayesian-trials and --bayesian-timeout-s to tune "
            "budget; combine with --stack-top-k to also fit a meta-learner "
            "blending the top-k base models."
        ),
    )
    parser.add_argument(
        "--bayesian-trials",
        type=int,
        default=30,
        metavar="N",
        help=(
            "Number of Bayesian trials. TPE typically converges within 20-50 "
            "trials on 5-10 dimensional spaces; 30 is the middle-ground "
            "default. Each trial runs a full walk-forward, so real wall-clock "
            "is roughly N × n_folds × per-fit-seconds. Use --bayesian-timeout-s "
            "to cap absolute wall-clock regardless of trial count."
        ),
    )
    parser.add_argument(
        "--bayesian-timeout-s",
        type=float,
        default=None,
        metavar="SECS",
        help=(
            "Optional hard wall-clock cap for the Bayesian sweep, in seconds. "
            "Optuna stops kicking off new trials when this fires; the in-flight "
            "trial finishes. Use as the autonomous-loop safety net (\"stop after "
            "30 minutes regardless of progress\")."
        ),
    )
    parser.add_argument(
        "--stack-top-k",
        type=int,
        default=0,
        metavar="K",
        help=(
            "R2.S3 stacking: after the Bayesian sweep, take the top-K COMPLETE "
            "trials, refit each via walk-forward, gather OOF predictions, and "
            "fit a CV-regularised meta-learner (LogisticRegressionCV for "
            "binary / RidgeCV for regression) on the OOF-prediction matrix. "
            "0 (default) disables stacking. Recommended values: 3-7. "
            "Requires --bayesian. Stacker output lands in the JSON summary "
            "under 'stacker' but is NOT yet persisted in the artifact (S5)."
        ),
    )
    parser.add_argument(
        "--no-track",
        action="store_true",
        help=(
            "Disable experiment-tracking client (R1.A). Default behaviour POSTs a "
            "row to the orchestrator's /experiments endpoint and logs metrics + "
            "tags throughout the run; --no-track skips all that HTTP traffic. "
            "Use for offline smoke runs or when the orchestrator is intentionally "
            "down."
        ),
    )
    parser.add_argument(
        "--tracking-strict",
        action="store_true",
        help=(
            "Promote tracking failures from WARNING to fatal. Default is "
            "tolerant: a tracking outage logs and continues, the run summary's "
            "tracking.degraded field flags the gap. --tracking-strict raises "
            "TrackingError on any /experiments/* failure — useful for tests + "
            "reviewer audit runs where missing tracking is itself disqualifying."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="On failure, embed Python traceback in the error JSON for post-mortem.",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    _setup_logging(settings.log_level)

    # --gauntlet implies --walk-forward (gates 2-5 read walk_forward).
    if args.gauntlet and not args.walk_forward:
        logging.info("--gauntlet implies --walk-forward; enabling walk-forward")
        args.walk_forward = True

    # Preserve declaration order for "all"; argparse already validated
    # single-name inputs against the same ordering.
    targets = list(SPECS.keys()) if args.model == "all" else [args.model]

    # Share the input-feature matrix across specs only when --model all.
    # Single-spec runs leave this None to keep behaviour bit-identical
    # to the M5a code path.
    feature_cache: FeatureCache | None = {} if args.model == "all" else None

    summaries: list[dict[str, Any]] = []
    any_failed = False

    # --register requires --no-write to be off: nothing to register if
    # we never wrote the artifact. Catch this up front rather than
    # POSTing a body with artifact_uri=None and confusing the orchestrator.
    if args.register and args.no_write:
        parser.error("--register requires the artifact to be written; remove --no-write")

    # --folds is a dev-velocity flag; combining it with --register or
    # --gauntlet would publish an artifact/verdict whose walk-forward
    # evidence is intentionally truncated. Fail fast rather than ship a
    # spec whose gauntlet block is mining-vulnerable.
    if args.folds is not None:
        if args.folds <= 0:
            parser.error(f"--folds must be a positive integer; got {args.folds}")
        if not args.walk_forward:
            parser.error("--folds requires --walk-forward")
        if args.register:
            parser.error("--folds is a dev-velocity flag; refusing to --register a truncated walk-forward")
        if args.gauntlet:
            parser.error("--folds is a dev-velocity flag; refusing to --gauntlet a truncated walk-forward")

    # R2.S4 — Bayesian search validation.
    if args.bayesian and args.search:
        parser.error(
            "--bayesian and --search are mutually exclusive — pick one search "
            "strategy. --search is the M5b grid; --bayesian is the R2 TPE sweep."
        )
    if args.bayesian_trials <= 0:
        parser.error(f"--bayesian-trials must be > 0; got {args.bayesian_trials}")
    if args.bayesian_timeout_s is not None and args.bayesian_timeout_s <= 0:
        parser.error(f"--bayesian-timeout-s must be > 0 when set; got {args.bayesian_timeout_s}")
    if args.stack_top_k < 0:
        parser.error(f"--stack-top-k must be >= 0; got {args.stack_top_k}")
    if args.stack_top_k > 0 and not args.bayesian:
        parser.error(
            "--stack-top-k requires --bayesian (the stacker needs the sweep's "
            "top-k for diversity; the grid only has 8 fixed points)."
        )

    for spec_name in targets:
        # One tracking client per spec — the client carries run_id state, so
        # mixing two specs onto a single client would clobber the second's
        # run with the first's id. ``ExperimentClient.disabled()`` is a true
        # no-op (no HTTP), used when --no-track is set.
        if args.no_track:
            tracking_client = ExperimentClient.disabled()
        else:
            tracking_client = ExperimentClient.from_settings(
                settings, strict=args.tracking_strict,
            )

        try:
            summary = _run_one_model(
                spec_name,
                do_search=args.search,
                do_walk_forward=args.walk_forward,
                do_gauntlet=args.gauntlet,
                do_write=not args.no_write,
                do_register=args.register,
                artifact_dir=settings.artifact_dir,
                orchestrator_url=settings.orchestrator_url,
                orchestrator_token=settings.orchestrator_token,
                orchestrator_timeout_s=settings.orchestrator_request_timeout_s,
                agent_name=settings.agent_name,
                tracking_client=tracking_client,
                feature_cache=feature_cache,
                last_n_folds=args.folds,
                allow_leakage=args.allow_leakage,
                do_bayesian=args.bayesian,
                bayesian_trials=args.bayesian_trials,
                bayesian_timeout_s=args.bayesian_timeout_s,
                stack_top_k=args.stack_top_k,
            )
            summary["tracking"] = {
                "enabled": not args.no_track,
                "run_id": tracking_client.run_id,
                "degraded": tracking_client.degraded,
            }
            summaries.append(summary)
        except Exception as exc:
            any_failed = True
            # Defensive: _run_one_model already finishes the run with full
            # locals (dataset_sha + leakage_report) via its own except
            # handler — this call is a client-side no-op via _finished
            # when that path fired. It only matters when the exception
            # came from BEFORE _run_one_model was entered (e.g. malformed
            # spec_name caught at get_spec). In that case run_id is None
            # and finish_run no-ops via _require_run.
            tracking_client.finish_run(
                "failed", error_message=f"{type(exc).__name__}: {exc}",
            )
            logging.exception("training failed for %s", spec_name)
            err_entry: dict[str, Any] = {
                "status": "error",
                "model": spec_name,
                "error": f"{type(exc).__name__}: {exc}",
                "tracking": {
                    "enabled": not args.no_track,
                    "run_id": tracking_client.run_id,
                    "degraded": tracking_client.degraded,
                },
            }
            if args.verbose:
                err_entry["traceback"] = traceback.format_exc()
            summaries.append(err_entry)

    output: dict[str, Any]
    if len(summaries) == 1:
        output = summaries[0]
    else:
        output = {
            "status": "partial" if any_failed else "ok",
            "n_models": len(summaries),
            "n_failed": sum(1 for s in summaries if s.get("status") != "ok"),
            "models": summaries,
        }

    json.dump(
        _sanitize_for_json(output),
        sys.stdout,
        indent=2,
        default=_json_default,
        allow_nan=False,
    )
    sys.stdout.write("\n")
    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
