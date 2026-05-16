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
from .gauntlet import GauntletError, gauntlet_to_dict, run_gauntlet
from .gauntlet_directional import (
    DirectionalGauntletError,
    directional_gauntlet_to_dict,
    run_directional_gauntlet,
)
from .loader import FeatureCache, load_dataset, load_stacked_dataset
from .register_client import RegisterError, register_with_orchestrator
from .search import grid_search_one, search_result_to_dict, tuned_spec
from .settings import get_settings
from .specs import SPECS, get_spec
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
    feature_cache: FeatureCache | None = None,
    last_n_folds: int | None = None,
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
    # M5g.5: ``load_stacked_dataset`` dispatches single- vs multi-interval
    # — single-interval specs see identical behaviour to load_dataset.
    ds = load_stacked_dataset(spec, feature_cache=feature_cache)

    search_summary: dict[str, Any] | None = None
    if do_search:
        result = grid_search_one(ds, spec)
        search_summary = search_result_to_dict(result)
        final_spec = tuned_spec(spec, result.best_overrides)
    else:
        final_spec = spec

    # WF1 fix: when --walk-forward, the saved booster is the last fold's
    # model (validated by the prior folds). When off, the 80/20 fit.
    # ``eval_kind`` in the payload documents which split produced the
    # metrics so M5d/M5e can read them unambiguously.
    if do_walk_forward:
        payload = train_via_walk_forward(ds, final_spec, last_n_folds=last_n_folds)
    else:
        payload = train_with_dataset(ds, final_spec)

    # Always-set keys (even when None) so downstream consumers don't
    # have to handle KeyError vs dict-with-content as two distinct shapes.
    payload["search"] = search_summary
    walk_forward_summary = payload["walk_forward"]

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
        "data_fingerprint": payload["data_fingerprint"],
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

    for spec_name in targets:
        try:
            summaries.append(_run_one_model(
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
                feature_cache=feature_cache,
                last_n_folds=args.folds,
            ))
        except Exception as exc:
            any_failed = True
            logging.exception("training failed for %s", spec_name)
            err_entry: dict[str, Any] = {
                "status": "error",
                "model": spec_name,
                "error": f"{type(exc).__name__}: {exc}",
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
