"""Client for the orchestrator's ``POST /models/register`` endpoint.

The training CLI's ``--register`` flag uses this to insert a
``model_registry`` row right after writing the artifact. No external
dependency added — uses stdlib ``urllib`` so blackheart-train doesn't
have to vendor httpx.

The mapping from the training payload to the orchestrator's request
body is centralized in :func:`build_register_request`. Two reasons to
keep it explicit rather than dumping the whole payload:

1. The orchestrator's pydantic model rejects extra keys silently for
   nested dicts but tightly validates the top-level fields. Mapping
   here means a payload schema change in blackheart-train doesn't
   cascade into an orchestrator validation failure.
2. The artifact's ``booster`` object never crosses the HTTP boundary
   (it's a LightGBM Booster, JSON-unserializable). The training side
   computes ``content_sha256`` and POSTs the addressable identity; the
   binary stays on disk for the live inference worker to load.
"""
from __future__ import annotations

import json
import logging
import random
import time
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)


# ── Retry configuration (M10 audit, 2026-05-16) ──────────────────────────
#
# Transient 5xx + connection-level failures are retried with exponential
# backoff + jitter; persistent 4xx are NOT retried (caller bug, won't be
# fixed by waiting). Three attempts cover typical orchestrator boot
# windows and brief network blips; longer outages surface as a
# RegisterError so the CLI's run summary captures the failure.

#: Maximum total attempts (initial + retries). 3 = one initial + two retries.
DEFAULT_MAX_ATTEMPTS: int = 3

#: Base delay between attempts, in seconds. Doubles on each retry
#: (DEFAULT_BACKOFF_BASE_S * 2 ** (attempt-1)) and adds ±25% jitter to
#: avoid thundering-herd retries when multiple training CLIs collide.
DEFAULT_BACKOFF_BASE_S: float = 1.0

#: Status codes considered transient and worth retrying. 5xx + 408
#: (request timeout) + 429 (rate-limit). Note 4xx (other than 408/429)
#: are NOT retried — they're caller bugs, will fail identically on retry.
_RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})


def _sleep_with_jitter(base_delay_s: float) -> None:
    """Sleep for ``base_delay_s * U[0.75, 1.25]``. Module-level so tests
    can monkeypatch the wait to zero. Pure side effect — no return."""
    time.sleep(base_delay_s * random.uniform(0.75, 1.25))


class RegisterError(RuntimeError):
    """Raised when the orchestrator rejects a model registration. The
    message carries the HTTP status code and the orchestrator's error
    envelope (which itself names the failing field — pydantic
    validation, V66 CHECK violation, etc.). Caught by the CLI which
    converts to an error entry in the run summary."""


def build_register_request(
    payload: dict[str, Any],
    artifact_info: dict[str, Any] | None,
) -> dict[str, Any]:
    """Translate a training payload + optional artifact_info into the
    orchestrator's request body schema.

    The orchestrator's :class:`ModelRegisterRequest` doesn't accept the
    full training payload (extra keys are tolerated by pydantic but the
    untyped ``walk_forward`` block is passed through as-is). We pick
    the load-bearing fields and shape them.
    """
    spec = dict(payload["spec"])
    body: dict[str, Any] = {
        # ``payload_version`` lets the orchestrator branch on artifact
        # schema (v1 = single LightGBM under ``booster``; v2 = single
        # booster OR full Ensemble under ``ensemble``). Defaulted to 1
        # so a registration from a pre-Phase-2 training run still maps
        # cleanly — the orchestrator's schema treats v1 as the legacy
        # single-model case.
        "payload_version": payload.get("payload_version", 1),
        "content_sha256": payload["content_sha256"],
        "artifact_uri": (artifact_info or {}).get("path"),
        "artifact_size_bytes": (artifact_info or {}).get("size_bytes"),
        "spec": {
            "name": spec["name"],
            "purpose": spec["purpose"],
            "symbol": spec["symbol"],
            "interval": spec.get("interval"),
            "label_feature": spec["label_feature"],
            "label_version": spec.get("label_version", 1),
            "objective": spec["objective"],
            # datetimes in the payload come back as datetime objects via
            # asdict; the JSON serializer below handles them via the
            # default=str fallback.
            "train_start": _to_iso(spec["train_start"]),
            "train_end": _to_iso(spec["train_end"]),
            "hyperparams": spec.get("hyperparams"),
            "derived_features": list(spec.get("derived_features", [])),
        },
        "feature_names": list(payload["feature_names"]),
        "metrics": dict(payload["metrics"]),
        "walk_forward": payload.get("walk_forward"),
        "gauntlet": (
            {"overall_verdict": payload["gauntlet"]["overall_verdict"]}
            if payload.get("gauntlet") else None
        ),
        "deployment_readiness": payload["deployment_readiness"],
        "data_fingerprint": payload.get("data_fingerprint"),
    }
    return body


def _to_iso(value: Any) -> str:
    """datetime → 'YYYY-MM-DDTHH:MM:SS'. Strings already ISO pass
    through. Anything else gets str()'d so urllib's json.dumps can
    encode it."""
    if hasattr(value, "isoformat"):
        return value.isoformat()
    s = str(value)
    # Some serializations produce "2024-12-01 00:00:00" with a space —
    # the orchestrator's pydantic parser accepts space OR T; we normalise
    # to T defensively in case a stricter parser shows up later.
    return s.replace(" ", "T")


def register_with_orchestrator(
    payload: dict[str, Any],
    artifact_info: dict[str, Any] | None,
    *,
    orchestrator_url: str,
    auth_token: str,
    agent_name: str,
    timeout_s: float = 30.0,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    backoff_base_s: float = DEFAULT_BACKOFF_BASE_S,
) -> dict[str, Any]:
    """POST the artifact metadata to ``{orchestrator_url}/models/register``.

    Returns the orchestrator's response dict on 2xx. Raises
    :class:`RegisterError` on any non-retryable failure with the status
    + response body so the CLI can surface a clear error.

    The endpoint is idempotent on ``content_sha256`` — re-registering
    the same artifact returns the existing row.

    M10 audit (2026-05-16): retries transient 5xx + 408/429 + network
    failures with exponential backoff (1s -> 2s -> 4s with ±25% jitter).
    Non-retryable failures (4xx other than 408/429) fail-fast on the
    first attempt because they reflect a caller bug (bad payload) that
    won't be fixed by waiting. Max ``max_attempts`` total attempts
    (default 3 = initial + 2 retries).
    """
    body_bytes = json.dumps(
        build_register_request(payload, artifact_info), default=str
    ).encode("utf-8")
    url = f"{orchestrator_url.rstrip('/')}/models/register"
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1; got {max_attempts}")

    last_error: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        req = urllib.request.Request(
            url,
            data=body_bytes,
            headers={
                "X-Orch-Token": auth_token,
                "X-Agent-Name": agent_name,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                body_text = resp.read().decode("utf-8")
                response = json.loads(body_text)
            logger.info(
                "model registered with orchestrator | sha=%s model_id=%s "
                "status=%s replay=%s attempts=%d",
                payload["content_sha256"][:12],
                response.get("model_id"),
                response.get("status"),
                response.get("idempotent_replay"),
                attempt,
            )
            return response
        except urllib.error.HTTPError as e:
            # HTTPError exposes status + response body. Retryable
            # codes get another shot; the rest fail-fast.
            err_body = e.read().decode("utf-8", errors="replace") if e.fp else ""
            if e.code not in _RETRYABLE_STATUS or attempt >= max_attempts:
                raise RegisterError(
                    f"orchestrator rejected registration: HTTP {e.code} "
                    f"{e.reason} (attempt {attempt}/{max_attempts}): {err_body}"
                ) from e
            logger.warning(
                "orchestrator transient HTTP %d %s (attempt %d/%d) — retrying",
                e.code, e.reason, attempt, max_attempts,
            )
            last_error = e
        except urllib.error.URLError as e:
            # Network-level failure (DNS, connection refused, timeout).
            # Always retryable up to max_attempts.
            if attempt >= max_attempts:
                raise RegisterError(
                    f"orchestrator unreachable at {url} after "
                    f"{max_attempts} attempt(s): {type(e).__name__}: {e.reason}"
                ) from e
            logger.warning(
                "orchestrator unreachable: %s (attempt %d/%d) — retrying",
                type(e).__name__, attempt, max_attempts,
            )
            last_error = e

        # Backoff before the next attempt. Exponential: 1s, 2s, 4s, ...
        # plus ±25% jitter via _sleep_with_jitter (module-level so tests
        # monkeypatch it to zero and run in milliseconds).
        delay = backoff_base_s * (2 ** (attempt - 1))
        _sleep_with_jitter(delay)

    # Unreachable in normal flow — the loop either returns on success
    # or raises on the final attempt. The fallback is a safety net for
    # ``max_attempts == 0``, already rejected above.
    raise RegisterError(
        f"orchestrator registration exhausted retries (last_error={last_error!r})"
    )
