"""Experiment-tracking client for the orchestrator's ``/experiments/*`` API
(V92 schema, R1.A of the ML/research uplift plan).

Adds first-class run tracking to every blackheart-train invocation. Mirrors
the HTTP/retry shape of :mod:`register_client` — stdlib urllib only, no
new dependency.

Design choice — failures are NON-FATAL by default. Tracking is observability;
its outages must not block a training run. If any tracking call fails after
exhausting retries, the client logs a warning, flips ``self.enabled = False``,
and all subsequent calls become no-ops. The CLI surfaces "tracking degraded"
in the run summary so the operator notices without the run being lost.

Pass ``strict=True`` to raise on any failure instead — useful for tests and
for the rare case where missing tracking IS a deal-breaker (e.g. a reviewer
audit run that demands a tracked artifact).

Typical lifecycle (also exposed as a context manager via :meth:`tracked_run`):

    client = ExperimentClient.from_settings()
    run_id = client.start_run(spec_name="regime_btc_v3", spec_symbol="BTCUSDT")
    try:
        ... do training ...
        client.log_metric("oof_auc", 0.62)
        client.log_metrics([{"name": "fold_auc", "value": 0.58, "fold_idx": 0}, ...])
        client.set_tags({"deployment_ready": True})
        client.finish_run("completed", content_sha256=sha)
    except Exception as exc:
        client.finish_run("failed", error_message=str(exc))
        raise
"""
from __future__ import annotations

import json
import logging
import math
import random
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)


DEFAULT_MAX_ATTEMPTS: int = 3
DEFAULT_BACKOFF_BASE_S: float = 1.0
_RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})


def _sleep_with_jitter(base_delay_s: float) -> None:
    time.sleep(base_delay_s * random.uniform(0.75, 1.25))


class TrackingError(RuntimeError):
    """Raised when an /experiments/* call fails and ``strict=True``."""


class ExperimentClient:
    """Thin urllib client for orchestrator /experiments/* endpoints.

    Stateful — carries ``run_id`` once :meth:`start_run` succeeds. All
    subsequent methods PATCH against that run_id.

    Tolerant by default: a failed call logs a WARNING, flips
    ``self.enabled = False``, and subsequent methods become no-ops. Use
    :attr:`degraded` to check after a run whether tracking succeeded
    end-to-end.

    Set ``strict=True`` to re-raise instead. ``--no-track`` at the CLI is
    implemented by constructing the client with ``enabled=False`` so no
    HTTP calls fire at all.
    """

    def __init__(
        self,
        *,
        orchestrator_url: str,
        auth_token: str,
        agent_name: str,
        timeout_s: float = 30.0,
        strict: bool = False,
        enabled: bool = True,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        backoff_base_s: float = DEFAULT_BACKOFF_BASE_S,
    ) -> None:
        self._url = orchestrator_url.rstrip("/")
        self._token = auth_token
        self._agent = agent_name
        self._timeout_s = timeout_s
        self._strict = strict
        self._max_attempts = max_attempts
        self._backoff_base_s = backoff_base_s
        self.enabled = enabled
        self.run_id: str | None = None
        # Set to True if any call fails-soft; the CLI reads this for the
        # run summary. Distinct from ``enabled``: a strict-mode failure
        # raises; tolerant-mode failure flips both ``enabled=False`` and
        # ``degraded=True``.
        self.degraded = False
        # Idempotency: tracked_run() calls finish on context exit; the CLI
        # may also call finish explicitly with content_sha256 / leakage
        # report. We let the FIRST call through and silently skip the rest,
        # so callers can interleave freely.
        self._finished = False

    # ── construction helpers ──────────────────────────────────────────────

    @classmethod
    def from_settings(cls, settings, *, enabled: bool = True, strict: bool = False) -> ExperimentClient:
        """Construct from a :class:`blackheart_train.settings.Settings` instance.
        Avoids the import-time dependency on Settings — handy for tests.
        """
        return cls(
            orchestrator_url=settings.orchestrator_url,
            auth_token=settings.orchestrator_token,
            agent_name=settings.agent_name,
            timeout_s=settings.orchestrator_request_timeout_s,
            enabled=enabled,
            strict=strict,
        )

    @classmethod
    def disabled(cls) -> ExperimentClient:
        """Construct a no-op client. Every method returns None / 0 without
        making HTTP calls. Used when --no-track is passed.
        """
        return cls(
            orchestrator_url="",
            auth_token="",
            agent_name="",
            enabled=False,
        )

    # ── core HTTP plumbing ────────────────────────────────────────────────

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        *,
        idempotency_key: str | None = None,
    ) -> dict[str, Any] | None:
        """POST / PATCH / GET against the orchestrator. Returns the parsed
        JSON body on 2xx, ``None`` on graceful-degradation.

        Retries transient failures (5xx, 408, 429, connection errors) up to
        ``self._max_attempts``. Non-retryable failures (4xx other than
        408/429) fail-fast — they reflect a caller bug.

        On final-attempt failure in tolerant mode, logs a WARNING + flips
        ``self.enabled=False`` + returns None. In strict mode, raises
        :class:`TrackingError`.

        Bug-#6 fix (2026-05-17): unified path for idempotent and
        non-idempotent calls. ``idempotency_key`` adds the header when
        provided; everything else is identical. Prior to this fix the
        two paths drifted (the idempotency path silently dropped the
        retry-WARNING log line) — a single helper closes that gap.
        """
        if not self.enabled:
            return None
        url = f"{self._url}{path}"
        data = json.dumps(body, default=str).encode("utf-8") if body is not None else None
        headers = {
            "X-Orch-Token": self._token,
            "X-Agent-Name": self._agent,
            "Content-Type": "application/json",
        }
        if idempotency_key is not None:
            headers["Idempotency-Key"] = idempotency_key

        last_error: BaseException | None = None
        for attempt in range(1, self._max_attempts + 1):
            req = urllib.request.Request(
                url, data=data, headers=headers, method=method,
            )
            try:
                with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
                    body_text = resp.read().decode("utf-8")
                    return json.loads(body_text) if body_text else {}
            except urllib.error.HTTPError as e:
                err_body = e.read().decode("utf-8", errors="replace") if e.fp else ""
                if e.code not in _RETRYABLE_STATUS or attempt >= self._max_attempts:
                    return self._handle_failure(
                        f"HTTP {e.code} {e.reason}: {err_body}", exc=e,
                    )
                logger.warning(
                    "tracking: transient HTTP %d %s (attempt %d/%d) — retrying",
                    e.code, e.reason, attempt, self._max_attempts,
                )
                last_error = e
            except urllib.error.URLError as e:
                if attempt >= self._max_attempts:
                    return self._handle_failure(
                        f"orchestrator unreachable: {type(e).__name__}: {e.reason}",
                        exc=e,
                    )
                logger.warning(
                    "tracking: orchestrator unreachable %s (attempt %d/%d) — retrying",
                    type(e).__name__, attempt, self._max_attempts,
                )
                last_error = e

            _sleep_with_jitter(self._backoff_base_s * (2 ** (attempt - 1)))

        return self._handle_failure(
            f"retries exhausted (last_error={last_error!r})", exc=last_error,
        )

    def _handle_failure(self, msg: str, *, exc: BaseException | None) -> None:
        """Tolerant mode → log + flip flags + return None.
        Strict mode → raise TrackingError.
        """
        full = f"experiment tracking failed: {msg}"
        if self._strict:
            raise TrackingError(full) from exc
        logger.warning("%s — tracking disabled for remainder of this run", full)
        self.enabled = False
        self.degraded = True
        return None

    # ── public API ────────────────────────────────────────────────────────

    def start_run(
        self,
        *,
        spec_name: str,
        spec_version: str | None = None,
        spec_symbol: str | None = None,
        spec_interval: str | None = None,
        spec_horizon_bars: int | None = None,
        git_sha: str | None = None,
        dataset_sha: str | None = None,
        params: dict[str, Any] | None = None,
        tags: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> str | None:
        """POST /experiments. Returns ``run_id`` (also stored on the client)
        or None if tracking is disabled/degraded.

        ``idempotency_key`` is optional but recommended — pass the CLI's
        invocation id so a retry of the same `python -m blackheart_train.cli`
        invocation lands on the same run_id.
        """
        if not self.enabled:
            return None
        if self.run_id is not None:
            # Multiple start_runs in one client = misuse. The Pythonic guard
            # is to make this an error in strict mode and a warning otherwise.
            if self._strict:
                raise TrackingError(f"start_run called twice (existing run_id={self.run_id})")
            logger.warning("tracking: start_run called twice — keeping existing run_id=%s", self.run_id)
            return self.run_id

        body = {
            "spec_name": spec_name,
            "spec_version": spec_version,
            "spec_symbol": spec_symbol,
            "spec_interval": spec_interval,
            "spec_horizon_bars": spec_horizon_bars,
            "git_sha": git_sha,
            "dataset_sha": dataset_sha,
            "params": params,
            "tags": tags,
        }
        response = self._request(
            "POST", "/experiments", body, idempotency_key=idempotency_key,
        )
        if response is None:
            return None
        self.run_id = response["run_id"]
        logger.info("tracking: started run_id=%s spec=%s", self.run_id, spec_name)
        return self.run_id

    def _require_run(self) -> bool:
        """Internal — returns True if a run_id is set AND enabled.
        False means the method should no-op.
        """
        if not self.enabled:
            return False
        if self.run_id is None:
            if self._strict:
                raise TrackingError("no active run_id — start_run not called or it failed")
            logger.warning("tracking: no active run_id — call dropped silently")
            return False
        return True

    def set_params(self, params: dict[str, Any]) -> None:
        if not self._require_run():
            return
        self._request("PATCH", f"/experiments/{self.run_id}/params", {"params": params})

    def set_tags(self, tags: dict[str, Any], *, merge: bool = True) -> None:
        if not self._require_run():
            return
        self._request(
            "PATCH", f"/experiments/{self.run_id}/tags",
            {"tags": tags, "merge": merge},
        )

    def log_metric(
        self,
        name: str,
        value: float,
        *,
        fold_idx: int | None = None,
        step: int | None = None,
    ) -> None:
        """Log a single metric. NaN/Inf are dropped silently — neither the
        DB CHECK nor the orchestrator's pydantic validator would accept
        them, and a tracking failure shouldn't sink a real metric batch.
        """
        if not self._require_run():
            return
        if not math.isfinite(value):
            logger.warning(
                "tracking: skipping non-finite metric %s=%r fold=%s",
                name, value, fold_idx,
            )
            return
        entry: dict[str, Any] = {"name": name, "value": float(value)}
        if fold_idx is not None:
            entry["fold_idx"] = fold_idx
        if step is not None:
            entry["step"] = step
        self._request(
            "PATCH", f"/experiments/{self.run_id}/metrics",
            {"metrics": [entry]},
        )

    def log_metrics(self, metrics: list[dict[str, Any]]) -> None:
        """Batch-log. Filters non-finite values out before sending —
        same rationale as :meth:`log_metric`.
        """
        if not self._require_run():
            return
        clean: list[dict[str, Any]] = []
        dropped = 0
        for m in metrics:
            v = m.get("value")
            if not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                dropped += 1
                continue
            entry: dict[str, Any] = {"name": m["name"], "value": float(v)}
            if m.get("fold_idx") is not None:
                entry["fold_idx"] = m["fold_idx"]
            if m.get("step") is not None:
                entry["step"] = m["step"]
            clean.append(entry)
        if dropped:
            logger.warning("tracking: dropped %d non-finite metric(s)", dropped)
        if not clean:
            return
        self._request(
            "PATCH", f"/experiments/{self.run_id}/metrics",
            {"metrics": clean},
        )

    def finish_run(
        self,
        status: str,
        *,
        content_sha256: str | None = None,
        leakage_report: dict[str, Any] | None = None,
        error_message: str | None = None,
        dataset_sha: str | None = None,
    ) -> None:
        """Transition the run to a terminal status.

        Status must be one of ``completed`` / ``failed`` / ``aborted``.
        ``content_sha256`` should be passed when the run produced a
        registered artifact — the V92 FK enforces referential integrity.
        ``dataset_sha`` is the coarse schema/range fingerprint from
        :func:`integrity.compute_dataset_sha` — computed post-load, so it
        lands at finish-time rather than start-time.
        """
        if status not in ("completed", "failed", "aborted"):
            raise ValueError(f"finish_run status must be terminal; got {status!r}")
        if not self._require_run():
            return
        if self._finished:
            logger.debug("tracking: finish_run already called for run_id=%s — skipping", self.run_id)
            return
        # Bug-#1 fix (2026-05-17): set _finished AFTER the HTTP call
        # returns non-None. If we set it pre-call and the request fails
        # tolerantly (network blip, orchestrator briefly down), _finished
        # would stay True and the row would be stuck in 'running' forever.
        # The natural retry path (tracked_run __exit__, manual operator
        # retry) becomes a silent no-op without this guard. After:
        #   - success → _finished=True, future calls correctly no-op
        #   - tolerant failure → _finished stays False, retry can fire
        #   - strict failure → raises, _finished stays False (irrelevant)
        response = self._request(
            "POST", f"/experiments/{self.run_id}/finish",
            {
                "status": status,
                "content_sha256": content_sha256,
                "leakage_report": leakage_report,
                "error_message": error_message,
                "dataset_sha": dataset_sha,
            },
        )
        if response is not None:
            self._finished = True


# ─────────────────────────────────────────────────────────────────────────
# Payload → metrics translation
# ─────────────────────────────────────────────────────────────────────────


def extract_run_metrics(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten a blackheart-train payload into a metric-log entry list.

    Run-level metrics (``fold_idx=None``) cover:
      * ``payload["metrics"]`` — final-fit scalars (auc, log_loss, ...)
      * ``payload["walk_forward"]["primary_mean"]`` / median / std
      * ``payload["walk_forward"]["metric_means"]`` — across-fold mean per metric

    Per-fold metrics: walk_forward.folds[*].metrics, with ``fold_idx``
    pulled from the fold's ``fold`` field.

    Non-finite values are NOT filtered here — the client's log_metrics()
    does that — so this function stays a pure shape transform.
    """
    out: list[dict[str, Any]] = []

    # Run-level scalars from the final-fit metrics dict.
    for k, v in (payload.get("metrics") or {}).items():
        if isinstance(v, (int, float)):
            out.append({"name": k, "value": float(v)})

    wf = payload.get("walk_forward") or {}

    # Walk-forward aggregates (primary mean/median/std + metric_means).
    if wf:
        for key in ("primary_mean", "primary_median", "primary_std"):
            v = wf.get(key)
            if isinstance(v, (int, float)):
                # primary metric name e.g. "wf_auc" — qualified so it doesn't
                # collide with the single-fit auc above.
                out.append({"name": f"wf_{key}", "value": float(v)})
        for k, v in (wf.get("metric_means") or {}).items():
            if isinstance(v, (int, float)):
                out.append({"name": f"wf_mean_{k}", "value": float(v)})

    # Per-fold metrics.
    for fold in wf.get("folds") or []:
        fidx = fold.get("fold")
        if fidx is None:
            continue
        for k, v in (fold.get("metrics") or {}).items():
            if isinstance(v, (int, float)):
                out.append({"name": k, "value": float(v), "fold_idx": int(fidx)})

    return out


def derive_summary_tags(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract a compact tag dict from a training payload.

    Tags are searchable JSONB in V92 and intended for filtering ("show me
    all runs with gauntlet=PASS and deployment_ready=true"). Keep the set
    small and stable — adding new keys later is cheap; removing is not.
    """
    tags: dict[str, Any] = {}
    if (g := payload.get("gauntlet")) is not None:
        verdict = g.get("overall_verdict") if isinstance(g, dict) else None
        if verdict is not None:
            tags["gauntlet_verdict"] = verdict
    if (dr := payload.get("deployment_readiness")) is not None:
        if isinstance(dr, dict) and "deployment_ready" in dr:
            tags["deployment_ready"] = bool(dr["deployment_ready"])
    eval_kind = payload.get("eval_kind")
    if eval_kind is not None:
        tags["eval_kind"] = eval_kind
    return tags


# ─────────────────────────────────────────────────────────────────────────
# Context-manager helper
# ─────────────────────────────────────────────────────────────────────────


@contextmanager
def tracked_run(
    client: ExperimentClient,
    *,
    spec_name: str,
    spec_version: str | None = None,
    spec_symbol: str | None = None,
    spec_interval: str | None = None,
    spec_horizon_bars: int | None = None,
    params: dict[str, Any] | None = None,
    tags: dict[str, Any] | None = None,
    git_sha: str | None = None,
    dataset_sha: str | None = None,
    idempotency_key: str | None = None,
):
    """Context manager around start_run / finish_run.

    On normal exit: finish_run(completed). On exception: finish_run(failed,
    error_message=str(exc)) and re-raises. The caller can override the
    terminal status by stamping ``client._final_status`` (e.g. for
    gauntlet-FAIL artifacts where the run completed cleanly but the
    operator wants the row tagged differently — usually unnecessary; the
    gauntlet verdict lives in tags).

    Yields the run_id string (may be None if tracking is disabled).
    """
    run_id = client.start_run(
        spec_name=spec_name,
        spec_version=spec_version,
        spec_symbol=spec_symbol,
        spec_interval=spec_interval,
        spec_horizon_bars=spec_horizon_bars,
        params=params,
        tags=tags,
        git_sha=git_sha,
        dataset_sha=dataset_sha,
        idempotency_key=idempotency_key,
    )
    try:
        yield run_id
    except BaseException as exc:
        client.finish_run("failed", error_message=f"{type(exc).__name__}: {exc}")
        raise
    else:
        # Caller hasn't already finished it — finish_run is a no-op when the
        # underlying row is already terminal (409), so the second call is
        # safe. The client suppresses the 409 via the tolerant-mode flag.
        client.finish_run("completed")
