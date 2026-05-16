"""Unit tests for the orchestrator register-client (CP1 fix).

No real HTTP — we monkeypatch ``urllib.request.urlopen`` so the tests
run in milliseconds and don't require the orchestrator to be up.
"""
from __future__ import annotations

import io
import json
import urllib.error

import pytest

from blackheart_train.register_client import (
    RegisterError,
    build_register_request,
    register_with_orchestrator,
)


# ── build_register_request ────────────────────────────────────────────────


def _sample_payload() -> dict:
    """Minimal training payload — shaped exactly like ``train._train_core``
    returns it."""
    from datetime import datetime
    return {
        "content_sha256": "a" * 64,
        "data_fingerprint": "b" * 64,
        "spec": {
            "name": "regime_btc_v2",
            "purpose": "regime",
            "symbol": "BTCUSDT",
            "interval": "1h",
            "label_feature": "label_regime_risk_on_24h",
            "label_version": 1,
            "objective": "binary",
            "train_start": datetime(2024, 12, 1),
            "train_end": datetime(2026, 5, 14),
            "val_fraction": 0.2,
            "hyperparams": {"num_leaves": 31, "random_state": 42},
            "derived_features": ("btc_log_return_24h",),
        },
        "feature_names": ("f1", "f2", "btc_log_return_24h"),
        "metrics": {"auc": 0.58, "log_loss": 0.65},
        "walk_forward": {
            "primary_metric": "auc", "primary_mean": 0.58,
            "primary_std": 0.07, "n_folds_run": 6,
        },
        "gauntlet": {"overall_verdict": "PASS"},
        "deployment_readiness": {
            "deployment_ready": False,
            "unregistered_input_features": ["btc_log_return_24h"],
            "unregistered_label": "label_regime_risk_on_24h",
        },
    }


def test_build_request_maps_required_fields():
    payload = _sample_payload()
    artifact = {"path": "/tmp/x.pkl", "size_bytes": 1234}
    body = build_register_request(payload, artifact)
    assert body["content_sha256"] == "a" * 64
    assert body["artifact_uri"] == "/tmp/x.pkl"
    assert body["artifact_size_bytes"] == 1234
    assert body["spec"]["purpose"] == "regime"
    assert body["spec"]["objective"] == "binary"
    assert body["spec"]["label_feature"] == "label_regime_risk_on_24h"
    assert body["spec"]["derived_features"] == ["btc_log_return_24h"]
    assert body["feature_names"] == ["f1", "f2", "btc_log_return_24h"]
    assert body["gauntlet"] == {"overall_verdict": "PASS"}
    assert body["deployment_readiness"]["deployment_ready"] is False


def test_build_request_isoformats_datetime():
    """train_start/train_end come in as datetime; the orchestrator's
    pydantic parser accepts ISO strings, so we serialise to ISO with
    a 'T' separator."""
    payload = _sample_payload()
    body = build_register_request(payload, None)
    assert body["spec"]["train_start"] == "2024-12-01T00:00:00"
    assert "T" in body["spec"]["train_end"]


def test_build_request_handles_missing_artifact_info():
    """--no-write run still allows --register? No (CLI guard). But the
    helper itself must not crash on artifact_info=None — the orchestrator's
    artifact_uri / artifact_size_bytes fields are optional."""
    body = build_register_request(_sample_payload(), None)
    assert body["artifact_uri"] is None
    assert body["artifact_size_bytes"] is None


def test_build_request_omits_gauntlet_when_none():
    payload = _sample_payload()
    payload["gauntlet"] = None
    body = build_register_request(payload, None)
    assert body["gauntlet"] is None


def test_build_request_serialises_tuple_feature_names_as_list():
    """LoadedDataset.feature_names is a tuple; JSON has no tuple type
    and the orchestrator's schema wants list[str]."""
    payload = _sample_payload()
    body = build_register_request(payload, None)
    assert isinstance(body["feature_names"], list)
    assert isinstance(body["spec"]["derived_features"], list)


def test_build_request_forwards_payload_version():
    """v2 (M5g.3 phase 2) payloads carry a ``payload_version`` field;
    the register body must forward it so the orchestrator can branch
    on artifact schema. A pre-Phase-2 payload lacking the field
    defaults to 1 — the legacy single-LightGBM shape."""
    payload = _sample_payload()
    payload["payload_version"] = 2
    body = build_register_request(payload, None)
    assert body["payload_version"] == 2

    # Pre-Phase-2 payload: key absent → defaults to 1.
    legacy = _sample_payload()
    legacy.pop("payload_version", None)
    body_legacy = build_register_request(legacy, None)
    assert body_legacy["payload_version"] == 1


# ── register_with_orchestrator (HTTP layer mocked) ────────────────────────


class _FakeResponse:
    def __init__(self, body: dict, status: int = 200):
        self._body = json.dumps(body).encode("utf-8")
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self._body


def test_register_success_returns_response_dict(monkeypatch):
    captured: dict = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode("utf-8"))
        captured["headers"] = dict(req.headers)
        return _FakeResponse({
            "model_id": "00000000-0000-0000-0000-000000000001",
            "content_sha256": "a" * 64,
            "status": "awaiting_operator_review",
            "version": 1,
            "registered_at": "2026-05-15T00:00:00",
            "idempotent_replay": False,
        })

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    payload = _sample_payload()
    artifact = {"path": "/tmp/x.pkl", "size_bytes": 999}
    result = register_with_orchestrator(
        payload, artifact,
        orchestrator_url="http://127.0.0.1:8082",
        auth_token="test-token",
        agent_name="test-agent",
    )
    assert result["model_id"] == "00000000-0000-0000-0000-000000000001"
    assert captured["url"] == "http://127.0.0.1:8082/models/register"
    # urllib's Request.headers keys are capitalized (the http lib normalises)
    assert captured["headers"].get("X-orch-token") == "test-token"
    assert captured["headers"].get("X-agent-name") == "test-agent"
    assert captured["body"]["content_sha256"] == "a" * 64


def test_register_raises_on_http_error(monkeypatch):
    """A 422 from the orchestrator (e.g. bad purpose) must surface as
    RegisterError carrying the response body so the operator sees the
    real reason."""
    def fake_urlopen(req, timeout=None):
        body = json.dumps({
            "error_code": "validation_failed",
            "details": {"errors": [{"loc": ["body", "spec", "purpose"]}]},
        }).encode("utf-8")
        raise urllib.error.HTTPError(
            url=req.full_url, code=422, msg="Unprocessable",
            hdrs=None, fp=io.BytesIO(body),
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(RegisterError, match="HTTP 422"):
        register_with_orchestrator(
            _sample_payload(), None,
            orchestrator_url="http://127.0.0.1:8082",
            auth_token="t", agent_name="a",
        )


def test_register_raises_on_connection_error(monkeypatch):
    """Orchestrator unreachable → clear RegisterError naming the URL.
    With M10 retry, the connection error is retried up to max_attempts;
    final failure carries the attempt count."""
    monkeypatch.setattr("blackheart_train.register_client._sleep_with_jitter", lambda d: None)
    def fake_urlopen(req, timeout=None):
        raise urllib.error.URLError("Connection refused")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(RegisterError, match="unreachable"):
        register_with_orchestrator(
            _sample_payload(), None,
            orchestrator_url="http://127.0.0.1:8082",
            auth_token="t", agent_name="a",
            max_attempts=2,
        )


# ── M10 retry behaviour (2026-05-16) ──────────────────────────────────────


def _failing_then_succeeding(failures: list, success_body: dict):
    """Build a fake urlopen that raises each item in ``failures`` (in
    order) on successive calls and then returns a 200 response on the
    next call. Used to drive the retry loop deterministically.
    """
    calls = {"n": 0}
    def fake_urlopen(req, timeout=None):
        n = calls["n"]
        calls["n"] = n + 1
        if n < len(failures):
            raise failures[n]
        return _FakeResponse(success_body)
    fake_urlopen.calls = calls  # type: ignore[attr-defined]
    return fake_urlopen


def test_retry_recovers_from_transient_503(monkeypatch):
    """One 503 then 200 → success on attempt 2. Sleep zeroed so the test
    is fast; the production behavior is unchanged."""
    monkeypatch.setattr("blackheart_train.register_client._sleep_with_jitter", lambda d: None)
    failures = [urllib.error.HTTPError(
        url="http://127.0.0.1:8082/models/register",
        code=503, msg="Service Unavailable",
        hdrs=None, fp=io.BytesIO(b"{}"),
    )]
    fake_urlopen = _failing_then_succeeding(failures, {
        "model_id": "00000000-0000-0000-0000-000000000002",
        "content_sha256": "a" * 64,
        "status": "awaiting_operator_review",
        "version": 1,
        "registered_at": "2026-05-15T00:00:00",
        "idempotent_replay": False,
    })
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result = register_with_orchestrator(
        _sample_payload(), None,
        orchestrator_url="http://127.0.0.1:8082",
        auth_token="t", agent_name="a",
    )
    assert result["model_id"] == "00000000-0000-0000-0000-000000000002"
    assert fake_urlopen.calls["n"] == 2  # one failure, one success


def test_retry_recovers_from_transient_urlerror(monkeypatch):
    """Network blip (URLError) then 200 → success on retry."""
    monkeypatch.setattr("blackheart_train.register_client._sleep_with_jitter", lambda d: None)
    failures = [urllib.error.URLError("Connection refused")]
    fake_urlopen = _failing_then_succeeding(failures, {
        "model_id": "00000000-0000-0000-0000-000000000003",
        "content_sha256": "a" * 64,
        "status": "awaiting_operator_review",
        "version": 1,
        "registered_at": "2026-05-15T00:00:00",
        "idempotent_replay": False,
    })
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result = register_with_orchestrator(
        _sample_payload(), None,
        orchestrator_url="http://127.0.0.1:8082",
        auth_token="t", agent_name="a",
    )
    assert result["model_id"] == "00000000-0000-0000-0000-000000000003"
    assert fake_urlopen.calls["n"] == 2


def test_no_retry_on_4xx_caller_error(monkeypatch):
    """400/422 from the orchestrator means caller-side payload bug;
    retrying just wastes time. Must fail-fast on attempt 1."""
    monkeypatch.setattr("blackheart_train.register_client._sleep_with_jitter", lambda d: None)
    calls = {"n": 0}
    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        raise urllib.error.HTTPError(
            url=req.full_url, code=400, msg="Bad Request",
            hdrs=None, fp=io.BytesIO(b'{"error_code":"validation_failed"}'),
        )
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(RegisterError, match="HTTP 400"):
        register_with_orchestrator(
            _sample_payload(), None,
            orchestrator_url="http://127.0.0.1:8082",
            auth_token="t", agent_name="a",
            max_attempts=5,  # generous cap to prove fail-fast ignores it
        )
    assert calls["n"] == 1, (
        f"4xx must fail-fast — saw {calls['n']} attempts"
    )


def test_retry_exhaustion_carries_status_code(monkeypatch):
    """All attempts fail with 503 → RegisterError carries status + count."""
    monkeypatch.setattr("blackheart_train.register_client._sleep_with_jitter", lambda d: None)
    calls = {"n": 0}
    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        raise urllib.error.HTTPError(
            url=req.full_url, code=503, msg="Service Unavailable",
            hdrs=None, fp=io.BytesIO(b'{}'),
        )
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(RegisterError, match="HTTP 503"):
        register_with_orchestrator(
            _sample_payload(), None,
            orchestrator_url="http://127.0.0.1:8082",
            auth_token="t", agent_name="a",
            max_attempts=3,
        )
    assert calls["n"] == 3


def test_retry_429_is_retryable(monkeypatch):
    """Rate-limit response (429) is retryable per HTTP semantics."""
    monkeypatch.setattr("blackheart_train.register_client._sleep_with_jitter", lambda d: None)
    failures = [urllib.error.HTTPError(
        url="http://127.0.0.1:8082/models/register",
        code=429, msg="Too Many Requests",
        hdrs=None, fp=io.BytesIO(b"{}"),
    )]
    fake_urlopen = _failing_then_succeeding(failures, {
        "model_id": "00000000-0000-0000-0000-000000000004",
        "content_sha256": "a" * 64,
        "status": "awaiting_operator_review",
        "version": 1,
        "registered_at": "2026-05-15T00:00:00",
        "idempotent_replay": False,
    })
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result = register_with_orchestrator(
        _sample_payload(), None,
        orchestrator_url="http://127.0.0.1:8082",
        auth_token="t", agent_name="a",
    )
    assert result["model_id"] == "00000000-0000-0000-0000-000000000004"


def test_max_attempts_zero_rejected():
    """Defensive: max_attempts must be >= 1."""
    with pytest.raises(ValueError, match="max_attempts must be >= 1"):
        register_with_orchestrator(
            _sample_payload(), None,
            orchestrator_url="http://127.0.0.1:8082",
            auth_token="t", agent_name="a",
            max_attempts=0,
        )
