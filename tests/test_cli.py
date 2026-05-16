"""Unit tests for cli.py helpers.

The interesting bit here is :func:`_sanitize_for_json` — RFC-7159 JSON
has no NaN/Infinity tokens, so any of our metric values that come back
NaN need to be replaced with ``null`` before serialisation. We test
this in isolation rather than running the full CLI because the
contract is "no NaN escapes to JSON," and that's easiest to assert at
the helper boundary.
"""
from __future__ import annotations

import json
import math

import pytest

from blackheart_train.cli import _json_default, _sanitize_for_json


def test_sanitize_replaces_nan_with_none():
    assert _sanitize_for_json(float("nan")) is None


def test_sanitize_replaces_inf_with_none():
    assert _sanitize_for_json(float("inf")) is None
    assert _sanitize_for_json(float("-inf")) is None


def test_sanitize_preserves_finite_floats():
    assert _sanitize_for_json(3.14) == 3.14
    assert _sanitize_for_json(0.0) == 0.0
    assert _sanitize_for_json(-1.5) == -1.5


def test_sanitize_walks_nested_dict():
    payload = {
        "metrics": {"auc": float("nan"), "loss": 0.5},
        "spec": {"name": "x", "lr": float("inf")},
    }
    out = _sanitize_for_json(payload)
    assert out["metrics"]["auc"] is None
    assert out["metrics"]["loss"] == 0.5
    assert out["spec"]["lr"] is None


def test_sanitize_walks_list_and_tuple():
    payload = [1.0, float("nan"), [float("inf"), 2.0]]
    out = _sanitize_for_json(payload)
    assert out == [1.0, None, [None, 2.0]]
    # Tuples become lists (JSON has no tuple type)
    out_tup = _sanitize_for_json((float("nan"), 1))
    assert out_tup == [None, 1]


def test_sanitize_preserves_non_float_primitives():
    """ints, strings, bools, None all pass through unchanged."""
    payload = {"i": 42, "s": "hello", "b": True, "n": None}
    assert _sanitize_for_json(payload) == payload


def test_sanitized_payload_round_trips_through_strict_json():
    """End-to-end: sanitised output must encode under allow_nan=False
    and decode without losing structure."""
    payload = {
        "metrics": {"auc": float("nan"), "rmse": 0.05},
        "runs": [
            {"metric": float("inf"), "ok": True},
            {"metric": 0.42, "ok": False},
        ],
    }
    sanitised = _sanitize_for_json(payload)
    encoded = json.dumps(sanitised, allow_nan=False)
    decoded = json.loads(encoded)
    assert decoded["metrics"]["auc"] is None
    assert decoded["runs"][0]["metric"] is None
    assert decoded["runs"][1]["metric"] == 0.42


def test_json_default_raises_on_unknown_type():
    """The strict encoder fallback must reject anything not explicitly
    handled, so a future payload type can't be silently str()-coerced."""

    class Custom:
        pass

    with pytest.raises(TypeError, match="not JSON serialisable"):
        _json_default(Custom())


def test_json_default_handles_datetime_and_path(tmp_path):
    import datetime as _dt

    out = _json_default(_dt.datetime(2026, 5, 15, 12, 0))
    assert out == "2026-05-15T12:00:00"
    out = _json_default(tmp_path)
    assert isinstance(out, str)
