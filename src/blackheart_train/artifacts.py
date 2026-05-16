"""Content-addressed artifact storage on local FS.

An artifact is identified by ``content_sha256`` — a hash over *only* the
content-defining fields (booster bytes + spec + feature_names + label info).
Run metadata (``trained_at``, ``metrics``, ``n_train_rows``…) lives inside
the payload but does NOT affect the sha. This means a re-train against
identical data with the same seed lands at the same path — true content
addressing.

Layout:

    <artifact_dir>/<sha256[:2]>/<sha256>.pkl

The 2-char shard prefix keeps any single directory from growing past a
few thousand entries (NTFS handles more, but FS browsers struggle).

Pickle protocol is pinned to **5** so the byte layout is stable across
Python minor-version bumps. ``HIGHEST_PROTOCOL`` floats and would silently
change the on-disk representation when we upgrade Python.

Integrity:

* ``compute_content_sha`` hashes a canonicalised JSON of the content
  fields (sorted keys, ``default=str`` for datetimes). The booster's
  text representation (``model_to_string()``) is part of the content —
  deterministic for a seeded LightGBM fit on identical data.
* ``read_artifact`` cross-checks ``payload["content_sha256"]`` against
  the filename's sha. A mismatch means the artifact was tampered with
  or written by a buggy producer; we refuse rather than silently use it.

Booster prediction semantics (so M5e's registry write + the live inference
worker know what they get back from ``Booster.predict``):

* ``objective='binary'``      → ``predict()`` returns class-1 probabilities.
* ``objective='regression'``  → ``predict()`` returns raw forecasts.
* ``objective='multiclass'``  → softmax probabilities per class (not used in M5a).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


_PICKLE_PROTOCOL = 5


def compute_content_sha(content: dict[str, Any]) -> str:
    """Hash ``content`` deterministically via canonical JSON.

    ``content`` should hold only the model-identity fields — typically
    ``{spec, feature_names, objective, label_feature, label_version,
    booster_model_str}``. Order of keys does not matter; ``sort_keys=True``
    canonicalises. ``default=str`` is acceptable here because callers are
    expected to pre-convert their payload (dataclass → asdict, tuple → list).
    """
    canonical = json.dumps(content, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def write_artifact(
    payload: dict[str, Any],
    content_sha: str,
    artifact_dir: Path,
) -> tuple[Path, int]:
    """Pickle ``payload`` at ``<artifact_dir>/<sha[:2]>/<sha>.pkl``.

    The caller must already have set ``payload["content_sha256"] = content_sha``
    so that :func:`read_artifact` can verify the round-trip. We assert the
    invariant here rather than silently writing a self-inconsistent file.

    Returns ``(abs_path, size_bytes)``. Idempotent: re-writing to an
    existing path is a no-op (we trust the filename's sha is the source
    of truth for identity).
    """
    if payload.get("content_sha256") != content_sha:
        raise ValueError(
            "payload['content_sha256'] must equal content_sha argument "
            f"(payload has {payload.get('content_sha256')!r}, "
            f"expected {content_sha!r})"
        )

    target = artifact_dir / content_sha[:2] / f"{content_sha}.pkl"
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        logger.info("artifact already exists | sha256=%s path=%s", content_sha, target)
        return target.resolve(), target.stat().st_size

    blob = pickle.dumps(payload, protocol=_PICKLE_PROTOCOL)
    # Unique tmp name keeps two parallel writers from clobbering each other.
    tmp = target.with_suffix(f".pkl.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}")
    try:
        tmp.write_bytes(blob)
        tmp.replace(target)
    finally:
        # Cleanup if replace() raised — tmp.unlink() is a no-op if missing.
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass

    size = target.stat().st_size
    logger.info("artifact written | sha256=%s path=%s bytes=%d", content_sha, target, size)
    return target.resolve(), size


def read_artifact(content_sha: str, artifact_dir: Path) -> dict[str, Any]:
    """Load an artifact and verify its content_sha matches the filename.

    Tampering detection: we don't hash the file bytes (that would change
    with every re-pickle even for identical models). Instead we trust the
    filename and assert the payload's self-reported ``content_sha256``
    matches. Mismatch = producer bug or human edit; refuse.

    v1 → v2 backfill: pre-Phase-2 (M5g.3 phase 1 and earlier) artifacts
    have no ``payload_version`` or ``ensemble`` key — they're implicitly
    v1 with the single LightGBM booster under ``payload["booster"]``.
    We backfill ``payload_version=1`` and ``ensemble=None`` so v2-aware
    consumers can branch on the version field without first probing for
    its existence. The on-disk pickle is not modified; the backfill is
    purely in the returned dict.
    """
    path = artifact_dir / content_sha[:2] / f"{content_sha}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"artifact not found: {path}")
    payload = pickle.loads(path.read_bytes())
    stored = payload.get("content_sha256")
    if stored != content_sha:
        raise ValueError(
            f"artifact content_sha mismatch at {path}: "
            f"filename says {content_sha}, payload says {stored!r}"
        )
    payload.setdefault("payload_version", 1)
    payload.setdefault("ensemble", None)
    return payload
