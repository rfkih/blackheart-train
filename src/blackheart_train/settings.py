"""Centralised settings via pydantic-settings.

All env vars prefixed ``TRAIN_`` to avoid collisions with the trading JVM
(``SPRING_*``) and blackheart-ingest (``INGEST_*``) when both run on the
same host.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TRAIN_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Postgres (shared with trading JVM) ─────────────────────────────────
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "trading_db"
    db_user: str = "blackheart_research"
    db_password: str = ""

    # ── Artifact storage ───────────────────────────────────────────────────
    artifact_dir: Path = Path("C:/Project/blackheart-train/artifacts")

    # ── Orchestrator (--register flag posts here) ──────────────────────────
    # The training worker does NOT call the orchestrator by default; only
    # when --register is passed. The orchestrator's POST /models/register
    # endpoint inserts a model_registry row keyed on content_sha256.
    orchestrator_url: str = "http://127.0.0.1:8082"
    orchestrator_token: str = "dev-sentinel-not-for-prod"
    orchestrator_request_timeout_s: float = 30.0
    agent_name: str = "blackheart-train"

    # ── Behaviour ──────────────────────────────────────────────────────────
    log_level: str = "INFO"

    def db_kwargs(self) -> dict[str, object]:
        return {
            "host": self.db_host,
            "port": self.db_port,
            "dbname": self.db_name,
            "user": self.db_user,
            "password": self.db_password,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
