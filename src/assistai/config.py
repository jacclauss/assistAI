"""Runtime configuration.

Operator-facing settings come from the environment. Agent definitions and tool
ACLs will be loaded from ``config/assistai.toml`` in phase 3; see
``config/assistai.example.toml`` for the intended shape.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

LogLevel = Literal["debug", "info", "warning", "error"]

FIREWORKS_INFERENCE_URL = "https://api.fireworks.ai/inference/v1"
FIREWORKS_ACCOUNT_URL = "https://api.fireworks.ai/v1"


class Settings(BaseSettings):
    """Environment-derived settings for the gateway process."""

    model_config = SettingsConfigDict(
        env_prefix="ASSISTAI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    log_level: LogLevel = "info"
    log_console: bool = False
    heartbeat_seconds: float = Field(default=60.0, gt=0)

    # Read from FIREWORKS_API_KEY rather than the ASSISTAI_ prefix, so the same
    # variable name works for any tooling that expects the provider default.
    fireworks_api_key: SecretStr | None = Field(default=None, alias="FIREWORKS_API_KEY")
    fireworks_base_url: str = FIREWORKS_INFERENCE_URL
    fireworks_account_url: str = FIREWORKS_ACCOUNT_URL

    manifest_path: Path | None = None
    max_tokens: int = Field(default=2048, gt=0)
    max_tool_rounds: int = Field(default=4, gt=0)
    request_timeout_seconds: float = Field(default=60.0, gt=0)
