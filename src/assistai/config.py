"""Runtime configuration.

Operator-facing settings come from the environment. Agent definitions and tool
ACLs are loaded from ``config/assistai.toml``; see ``config/assistai.example.toml``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from assistai.signal.numbers import normalize_e164

LogLevel = Literal["debug", "info", "warning", "error"]
DmPolicy = Literal["pairing", "allowlist"]

FIREWORKS_INFERENCE_URL = "https://api.fireworks.ai/inference/v1"
FIREWORKS_ACCOUNT_URL = "https://api.fireworks.ai/v1"
SIGNAL_DEFAULT_URL = "http://signal-cli:8080"


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
    state_dir: Path = Path("state")
    # How long a channel gets to drain after SIGTERM. Docker kills at 10s by
    # default, so anything longer means being killed rather than exiting.
    shutdown_grace_seconds: float = Field(default=5.0, gt=0)

    # Read from FIREWORKS_API_KEY rather than the ASSISTAI_ prefix, so the same
    # variable name works for any tooling that expects the provider default.
    fireworks_api_key: SecretStr | None = Field(default=None, alias="FIREWORKS_API_KEY")
    fireworks_base_url: str = FIREWORKS_INFERENCE_URL
    fireworks_account_url: str = FIREWORKS_ACCOUNT_URL

    manifest_path: Path | None = None
    max_tokens: int = Field(default=2048, gt=0)
    max_tool_rounds: int = Field(default=4, gt=0)
    request_timeout_seconds: float = Field(default=60.0, gt=0)
    temperature: float = Field(default=0.2, ge=0, le=2)
    # Concurrent model turns across all senders. The Pi has finite memory and
    # every turn is billable.
    max_concurrent_turns: int = Field(default=4, gt=0)

    # Agent roster. Required once Signal is on; the REPL does not need it.
    agents_config_path: Path | None = None

    # Signal is off until an account is set. The default URL is the compose
    # service name; host-mode against compose.dev binds 127.0.0.1:8080.
    signal_base_url: str = SIGNAL_DEFAULT_URL
    signal_account: str | None = None
    # Stored as a comma-separated string so pydantic-settings does not
    # JSON-decode the env var (it would for a tuple/list type).
    signal_allow_from: str = ""
    signal_dm_policy: DmPolicy = "pairing"
    signal_device_name: str = "assistai"
    signal_pairing_ttl_seconds: float = Field(default=900.0, gt=0)
    signal_reconnect_seconds: float = Field(default=2.0, gt=0)
    # signal-cli in json-rpc mode is slow to start; wait for it rather than
    # exiting and letting the container runtime restart-loop us.
    signal_startup_timeout_seconds: float = Field(default=120.0, ge=0)
    # Inference is metered, so cap what one sender can spend per minute and how
    # much text one message can push into context.
    signal_rate_limit_per_minute: int = Field(default=12, gt=0)
    signal_max_inbound_chars: int = Field(default=4000, gt=0)
    signal_pairing_replies_per_hour: int = Field(default=3, gt=0)

    @field_validator("signal_account", mode="before")
    @classmethod
    def _account(cls, value: object) -> object:
        if value is None or value == "":
            return None
        if isinstance(value, str):
            return normalize_e164(value)
        return value

    @field_validator("signal_allow_from", mode="before")
    @classmethod
    def _allow_from(cls, value: object) -> object:
        if value is None or value == "" or value == ():
            return ""
        if isinstance(value, str):
            parts = [normalize_e164(part) for part in value.split(",") if part.strip()]
            return ",".join(parts)
        if isinstance(value, (list, tuple)):
            return ",".join(normalize_e164(str(part)) for part in value)
        return value

    @property
    def allow_from(self) -> tuple[str, ...]:
        """Normalized E.164 numbers permitted to talk to the bot."""
        if not self.signal_allow_from:
            return ()
        return tuple(self.signal_allow_from.split(","))
