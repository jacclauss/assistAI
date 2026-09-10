from __future__ import annotations

import pytest
from pydantic import ValidationError

from assistai.config import Settings


def test_defaults_are_conservative() -> None:
    settings = Settings()

    assert settings.log_level == "info"
    assert settings.log_console is False
    assert settings.heartbeat_seconds == 60.0
    assert settings.fireworks_api_key is None
    assert settings.max_tokens == 2048
    assert settings.max_tool_rounds == 4
    assert settings.manifest_path is None
    assert settings.signal_account is None
    assert settings.allow_from == ()
    assert settings.signal_dm_policy == "allowlist"
    assert settings.signal_base_url == "http://signal-cli:8080"
    assert settings.history_keep == 30
    assert settings.history_max_age_seconds == 14 * 24 * 3600


def test_environment_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASSISTAI_LOG_LEVEL", "debug")
    monkeypatch.setenv("ASSISTAI_HEARTBEAT_SECONDS", "5")

    settings = Settings()

    assert settings.log_level == "debug"
    assert settings.heartbeat_seconds == 5.0


def test_fireworks_key_read_without_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FIREWORKS_API_KEY", "fw-secret")

    settings = Settings()

    assert settings.fireworks_api_key is not None
    assert settings.fireworks_api_key.get_secret_value() == "fw-secret"


def test_secret_does_not_leak_into_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FIREWORKS_API_KEY", "fw-secret")

    settings = Settings()

    assert "fw-secret" not in repr(settings)
    assert "fw-secret" not in str(settings)


def test_unknown_log_level_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASSISTAI_LOG_LEVEL", "chatty")

    with pytest.raises(ValidationError):
        Settings()


def test_nonpositive_heartbeat_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(heartbeat_seconds=0)


def test_signal_allow_from_parses_csv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASSISTAI_SIGNAL_ACCOUNT", "+1 (555) 555-0100")
    monkeypatch.setenv("ASSISTAI_SIGNAL_ALLOW_FROM", "+15555550101, +1-555-555-0102")

    settings = Settings()

    assert settings.signal_account == "+15555550100"
    assert settings.allow_from == ("+15555550101", "+15555550102")


def test_invalid_signal_account_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASSISTAI_SIGNAL_ACCOUNT", "not-a-number")

    with pytest.raises(ValidationError):
        Settings()
