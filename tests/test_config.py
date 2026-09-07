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
