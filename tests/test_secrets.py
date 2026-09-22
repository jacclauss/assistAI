"""The calendar password comes from the Keychain or a private file, not .env."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from structlog.testing import capture_logs

from assistai.config import Settings
from assistai.errors import SecretsError
from assistai.secrets import (
    apply_caldav_password,
    read_keychain_password,
    read_stored_caldav_password,
)
from tests.fakes import settings


def _configured(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "caldav_username": "shared@icloud.com",
        "caldav_calendar": "Home",
    }
    values.update(overrides)
    return settings(**values)


def test_private_file_beats_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "caldav-password"
    path.write_text("from-file\n", encoding="utf-8")
    path.chmod(0o600)
    monkeypatch.setattr("assistai.secrets.read_keychain_password", lambda: None)

    updated = apply_caldav_password(
        _configured(caldav_password="from-env", caldav_password_file=path)
    )

    assert updated.caldav_password is not None
    assert updated.caldav_password.get_secret_value() == "from-file"


def test_keychain_beats_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("assistai.secrets.read_keychain_password", lambda: "from-keychain")

    updated = apply_caldav_password(_configured(caldav_password="from-env"))

    assert updated.caldav_password is not None
    assert updated.caldav_password.get_secret_value() == "from-keychain"


def test_environment_is_a_warned_last_resort(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("assistai.secrets.read_keychain_password", lambda: None)
    monkeypatch.setattr("assistai.secrets.default_password_path", lambda: tmp_path / "missing")

    with capture_logs() as logs:
        updated = apply_caldav_password(_configured(caldav_password="from-env"))

    assert updated.caldav_password is not None
    assert updated.caldav_password.get_secret_value() == "from-env"
    assert any(entry.get("event") == "secrets.caldav_password_in_environment" for entry in logs)
    assert all("from-env" not in str(entry) for entry in logs)


def test_unconfigured_calendar_does_not_read_a_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse() -> str:
        raise AssertionError("looked up a secret")

    monkeypatch.setattr("assistai.secrets.read_keychain_password", refuse)

    assert apply_caldav_password(settings()).caldav_password is None


def test_loose_file_is_refused_without_echoing_it(tmp_path: Path) -> None:
    path = tmp_path / "caldav-password"
    path.write_text("app-secret-password\n", encoding="utf-8")
    path.chmod(0o644)

    with pytest.raises(SecretsError, match="chmod 600") as caught:
        read_stored_caldav_password(settings(caldav_password_file=path))

    assert "app-secret-password" not in str(caught.value)


def test_missing_or_non_regular_file_is_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("assistai.secrets.read_keychain_password", lambda: None)
    missing = tmp_path / "missing"
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)

    assert read_stored_caldav_password(settings(caldav_password_file=missing)) is None
    assert read_stored_caldav_password(settings(caldav_password_file=fifo)) is None


def test_directory_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SecretsError, match="directory"):
        read_stored_caldav_password(settings(caldav_password_file=tmp_path))


def test_symlink_is_refused(tmp_path: Path) -> None:
    target = tmp_path / "real"
    target.write_text("app-secret-password\n", encoding="utf-8")
    target.chmod(0o600)
    link = tmp_path / "link"
    link.symlink_to(target)

    with pytest.raises(SecretsError, match="symlink") as caught:
        read_stored_caldav_password(settings(caldav_password_file=link))

    assert "app-secret-password" not in str(caught.value)


def test_file_inside_the_project_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    package = root / "src" / "assistai"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    secret = root / "caldav-password"
    secret.write_text("app-secret-password\n", encoding="utf-8")
    secret.chmod(0o600)

    with pytest.raises(SecretsError, match="outside this project") as caught:
        apply_caldav_password(settings(caldav_password_file=secret))

    assert "app-secret-password" not in str(caught.value)


def test_empty_loose_file_is_unset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("assistai.secrets.read_keychain_password", lambda: None)
    path = tmp_path / "caldav-password"
    path.write_text("", encoding="utf-8")
    path.chmod(0o644)

    assert read_stored_caldav_password(settings(caldav_password_file=path)) is None


def test_trailing_crlf_is_one_newline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("assistai.secrets.read_keychain_password", lambda: None)
    path = tmp_path / "caldav-password"
    path.write_bytes(b"app-secret-password\r\n")
    path.chmod(0o600)

    assert read_stored_caldav_password(settings(caldav_password_file=path)) == "app-secret-password"


def test_oversize_file_is_refused_without_echoing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("assistai.secrets.read_keychain_password", lambda: None)
    path = tmp_path / "caldav-password"
    path.write_bytes(b"a" * 5000)
    path.chmod(0o600)

    with pytest.raises(SecretsError, match="too large") as caught:
        read_stored_caldav_password(settings(caldav_password_file=path))

    assert "aaaa" not in str(caught.value)


def test_writable_directory_is_refused(tmp_path: Path) -> None:
    parent = tmp_path / "open"
    parent.mkdir()
    parent.chmod(0o777)
    path = parent / "caldav-password"
    path.write_text("app-secret-password\n", encoding="utf-8")
    path.chmod(0o600)

    with pytest.raises(SecretsError, match="writable by other users") as caught:
        read_stored_caldav_password(settings(caldav_password_file=path))

    assert "app-secret-password" not in str(caught.value)


def test_sticky_directory_can_hold_the_file(tmp_path: Path) -> None:
    parent = tmp_path / "sticky"
    parent.mkdir()
    parent.chmod(0o1777)
    path = parent / "caldav-password"
    path.write_text("from-file\n", encoding="utf-8")
    path.chmod(0o600)

    assert read_stored_caldav_password(settings(caldav_password_file=path)) == "from-file"


def test_default_symlink_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setattr("assistai.secrets.read_keychain_password", lambda: None)
    parent = tmp_path / ".config" / "assistai"
    parent.mkdir(parents=True)
    target = tmp_path / "real"
    target.write_text("app-secret-password\n", encoding="utf-8")
    target.chmod(0o600)
    link = parent / "caldav-password"
    link.symlink_to(target)

    with pytest.raises(SecretsError, match="symlink") as caught:
        read_stored_caldav_password(settings())

    assert "app-secret-password" not in str(caught.value)


def test_utf8_bom_is_not_part_of_the_password(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("assistai.secrets.read_keychain_password", lambda: None)
    path = tmp_path / "caldav-password"
    path.write_bytes(b"\xef\xbb\xbfapp-secret-password\n")
    path.chmod(0o600)

    assert read_stored_caldav_password(settings(caldav_password_file=path)) == "app-secret-password"


def test_default_file_lives_under_the_home_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setattr("assistai.secrets.read_keychain_password", lambda: None)
    path = tmp_path / ".config" / "assistai" / "caldav-password"
    path.parent.mkdir(parents=True)
    path.write_text("from-home\n", encoding="utf-8")
    path.chmod(0o600)

    assert read_stored_caldav_password(settings()) == "from-home"


def test_keychain_strips_one_trailing_newline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr("assistai.secrets.sys.platform", "darwin")

    def fake_run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        assert argv[0] == "/usr/bin/security"
        assert argv[-1] == "-w"
        return subprocess.CompletedProcess(argv, 0, stdout=b"app-secret-password\n")

    monkeypatch.setattr("assistai.secrets.subprocess.run", fake_run)

    assert read_keychain_password() == "app-secret-password"


def test_missing_keychain_item_is_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr("assistai.secrets.sys.platform", "darwin")
    monkeypatch.setattr(
        "assistai.secrets.subprocess.run",
        lambda argv, **_kwargs: subprocess.CompletedProcess(argv, 44, stdout=b""),
    )

    assert read_keychain_password() is None


def test_pytest_does_not_open_the_home_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/test_secrets.py::test")
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    path = tmp_path / ".config" / "assistai" / "caldav-password"
    path.parent.mkdir(parents=True)
    path.write_text("from-home\n", encoding="utf-8")
    path.chmod(0o600)

    assert apply_caldav_password(_configured()).caldav_password is None


def test_keychain_failure_does_not_log_the_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr("assistai.secrets.sys.platform", "darwin")
    monkeypatch.setattr(
        "assistai.secrets.subprocess.run",
        lambda argv, **_kwargs: subprocess.CompletedProcess(
            argv, 128, stdout=b"app-secret-password\n"
        ),
    )

    with capture_logs() as logs:
        assert read_keychain_password() is None

    assert any(entry.get("event") == "secrets.keychain_unavailable" for entry in logs)
    assert all("app-secret-password" not in str(entry) for entry in logs)


def test_keychain_is_not_consulted_off_mac_or_during_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        raise AssertionError("called security")

    monkeypatch.setattr("assistai.secrets.subprocess.run", refuse)
    monkeypatch.setattr("assistai.secrets.sys.platform", "linux")

    assert read_keychain_password() is None

    monkeypatch.setattr("assistai.secrets.sys.platform", "darwin")

    assert read_keychain_password() is None
