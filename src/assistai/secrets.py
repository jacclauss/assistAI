"""Calendar password lookup. Not the project tree, and not the model prompt.

On a Mac the gateway reads the macOS Keychain (service ``assistai``, account
``caldav``). On the Pi it reads a regular file outside the repo, mode 600.
An environment value is only used when neither of those is set, and that path
logs a warning because a file in this project is readable by a coding agent.
"""

from __future__ import annotations

import errno
import os
import stat
import subprocess
import sys
from pathlib import Path

import structlog
from pydantic import SecretStr

from assistai.config import Settings
from assistai.errors import SecretsError

log = structlog.get_logger(__name__)

KEYCHAIN_SERVICE = "assistai"
KEYCHAIN_ACCOUNT = "caldav"
# Long enough for the macOS Allow prompt. `security` returns immediately when
# the item is missing or this process is already allowed.
_KEYCHAIN_TIMEOUT_SECONDS = 60.0
# App-specific passwords are short. A later OAuth refresh token still fits.
_MAX_SECRET_BYTES = 4096


def default_password_path() -> Path:
    """Outside the repo, so opening the project does not open the secret."""
    return Path.home() / ".config" / "assistai" / "caldav-password"


def apply_caldav_password(settings: Settings) -> Settings:
    """Fill ``caldav_password`` from the Keychain or a private file."""
    path = _candidate_file(settings)
    # Reject an unsafe file even when the calendar name is still blank, so a
    # secret dropped in the project is not ignored until setup is finished.
    file_secret = _load_file(path) if path is not None else None
    if not settings.caldav_username.strip() or not settings.caldav_calendar.strip():
        return settings
    keychain = read_keychain_password()
    if keychain:
        return settings.model_copy(update={"caldav_password": SecretStr(keychain)})
    if file_secret:
        return settings.model_copy(update={"caldav_password": SecretStr(file_secret)})
    current = settings.caldav_password
    if current is not None and current.get_secret_value():
        log.warning(
            "secrets.caldav_password_in_environment",
            hint=(
                "move the app-specific password into the macOS Keychain "
                "or a mode 600 file outside this repo"
            ),
        )
    return settings


def read_stored_caldav_password(settings: Settings) -> str | None:
    """Keychain first, then the configured file, then the default path."""
    path = _candidate_file(settings)
    file_secret = _load_file(path) if path is not None else None
    keychain = read_keychain_password()
    if keychain:
        return keychain
    return file_secret


def read_keychain_password() -> str | None:
    """The app-specific password, or None when Keychain has no such item."""
    if sys.platform != "darwin":
        return None
    # A test must not touch the login keychain. A prompt would hang the suite,
    # and a hit would copy the real app-specific password into the test process.
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return None
    try:
        completed = subprocess.run(  # noqa: S603 — argv is a fixed Keychain lookup, not user input
            [
                "/usr/bin/security",
                "find-generic-password",
                "-s",
                KEYCHAIN_SERVICE,
                "-a",
                KEYCHAIN_ACCOUNT,
                "-w",
            ],
            check=False,
            capture_output=True,
            timeout=_KEYCHAIN_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        log.warning("secrets.keychain_unavailable", reason="timeout")
        return None
    except OSError:
        log.warning("secrets.keychain_unavailable", reason="unavailable")
        return None
    if completed.returncode != 0:
        # 44 is "item not found", which is the normal path on a machine that
        # uses the file instead. Anything else is a denied prompt or a failure.
        if completed.returncode != 44:
            log.warning("secrets.keychain_unavailable", returncode=completed.returncode)
        return None
    if len(completed.stdout) > _MAX_SECRET_BYTES + 2:
        log.warning("secrets.keychain_rejected", reason="too large")
        return None
    try:
        secret = completed.stdout.decode("utf-8")
    except UnicodeDecodeError:
        log.warning("secrets.keychain_rejected", reason="not text")
        return None
    secret = _strip_one_newline(secret)
    if len(secret.encode("utf-8")) > _MAX_SECRET_BYTES:
        log.warning("secrets.keychain_rejected", reason="too large")
        return None
    return secret or None


def _candidate_file(settings: Settings) -> Path | None:
    if settings.caldav_password_file is not None:
        return settings.caldav_password_file
    # The default path is the operator's real file. Tests must not open it.
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return None
    default = default_password_path()
    try:
        default.lstat()
    except FileNotFoundError:
        return None
    except OSError:
        # Present but not stat-able. _load_file raises a message without the secret.
        return default
    return default


def _load_file(path: Path) -> str | None:
    """The file's text, or None when there is nothing to read.

    Raises when the path is inside this project, is a symlink or directory,
    or is a non-empty file other users can read.
    """
    if _inside_source_tree(path):
        raise SecretsError("calendar password file must live outside this project")
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    except OSError:
        raise SecretsError(f"calendar password file {path} is not readable") from None
    if stat.S_ISLNK(info.st_mode):
        raise SecretsError(f"calendar password file {path} must be a regular file, not a symlink")
    if stat.S_ISDIR(info.st_mode):
        raise SecretsError(
            f"calendar password file {path} is a directory; "
            "remove it and use a mode 600 file outside this project"
        )
    if not stat.S_ISREG(info.st_mode):
        return None
    # Check the directory before opening. A replaceable directory can swap in a
    # FIFO, and opening that would block the gateway.
    _reject_replaceable_parent(path)
    return _read_regular(path)


def _inside_source_tree(path: Path) -> bool:
    """True when ``path`` sits in a checkout or the gateway image source."""
    candidates = [path.absolute()]
    try:
        resolved = path.resolve()
    except OSError:
        resolved = None
    if resolved is not None and resolved not in candidates:
        candidates.append(resolved)
    for candidate in candidates:
        for parent in candidate.parents:
            try:
                inside = (parent / "src" / "assistai" / "__init__.py").is_file()
            except OSError:
                continue
            if inside:
                return True
    return False


def _read_regular(path: Path) -> str | None:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise SecretsError(
                f"calendar password file {path} must be a regular file, not a symlink"
            ) from None
        raise SecretsError(f"calendar password file {path} is not readable") from None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            return None
        # An empty placeholder (including a null mount that shows up as an
        # empty mode-0666 file) is "not set". A non-empty loose file is not.
        if info.st_size == 0:
            return None
        if info.st_mode & 0o077:
            raise SecretsError(
                f"calendar password file {path} is readable by other users; chmod 600 {path}"
            )
        # Two extra bytes leave room for one trailing CR LF before the cap.
        if info.st_size > _MAX_SECRET_BYTES + 2:
            raise SecretsError("calendar password file is too large")
        raw = _read_bounded(fd)
    finally:
        os.close(fd)
    if len(raw) > _MAX_SECRET_BYTES + 2:
        raise SecretsError("calendar password file is too large")
    try:
        secret = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise SecretsError("calendar password file is not text") from None
    secret = _strip_one_newline(secret)
    if len(secret.encode("utf-8")) > _MAX_SECRET_BYTES:
        raise SecretsError("calendar password file is too large")
    return secret or None


def _reject_replaceable_parent(path: Path) -> None:
    """A group-writable directory lets someone else swap the password file."""
    try:
        mode = path.parent.stat().st_mode
    except OSError:
        raise SecretsError(f"calendar password file {path} is not readable") from None
    if stat.S_ISDIR(mode) and (mode & 0o022) and not (mode & stat.S_ISVTX):
        raise SecretsError(f"calendar password directory {path.parent} is writable by other users")


def _read_bounded(fd: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    limit = _MAX_SECRET_BYTES + 3
    while total < limit:
        part = os.read(fd, limit - total)
        if not part:
            break
        chunks.append(part)
        total += len(part)
    return b"".join(chunks)


def _strip_one_newline(secret: str) -> str:
    """Drop the single trailing newline a file or `security -w` adds."""
    if secret.endswith("\r\n"):
        return secret[:-2]
    if secret.endswith("\n"):
        return secret[:-1]
    return secret
