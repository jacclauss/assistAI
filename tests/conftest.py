from __future__ import annotations

import os
from pathlib import Path

import pytest

from assistai.sysclock import ClockState, KernelClockSync


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep settings tests off the developer's real environment.

    Clears ambient AssistAI variables and moves to an empty directory so the
    relative ``.env`` lookup finds nothing. Without this, results depend on
    whoever's machine is running the suite.
    """
    for key in list(os.environ):
        if key == "ASSISTAI_LIVE":
            continue
        if key.startswith("ASSISTAI_") or key == "FIREWORKS_API_KEY":
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)


@pytest.fixture(autouse=True)
def synced_kernel_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Report a synchronized kernel clock unless a test injects its own.

    The gateway waits up to two minutes for an unsynchronized clock. Whether
    the machine running the suite has NTP is not the code under test.
    """
    monkeypatch.setattr(KernelClockSync, "state", lambda _self: ClockState.SYNCED)
