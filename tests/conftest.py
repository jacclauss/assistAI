from __future__ import annotations

import os
from pathlib import Path

import pytest


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
