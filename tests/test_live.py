"""Optional live checks against Fireworks. Skipped unless ASSISTAI_LIVE=1."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import SecretStr

from assistai.chat import chat_once
from assistai.config import Settings
from assistai.inference.client import FireworksClient
from assistai.manifest import load_manifest

pytestmark = pytest.mark.live


def _live_enabled() -> bool:
    return os.environ.get("ASSISTAI_LIVE") == "1"


@pytest.fixture
def live_settings(repo_root: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    if not _live_enabled():
        pytest.skip("set ASSISTAI_LIVE=1 to hit Fireworks")
    key = _key_from_dotenv(repo_root / ".env")
    if key is None:
        pytest.skip("no FIREWORKS_API_KEY in repo .env")
    return Settings(
        FIREWORKS_API_KEY=SecretStr(key),
        manifest_path=repo_root / "manifest.toml",
        max_tokens=32,
    )


def _key_from_dotenv(path: Path) -> str | None:
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("FIREWORKS_API_KEY="):
            value = line.split("=", 1)[1].strip()
            return value or None
    return None


async def test_primary_model_replies(live_settings: Settings) -> None:
    text = await chat_once(
        live_settings,
        "Reply with the single word pong and nothing else.",
        use_tools=False,
        write=lambda _s: None,
    )

    assert "pong" in text.lower()


async def test_validate_pinned_refs(live_settings: Settings, repo_root: Path) -> None:
    manifest = load_manifest(repo_root / "manifest.toml")
    client = FireworksClient(live_settings)
    try:
        await client.validate(manifest.required_refs())
    finally:
        await client.aclose()
