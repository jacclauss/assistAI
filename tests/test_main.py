from __future__ import annotations

import pytest

import assistai.__main__ as cli
from assistai.errors import MissingAPIKeyError


def test_unknown_command_starts_gateway(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[str] = []

    async def fake_gateway() -> None:
        called.append("gateway")

    monkeypatch.setattr(cli, "_gateway", fake_gateway)
    cli.main([])

    assert called == ["gateway"]


def test_chat_once_dispatches(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[str | None, bool]] = []

    async def fake_chat(once: str | None, *, use_tools: bool) -> None:
        seen.append((once, use_tools))

    monkeypatch.setattr(cli, "_chat", fake_chat)
    cli.main(["chat", "--once", "ping", "--no-tools"])

    assert seen == [("ping", False)]


def test_assistai_error_exits_two(monkeypatch: pytest.MonkeyPatch) -> None:
    async def boom() -> None:
        raise MissingAPIKeyError("no key")

    monkeypatch.setattr(cli, "_gateway", boom)
    with pytest.raises(SystemExit) as exc:
        cli.main([])

    assert exc.value.code == 2
