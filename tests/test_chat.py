from __future__ import annotations

from tests.fakes import (
    catalog_then_completions,
    client_for,
    completion_stream,
    manifest,
    settings,
    text_event,
)

from assistai.chat import chat_once, chat_repl


async def test_once_returns_final_text() -> None:
    client = client_for(lambda _req: completion_stream(text_event("pong", finish="stop")))
    chunks: list[str] = []

    text = await chat_once(
        settings(),
        "ping",
        client=client,
        manifest=manifest(),
        use_tools=False,
        write=chunks.append,
    )

    assert text == "pong"
    assert "pong" in "".join(chunks)
    await client.aclose()


async def test_repl_quit_and_reset() -> None:
    client = client_for(
        catalog_then_completions(
            (text_event("first", finish="stop"),),
            (text_event("second", finish="stop"),),
        )
    )
    prompts = iter(["hello", "/reset", "again", "/quit"])
    lines: list[str] = []

    await chat_repl(
        settings(),
        client=client,
        manifest=manifest(),
        use_tools=False,
        read_line=lambda _prompt: next(prompts),
        write=lambda s: None,
        writeln=lines.append,
    )

    assert any("history cleared" in line for line in lines)
    await client.aclose()


async def test_repl_eof_exits_cleanly() -> None:
    client = client_for(catalog_then_completions())

    def boom(_prompt: str) -> str:
        raise EOFError

    await chat_repl(
        settings(),
        client=client,
        manifest=manifest(),
        use_tools=False,
        read_line=boom,
        write=lambda s: None,
        writeln=lambda s: None,
    )
    await client.aclose()
