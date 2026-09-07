"""Terminal REPL for phase 1. Multi-turn conversation against Fireworks."""

from __future__ import annotations

from collections.abc import Callable

from assistai.config import Settings
from assistai.conversation import SYSTEM_PROMPT, last_assistant_text
from assistai.inference.client import FireworksClient
from assistai.inference.loop import run_turn
from assistai.inference.tools import default_registry
from assistai.inference.types import Message, TextDelta
from assistai.manifest import Manifest, load_manifest, resolve_manifest_path


async def chat_once(
    settings: Settings,
    text: str,
    *,
    client: FireworksClient | None = None,
    manifest: Manifest | None = None,
    use_tools: bool = True,
    write: Callable[[str], None] = lambda s: print(s, end="", flush=True),
) -> str:
    """Run a single user turn and return the assistant's final text."""
    owns_client = client is None
    if client is None:
        client = FireworksClient(settings)
    if manifest is None:
        manifest = load_manifest(resolve_manifest_path(settings))
    messages = [
        Message(role="system", content=SYSTEM_PROMPT),
        Message(role="user", content=text),
    ]
    registry = default_registry() if use_tools else None
    try:
        await run_turn(
            client,
            manifest.primary,
            messages,
            registry,
            max_tool_rounds=settings.max_tool_rounds,
            on_delta=lambda delta: write(delta.text),
        )
    finally:
        if owns_client:
            await client.aclose()
    write("\n")
    return last_assistant_text(messages)


async def chat_repl(
    settings: Settings,
    *,
    client: FireworksClient | None = None,
    manifest: Manifest | None = None,
    use_tools: bool = True,
    read_line: Callable[[str], str] = input,
    write: Callable[[str], None] = lambda s: print(s, end="", flush=True),
    writeln: Callable[[str], None] = print,
) -> None:
    """Interactive loop. ``/quit`` and ``/reset`` are the only commands."""
    owns_client = client is None
    if client is None:
        client = FireworksClient(settings)
    if manifest is None:
        manifest = load_manifest(resolve_manifest_path(settings))
    writeln(f"AssistAI chat  model={manifest.primary.ref}")
    writeln("Type /quit to exit, /reset to clear history.")
    messages = [Message(role="system", content=SYSTEM_PROMPT)]
    registry = default_registry() if use_tools else None
    try:
        await client.validate(manifest.required_refs())
        while True:
            try:
                line = read_line("> ").strip()
            except EOFError:
                writeln("")
                return
            if not line:
                continue
            if line in {"/quit", "/exit"}:
                return
            if line == "/reset":
                messages[:] = [Message(role="system", content=SYSTEM_PROMPT)]
                writeln("history cleared")
                continue
            messages.append(Message(role="user", content=line))
            await run_turn(
                client,
                manifest.primary,
                messages,
                registry,
                max_tool_rounds=settings.max_tool_rounds,
                on_delta=_printer(write),
            )
            write("\n")
    finally:
        if owns_client:
            await client.aclose()


def _printer(write: Callable[[str], None]) -> Callable[[TextDelta], None]:
    def on_delta(delta: TextDelta) -> None:
        write(delta.text)

    return on_delta
