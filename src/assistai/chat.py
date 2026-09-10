"""Terminal REPL. Multi-turn conversation against Fireworks, through the broker."""

from __future__ import annotations

from collections.abc import Callable

from assistai.agents import BrokerPolicy, local_agent, system_prompt_for
from assistai.broker import BoundSurface, Scope, ToolBroker, ToolCatalog, builtin_catalog
from assistai.config import Settings
from assistai.conversation import last_assistant_text
from assistai.inference.client import FireworksClient
from assistai.inference.loop import run_turn
from assistai.inference.types import Message, TextDelta
from assistai.manifest import Manifest, load_manifest, resolve_manifest_path

_REPL_AGENT = local_agent("repl")


async def chat_once(
    settings: Settings,
    text: str,
    *,
    client: FireworksClient | None = None,
    manifest: Manifest | None = None,
    catalog: ToolCatalog | None = None,
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
        Message(role="system", content=system_prompt_for(_REPL_AGENT)),
        Message(role="user", content=text),
    ]
    try:
        await run_turn(
            client,
            manifest.primary,
            messages,
            _surface(messages, catalog) if use_tools else None,
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
    messages = [Message(role="system", content=system_prompt_for(_REPL_AGENT))]
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
                messages[:] = [Message(role="system", content=system_prompt_for(_REPL_AGENT))]
                writeln("history cleared")
                continue
            messages.append(Message(role="user", content=line))
            await run_turn(
                client,
                manifest.primary,
                messages,
                _surface(messages) if use_tools else None,
                max_tool_rounds=settings.max_tool_rounds,
                on_delta=_printer(write),
            )
            write("\n")
    finally:
        if owns_client:
            await client.aclose()


def _broker(catalog: ToolCatalog | None = None) -> ToolBroker:
    """Fresh catalog each call so later tools are not frozen at import."""
    return ToolBroker(
        catalog if catalog is not None else builtin_catalog(),
        BrokerPolicy.defaults(),
    )


def _surface(messages: list[Message], catalog: ToolCatalog | None = None) -> BoundSurface:
    """A brokered view of the REPL agent. Taint is seeded from history."""
    return _broker(catalog).for_agent(_REPL_AGENT, Scope.from_history(messages))


def _printer(write: Callable[[str], None]) -> Callable[[TextDelta], None]:
    def on_delta(delta: TextDelta) -> None:
        write(delta.text)

    return on_delta
