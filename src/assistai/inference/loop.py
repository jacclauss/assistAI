"""One conversation turn: stream a completion, run tools, stream again.

The loop only executes tools the caller put on the surface. The broker sits in
front of ``execute`` and is what enforces per-agent ACLs.
"""

from __future__ import annotations

from collections.abc import Callable

import structlog

from assistai.errors import ToolLoopError
from assistai.inference.client import FireworksClient
from assistai.inference.tools import ToolSurface
from assistai.inference.types import Message, TextDelta
from assistai.manifest import ModelPin

log = structlog.get_logger(__name__)

DeltaCallback = Callable[[TextDelta], None]


async def run_turn(
    client: FireworksClient,
    pin: ModelPin,
    messages: list[Message],
    registry: ToolSurface | None,
    *,
    max_tool_rounds: int,
    on_delta: DeltaCallback | None = None,
) -> list[Message]:
    """Append assistant (and any tool) messages for one user turn.

    ``messages`` is mutated and also returned so callers can ignore the return
    if they already hold the list.
    """
    tools = registry.specs() if registry is not None else None
    rounds = 0
    # Anything the model says with untrusted output in context is derived from
    # it. Marking those messages too keeps the conversation tainted after the
    # raw tool result is trimmed away, since a summary can carry the injection
    # just as well as the page did.
    derived_from_untrusted = any(message.untrusted for message in messages)
    while True:
        completion = await client.complete(pin, messages, tools, on_delta=on_delta)
        if completion.usage is not None:
            log.info(
                "inference.usage",
                model=pin.ref,
                prompt_tokens=completion.usage.prompt_tokens,
                completion_tokens=completion.usage.completion_tokens,
            )
        assistant = Message(
            role="assistant",
            content=completion.content or None,
            tool_calls=list(completion.tool_calls),
            untrusted=derived_from_untrusted,
        )
        messages.append(assistant)
        if not completion.tool_calls:
            return messages
        if registry is None:
            raise ToolLoopError("model requested a tool but no registry was provided")
        rounds += 1
        if rounds > max_tool_rounds:
            raise ToolLoopError(f"exceeded max_tool_rounds ({max_tool_rounds})")
        for call in completion.tool_calls:
            log.info("tool.invoked", name=call.name, call_id=call.id)
            result = await registry.execute(call)
            derived_from_untrusted = derived_from_untrusted or result.untrusted
            messages.append(
                Message(
                    role="tool",
                    content=result.content,
                    tool_call_id=call.id,
                    untrusted=result.untrusted,
                )
            )
