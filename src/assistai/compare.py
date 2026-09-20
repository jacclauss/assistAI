"""Score candidate models against the real tool schemas.

Benchmarks do not predict how a model handles this broker's JSON. The
comparison advertises ``get_time`` — a stable, side-effect-free schema — and
records whether the model called it with valid arguments. One failing
candidate does not abort the rest.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import structlog

from assistai.agents import AgentSpec, BrokerPolicy, local_agent
from assistai.broker import BoundSurface, ToolBroker, builtin_catalog
from assistai.config import Settings
from assistai.errors import AssistAIError, ToolLoopError
from assistai.inference.client import FireworksClient
from assistai.inference.loop import run_turn
from assistai.inference.types import Message
from assistai.manifest import Manifest, ModelPin

log = structlog.get_logger(__name__)

_PROMPT = (
    "Call the get_time tool now. Do not guess the time. Do not answer in prose "
    "until the tool result is in context."
)


@dataclass(frozen=True)
class CompareResult:
    """One model's attempt at the real schema."""

    name: str
    ref: str
    available: bool
    tool_called: bool
    arguments_valid: bool
    finished: bool
    error: str | None
    latency_ms: float


def eval_agent() -> AgentSpec:
    """Synthetic agent used only for the bake-off. Not reachable over Signal."""
    return local_agent("eval", tools=("get_time",))


async def compare_models(
    settings: Settings,
    manifest: Manifest,
    *,
    client: FireworksClient | None = None,
    pins: tuple[ModelPin, ...] | None = None,
) -> list[CompareResult]:
    """Run the primary and every candidate against ``get_time``."""
    owns = client is None
    if client is None:
        client = FireworksClient(settings.model_copy(update={"temperature": 0}))
    targets = pins if pins is not None else (manifest.primary, *manifest.candidates.values())
    catalog = builtin_catalog()
    broker = ToolBroker(catalog, BrokerPolicy(tainted_sinks_denied=frozenset()))
    agent = eval_agent()
    results: list[CompareResult] = []
    try:
        for pin in targets:
            results.append(
                await _score(
                    client,
                    pin,
                    broker.for_agent(agent),
                    max_tool_rounds=settings.max_tool_rounds,
                )
            )
    finally:
        if owns:
            await client.aclose()
    return results


def format_report(results: list[CompareResult]) -> str:
    """Human-readable table. Keep it tight: this prints over SSH."""
    header = f"{'name':<14} {'ok':<4} {'tool':<6} {'args':<6} {'ms':>7}  error"
    lines = [header, "-" * len(header)]
    for row in results:
        ok = (
            "yes"
            if row.available and row.tool_called and row.arguments_valid and row.finished
            else "no"
        )
        lines.append(
            f"{row.name:<14} {ok:<4} "
            f"{_yn(row.tool_called):<6} {_yn(row.arguments_valid):<6} "
            f"{row.latency_ms:7.0f}  {row.error or ''}"
        )
    return "\n".join(lines)


def _yn(value: bool) -> str:
    return "yes" if value else "no"


async def _score(
    client: FireworksClient,
    pin: ModelPin,
    surface: BoundSurface,
    *,
    max_tool_rounds: int,
) -> CompareResult:
    started = time.monotonic()
    first = await _attempt(client, pin, surface, max_tool_rounds=max_tool_rounds)
    chosen = first
    # Tool calling is stochastic, so one miss is not a verdict. A model the
    # account cannot serve will not heal on retry, so do not pay for a second.
    retryable = first.available and not (first.tool_called and first.arguments_valid)
    if retryable:
        second = await _attempt(client, pin, surface, max_tool_rounds=max_tool_rounds)
        if second.tool_called and second.arguments_valid and second.finished:
            chosen = second
    return CompareResult(
        name=chosen.name,
        ref=chosen.ref,
        available=chosen.available,
        tool_called=chosen.tool_called,
        arguments_valid=chosen.arguments_valid,
        finished=chosen.finished,
        error=chosen.error,
        latency_ms=(time.monotonic() - started) * 1000,
    )


async def _attempt(
    client: FireworksClient,
    pin: ModelPin,
    surface: BoundSurface,
    *,
    max_tool_rounds: int,
) -> CompareResult:
    started = time.monotonic()
    messages = [
        Message(
            role="system",
            content="You are a test harness. Call get_time when asked for the time.",
        ),
        Message(role="user", content=_PROMPT),
    ]
    try:
        await run_turn(client, pin, messages, surface, max_tool_rounds=max_tool_rounds)
    except ToolLoopError as exc:
        # The model answered; it just could not drive the loop. That is a score
        # of zero for this model, not an unreachable ref, and the distinction
        # decides whether to fix the manifest or pick a different model.
        return CompareResult(
            name=pin.name,
            ref=pin.ref,
            available=True,
            tool_called=_called_get_time(messages),
            arguments_valid=False,
            finished=False,
            error=type(exc).__name__,
            latency_ms=(time.monotonic() - started) * 1000,
        )
    except AssistAIError as exc:
        return CompareResult(
            name=pin.name,
            ref=pin.ref,
            available=False,
            tool_called=False,
            arguments_valid=False,
            finished=False,
            error=type(exc).__name__,
            latency_ms=(time.monotonic() - started) * 1000,
        )
    tool_called = _called_get_time(messages)
    arguments_valid = False
    for message in messages:
        if message.role == "assistant" and message.tool_calls:
            for call in message.tool_calls:
                if call.name == "get_time":
                    arguments_valid = _is_object(call.arguments)
        if message.role == "tool":
            try:
                parsed = json.loads(message.content or "")
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and "utc" in parsed:
                arguments_valid = True
    finished = any(message.role == "assistant" and not message.tool_calls for message in messages)
    elapsed = (time.monotonic() - started) * 1000
    ok = tool_called and arguments_valid and finished
    log.info(
        "compare.scored",
        model=pin.ref,
        tool_called=tool_called,
        arguments_valid=arguments_valid,
        finished=finished,
        latency_ms=round(elapsed),
    )
    return CompareResult(
        name=pin.name,
        ref=pin.ref,
        available=True,
        tool_called=tool_called,
        arguments_valid=arguments_valid,
        finished=finished,
        error=None if ok else "no_tool_call",
        latency_ms=elapsed,
    )


def _called_get_time(messages: list[Message]) -> bool:
    return any(
        call.name == "get_time"
        for message in messages
        if message.role == "assistant"
        for call in message.tool_calls
    )


def _is_object(arguments: str) -> bool:
    try:
        parsed = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return False
    return isinstance(parsed, dict)
