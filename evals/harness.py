"""Golden behaviour cases against a real model, with every tool faked.

The model, system prompt, catalog, and broker are the production ones. Tool
handlers return the case's canned fixtures, so a case can say "the inbox has
7 unread" or "this page carries an injection" and grade what the model did
with it. Nothing reaches Gmail, iCloud, the web, or a phone.

A case grades behaviour, not wording: which tools were called, which were
staged, and a few substrings the reply must or must not contain.
"""

from __future__ import annotations

import json
import math
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from assistai.agents import AgentSpec, Binding, BrokerPolicy, load_household, system_prompt_for
from assistai.broker import Scope, ToolBroker, ToolCatalog, builtin_catalog
from assistai.config import Settings
from assistai.conversation import last_assistant_text
from assistai.errors import AssistAIError
from assistai.inference.client import DeltaCallback, FireworksClient
from assistai.inference.loop import run_turn
from assistai.inference.tools import ToolHandler
from assistai.inference.types import Completion, Message, Role, ToolSpec
from assistai.manifest import ModelPin

CASES_DIR = Path(__file__).resolve().parent / "cases"
_DEFAULT_RESULT = json.dumps({"ok": True})
_HISTORY_ROLES: frozenset[Role] = frozenset({"user", "assistant"})


class CaseError(ValueError):
    """A case file that cannot be graded as written."""


@dataclass(frozen=True)
class Expect:
    calls: tuple[str, ...] = ()
    no_calls: tuple[str, ...] = ()
    staged: tuple[str, ...] = ()
    no_staged: tuple[str, ...] = ()
    reply_contains: tuple[str, ...] = ()
    reply_excludes: tuple[str, ...] = ()
    max_tool_calls: int | None = None


@dataclass(frozen=True)
class Case:
    id: str
    user: str
    tools: tuple[str, ...]
    web_access: bool
    expect: Expect
    fixtures: dict[str, str] = field(default_factory=dict)
    history: tuple[Message, ...] = ()
    tags: tuple[str, ...] = ()
    pass_rate: float = 1.0


@dataclass(frozen=True)
class Trial:
    passed: bool
    failures: tuple[str, ...]
    calls: tuple[str, ...]
    staged: tuple[str, ...]
    reply: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float


@dataclass(frozen=True)
class CaseResult:
    case: Case
    trials: tuple[Trial, ...]

    @property
    def passes(self) -> int:
        return sum(1 for trial in self.trials if trial.passed)

    @property
    def passed(self) -> bool:
        return self.passes >= math.ceil(len(self.trials) * self.case.pass_rate)


class _MeteredClient(FireworksClient):
    """Counts tokens per trial. The loop logs usage but does not return it."""

    prompt_tokens = 0
    completion_tokens = 0

    async def complete(
        self,
        pin: ModelPin,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        on_delta: DeltaCallback | None = None,
    ) -> Completion:
        completion = await super().complete(pin, messages, tools, on_delta=on_delta)
        if completion.usage is not None:
            self.prompt_tokens += completion.usage.prompt_tokens
            self.completion_tokens += completion.usage.completion_tokens
        return completion


def load_cases(
    directory: Path = CASES_DIR, *, roster: Path, catalog: ToolCatalog | None = None
) -> list[Case]:
    """Parse every case file. ``roster`` supplies the default tool list."""
    known = (catalog or builtin_catalog()).names()
    household = load_household(roster, known_tools=known)
    default = household.agents[0]
    cases = [
        _parse_case(path, known=known, default=default) for path in sorted(directory.glob("*.toml"))
    ]
    seen: set[str] = set()
    for case in cases:
        if case.id in seen:
            raise CaseError(f"duplicate case id {case.id}")
        seen.add(case.id)
    return cases


async def run_case(
    client: FireworksClient, pin: ModelPin, case: Case, *, trials: int, max_tool_rounds: int
) -> CaseResult:
    if trials < 1:
        # Zero trials passes every case with nothing run.
        raise CaseError(f"trials must be at least 1, not {trials}")
    results = [
        await _trial(client, pin, case, max_tool_rounds=max_tool_rounds) for _ in range(trials)
    ]
    return CaseResult(case=case, trials=tuple(results))


def metered(settings: Settings) -> FireworksClient:
    return _MeteredClient(settings)


def grade(
    case: Case, *, calls: tuple[str, ...], staged: tuple[str, ...], reply: str
) -> tuple[str, ...]:
    """Every expectation the trial missed. Empty means it passed."""
    expect = case.expect
    failures: list[str] = []
    lowered = reply.lower()
    failures += [f"did not call {name}" for name in expect.calls if name not in calls]
    failures += [f"called {name}" for name in expect.no_calls if name in calls]
    failures += [f"did not stage {name}" for name in expect.staged if name not in staged]
    failures += [f"staged {name}" for name in expect.no_staged if name in staged]
    failures += [
        f"reply lacks {text!r}" for text in expect.reply_contains if text.lower() not in lowered
    ]
    failures += [
        f"reply contains {text!r}" for text in expect.reply_excludes if text.lower() in lowered
    ]
    if expect.max_tool_calls is not None and len(calls) > expect.max_tool_calls:
        failures.append(f"{len(calls)} tool calls > {expect.max_tool_calls}")
    return tuple(failures)


async def _trial(
    client: FireworksClient, pin: ModelPin, case: Case, *, max_tool_rounds: int
) -> Trial:
    agent = AgentSpec(
        name="jacob",
        binding=Binding(channel="local", peer="eval"),
        reads=("own", "shared"),
        writes=("own", "shared"),
        tools=case.tools,
        web_access=case.web_access,
    )
    catalog = builtin_catalog()
    for name in catalog.names():
        catalog.bind(name, _fixture(case.fixtures.get(name, _DEFAULT_RESULT)))
    broker = ToolBroker(catalog, BrokerPolicy.defaults())
    messages = [Message(role="system", content=system_prompt_for(agent)), *case.history]
    surface = broker.for_agent(agent, Scope.from_history(messages))
    messages.append(Message(role="user", content=case.user, created_at=time.time()))
    if isinstance(client, _MeteredClient):
        client.prompt_tokens = client.completion_tokens = 0
    started = time.monotonic()
    error: str | None = None
    try:
        await run_turn(client, pin, messages, surface, max_tool_rounds=max_tool_rounds)
    except AssistAIError as exc:
        error = f"{type(exc).__name__}: {exc}"
    elapsed = (time.monotonic() - started) * 1000
    calls = tuple(
        call.name
        for message in messages
        if message.role == "assistant"
        for call in message.tool_calls
    )
    staged = tuple(call.name for call in surface.take_staged())
    reply = last_assistant_text(messages)
    failures = grade(case, calls=calls, staged=staged, reply=reply)
    if error is not None:
        failures = (error, *failures)
    metered_client = client if isinstance(client, _MeteredClient) else None
    return Trial(
        passed=not failures,
        failures=failures,
        calls=calls,
        staged=staged,
        reply=reply,
        prompt_tokens=metered_client.prompt_tokens if metered_client else 0,
        completion_tokens=metered_client.completion_tokens if metered_client else 0,
        latency_ms=elapsed,
    )


def _fixture(content: str) -> ToolHandler:
    async def handler(_arguments: dict[str, Any]) -> str:
        return content

    return handler


def _parse_case(path: Path, *, known: frozenset[str], default: AgentSpec) -> Case:
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise CaseError(f"{path.name}: {exc}") from exc
    case_id = raw.get("id")
    if case_id != path.stem:
        raise CaseError(f"{path.name}: id must equal the file name ({path.stem})")
    user = raw.get("user")
    if not isinstance(user, str) or not user.strip():
        raise CaseError(f"{path.name}: user must be a non-empty string")
    tools = tuple(raw.get("tools", default.tools))
    expect_raw = raw.get("expect", {})
    expect = Expect(
        calls=tuple(expect_raw.get("calls", ())),
        no_calls=tuple(expect_raw.get("no_calls", ())),
        staged=tuple(expect_raw.get("staged", ())),
        no_staged=tuple(expect_raw.get("no_staged", ())),
        reply_contains=tuple(expect_raw.get("reply_contains", ())),
        reply_excludes=tuple(expect_raw.get("reply_excludes", ())),
        max_tool_calls=expect_raw.get("max_tool_calls"),
    )
    fixtures = {str(name): str(value) for name, value in raw.get("fixtures", {}).items()}
    named = {
        *tools,
        *expect.calls,
        *expect.no_calls,
        *expect.staged,
        *expect.no_staged,
        *fixtures,
    }
    unknown = sorted(name for name in named if name not in known)
    if unknown:
        raise CaseError(f"{path.name}: unknown tools {unknown}")
    history: list[Message] = []
    for item in raw.get("history", ()):
        role = item.get("role")
        if role not in _HISTORY_ROLES:
            raise CaseError(f"{path.name}: history role must be user or assistant")
        history.append(
            Message(
                role=role,
                content=str(item.get("content", "")),
                untrusted=bool(item.get("untrusted", False)),
                created_at=time.time(),
            )
        )
    pass_rate = float(raw.get("pass_rate", 1.0))
    if not 0 < pass_rate <= 1:
        raise CaseError(f"{path.name}: pass_rate must be in (0, 1]")
    return Case(
        id=case_id,
        user=user,
        tools=tools,
        web_access=bool(raw.get("web_access", default.web_access)),
        expect=expect,
        fixtures=fixtures,
        history=tuple(history),
        tags=tuple(raw.get("tags", ())),
        pass_rate=pass_rate,
    )
