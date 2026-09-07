"""The security boundary. Agents never execute tools; they ask the broker.

A bound surface advertises only the agent's allowlist, refuses anything else
before the handler runs, and taints the conversation when an untrusted tool
returns. Once tainted, configured sinks are refused even if they sit on the
allowlist, and the taint outlives the turn because the untrusted text does.
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Literal

import structlog

from assistai.agents import AgentSpec, BrokerPolicy
from assistai.inference.tools import (
    GET_TIME_SPEC,
    ToolHandler,
    ToolRegistry,
    ToolResult,
    get_time,
)
from assistai.inference.types import Message, ToolCall, ToolSpec

log = structlog.get_logger(__name__)

DenyReason = Literal["unknown", "acl", "web_access", "taint"]

# The Pi runs this for weeks. Keep enough audit history to explain a refusal
# without letting a long uptime grow the process.
AUDIT_LIMIT = 1000


@dataclass(frozen=True)
class ToolMeta:
    """Broker-only attributes. Never sent to the model."""

    trusted: bool = True
    sink: str | None = None
    web: bool = False


@dataclass(frozen=True)
class AuditEvent:
    """One allow or deny. Tests read this; production logs it."""

    agent: str
    tool: str
    action: Literal["allow", "deny"]
    reason: DenyReason | None = None


@dataclass
class Scope:
    """Taint for one conversation, not one turn.

    Untrusted text stays in history after the turn that fetched it, so a scope
    that reset each turn would let the *next* message reach a privileged sink
    with the attacker's instructions still in context.
    """

    tainted: bool = False

    @classmethod
    def from_history(cls, messages: Iterable[Message]) -> Scope:
        return cls(tainted=any(message.untrusted for message in messages))


class ToolCatalog:
    """Every tool the process knows how to run, plus the metadata the broker needs."""

    def __init__(self) -> None:
        self._registry = ToolRegistry()
        self._meta: dict[str, ToolMeta] = {}

    def add(
        self,
        spec: ToolSpec,
        handler: ToolHandler,
        *,
        trusted: bool = True,
        sink: str | None = None,
        web: bool = False,
    ) -> None:
        self._registry.register(spec, handler)
        self._meta[spec.name] = ToolMeta(trusted=trusted, sink=sink, web=web)

    def names(self) -> frozenset[str]:
        return self._registry.names()

    def spec(self, name: str) -> ToolSpec | None:
        return self._registry.spec(name)

    def meta(self, name: str) -> ToolMeta:
        return self._meta.get(name, ToolMeta())

    async def execute(self, call: ToolCall) -> ToolResult:
        return await self._registry.execute(call)


class ToolBroker:
    """Builds a per-agent surface the loop can call."""

    def __init__(self, catalog: ToolCatalog, policy: BrokerPolicy) -> None:
        self._catalog = catalog
        self._policy = policy
        self.audit: deque[AuditEvent] = deque(maxlen=AUDIT_LIMIT)

    def for_agent(self, agent: AgentSpec, scope: Scope | None = None) -> BoundSurface:
        return BoundSurface(
            catalog=self._catalog,
            policy=self._policy,
            agent=agent,
            scope=scope or Scope(),
            record=self.record,
        )

    def record(self, event: AuditEvent) -> None:
        self.audit.append(event)
        if event.action == "deny":
            log.warning(
                "broker.denied",
                agent=event.agent,
                tool=event.tool,
                reason=event.reason,
            )
            return
        log.info("broker.executed", agent=event.agent, tool=event.tool)


class BoundSurface:
    """The loop's view of one agent, scoped to one conversation."""

    def __init__(
        self,
        *,
        catalog: ToolCatalog,
        policy: BrokerPolicy,
        agent: AgentSpec,
        scope: Scope,
        record: Callable[[AuditEvent], None],
    ) -> None:
        self._catalog = catalog
        self._policy = policy
        self._agent = agent
        self._scope = scope
        self._record = record

    def specs(self) -> list[ToolSpec]:
        allowed: list[ToolSpec] = []
        for name in self._agent.tools:
            spec = self._catalog.spec(name)
            if spec is None:
                continue
            meta = self._catalog.meta(name)
            if meta.web and not self._agent.web_access:
                continue
            allowed.append(spec)
        return allowed

    async def execute(self, call: ToolCall) -> ToolResult:
        denied = self._deny_reason(call.name)
        if denied is not None:
            self._record(
                AuditEvent(agent=self._agent.name, tool=call.name, action="deny", reason=denied)
            )
            body = json.dumps({"error": "tool_denied", "name": call.name, "reason": denied})
            return ToolResult(body)
        result = await self._catalog.execute(call)
        meta = self._catalog.meta(call.name)
        if not meta.trusted:
            self._scope.tainted = True
        self._record(AuditEvent(agent=self._agent.name, tool=call.name, action="allow"))
        return ToolResult(result.content, untrusted=not meta.trusted)

    def _deny_reason(self, name: str) -> DenyReason | None:
        if name not in self._catalog.names():
            return "unknown"
        if name not in self._agent.tools:
            return "acl"
        meta = self._catalog.meta(name)
        if meta.web and not self._agent.web_access:
            return "web_access"
        if (
            self._scope.tainted
            and meta.sink is not None
            and meta.sink in self._policy.tainted_sinks_denied
        ):
            return "taint"
        return None


def builtin_catalog() -> ToolCatalog:
    """The tools this build actually implements. Signal ACLs start empty anyway."""
    catalog = ToolCatalog()
    catalog.add(GET_TIME_SPEC, get_time, trusted=True)
    return catalog
