"""Household and broker doubles for tests."""

from __future__ import annotations

from assistai.agents import AgentSpec, Binding, BrokerPolicy, Household
from assistai.broker import ToolBroker, builtin_catalog

_DEFAULT_SINKS = frozenset({"shared:write", "shared:publish", "message:other_peer"})


def agent(
    name: str,
    peer: str,
    *,
    tools: tuple[str, ...] = (),
    web_access: bool = False,
    reads: tuple[str, ...] = ("own", "shared"),
    writes: tuple[str, ...] = ("own", "shared:publish"),
) -> AgentSpec:
    return AgentSpec(
        name=name,
        binding=Binding(channel="signal", peer=peer),
        reads=reads,
        writes=writes,
        tools=tools,
        web_access=web_access,
    )


def household(*agents: AgentSpec, sinks: frozenset[str] | None = None) -> Household:
    roster = agents or (
        agent("jacob", "+15555550101"),
        agent("spouse", "+15555550102"),
    )
    return Household(
        agents=roster, broker=BrokerPolicy(tainted_sinks_denied=sinks or _DEFAULT_SINKS)
    )


def broker_for(home: Household | None = None) -> ToolBroker:
    return ToolBroker(builtin_catalog(), (home or household()).broker)
