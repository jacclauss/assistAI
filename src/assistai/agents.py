"""Household agents: identity, Signal bindings, and tool ACLs.

Loaded from ``config/assistai.toml``. Being allowed to message the bot is not
the same as having an agent: the allowlist admits a number, this file decides
who they become. Unbound numbers never reach a model. Every agent binds to a
Signal DM; an organizer process is not a Signal identity.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

from assistai.config import Settings
from assistai.errors import HouseholdConfigError
from assistai.signal.numbers import InvalidNumberError, normalize_e164

_AGENT_NAME = re.compile(r"^[a-z][a-z0-9_]{0,31}$")

_DEFAULT_TAINT_SINKS = frozenset({"shared:write", "shared:publish", "message:other_peer"})


@dataclass(frozen=True)
class Binding:
    """Where this agent is reachable.

    Household agents are Signal DMs. The REPL and bake-off use ``local``, which
    ``is_signal_dm`` rejects, so stuffing that spec into a Household cannot
    make a phone number reach it.
    """

    channel: str
    peer: str

    @property
    def is_signal_dm(self) -> bool:
        return self.channel == "signal" and not self.peer.startswith("group:")


@dataclass(frozen=True)
class AgentSpec:
    """One named agent with an allowlist of tools it may even see."""

    name: str
    binding: Binding
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    tools: tuple[str, ...]
    web_access: bool


@dataclass(frozen=True)
class BrokerPolicy:
    """Sinks the broker refuses once a turn has ingested untrusted content."""

    tainted_sinks_denied: frozenset[str]

    @classmethod
    def defaults(cls) -> BrokerPolicy:
        return cls(tainted_sinks_denied=_DEFAULT_TAINT_SINKS)


@dataclass(frozen=True)
class Household:
    """The complete agent roster and broker policy from disk."""

    agents: tuple[AgentSpec, ...]
    broker: BrokerPolicy

    def agent_for_signal_dm(self, number: str) -> AgentSpec | None:
        for agent in self.agents:
            if agent.binding.is_signal_dm and agent.binding.peer == number:
                return agent
        return None

    def dm_peers(self) -> tuple[str, ...]:
        return tuple(agent.binding.peer for agent in self.agents if agent.binding.is_signal_dm)


def local_agent(name: str, *, tools: tuple[str, ...] = ("get_time",)) -> AgentSpec:
    """Not reachable over Signal. Used by the terminal REPL and the bake-off."""
    return AgentSpec(
        name=name,
        binding=Binding(channel="local", peer=name),
        reads=(),
        writes=(),
        tools=tools,
        web_access=False,
    )


def system_prompt_for(agent: AgentSpec) -> str:
    """Identity the model sees. Distinct per agent so routing is observable."""
    return (
        f"You are AssistAI agent '{agent.name}', a concise household assistant. "
        "Use only the tools listed in this request. Do not invent tool results. "
        "When you use the web, cite the URLs you relied on."
    )


def resolve_household_path(settings: Settings) -> Path:
    """Find ``config/assistai.toml`` the same way the manifest is found."""
    if settings.agents_config_path is not None:
        return settings.agents_config_path
    cwd = Path.cwd() / "config" / "assistai.toml"
    if cwd.is_file():
        return cwd
    container = Path("/app/config/assistai.toml")
    if container.is_file():
        return container
    raise HouseholdConfigError(
        "config/assistai.toml not found; copy config/assistai.example.toml and set the numbers"
    )


def load_household(path: Path, *, known_tools: frozenset[str] | None = None) -> Household:
    """Parse and validate the agent roster in ``path``."""
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise HouseholdConfigError(f"agent config not found: {path}") from exc
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise HouseholdConfigError(f"agent config unreadable: {path}") from exc

    agents_raw = raw.get("agents")
    if not isinstance(agents_raw, dict) or not agents_raw:
        raise HouseholdConfigError("config is missing an [agents] table")

    agents: list[AgentSpec] = []
    seen_names: set[str] = set()
    seen_bindings: set[tuple[str, str]] = set()
    for name, block in agents_raw.items():
        agent = _parse_agent(name, block, known_tools=known_tools)
        if agent.name in seen_names:
            raise HouseholdConfigError(f"duplicate agent name: {agent.name}")
        key = (agent.binding.channel, agent.binding.peer)
        if key in seen_bindings:
            raise HouseholdConfigError(
                f"agents {agent.name} and another share binding {agent.binding.peer}"
            )
        seen_names.add(agent.name)
        seen_bindings.add(key)
        agents.append(agent)

    dms = [agent for agent in agents if agent.binding.is_signal_dm]
    if len(dms) < 2:
        raise HouseholdConfigError(
            "need at least two Signal DM agents so two numbers reach two identities"
        )

    broker_raw = raw.get("broker")
    policy = _parse_broker(broker_raw)
    return Household(agents=tuple(agents), broker=policy)


def _parse_agent(name: object, block: object, *, known_tools: frozenset[str] | None) -> AgentSpec:
    if not isinstance(name, str) or not _AGENT_NAME.fullmatch(name):
        raise HouseholdConfigError(f"invalid agent name: {name!r}")
    if not isinstance(block, dict):
        raise HouseholdConfigError(f"agents.{name} must be a table")
    binding = _parse_binding(name, block.get("binds_to"))
    reads = _string_tuple(block.get("reads"), field=f"agents.{name}.reads")
    writes = _string_tuple(block.get("writes"), field=f"agents.{name}.writes")
    tools = _dedupe(_string_tuple(block.get("tools"), field=f"agents.{name}.tools"))
    if known_tools is not None:
        unknown = [tool for tool in tools if tool not in known_tools]
        if unknown:
            raise HouseholdConfigError(
                f"agents.{name}.tools names unknown tools: {', '.join(unknown)}"
            )
    # Deny by default: an agent added without thinking about the web does not
    # get it. Forgetting this line must not hand out a browser.
    web_access = block.get("web_access", False)
    if not isinstance(web_access, bool):
        raise HouseholdConfigError(f"agents.{name}.web_access must be a boolean")
    return AgentSpec(
        name=name,
        binding=binding,
        reads=reads,
        writes=writes,
        tools=tools,
        web_access=web_access,
    )


def _parse_binding(name: str, raw: object) -> Binding:
    if not isinstance(raw, dict):
        raise HouseholdConfigError(f"agents.{name}.binds_to must be a table")
    channel = raw.get("channel")
    peer = raw.get("peer")
    if channel != "signal":
        raise HouseholdConfigError(f"agents.{name}.binds_to.channel must be 'signal'")
    if not isinstance(peer, str) or not peer.strip():
        raise HouseholdConfigError(f"agents.{name}.binds_to.peer must be a non-empty string")
    peer = peer.strip()
    if peer.startswith("group:"):
        raise HouseholdConfigError(
            f"agents.{name} must bind to a Signal DM; group routing is not built"
        )
    try:
        return Binding(channel="signal", peer=normalize_e164(peer))
    except InvalidNumberError as exc:
        raise HouseholdConfigError(
            f"agents.{name}.binds_to.peer is not an E.164 number: {peer}"
        ) from exc


def _parse_broker(raw: object) -> BrokerPolicy:
    if raw is None:
        return BrokerPolicy.defaults()
    if not isinstance(raw, dict):
        raise HouseholdConfigError("broker must be a table")
    sinks = raw.get("tainted_sinks_denied")
    if sinks is None:
        return BrokerPolicy.defaults()
    if not isinstance(sinks, list) or not all(isinstance(item, str) and item for item in sinks):
        raise HouseholdConfigError("broker.tainted_sinks_denied must be a list of strings")
    return BrokerPolicy(tainted_sinks_denied=frozenset(sinks))


def _string_tuple(raw: object, *, field: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise HouseholdConfigError(f"{field} must be a list of strings")
    return tuple(raw)


def _dedupe(values: tuple[str, ...]) -> tuple[str, ...]:
    """Preserve order; a repeated tool must not be advertised twice."""
    return tuple(dict.fromkeys(values))
