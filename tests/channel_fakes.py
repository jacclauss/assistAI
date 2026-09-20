"""Shared Signal channel construction for unit tests."""

from __future__ import annotations

from pathlib import Path

from assistai.agents import Household
from assistai.broker import ToolBroker
from assistai.inference.client import FireworksClient
from assistai.signal.channel import SignalChannel
from assistai.signal.policy import AccessPolicy
from assistai.store import Store
from tests.agent_fakes import agent, broker_for, household
from tests.fakes import manifest
from tests.signal_fakes import FakeSignal, signal_settings


def store_for(tmp_path: Path) -> Store:
    return Store(tmp_path / "assistai.sqlite")


def policy_for(
    tmp_path: Path,
    bootstrap: tuple[str, ...] = ("+15555550101",),
    *,
    store: Store | None = None,
) -> AccessPolicy:
    return AccessPolicy(
        bootstrap,
        store=store or store_for(tmp_path),
        pairing_ttl_seconds=60,
    )


def channel_for(
    tmp_path: Path,
    signal: FakeSignal,
    fireworks: FireworksClient | None,
    *,
    jacob_tools: tuple[str, ...] = (),
    home: Household | None = None,
    broker: ToolBroker | None = None,
    store: Store | None = None,
    **overrides: object,
) -> SignalChannel:
    settings = signal_settings(state_dir=tmp_path, **overrides)
    peers = settings.allow_from or ("+15555550101",)
    roster = home or household(
        agent("jacob", peers[0], tools=jacob_tools),
        agent("spouse", peers[1] if len(peers) > 1 else "+15555550102"),
    )
    db = store or store_for(tmp_path)
    return SignalChannel(
        settings,
        signal,
        policy=policy_for(tmp_path, settings.allow_from, store=db),
        fireworks=fireworks,
        manifest=manifest() if fireworks is not None else None,
        household=roster,
        broker=broker or broker_for(roster),
        store=db,
    )
