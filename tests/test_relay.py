from __future__ import annotations

import json
from pathlib import Path

import pytest

from assistai.agents import AgentSpec
from assistai.errors import StoreError
from assistai.inference.types import Message
from assistai.relay import (
    RelayError,
    bind_relay,
    format_outbound,
    format_record,
    parse_body,
    using_agent,
)
from assistai.signal.client import MAX_SEND_CHARS
from assistai.store import Store
from tests.agent_fakes import agent, household
from tests.signal_fakes import FakeSignal


def test_outbound_is_labelled_as_a_relay() -> None:
    assert format_outbound("jacob", "pick up milk") == "From Jacob:\npick up milk"


def test_history_record_is_structured() -> None:
    record = json.loads(format_record("jacob", "pick up milk"))
    assert record["kind"] == "relay"
    assert record["from"] == "jacob"
    assert record["body"] == "pick up milk"
    assert record["attachments"] == []


def test_empty_body_is_rejected() -> None:
    with pytest.raises(RelayError):
        parse_body({"body": "   "})


def test_labelled_body_fits_in_one_signal_message() -> None:
    """A max-length body plus the From line must not split across two sends."""
    prefix = format_outbound("jacob", "")
    body = "x" * (MAX_SEND_CHARS - len(prefix))
    labelled = format_outbound("jacob", parse_body({"body": body}, from_name="jacob"))
    assert len(labelled) == MAX_SEND_CHARS
    with pytest.raises(RelayError):
        parse_body({"body": body + "y"}, from_name="jacob")


async def test_handler_sends_then_injects(tmp_path: Path) -> None:
    store = Store(tmp_path / "assistai.sqlite")
    signal = FakeSignal()
    jacob = agent("jacob", "+15555550101", tools=("relay",))
    spouse = agent("spouse", "+15555550102", tools=("relay",))
    home = household(jacob, spouse)
    injected: list[tuple[str, str]] = []

    async def inject(recipient: AgentSpec, record: str) -> None:
        injected.append((recipient.name, record))
        messages = store.load_history(recipient.name)
        messages.append(Message(role="user", content=record, untrusted=True, created_at=1.0))
        store.save_history(recipient.name, messages)

    handler = bind_relay(household=home, signal=signal, inject=inject)
    with using_agent(jacob):
        result = json.loads(await handler({"body": "pick up milk"}))

    assert result["ok"] is True
    assert signal.sent == [("+15555550102", "From Jacob:\npick up milk")]
    assert injected[0][0] == "spouse"
    loaded = store.load_history("spouse")
    assert loaded[-1].untrusted is True
    assert "pick up milk" in (loaded[-1].content or "")
    store.close()


async def test_failed_inject_still_delivered(
    tmp_path: Path,
) -> None:
    signal = FakeSignal()
    jacob = agent("jacob", "+15555550101", tools=("relay",))
    home = household(jacob, agent("spouse", "+15555550102", tools=("relay",)))

    async def boom(_recipient: AgentSpec, _record: str) -> None:
        raise StoreError("unreadable")

    handler = bind_relay(household=home, signal=signal, inject=boom)
    with using_agent(jacob), pytest.raises(RelayError, match="could not be recorded"):
        await handler({"body": "pick up milk"})

    assert signal.sent == [("+15555550102", "From Jacob:\npick up milk")]


async def test_cannot_relay_without_another_household_member() -> None:
    jacob = agent("jacob", "+15555550101", tools=("relay",))
    handler = bind_relay(household=household(jacob), signal=FakeSignal(), inject=_noop_inject)
    with using_agent(jacob), pytest.raises(RelayError, match="no other household member"):
        await handler({"body": "pick up milk"})


async def _noop_inject(_recipient: AgentSpec, _record: str) -> None:
    raise AssertionError("must not inject")
