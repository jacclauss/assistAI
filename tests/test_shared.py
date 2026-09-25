"""Shared lists: ACL, one writer, a yes applies the stored change."""

from __future__ import annotations

from pathlib import Path

import pytest

from assistai.agents import load_household
from assistai.broker import ToolBroker, builtin_catalog
from assistai.errors import HouseholdConfigError, SharedError
from assistai.relay import using_agent
from assistai.shared.organizer import Organizer
from assistai.shared.records import parse_change
from assistai.shared.tools import SHARED_CHANGE, SHARED_LISTS, bind_shared
from assistai.staging import format_proposal
from assistai.store import Store
from tests.agent_fakes import agent
from tests.fakes import tool_call


def _organizer(tmp_path: Path) -> tuple[Organizer, Store]:
    store = Store(tmp_path / "assistai.sqlite")
    return Organizer(store), store


def test_a_list_is_shared_after_one_confirmed_change(tmp_path: Path) -> None:
    organizer, store = _organizer(tmp_path)
    jacob = agent("jacob", "+15555550101", writes=("shared",), reads=("shared",))
    spouse = agent("spouse", "+15555550102", writes=(), reads=("shared",))
    organizer.apply(jacob, parse_change({"list": "shopping", "add": ["milk", "eggs"]}))
    text = organizer.read(spouse, "shopping")
    assert "milk" in text
    assert "eggs" in text
    assert "[open]" in text
    store.close()


def test_a_reader_cannot_change_a_list(tmp_path: Path) -> None:
    organizer, store = _organizer(tmp_path)
    spouse = agent("spouse", "+15555550102", writes=(), reads=("shared",))
    with pytest.raises(SharedError, match="cannot change"):
        organizer.apply(spouse, parse_change({"list": "shopping", "add": ["milk"]}))
    assert organizer.read(spouse) == "No lists yet."
    store.close()


def test_an_agent_without_the_grant_cannot_read(tmp_path: Path) -> None:
    organizer, store = _organizer(tmp_path)
    stranger = agent("spouse", "+15555550102", reads=(), writes=())
    with pytest.raises(SharedError, match="cannot read"):
        organizer.read(stranger)
    store.close()


def test_a_private_list_is_invisible_to_the_other_person(tmp_path: Path) -> None:
    organizer, store = _organizer(tmp_path)
    jacob = agent("jacob", "+15555550101", writes=("own", "shared"), reads=("own", "shared"))
    spouse = agent("spouse", "+15555550102", writes=("own", "shared"), reads=("own", "shared"))
    organizer.apply(
        jacob, parse_change({"list": "shopping", "add": ["milk"], "private": True})
    )
    organizer.apply(jacob, parse_change({"list": "shopping", "add": ["bread"]}))
    private = organizer.read(jacob, "shopping", private=True)
    assert "milk" in private
    assert "bread" not in private
    shared = organizer.read(spouse, "shopping")
    assert "bread" in shared
    assert "milk" not in shared
    with pytest.raises(SharedError, match="there is no list named gifts"):
        organizer.read(spouse, "gifts", private=True)
    with pytest.raises(SharedError, match="there is no list named shopping"):
        organizer.read(spouse, "shopping", private=True)
    visible = organizer.read(spouse)
    assert "milk" not in visible
    assert "(private)" not in visible
    store.close()


def test_sharing_a_private_list_is_its_own_change(tmp_path: Path) -> None:
    organizer, store = _organizer(tmp_path)
    jacob = agent("jacob", "+15555550101", writes=("own", "shared"), reads=("own", "shared"))
    spouse = agent("spouse", "+15555550102", writes=("own",), reads=("own", "shared"))
    organizer.apply(jacob, parse_change({"list": "gifts", "add": ["book"], "private": True}))
    with pytest.raises(SharedError, match="there is no list named gifts"):
        organizer.read(spouse, "gifts")
    organizer.apply(jacob, parse_change({"list": "gifts", "share": True}))
    assert "book" in organizer.read(spouse, "gifts")
    store.close()


def test_checking_an_item_off_does_not_partially_apply(tmp_path: Path) -> None:
    organizer, store = _organizer(tmp_path)
    jacob = agent("jacob", "+15555550101", writes=("shared",), reads=("shared",))
    organizer.apply(jacob, parse_change({"list": "shopping", "add": ["milk"]}))
    with pytest.raises(SharedError, match="eggs"):
        organizer.apply(
            jacob,
            parse_change({"list": "shopping", "done": ["milk"], "remove": ["eggs"]}),
        )
    text = organizer.read(jacob, "shopping")
    assert "[open] milk" in text
    store.close()


def test_shared_change_stages_and_jobs_cannot_see_it() -> None:
    import asyncio

    jacob = agent(
        "jacob",
        "+15555550101",
        tools=(SHARED_LISTS, SHARED_CHANGE),
        writes=("shared",),
    )
    from tests.agent_fakes import household

    catalog = builtin_catalog()
    home = household(jacob, agent("spouse", "+15555550102"))
    surface = ToolBroker(catalog, home.broker).for_agent(jacob)
    with using_agent(jacob):
        staged = asyncio.run(
            surface.execute(
                tool_call(SHARED_CHANGE, '{"list": "shopping", "add": ["milk"]}')
            )
        )
    assert "staged" in staged.content
    assert "shopping" in format_proposal(surface.take_staged(), tainted=True)
    report = ToolBroker(catalog, home.broker).for_agent(jacob, report_only=True)
    assert SHARED_CHANGE not in {spec.name for spec in report.specs()}
    assert SHARED_LISTS in {spec.name for spec in report.specs()}


def test_organizer_is_not_a_signal_agent(tmp_path: Path) -> None:
    path = tmp_path / "assistai.toml"
    path.write_text(
        """
[agents.organizer]
binds_to = { channel = "signal", peer = "+15555550101" }
tools = ["shared_lists"]

[agents.spouse]
binds_to = { channel = "signal", peer = "+15555550102" }
tools = ["get_time"]
""",
        encoding="utf-8",
    )
    with pytest.raises(HouseholdConfigError, match="organizer"):
        load_household(path, known_tools=builtin_catalog().names())


def test_the_tool_uses_the_active_agent(tmp_path: Path) -> None:
    organizer, store = _organizer(tmp_path)
    jacob = agent("jacob", "+15555550101", writes=("shared",), reads=("shared",))
    handlers = bind_shared(organizer)
    with using_agent(jacob):
        import asyncio

        asyncio.run(handlers[SHARED_CHANGE]({"list": "shopping", "add": ["milk"]}))
        text = asyncio.run(handlers[SHARED_LISTS]({"list": "shopping"}))
    assert "milk" in text
    store.close()

