from __future__ import annotations

from pathlib import Path

import pytest

from assistai.agents import load_household, resolve_household_path
from assistai.broker import builtin_catalog
from assistai.config import Settings
from assistai.errors import HouseholdConfigError


def _write(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def test_example_roster_loads(repo_root: Path) -> None:
    household = load_household(
        repo_root / "config" / "assistai.example.toml",
        known_tools=builtin_catalog().names(),
    )

    assert {agent.name for agent in household.agents} == {"jacob", "spouse", "organizer"}
    jacob = household.agent_for_signal_dm("+15555550101")
    spouse = household.agent_for_signal_dm("+15555550102")
    assert jacob is not None and spouse is not None
    assert jacob.name == "jacob"
    assert spouse.name == "spouse"
    assert household.agent_for_signal_dm("+15555550199") is None
    assert jacob.tools == ()
    assert spouse.tools == ()
    organizer = next(agent for agent in household.agents if agent.name == "organizer")
    assert organizer.web_access is False
    assert organizer.binding.peer == "group:household"
    assert jacob.web_access is True
    assert spouse.web_access is True


def test_web_access_is_off_unless_asked_for(tmp_path: Path) -> None:
    """Forgetting the key must not hand an agent the internet."""
    path = _write(
        tmp_path / "web.toml",
        """
[agents.jacob]
binds_to = { channel = "signal", peer = "+15555550101" }
[agents.spouse]
binds_to = { channel = "signal", peer = "+15555550102" }
web_access = true
""",
    )

    household = load_household(path)

    jacob = household.agent_for_signal_dm("+15555550101")
    spouse = household.agent_for_signal_dm("+15555550102")
    assert jacob is not None and spouse is not None
    assert jacob.web_access is False
    assert spouse.web_access is True


def test_repeated_tools_are_advertised_once(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "dupe-tools.toml",
        """
[agents.jacob]
binds_to = { channel = "signal", peer = "+15555550101" }
tools = ["get_time", "get_time"]
[agents.spouse]
binds_to = { channel = "signal", peer = "+15555550102" }
tools = []
""",
    )

    household = load_household(path, known_tools=builtin_catalog().names())

    jacob = household.agent_for_signal_dm("+15555550101")
    assert jacob is not None
    assert jacob.tools == ("get_time",)


def test_two_dm_agents_required(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "one.toml",
        """
[agents.jacob]
binds_to = { channel = "signal", peer = "+15555550101" }
tools = []
""",
    )

    with pytest.raises(HouseholdConfigError, match="at least two"):
        load_household(path)


def test_duplicate_binding_rejected(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "dup.toml",
        """
[agents.jacob]
binds_to = { channel = "signal", peer = "+15555550101" }
[agents.spouse]
binds_to = { channel = "signal", peer = "+15555550101" }
""",
    )

    with pytest.raises(HouseholdConfigError, match="share binding"):
        load_household(path)


def test_unknown_tool_rejected(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "tools.toml",
        """
[agents.jacob]
binds_to = { channel = "signal", peer = "+15555550101" }
tools = ["shell"]
[agents.spouse]
binds_to = { channel = "signal", peer = "+15555550102" }
tools = []
""",
    )

    with pytest.raises(HouseholdConfigError, match="unknown tools"):
        load_household(path, known_tools=builtin_catalog().names())


def test_invalid_peer_rejected(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "peer.toml",
        """
[agents.jacob]
binds_to = { channel = "signal", peer = "not-a-number" }
[agents.spouse]
binds_to = { channel = "signal", peer = "+15555550102" }
""",
    )

    with pytest.raises(HouseholdConfigError, match=r"E\.164"):
        load_household(path)


_TWO_GOOD_AGENTS = """
[agents.jacob]
binds_to = { channel = "signal", peer = "+15555550101" }
[agents.spouse]
binds_to = { channel = "signal", peer = "+15555550102" }
"""


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        pytest.param("", "missing an \\[agents\\] table", id="no-agents-table"),
        pytest.param("[agents]\n", "missing an \\[agents\\] table", id="empty-agents-table"),
        pytest.param('[agents]\njacob = "nope"\n', "must be a table", id="agent-not-a-table"),
        pytest.param(
            '[agents."Jacob Smith"]\nbinds_to = { channel = "signal", peer = "+15555550101" }\n',
            "invalid agent name",
            id="bad-agent-name",
        ),
        pytest.param(
            "[agents.jacob]\nbinds_to = 5\n",
            "binds_to must be a table",
            id="binding-not-a-table",
        ),
        pytest.param(
            '[agents.jacob]\nbinds_to = { channel = "sms", peer = "+15555550101" }\n',
            "channel must be 'signal'",
            id="wrong-channel",
        ),
        pytest.param(
            '[agents.jacob]\nbinds_to = { channel = "signal", peer = "  " }\n',
            "peer must be a non-empty string",
            id="blank-peer",
        ),
        pytest.param(
            '[agents.jacob]\nbinds_to = { channel = "signal", peer = "group:Household!" }\n',
            "not a group id",
            id="bad-group-id",
        ),
        pytest.param(
            _TWO_GOOD_AGENTS + 'reads = "own"\n',
            "reads must be a list of strings",
            id="reads-not-a-list",
        ),
        pytest.param(
            _TWO_GOOD_AGENTS + "web_access = 1\n",
            "web_access must be a boolean",
            id="web-access-not-a-bool",
        ),
        pytest.param(
            _TWO_GOOD_AGENTS + "[broker]\ntainted_sinks_denied = 3\n",
            "must be a list of strings",
            id="sinks-not-a-list",
        ),
        pytest.param(
            _TWO_GOOD_AGENTS + '[broker]\ntainted_sinks_denied = ["ok", ""]\n',
            "must be a list of strings",
            id="sinks-with-a-blank",
        ),
    ],
)
def test_malformed_config_is_rejected_with_a_useful_message(
    tmp_path: Path, body: str, expected: str
) -> None:
    """A bad roster must fail at load, not silently grant or drop authority."""
    path = _write(tmp_path / "bad.toml", body)

    with pytest.raises(HouseholdConfigError, match=expected):
        load_household(path)


def test_unreadable_config_is_reported(tmp_path: Path) -> None:
    with pytest.raises(HouseholdConfigError, match="not found"):
        load_household(tmp_path / "missing.toml")


def test_invalid_toml_is_reported(tmp_path: Path) -> None:
    path = _write(tmp_path / "broken.toml", "[agents\n")

    with pytest.raises(HouseholdConfigError, match="unreadable"):
        load_household(path)


def test_broker_defaults_deny_the_standard_sinks(tmp_path: Path) -> None:
    """Omitting [broker] must not mean 'nothing is denied'."""
    path = _write(tmp_path / "nobroker.toml", _TWO_GOOD_AGENTS)

    household = load_household(path)

    assert household.broker.tainted_sinks_denied == frozenset(
        {"shared:write", "shared:publish", "message:other_peer"}
    )


def test_a_group_binding_is_not_a_dm(tmp_path: Path) -> None:
    """The organizer must not be reachable by texting the bot directly."""
    path = _write(
        tmp_path / "group.toml",
        _TWO_GOOD_AGENTS
        + """
[agents.organizer]
binds_to = { channel = "signal", peer = "group:household" }
""",
    )

    household = load_household(path)

    organizer = next(agent for agent in household.agents if agent.name == "organizer")
    assert organizer.binding.is_signal_dm is False
    assert household.dm_peers() == ("+15555550101", "+15555550102")


def test_peers_are_normalized_before_matching(tmp_path: Path) -> None:
    """A config written with dashes must still match what signal-cli sends."""
    path = _write(
        tmp_path / "loose.toml",
        """
[agents.jacob]
binds_to = { channel = "signal", peer = "+1 (555) 555-0101" }
[agents.spouse]
binds_to = { channel = "signal", peer = "+15555550102" }
""",
    )

    household = load_household(path)

    assert household.agent_for_signal_dm("+15555550101") is not None


def test_explicit_path_wins(tmp_path: Path) -> None:
    chosen = tmp_path / "elsewhere.toml"

    resolved = resolve_household_path(Settings(agents_config_path=chosen))

    assert resolved == chosen


def test_config_is_found_next_to_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "config").mkdir()
    expected = tmp_path / "config" / "assistai.toml"
    expected.write_text(_TWO_GOOD_AGENTS, encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert resolve_household_path(Settings()) == expected


def test_missing_config_names_the_example_to_copy() -> None:
    """The error is the only instruction an operator gets at this point."""
    with pytest.raises(HouseholdConfigError, match=r"assistai\.example\.toml"):
        resolve_household_path(Settings())


def test_get_time_is_allowed_on_an_acl(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "clock.toml",
        """
[agents.jacob]
binds_to = { channel = "signal", peer = "+15555550101" }
tools = ["get_time"]
[agents.spouse]
binds_to = { channel = "signal", peer = "+15555550102" }
tools = []
""",
    )

    household = load_household(path, known_tools=builtin_catalog().names())

    jacob = household.agent_for_signal_dm("+15555550101")
    spouse = household.agent_for_signal_dm("+15555550102")
    assert jacob is not None
    assert spouse is not None
    assert jacob.tools == ("get_time",)
    assert spouse.tools == ()
