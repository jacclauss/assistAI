from __future__ import annotations

from pathlib import Path

import pytest

from assistai.agents import BrokerPolicy, load_household, local_agent, resolve_household_path
from assistai.broker import builtin_catalog
from assistai.config import Settings
from assistai.errors import HouseholdConfigError
from tests.agent_fakes import agent as household_agent
from tests.agent_fakes import household as make_household


def _write(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def test_example_roster_loads(repo_root: Path) -> None:
    household = load_household(
        repo_root / "config" / "assistai.example.toml",
        known_tools=builtin_catalog().names(),
    )

    assert {agent.name for agent in household.agents} == {"jacob", "spouse"}
    jacob = household.agent_for_signal_dm("+15555550101")
    spouse = household.agent_for_signal_dm("+15555550102")
    assert jacob is not None and spouse is not None
    assert jacob.name == "jacob"
    assert spouse.name == "spouse"
    assert household.agent_for_signal_dm("+15555550199") is None
    assert jacob.tools == (
        "get_time",
        "relay",
        "job_create",
        "job_list",
        "job_cancel",
        "job_reschedule",
        "web_search",
        "web_fetch",
        "calendar_today",
        "calendar_add",
    )
    assert spouse.tools == (
        "get_time",
        "relay",
        "job_create",
        "job_list",
        "job_cancel",
        "job_reschedule",
        "web_search",
        "web_fetch",
        "calendar_today",
        "calendar_add",
    )
    assert jacob.writes == ()
    assert spouse.writes == ()
    assert jacob.web_access is True
    assert spouse.web_access is True
    assert all(agent.binding.is_signal_dm for agent in household.agents)


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
            '[agents.jacob]\nbinds_to = { channel = "signal", peer = "group:household" }\n',
            "must bind to a Signal DM",
            id="group-binding",
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

    assert household.broker == BrokerPolicy.defaults()


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


def test_local_agent_is_not_reachable_over_signal() -> None:
    """Stuffing the REPL spec into a Household still cannot bind a phone number."""
    repl = local_agent("repl")
    home = make_household(
        household_agent("jacob", "+15555550101"),
        household_agent("spouse", "+15555550102"),
        repl,
    )

    assert repl.binding.is_signal_dm is False
    assert repl.web_access is False
    assert repl.tools == ("get_time",)
    assert home.agent_for_signal_dm(repl.binding.peer) is None
    assert repl.binding.peer not in home.dm_peers()


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
