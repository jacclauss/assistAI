from __future__ import annotations

import json

from tests.fakes import tool_call

from assistai.inference.tools import default_registry


async def test_get_time_returns_utc() -> None:
    registry = default_registry()

    result = json.loads(await registry.execute(tool_call()))

    assert "T" in result["utc"]
    assert result["utc"].endswith("+00:00") or result["utc"].endswith("Z")


async def test_unknown_tool_is_an_error_string() -> None:
    registry = default_registry()

    result = json.loads(await registry.execute(tool_call(name="shell")))

    assert result["error"] == "unknown_tool"
    assert result["name"] == "shell"


async def test_invalid_arguments_are_an_error_string() -> None:
    registry = default_registry()

    result = json.loads(await registry.execute(tool_call(arguments="{nope")))

    assert result["error"] == "invalid_arguments"
