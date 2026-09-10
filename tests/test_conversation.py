from __future__ import annotations

from assistai.conversation import apply_window, last_assistant_text, safe_trim
from assistai.inference.types import Message, ToolCall


def _tool_exchange(call_id: str = "c1") -> list[Message]:
    return [
        Message(role="user", content="what time is it"),
        Message(
            role="assistant",
            tool_calls=[ToolCall(id=call_id, name="get_time", arguments="{}")],
        ),
        Message(role="tool", content='{"utc":"..."}', tool_call_id=call_id),
        Message(role="assistant", content="noon"),
    ]


def _valid(messages: list[Message]) -> None:
    """A tool message must follow its assistant tool_calls, and vice versa."""
    for i, message in enumerate(messages):
        if message.role == "tool":
            prev = messages[i - 1] if i > 0 else None
            assert prev is not None, f"orphan tool message at {i}"
            assert prev.role in {"assistant", "tool"}, f"orphan tool message at {i}"
            if prev.role == "assistant":
                assert prev.tool_calls, f"tool message at {i} follows a plain assistant"
        if message.role == "assistant" and message.tool_calls:
            nxt = messages[i + 1] if i + 1 < len(messages) else None
            assert nxt is not None and nxt.role == "tool", f"dangling tool call at {i}"


def test_short_history_untouched() -> None:
    messages = [Message(role="system", content="sys"), Message(role="user", content="hi")]

    safe_trim(messages, 30)

    assert len(messages) == 2


def test_system_prompt_is_always_kept() -> None:
    messages = [Message(role="system", content="sys")]
    messages += [Message(role="user", content=f"m{i}") for i in range(100)]

    safe_trim(messages, 10)

    assert messages[0].role == "system"
    assert len(messages) <= 11


def test_trim_never_orphans_a_tool_message() -> None:
    """Sweep the cut across a tool exchange; every position must stay valid.

    A cut landing between an assistant tool_call and its result used to leave a
    tool message the provider rejects.
    """
    for tail in range(0, 40):
        messages = [Message(role="system", content="sys")]
        messages += [Message(role="user", content=f"m{i}") for i in range(40)]
        messages += _tool_exchange()
        messages += [Message(role="user", content=f"t{i}") for i in range(tail)]

        safe_trim(messages, 30)

        assert messages[0].role == "system"
        _valid(messages)


def test_last_assistant_text_skips_tool_requests() -> None:
    messages = [
        Message(role="user", content="hi"),
        Message(role="assistant", tool_calls=[ToolCall(id="c1", name="get_time", arguments="{}")]),
        Message(role="tool", content="{}", tool_call_id="c1"),
        Message(role="assistant", content="final"),
    ]

    assert last_assistant_text(messages) == "final"


def test_last_assistant_text_when_absent() -> None:
    assert last_assistant_text([Message(role="user", content="hi")]) == ""


def test_age_trim_drops_old_untrusted_messages() -> None:
    """Taint clears when the labelled messages fall out of the window."""
    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="old fetch", created_at=1.0, untrusted=True),
        Message(role="assistant", content="poisoned", created_at=2.0, untrusted=True),
        Message(role="user", content="new", created_at=100.0),
        Message(role="assistant", content="fresh", created_at=101.0),
    ]

    apply_window(messages, keep=30, max_age_seconds=10, now=110.0)

    assert [message.content for message in messages] == ["sys", "new", "fresh"]
    assert not any(message.untrusted for message in messages)


def test_age_trim_does_not_split_a_tool_exchange() -> None:
    messages = [Message(role="system", content="sys")]
    messages += [
        Message(role="user", content="old", created_at=1.0),
        Message(role="assistant", content="ok", created_at=2.0),
    ]
    messages += [
        Message(role="user", content="fetch", created_at=50.0),
        Message(
            role="assistant",
            tool_calls=[ToolCall(id="c1", name="web_fetch", arguments="{}")],
            created_at=51.0,
        ),
        Message(role="tool", content="page", tool_call_id="c1", untrusted=True, created_at=52.0),
        Message(role="assistant", content="summary", untrusted=True, created_at=53.0),
    ]

    apply_window(messages, keep=30, max_age_seconds=10, now=60.0)

    _valid(messages)
    assert messages[1].content == "fetch"
    assert any(message.untrusted for message in messages)
