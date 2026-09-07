"""Wire types for chat completions and tool calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class ToolSpec:
    """An OpenAI-compatible function tool the model may call."""

    name: str
    description: str
    parameters: dict[str, Any]

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(frozen=True)
class ToolCall:
    """A completed tool invocation requested by the model."""

    id: str
    name: str
    arguments: str


@dataclass
class Message:
    """One turn in the conversation sent to or received from the model."""

    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None

    def to_openai(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role}
        if self.role == "tool":
            payload["tool_call_id"] = self.tool_call_id
            payload["content"] = self.content or ""
            return payload
        if self.tool_calls:
            payload["content"] = self.content
            payload["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in self.tool_calls
            ]
            return payload
        payload["content"] = self.content or ""
        return payload


@dataclass(frozen=True)
class Completion:
    """A fully accumulated streamed (or probed) model response."""

    content: str
    tool_calls: list[ToolCall]
    finish_reason: str | None


@dataclass(frozen=True)
class TextDelta:
    """A streamed content fragment for the REPL to print."""

    text: str
