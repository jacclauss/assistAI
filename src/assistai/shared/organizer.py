"""The only process that reads or writes household lists.

It is not a Signal identity and it has no web access. Agents ask it through
tools; it checks the agent's read or write grant and then touches the store.
A private list is visible only to the agent who owns it.
"""

from __future__ import annotations

import time
from typing import Protocol

from assistai.agents import AgentSpec
from assistai.errors import SharedError, StoreError
from assistai.shared.records import ListChange

MAX_LISTS = 30
MAX_OPEN = 50

_MISSING = "there is no list named {name}"


class SharedStore(Protocol):
    def shared_snapshot(self) -> list[tuple[str, str | None, list[tuple[str, bool]]]]: ...

    def apply_shared(
        self,
        *,
        name: str,
        add: tuple[str, ...],
        done: tuple[str, ...],
        remove: tuple[str, ...],
        actor: str,
        owner: str | None,
        share: bool,
        now: float,
    ) -> None: ...


class Organizer:
    """Household lists. Never browses, never sends, never binds to a phone."""

    def __init__(self, store: SharedStore) -> None:
        self._store = store

    def read(self, agent: AgentSpec, name: str = "", *, private: bool = False) -> str:
        if "shared" not in agent.reads and "own" not in agent.reads:
            raise SharedError("this agent cannot read lists")
        rows = [
            row
            for row in self._store.shared_snapshot()
            if _visible(agent, row[1], private=private, named=bool(name))
        ]
        if name:
            wanted = name.casefold()
            rows = [row for row in rows if row[0].casefold() == wanted]
            if not rows:
                raise SharedError(_MISSING.format(name=name))
        if not rows:
            return "No lists yet."
        return "\n\n".join(_format_list(title, owner, items) for title, owner, items in rows)

    def apply(self, agent: AgentSpec, change: ListChange) -> str:
        _require_write(agent, change)
        owner = agent.name if change.private else None
        rows = [
            row
            for row in self._store.shared_snapshot()
            if _visible(agent, row[1], private=change.private or change.share, named=True)
        ]
        current = next((row for row in rows if row[0].casefold() == change.name.casefold()), None)
        if change.share:
            if current is None or current[1] != agent.name:
                raise SharedError(_MISSING.format(name=change.name))
        elif current is None:
            if not change.add or change.done or change.remove:
                raise SharedError(_MISSING.format(name=change.name))
            scope = [row for row in rows if (row[1] is None) == (owner is None)]
            if len(scope) >= MAX_LISTS:
                raise SharedError(f"there can be at most {MAX_LISTS} lists of that kind")
        else:
            open_items = sum(1 for _text, done in current[2] if not done)
            if open_items + len(change.add) > MAX_OPEN:
                raise SharedError(f"a list can have at most {MAX_OPEN} open items")
        try:
            self._store.apply_shared(
                name=change.name,
                add=change.add,
                done=change.done,
                remove=change.remove,
                actor=agent.name,
                owner=owner,
                share=change.share,
                now=time.time(),
            )
        except StoreError as exc:
            raise SharedError(str(exc)) from exc
        if change.share:
            return f"Shared {change.name}."
        return f"Updated {change.name}."


def _visible(agent: AgentSpec, owner: str | None, *, private: bool, named: bool) -> bool:
    """A named lookup stays inside one scope so the other person's names do not leak."""
    if named and private:
        return owner == agent.name and "own" in agent.reads
    if named:
        return owner is None and "shared" in agent.reads
    if private:
        return owner == agent.name and "own" in agent.reads
    if owner is None:
        return "shared" in agent.reads
    return owner == agent.name and "own" in agent.reads


def _require_write(agent: AgentSpec, change: ListChange) -> None:
    if (change.private or change.share) and "own" not in agent.writes:
        raise SharedError("this agent cannot change private lists")
    if (change.share or not change.private) and "shared" not in agent.writes:
        raise SharedError("this agent cannot change shared lists")


def _format_list(name: str, owner: str | None, items: list[tuple[str, bool]]) -> str:
    scope = "private" if owner else "shared"
    lines = [f"{name} ({scope})"]
    if not items:
        lines.append("- (empty)")
        return "\n".join(lines)
    for text, done in items:
        mark = "done" if done else "open"
        lines.append(f"- [{mark}] {text}")
    return "\n".join(lines)
