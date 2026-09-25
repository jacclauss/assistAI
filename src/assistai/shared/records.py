"""Validate a shared-list change before it is stored."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from assistai.errors import SharedError

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 .'-]{0,39}$")
MAX_ITEMS = 20


@dataclass(frozen=True)
class ListChange:
    name: str
    add: tuple[str, ...] = ()
    done: tuple[str, ...] = ()
    remove: tuple[str, ...] = ()
    private: bool = False
    share: bool = False

    def label_text(self) -> str:
        if self.share:
            return f"Share your private list {self.name} with the household."
        scope = "private" if self.private else "shared"
        lines = [f"{self.name} ({scope}):"]
        lines.extend(f"- add {item}" for item in self.add)
        lines.extend(f"- done {item}" for item in self.done)
        lines.extend(f"- remove {item}" for item in self.remove)
        return "\n".join(lines)


def parse_change(arguments: dict[str, Any]) -> ListChange:
    name = arguments.get("list")
    if not isinstance(name, str) or _NAME.fullmatch(name.strip()) is None:
        raise SharedError("list name must be a short household name")
    private = _flag(arguments.get("private", False), field="private")
    share = _flag(arguments.get("share", False), field="share")
    if private and share:
        raise SharedError("a list cannot be private and shared in one change")
    add = _items(arguments.get("add", []))
    done = _items(arguments.get("done", []))
    remove = _items(arguments.get("remove", []))
    if share and (add or done or remove):
        raise SharedError("share a private list on its own")
    _reject_overlap(add, done, remove)
    if not share and not add and not done and not remove:
        raise SharedError("say what to add, check off, or remove")
    if len(add) + len(done) + len(remove) > MAX_ITEMS:
        raise SharedError(f"a change can touch at most {MAX_ITEMS} items")
    return ListChange(
        name=name.strip(),
        add=add,
        done=done,
        remove=remove,
        private=private,
        share=share,
    )


def _reject_overlap(add: tuple[str, ...], done: tuple[str, ...], remove: tuple[str, ...]) -> None:
    """One item, one operation. Otherwise a yes can check off a row that was also removed."""
    folded = {
        "add": {item.casefold() for item in add},
        "done": {item.casefold() for item in done},
        "remove": {item.casefold() for item in remove},
    }
    for left, right in (("add", "done"), ("add", "remove"), ("done", "remove")):
        both = folded[left] & folded[right]
        if both:
            raise SharedError(f"{next(iter(both))!r} cannot be both {left} and {right}")


def _flag(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise SharedError(f"{field} must be true or false")
    return value


def _items(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise SharedError("items must be a list of text")
    found: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            raise SharedError("items must be text")
        text = " ".join(item.split())
        if not text or len(text) > 120:
            raise SharedError("each item needs a short description")
        key = text.casefold()
        if key in seen:
            raise SharedError(f"{text!r} is listed twice")
        seen.add(key)
        found.append(text)
    return tuple(found)
