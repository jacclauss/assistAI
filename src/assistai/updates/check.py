"""Compare image pins with published tags. This module does not fetch or write."""

from __future__ import annotations

import re
from dataclasses import dataclass

from assistai.manifest import ImagePin

_VERSION = re.compile(r"^\d+(?:\.\d+)*$")
_DIGEST = re.compile(r"sha256:[a-f0-9]{64}")


@dataclass(frozen=True)
class PublishedTag:
    name: str
    digest: str = ""


@dataclass(frozen=True)
class TagPage:
    """Tags from one repository. ``complete`` is false when the page cap stopped the walk."""

    tags: tuple[PublishedTag, ...]
    complete: bool


@dataclass(frozen=True)
class UpdateFinding:
    name: str
    priority: str
    detail: str


def assess(
    pin: ImagePin,
    tags: tuple[PublishedTag, ...] | list[PublishedTag],
    *,
    complete: bool,
) -> UpdateFinding | None:
    """A newer high-priority release, a missing tag, or a moved digest.

    An empty digest is unverified, not stale. A normal pin is not nudged
    because some other tag on that repository is newer. A truncated tag list
    cannot prove a tag is gone.
    """
    by_name = {tag.name: tag for tag in tags}
    current = by_name.get(pin.tag)
    newer = _newer_release(pin, tags)
    if current is None:
        if newer is not None:
            return UpdateFinding(
                name=pin.name,
                priority=pin.priority,
                detail=f"pinned {pin.ref}, upstream {newer}",
            )
        if not complete:
            return None
        return UpdateFinding(
            name=pin.name,
            priority=pin.priority,
            detail=f"pinned {pin.ref}, and that tag is no longer published",
        )
    if newer is not None:
        return UpdateFinding(
            name=pin.name,
            priority=pin.priority,
            detail=f"pinned {pin.ref}, upstream {newer}",
        )
    if pin.digest and current.digest and current.digest != pin.digest:
        moved = current.digest if _DIGEST.fullmatch(current.digest) else "a new digest"
        return UpdateFinding(
            name=pin.name,
            priority=pin.priority,
            detail=f"pinned {pin.ref} moved from {pin.digest} to {moved}",
        )
    return None


def format_report(findings: tuple[UpdateFinding, ...] | list[UpdateFinding]) -> str:
    """What the operator reads. The first line is the promise that nothing changed."""
    if not findings:
        return ""
    ordered = sorted(findings, key=lambda item: (item.priority != "high", item.name))
    lines = ["Update check. Nothing was installed or changed."]
    for item in ordered:
        rank = "high priority" if item.priority == "high" else "normal"
        lines.append(f"{item.name} is stale ({rank}): {item.detail}.")
    return "\n".join(lines)


def _newer_release(
    pin: ImagePin, tags: tuple[PublishedTag, ...] | list[PublishedTag]
) -> str | None:
    """High-priority version pins only. ``latest`` and variant tags are not releases."""
    if pin.priority != "high":
        return None
    current = _version_key(pin.tag)
    if current is None:
        return None
    newest: tuple[int, ...] | None = None
    newest_name = ""
    for tag in tags:
        key = _version_key(tag.name)
        if key is None:
            continue
        if newest is None or key > newest:
            newest = key
            newest_name = tag.name
    if newest is None or newest <= current:
        return None
    return newest_name


def _version_key(tag: str) -> tuple[int, ...] | None:
    if _VERSION.fullmatch(tag) is None:
        return None
    return tuple(int(part) for part in tag.split("."))
