"""America/Chicago instants and a settable clock for time tests.

The household is in Chicago. 2026-11-01 repeats 1:00-1:59 AM (CDT, then
CST) and 2027-03-14 skips 2:00-2:59 AM. Tests name local wall times on
those days and let this module turn them into the instants the code stores.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from zoneinfo import ZoneInfo

from assistai.clock import HouseholdClock
from assistai.sysclock import ClockState

CHICAGO = ZoneInfo("America/Chicago")


def chicago(
    year: int, month: int, day: int, hour: int = 0, minute: int = 0, *, fold: int = 0
) -> float:
    """Unix time of a Chicago wall time. ``fold=1`` is the second 1:30 on fall-back day."""
    return datetime(year, month, day, hour, minute, tzinfo=CHICAGO, fold=fold).timestamp()


def wall(moment: float) -> str:
    """The Chicago wall time of an instant, with its abbreviation.

    Asserting on this text, not on floats, makes a failure read as the bug:
    "2026-11-01 07:00:00 CST" where 08:00 was expected.
    """
    return datetime.fromtimestamp(moment, CHICAGO).strftime("%Y-%m-%d %H:%M:%S %Z")


class FakeClock:
    """A wall clock a test moves by hand.

    Callable, so it drops in wherever the code takes a ``Callable[[], float]``
    clock: ``JobRunner(clock=...)`` or ``monkeypatch.setattr("assistai.jobs._now", ...)``.
    """

    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def set(self, moment: float) -> None:
        self.now = moment

    def advance(self, seconds: float) -> None:
        self.now += seconds


def chicago_clock(now: float | Callable[[], float]) -> HouseholdClock:
    """The household clock in Chicago, stopped at ``now`` or driven by a ``FakeClock``."""
    if callable(now):
        return HouseholdClock(zone=CHICAGO, now=now)
    moment = now
    return HouseholdClock(zone=CHICAGO, now=lambda: moment)


class FakeClockSync:
    """The kernel's sync state as a test sets it. Counts how often it was asked."""

    def __init__(self, state: ClockState = ClockState.SYNCED) -> None:
        self.current = state
        self.asked = 0

    def state(self) -> ClockState:
        self.asked += 1
        return self.current
