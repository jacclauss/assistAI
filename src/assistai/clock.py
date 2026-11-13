"""The household clock. Every household local time is read and written here.

A person names times on the clock face in the household zone, and a daily
job keeps its hour across daylight saving (decision 0002). Turning those
into instants, and instants back into words, happens in one place so a
local-time rule cannot differ between jobs, relays, the calendar, and the
previews a person says yes to. Nothing else calls ``strftime``,
``datetime.fromtimestamp``, or ``replace(tzinfo=...)`` on a household time.

Errors are ``JobError`` with model-facing text. Callers in relay and
calendar wrap them in their own error so the tool registry passes the
message on unchanged.
"""

from __future__ import annotations

import datetime as dt
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from zoneinfo import ZoneInfo

from assistai.errors import JobError

# A time named a moment ago is what the person meant, not a mistake. The
# model and the tool round trip take seconds, so "at 8:53" said at 8:53
# still fires now instead of being refused or moved to tomorrow.
PAST_TOLERANCE_SECONDS = 60

_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
DAY_WORDS = ("today", "tomorrow", *_WEEKDAYS)

# The time arguments of job_create, relay, and job_reschedule, as the model
# sees them. They take the person's words, not an instant: the server turns
# those into an instant with this module's rules.
AT_SCHEMA: dict[str, object] = {
    "type": "object",
    "description": (
        "A clock time the person named, in their words. 'tomorrow at 8' is day "
        "tomorrow and hour 8. 'Monday at 7:30' is day monday, hour 7, minute 30. "
        "'December 7 at 9' is date 2026-12-07 and hour 9. A bare 'at 8' is hour 8 "
        "with neither day nor date. Needs get_time in this turn."
    ),
    "properties": {
        "hour": {"type": "integer", "minimum": 0, "maximum": 23},
        "minute": {"type": "integer", "minimum": 0, "maximum": 59, "default": 0},
        "day": {
            "type": "string",
            "enum": list(DAY_WORDS),
            "description": "today, tomorrow, or a weekday. Omit when date is set.",
        },
        "date": {
            "type": "string",
            "description": "A calendar date, YYYY-MM-DD. Omit when day is set.",
        },
        "fold": {
            "type": "integer",
            "enum": [0, 1],
            "description": (
                "Written by the server when it stores the call, for an hour the "
                "clocks repeat. Do not set it."
            ),
        },
    },
    "required": ["hour"],
    "additionalProperties": False,
}

DELAY_SCHEMA: dict[str, object] = {
    "type": "object",
    "description": (
        "Only for a duration the person said: 'in two hours', 'in 50 days'. "
        "The server counts it. At least one minute in total."
    ),
    "properties": {
        "days": {"type": "integer", "minimum": 0},
        "hours": {"type": "integer", "minimum": 0},
        "minutes": {"type": "integer", "minimum": 0},
    },
    "additionalProperties": False,
}

EVERY_DAYS_SCHEMA: dict[str, object] = {
    "type": "integer",
    "minimum": 1,
    "description": (
        "Whole days between runs: 1 is daily, 7 is weekly. The run keeps the "
        "local time in at across daylight saving. Needs at."
    ),
}

EVERY_MINUTES_SCHEMA: dict[str, object] = {
    "type": "integer",
    "minimum": 1,
    "description": (
        "Elapsed minutes between runs, such as 90. Not for whole days: use every_days."
    ),
}


@dataclass(frozen=True)
class HouseholdClock:
    """The household zone and the wall clock. Built once at startup."""

    zone: ZoneInfo
    now: Callable[[], float] = time.time

    def today(self) -> dt.date:
        """The household's local date, which is not the UTC date in the evening."""
        return _local(self.now(), self.zone).date()


@dataclass(frozen=True, kw_only=True)
class LocalAt:
    """A clock time as the person named it.

    ``day`` is ``today``, ``tomorrow``, or a weekday name; ``date`` is a
    calendar date. At most one is set; with neither, it is the next time the
    clock shows ``hour:minute``. ``fold`` picks the occurrence of a time the
    fall-back hour repeats and is written by the server, not the model.
    """

    date: dt.date | None = None
    day: str | None = None
    hour: int
    minute: int = 0
    fold: int | None = None


@dataclass(frozen=True)
class Delay:
    """A spoken duration counted from when it is confirmed."""

    days: int = 0
    hours: int = 0
    minutes: int = 0

    @property
    def seconds(self) -> int:
        return ((self.days * 24 + self.hours) * 60 + self.minutes) * 60


@dataclass(frozen=True)
class Schedule:
    """When a job runs, in its own zone.

    ``anchor_at`` is the instant of the first scheduled run. A whole-day
    repeat has ``every_days`` and ``at_local`` ("HH:MM") and keeps that clock
    time across daylight saving; a minute repeat has ``every_seconds`` and is
    elapsed time. A one-time job has neither. A schedule that mixes the two,
    or has a gap under one unit, cannot be built.
    """

    zone: ZoneInfo
    anchor_at: float
    every_days: int | None = None
    at_local: str | None = None
    every_seconds: int | None = None

    def __post_init__(self) -> None:
        if self.every_days is not None and self.every_seconds is not None:
            raise JobError("a schedule repeats in days or in seconds, not both")
        if self.every_days is not None:
            if self.every_days < 1:
                raise JobError("every_days must be at least 1")
            if self.at_local is None:
                raise JobError("a whole-day repeat needs a local time")
            _parse_hhmm(self.at_local)
        elif self.at_local is not None:
            raise JobError("at_local belongs to a whole-day repeat")
        if self.every_seconds is not None and self.every_seconds < 1:
            raise JobError("every_seconds must be at least 1")

    @property
    def repeats(self) -> bool:
        return self.every_days is not None or self.every_seconds is not None


def resolve_at(at: LocalAt, *, clock: HouseholdClock, on: dt.date | None = None) -> float:
    """The instant the person meant by a named clock time.

    A ``day`` is read against the household date, never the UTC date, so
    "tomorrow at 8" at 11:30 PM is the local tomorrow. A weekday is today
    while that time is still more than a minute ahead, so "Monday at 7:30"
    on Monday morning is not next week. A bare time is on ``on`` when it is
    given, or the day after once that has passed, so moving a December
    reminder "to 9" keeps December; without ``on`` it is the next time the
    clock shows it. A time the spring change skips is refused with the
    reason; a time the fall-back hour repeats is the first occurrence
    unless ``fold`` is 1 or the first has passed. A time more than a minute
    past is refused with the household's now in words, so the model can
    correct the day instead of guessing.
    """
    _check_at(at)
    now = clock.now()
    today = _local(now, clock.zone).date()
    base = today if on is None else max(on, today)
    earliest = now - PAST_TOLERANCE_SECONDS
    try:
        if at.date is not None:
            day = at.date
        elif at.day == "today":
            day = today
        elif at.day == "tomorrow":
            day = today + dt.timedelta(days=1)
        elif at.day is not None:
            ahead = (_WEEKDAYS.index(at.day) - today.weekday()) % 7
            if ahead == 0 and _latest(today, at, clock.zone) <= now + PAST_TOLERANCE_SECONDS:
                ahead = 7
            day = today + dt.timedelta(days=ahead)
        elif _latest(base, at, clock.zone) < earliest:
            day = base + dt.timedelta(days=1)
        else:
            day = base
    except OverflowError as exc:
        raise JobError("that date is too far out to schedule") from exc
    moment = _occurrence(day, at, clock.zone, earliest=earliest)
    if moment < earliest:
        raise JobError(
            f"{_full(moment, clock.zone, year=True)} has already passed. "
            f"It is now {_full(now, clock.zone, year=True)} household time ({clock.zone.key}). "
            "Name a later day or time."
        )
    return moment


def fix_at(at: LocalAt, *, clock: HouseholdClock, on: dt.date | None = None) -> LocalAt:
    """``at`` pinned to the date it means now, with ``fold`` when that time happens twice.

    "Tomorrow" in a preview shown at 11:58 PM is a different day at a
    12:03 AM yes, so a staged time is stored as the date the person was
    shown. ``on`` is read as in ``resolve_at``. ``fold`` is written only for
    the fall-back hour, where the date and clock time alone do not say
    which occurrence was meant. Refuses what ``resolve_at`` refuses.
    """
    moment = resolve_at(at, clock=clock, on=on)
    local = _local(moment, clock.zone)
    return LocalAt(
        date=local.date(),
        hour=at.hour,
        minute=at.minute,
        fold=local.fold if _repeated(local) else None,
    )


def say_at(at: LocalAt) -> str:
    """A named time as the person can check it, before it is resolved to an instant.

    A date carries its weekday and year, so a date the model got wrong reads
    as wrong. Which occurrence of a fall-back hour is meant is spelled out,
    because the clock time alone shows it twice.
    """
    text = _clock_text(dt.time(at.hour, at.minute))
    if at.fold == 0:
        text = f"{text} (the first one, before the clocks fall back)"
    elif at.fold == 1:
        text = f"{text} (the second one, after the clocks fall back)"
    if at.date is not None:
        return f"{_date_text(at.date, year=True)} at {text}"
    if at.day in ("today", "tomorrow"):
        return f"{at.day} at {text}"
    if at.day is not None:
        return f"{at.day.capitalize()} at {text}"
    return f"the next {text}"


def say_every(
    *, every_days: int | None, every_minutes: int | None, at: LocalAt | None = None
) -> str:
    """How often a requested repeat runs, in words, before it is stored as a schedule.

    A whole-day repeat names the clock time it keeps. Without ``at`` it keeps
    the time the job already runs at, and says so.
    """
    if every_minutes is not None:
        phrase = say_duration(every_minutes * 60)
        return f"every {phrase[2:]}" if phrase.startswith("1 ") else f"every {phrase}"
    if every_days is None:
        return "once"
    at_text = _clock_text(dt.time(at.hour, at.minute)) if at is not None else "the time it runs now"
    if every_days == 1:
        return f"every day at {at_text}"
    if every_days == 7 and at is not None and at.date is not None:
        return f"every {_weekday(at.date)} at {at_text}"
    if every_days == 7 and at is not None and at.day is not None and at.day in _WEEKDAYS:
        return f"every {at.day.capitalize()} at {at_text}"
    return f"every {every_days} days at {at_text}"


def weekday_of(day: dt.date) -> str:
    """The weekday name of a date, so the model reads it instead of working it out."""
    return _weekday(day)


def next_slot(schedule: Schedule, *, after: float) -> float:
    """The earliest run of a repeating schedule strictly after ``after``.

    Whole-day slots are the local dates ``anchor_date + k * every_days`` at
    ``at_local``. A slot the spring change skips runs one clock hour later
    that day only; a slot the fall-back hour repeats runs at its first
    occurrence only. Minute slots are ``anchor_at + k * every_seconds``.
    ``k`` is computed and then corrected by at most two steps, so a job idle
    for years costs the same as one that ran yesterday. A one-time schedule
    has no next slot and raises ``JobError``.
    """
    if schedule.every_seconds is not None:
        step = schedule.every_seconds
        if after < schedule.anchor_at:
            return schedule.anchor_at
        k = math.floor((after - schedule.anchor_at) / step) + 1
        k = _settle(lambda n: schedule.anchor_at + n * step, k, after)
        return schedule.anchor_at + k * step
    if schedule.every_days is None or schedule.at_local is None:
        raise JobError("a one-time job has no next run")
    every = schedule.every_days
    hour, minute = _parse_hhmm(schedule.at_local)
    anchor_date = _local(schedule.anchor_at, schedule.zone).date()

    def slot(n: int) -> float:
        day = anchor_date + dt.timedelta(days=n * every)
        # fold=0 reads a skipped time with the offset before the change and
        # picks the first of a repeated time: both rules of decision 0002.
        wall = dt.datetime.combine(day, dt.time(hour, minute), tzinfo=schedule.zone)
        return wall.timestamp()

    behind = (_local(after, schedule.zone).date() - anchor_date).days
    try:
        return slot(_settle(slot, max(0, behind // every), after))
    except OverflowError as exc:
        raise JobError("the next run is too far out to schedule") from exc


def slots_between(schedule: Schedule, *, after: float, upto: float) -> int:
    """How many slots fall strictly after ``after`` and no later than ``upto``.

    Counted from slot numbers, not by walking, so a job idle for a year
    costs the same as one idle for a minute.
    """
    if upto <= after:
        return 0
    first = next_slot(schedule, after=after)
    if first > upto:
        return 0
    beyond = next_slot(schedule, after=upto)
    return _slot_number(schedule, beyond) - _slot_number(schedule, first)


def shifted_by_spring_gap(schedule: Schedule, slot: float) -> bool:
    """True when a whole-day slot runs off its clock time because the spring change skipped it.

    That is the one day decision 0002 lets a daily job run an hour late,
    and the operator should be able to see it happened.
    """
    if schedule.every_days is None or schedule.at_local is None:
        return False
    return at_local_of(slot, schedule.zone) != schedule.at_local


def say_when(instant: float, *, clock: HouseholdClock) -> str:
    """An instant as a person reads it, relative to the household's today.

    A time the fall-back hour repeats carries its zone abbreviation, so
    "1:40 AM CST" cannot be read as the earlier 1:40.
    """
    local = _local(instant, clock.zone)
    at = _clock_text(local)
    ahead = (local.date() - clock.today()).days
    if ahead == 0:
        return f"today at {at}"
    if ahead == 1:
        return f"tomorrow at {at} ({_date_text(local.date())})"
    if 1 < ahead <= 6:
        return f"{_weekday(local.date())} at {at} ({_month_day(local.date())})"
    return f"{_date_text(local.date(), year=local.year != clock.today().year)} at {at}"


def say_repeat(schedule: Schedule, *, clock: HouseholdClock) -> str:
    """How often a schedule runs, in words.

    A schedule kept in a zone other than the household's names that zone,
    because its clock time is not the one on the household's wall.
    """
    if schedule.every_seconds is not None:
        phrase = say_duration(schedule.every_seconds)
        return f"every {phrase[2:]}" if phrase.startswith("1 ") else f"every {phrase}"
    if schedule.every_days is None or schedule.at_local is None:
        return "once"
    hour, minute = _parse_hhmm(schedule.at_local)
    at = _clock_text(dt.time(hour, minute))
    if schedule.zone.key != clock.zone.key:
        at = f"{at} {schedule.zone.key}"
    if schedule.every_days == 1:
        return f"every day at {at}"
    if schedule.every_days == 7:
        weekday = _weekday(_local(schedule.anchor_at, schedule.zone).date())
        return f"every {weekday} at {at}"
    return f"every {schedule.every_days} days at {at}"


def say_duration(seconds: int) -> str:
    """A duration in the largest whole unit that fits it exactly."""
    if seconds % 86400 == 0:
        count = seconds // 86400
        unit = "day" if count == 1 else "days"
        return f"{count} {unit}"
    if seconds % 3600 == 0:
        count = seconds // 3600
        unit = "hour" if count == 1 else "hours"
        return f"{count} {unit}"
    if seconds % 60 == 0:
        count = seconds // 60
        unit = "minute" if count == 1 else "minutes"
        return f"{count} {unit}"
    return f"{seconds} seconds"


def say_day(day: dt.date, *, clock: HouseholdClock) -> str:
    """A date with its weekday, and its relation to today when it is near.

    The relative word makes a date read from the wrong zone visible: a
    calendar read for UTC's date says "tomorrow" in a Chicago evening.
    """
    ahead = (day - clock.today()).days
    text = _date_text(day, year=day.year != clock.today().year)
    if ahead == 0:
        return f"today, {text}"
    if ahead == -1:
        return f"yesterday, {text}"
    if ahead == 1:
        return f"tomorrow, {text}"
    return text


def local_reading(instant: float, *, clock: HouseholdClock) -> tuple[str, str]:
    """An instant as ISO-8601 local time and its UTC offset, for operator logs.

    The offset is spelled out so a log line shows which side of a daylight
    saving change the gateway believes it is on. People read ``say_when``.
    """
    local = _local(instant, clock.zone)
    offset = local.utcoffset() or dt.timedelta(0)
    minutes = int(offset.total_seconds()) // 60
    sign = "-" if minutes < 0 else "+"
    hours, rest = divmod(abs(minutes), 60)
    return local.isoformat(timespec="seconds"), f"{sign}{hours:02d}:{rest:02d}"


def at_local_of(instant: float, zone: ZoneInfo) -> str:
    """The "HH:MM" clock time ``instant`` shows in ``zone``.

    A whole-day repeat is stored as this text, so the hour it keeps is the
    one on the household's wall, not a count of seconds from UTC.
    """
    try:
        local = _local(instant, zone)
    except (OverflowError, OSError, ValueError) as exc:
        raise JobError("that time is too far out to schedule") from exc
    return f"{local.hour:02d}:{local.minute:02d}"


def local_day(instant: float, zone: ZoneInfo) -> dt.date:
    """The date ``instant`` falls on in ``zone``, which is not the UTC date in the evening."""
    return _local(instant, zone).date()


def start_of_day(day: dt.date, zone: ZoneInfo) -> float:
    """The instant ``day`` begins in ``zone``, so a listing from a date is the household's date."""
    return dt.datetime.combine(day, dt.time(0, 0), tzinfo=zone).timestamp()


def _check_at(at: LocalAt) -> None:
    if not 0 <= at.hour <= 23:
        raise JobError("hour must be 0 to 23")
    if not 0 <= at.minute <= 59:
        raise JobError("minute must be 0 to 59")
    if at.fold not in (None, 0, 1):
        raise JobError("fold must be 0 or 1")
    if at.date is not None and at.day is not None:
        raise JobError("name a day or a date, not both")
    if at.day is not None and at.day not in DAY_WORDS:
        raise JobError("day must be today, tomorrow, or a weekday name")


def _parse_hhmm(text: str) -> tuple[int, int]:
    hour_text, sep, minute_text = text.partition(":")
    if (
        not sep
        or len(hour_text) != 2
        or len(minute_text) != 2
        or not hour_text.isdigit()
        or not minute_text.isdigit()
    ):
        raise JobError(f"at_local must be HH:MM, not {text!r}")
    hour, minute = int(hour_text), int(minute_text)
    if hour > 23 or minute > 59:
        raise JobError(f"at_local must be HH:MM, not {text!r}")
    return hour, minute


def _local(instant: float, zone: ZoneInfo) -> dt.datetime:
    return dt.datetime.fromtimestamp(instant, zone)


def _walls(day: dt.date, at: LocalAt, zone: ZoneInfo) -> tuple[dt.datetime, dt.datetime]:
    naive = dt.datetime.combine(day, dt.time(at.hour, at.minute))
    return naive.replace(tzinfo=zone, fold=0), naive.replace(tzinfo=zone, fold=1)


def _latest(day: dt.date, at: LocalAt, zone: ZoneInfo) -> float:
    """The last instant ``day`` can mean ``at``, so a repeated hour is past only when both are."""
    first, second = _walls(day, at, zone)
    return max(first.timestamp(), second.timestamp())


def _occurrence(day: dt.date, at: LocalAt, zone: ZoneInfo, *, earliest: float) -> float:
    first, second = _walls(day, at, zone)
    before, after = first.utcoffset(), second.utcoffset()
    if before is None or after is None or before == after:
        return first.timestamp()
    if before < after:
        start, end = _jump(first, second, zone, before=before, after=after)
        raise JobError(
            f"{_clock_text(first)} does not exist on {_date_text(day, year=True)}: "
            f"clocks go from {start} to {end}. Name another time."
        )
    if at.fold == 1:
        return second.timestamp()
    if at.fold is None and first.timestamp() < earliest:
        return second.timestamp()
    return first.timestamp()


def _jump(
    first: dt.datetime,
    second: dt.datetime,
    zone: ZoneInfo,
    *,
    before: dt.timedelta,
    after: dt.timedelta,
) -> tuple[str, str]:
    """The wall times either side of the spring change around a skipped time.

    zoneinfo does not expose its transitions, so the change is found by
    bisecting the seconds between the two readings of the skipped time.
    """
    low, high = math.floor(second.timestamp()), math.ceil(first.timestamp())
    while high - low > 1:
        middle = (low + high) // 2
        if _local(middle, zone).utcoffset() == before:
            low = middle
        else:
            high = middle
    change = dt.datetime.fromtimestamp(high, dt.UTC).replace(tzinfo=None)
    return _bare_clock(change + before), _bare_clock(change + after)


def _slot_number(schedule: Schedule, slot: float) -> int:
    """The ``k`` of a slot ``next_slot`` returned. A shifted spring slot keeps its date."""
    if schedule.every_seconds is not None:
        return round((slot - schedule.anchor_at) / schedule.every_seconds)
    every = schedule.every_days or 1
    anchor_date = _local(schedule.anchor_at, schedule.zone).date()
    return (_local(slot, schedule.zone).date() - anchor_date).days // every


def _settle(slot: Callable[[int], float], k: int, after: float) -> int:
    # The computed k can be one or two off: a date boundary in the zone, or
    # a slot shifted out of the spring gap. Never more.
    for _ in range(2):
        if slot(k) > after:
            break
        k += 1
    for _ in range(2):
        if k > 0 and slot(k - 1) > after:
            k -= 1
    if slot(k) <= after:
        raise JobError("the schedule has no run after that time")
    return k


def _clock_text(local: dt.datetime | dt.time) -> str:
    text = f"{_bare_clock(local)} {'AM' if local.hour < 12 else 'PM'}"
    if isinstance(local, dt.datetime) and _repeated(local):
        return f"{text} {local.tzname()}"
    return text


def _bare_clock(local: dt.datetime | dt.time) -> str:
    return f"{local.hour % 12 or 12}:{local.minute:02d}"


def _repeated(local: dt.datetime) -> bool:
    """True for a wall time the fall-back hour shows twice, not for one the spring change skips."""
    first, second = local.replace(fold=0).utcoffset(), local.replace(fold=1).utcoffset()
    return first is not None and second is not None and first > second


def _weekday(day: dt.date) -> str:
    return _WEEKDAYS[day.weekday()].capitalize()


def _month_day(day: dt.date) -> str:
    return f"{_MONTHS[day.month - 1]} {day.day}"


def _date_text(day: dt.date, *, year: bool = False) -> str:
    text = f"{_weekday(day)}, {_month_day(day)}"
    return f"{text}, {day.year}" if year else text


def _full(instant: float, zone: ZoneInfo, *, year: bool) -> str:
    local = _local(instant, zone)
    return f"{_date_text(local.date(), year=year)} at {_clock_text(local)}"
