"""Validate one event the model wants added. No calendar is chosen here."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

from assistai.calendar.parse import MAX_DESCRIPTION, MAX_LOCATION, MAX_SUMMARY
from assistai.clock import HouseholdClock, say_day, say_span, wall_time
from assistai.errors import CalendarError, JobError

_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TIME = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?$")
_MAX_SPAN = timedelta(days=366)
_MAX_YEARS = 10
_DEFAULT_LENGTH = timedelta(hours=1)


@dataclass(frozen=True)
class CalendarDraft:
    """A confirmed-shaped event. All-day ``end`` is the last inclusive day.

    A timed ``start`` and ``end`` are aware, on the household's wall, so
    what is previewed, written, and confirmed is one instant each.
    """

    summary: str
    all_day: bool
    start: date | datetime
    end: date | datetime
    location: str = ""
    description: str = ""

    def label(self, clock: HouseholdClock) -> str:
        """When the event is, in the household's words: the day relative to today, and the hour.

        A start in the hour the clocks repeat names its zone abbreviation,
        so the person can see which of the two it is.
        """
        if self.all_day:
            start_day = _as_date(self.start)
            end_day = _as_date(self.end)
            if start_day == end_day:
                return f"{say_day(start_day, clock=clock)} (all day)"
            return (
                f"{say_day(start_day, clock=clock)} through "
                f"{say_day(end_day, clock=clock)} (all day)"
            )
        start, end = _instants(self.start, self.end)
        return say_span(start, end, clock=clock)


def parse_add(arguments: dict[str, Any], *, clock: HouseholdClock) -> CalendarDraft:
    """Reject a draft that is not one event on the shared calendar.

    Times are read on ``clock``'s wall. One the spring change skips is
    refused with the reason; one the fall-back hour repeats is the first.
    The year bound is the household's year, which is not the UTC year on
    New Year's Eve evening.
    """
    summary = _text(arguments.get("summary"), field="summary", limit=MAX_SUMMARY, required=True)
    location = _text(arguments.get("location", ""), field="location", limit=MAX_LOCATION)
    description = _text(
        arguments.get("description", ""),
        field="description",
        limit=MAX_DESCRIPTION,
        newlines=True,
    )
    all_day, start = _moment(arguments.get("start"), field="start", clock=clock)
    if "all_day" in arguments and arguments["all_day"] is not None:
        flag = arguments["all_day"]
        if not isinstance(flag, bool):
            raise CalendarError("all_day must be true or false")
        if flag:
            all_day = True
            start = _as_date(start)
        elif all_day:
            raise CalendarError("start needs a time")
    _check_year(_as_date(start), today=clock.today())
    if "end" in arguments and arguments["end"] not in (None, ""):
        end_all_day, end = _moment(arguments.get("end"), field="end", clock=clock)
        if all_day:
            end = _as_date(end)
        elif end_all_day:
            raise CalendarError("end needs a time")
    elif all_day:
        end = _as_date(start)
    elif isinstance(start, datetime):
        # Aware arithmetic in one zone is wall-clock arithmetic: 1:30 AM CDT
        # plus an hour would be 2:30 AM CST, two real hours on fall-back night.
        end = (start.astimezone(UTC) + _DEFAULT_LENGTH).astimezone(clock.zone)
    else:
        raise CalendarError("start needs a time")
    _check_span(all_day, start, end)
    return CalendarDraft(
        summary=summary,
        all_day=all_day,
        start=_as_date(start) if all_day else start,
        end=_as_date(end) if all_day else end,
        location=location,
        description=description,
    )


def _moment(value: object, *, field: str, clock: HouseholdClock) -> tuple[bool, date | datetime]:
    shape = f"{field} must be YYYY-MM-DD or YYYY-MM-DDTHH:MM"
    if not isinstance(value, str):
        raise CalendarError(shape)
    text = value.strip()
    try:
        if _DATE.fullmatch(text):
            return True, date.fromisoformat(text)
        if not _TIME.fullmatch(text):
            raise CalendarError(shape)
        floating = datetime.fromisoformat(text)
    except ValueError as exc:
        raise CalendarError(shape) from exc
    try:
        return False, wall_time(floating, clock.zone)
    except JobError as exc:
        raise CalendarError(str(exc)) from exc


def _text(
    value: object,
    *,
    field: str,
    limit: int,
    required: bool = False,
    newlines: bool = False,
) -> str:
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise CalendarError(f"{field} must be text")
    text = value.strip()
    if required and not text:
        raise CalendarError(f"{field} is required")
    if any(ord(ch) < 32 and ch not in ("\n", "\t") for ch in text):
        raise CalendarError(f"{field} has characters that cannot be stored")
    if not newlines and any(ch in text for ch in "\r\n"):
        raise CalendarError(f"{field} must be a single line")
    if len(text) > limit:
        raise CalendarError(f"{field} is too long")
    return text


def _as_date(value: date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    return value


def _check_year(day: date, *, today: date) -> None:
    if day.year < today.year - _MAX_YEARS or day.year > today.year + _MAX_YEARS:
        raise CalendarError("that date is too far away")


def _instants(start: date | datetime, end: date | datetime) -> tuple[float, float]:
    # Aware times in one zone compare and subtract on the wall clock, which
    # is an hour off across a change, so a timed event is measured in instants.
    if not isinstance(start, datetime) or not isinstance(end, datetime):
        raise CalendarError("start needs a time")
    if start.tzinfo is None or end.tzinfo is None:
        raise CalendarError("a timed event needs the household zone")
    return start.timestamp(), end.timestamp()


def _check_span(all_day: bool, start: date | datetime, end: date | datetime) -> None:
    start_day = _as_date(start)
    if all_day:
        if _as_date(end) < start_day:
            raise CalendarError("end is before start")
        if _as_date(end) - start_day > _MAX_SPAN:
            raise CalendarError("that event is longer than a year")
        return
    start_at, end_at = _instants(start, end)
    if end_at <= start_at:
        raise CalendarError("end is before start")
    if end_at - start_at > _MAX_SPAN.total_seconds():
        raise CalendarError("that event is longer than a year")
