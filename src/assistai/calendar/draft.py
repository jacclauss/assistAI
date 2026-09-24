"""Validate one event the model wants added. No calendar is chosen here."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

from assistai.calendar.parse import MAX_DESCRIPTION, MAX_LOCATION, MAX_SUMMARY
from assistai.errors import CalendarError

_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TIME = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?$")
_MAX_SPAN = timedelta(days=366)
_MAX_YEARS = 10


@dataclass(frozen=True)
class CalendarDraft:
    """A confirmed-shaped event. All-day ``end`` is the last inclusive day."""

    summary: str
    all_day: bool
    start: date | datetime
    end: date | datetime
    location: str = ""
    description: str = ""

    def label(self) -> str:
        if self.all_day:
            start_day = _as_date(self.start)
            end_day = _as_date(self.end)
            if start_day == end_day:
                return f"{start_day.isoformat()} (all day)"
            return f"{start_day.isoformat()} through {end_day.isoformat()} (all day)"
        if not isinstance(self.start, datetime) or not isinstance(self.end, datetime):
            raise CalendarError("start needs a time")
        start_text = self.start.strftime("%Y-%m-%d %H:%M")
        if self.start.date() == self.end.date():
            end_text = self.end.strftime("%H:%M")
        else:
            end_text = self.end.strftime("%Y-%m-%d %H:%M")
        return f"{start_text}-{end_text} (household local time)"


def parse_add(arguments: dict[str, Any], *, today: date | None = None) -> CalendarDraft:
    """Reject a draft that is not one event on the shared calendar."""
    summary = _text(arguments.get("summary"), field="summary", limit=MAX_SUMMARY, required=True)
    location = _text(arguments.get("location", ""), field="location", limit=MAX_LOCATION)
    description = _text(
        arguments.get("description", ""),
        field="description",
        limit=MAX_DESCRIPTION,
        newlines=True,
    )
    all_day, start = _moment(arguments.get("start"), field="start")
    if "all_day" in arguments and arguments["all_day"] is not None:
        flag = arguments["all_day"]
        if not isinstance(flag, bool):
            raise CalendarError("all_day must be true or false")
        if flag:
            all_day = True
            start = _as_date(start)
        elif all_day:
            raise CalendarError("start needs a time")
    if "end" in arguments and arguments["end"] not in (None, ""):
        end_all_day, end = _moment(arguments.get("end"), field="end")
        if all_day:
            end = _as_date(end)
        elif end_all_day:
            raise CalendarError("end needs a time")
    elif all_day:
        end = _as_date(start)
    elif isinstance(start, datetime):
        end = start + timedelta(hours=1)
    else:
        raise CalendarError("start needs a time")
    _check_span(all_day, start, end, today=today or date.today())
    return CalendarDraft(
        summary=summary,
        all_day=all_day,
        start=_as_date(start) if all_day else start,
        end=_as_date(end) if all_day else end,
        location=location,
        description=description,
    )


def _moment(value: object, *, field: str) -> tuple[bool, date | datetime]:
    if not isinstance(value, str):
        raise CalendarError(f"{field} must be YYYY-MM-DD or YYYY-MM-DDTHH:MM")
    text = value.strip()
    try:
        if _DATE.fullmatch(text):
            return True, date.fromisoformat(text)
        if _TIME.fullmatch(text):
            return False, datetime.fromisoformat(text)
    except ValueError as exc:
        raise CalendarError(f"{field} must be YYYY-MM-DD or YYYY-MM-DDTHH:MM") from exc
    raise CalendarError(f"{field} must be YYYY-MM-DD or YYYY-MM-DDTHH:MM")


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


def _check_span(
    all_day: bool,
    start: date | datetime,
    end: date | datetime,
    *,
    today: date,
) -> None:
    start_day = _as_date(start)
    if start_day.year < today.year - _MAX_YEARS or start_day.year > today.year + _MAX_YEARS:
        raise CalendarError("that date is too far away")
    if all_day:
        if _as_date(end) < start_day:
            raise CalendarError("end is before start")
        if _as_date(end) - start_day > _MAX_SPAN:
            raise CalendarError("that event is longer than a year")
        return
    if not isinstance(start, datetime) or not isinstance(end, datetime):
        raise CalendarError("start needs a time")
    if end <= start:
        raise CalendarError("end is before start")
    if end - start > _MAX_SPAN:
        raise CalendarError("that event is longer than a year")
