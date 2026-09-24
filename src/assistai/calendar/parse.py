"""Turn expanded VEVENT bodies into events on one local day."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from icalendar import Calendar
from icalendar.error import IncompleteComponent, InvalidCalendar

from assistai.errors import CalendarError

MAX_SUMMARY = 200
MAX_LOCATION = 200
MAX_DESCRIPTION = 400


class CalendarEvent:
    """One event the model may recite. Times are in the household timezone."""

    def __init__(
        self,
        *,
        summary: str,
        start: datetime,
        end: datetime,
        all_day: bool,
        location: str,
        description: str,
    ) -> None:
        self.summary = summary
        self.start = start
        self.end = end
        self.all_day = all_day
        self.location = location
        self.description = description


def events_on(ics: str, *, day: date, zone: ZoneInfo) -> list[CalendarEvent]:
    """Events in ``ics`` that overlap ``day`` in ``zone``. Cancelled ones drop out."""
    try:
        parsed: object = Calendar.from_ical(ics, multiple=True)
    except (ValueError, TypeError) as exc:
        raise CalendarError("calendar returned unreadable data") from exc
    if not isinstance(parsed, list) or not all(isinstance(item, Calendar) for item in parsed):
        raise CalendarError("calendar returned unreadable data")
    found: list[CalendarEvent] = []
    for calendar in parsed:
        for component in calendar.walk("VEVENT"):
            event = _event(component, zone)
            if event is None or not _overlaps(event, day, zone):
                continue
            found.append(event)
    found.sort(key=lambda item: (item.start, item.summary))
    return found


def _event(component: Any, zone: ZoneInfo) -> CalendarEvent | None:
    status = component.get("status")
    if status is not None and str(status).strip().upper() == "CANCELLED":
        return None
    # .end honors DURATION when DTEND is absent, and the RFC defaults when
    # both are absent. An event that sets both is invalid and is dropped.
    try:
        start_raw = component.start
        end_raw = component.end
    except (IncompleteComponent, InvalidCalendar):
        return None
    if not isinstance(start_raw, (datetime, date)) or not isinstance(end_raw, (datetime, date)):
        return None
    all_day = not isinstance(start_raw, datetime)
    start = _localize(start_raw, zone)
    end = _localize(end_raw, zone)
    if end < start:
        end = start
    return CalendarEvent(
        summary=_text(component.get("summary"), MAX_SUMMARY) or "(no title)",
        start=start,
        end=end,
        all_day=all_day,
        location=_text(component.get("location"), MAX_LOCATION),
        description=_text(component.get("description"), MAX_DESCRIPTION),
    )


def _localize(value: datetime | date, zone: ZoneInfo) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=zone)
        return value.astimezone(zone)
    return datetime.combine(value, time.min, tzinfo=zone)


def _overlaps(event: CalendarEvent, day: date, zone: ZoneInfo) -> bool:
    day_start = datetime.combine(day, time.min, tzinfo=zone)
    day_end = day_start + timedelta(days=1)
    return event.start < day_end and event.end > day_start


def _text(value: object, limit: int) -> str:
    # A repeated property (two SUMMARY lines) comes back as a list.
    if isinstance(value, list):
        value = value[0] if value else None
    if value is None:
        return ""
    text = " ".join(str(value).split())
    return text[:limit]
