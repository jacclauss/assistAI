"""Build one VEVENT. User text is escaped by the calendar library, never spliced."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

from icalendar import Calendar, Event

from assistai.calendar.draft import CalendarDraft
from assistai.errors import CalendarError


def render_event(draft: CalendarDraft, *, uid: str, zone: ZoneInfo, stamp: datetime) -> str:
    """ICS for ``draft``. All-day end is stored as the exclusive next day."""
    calendar = Calendar()
    calendar.add("prodid", "-//AssistAI//EN")
    calendar.add("version", "2.0")
    event = Event()
    event.add("uid", uid)
    event.add("dtstamp", stamp.astimezone(UTC))
    event.add("summary", draft.summary)
    if draft.all_day:
        event.add("dtstart", _day(draft.start))
        event.add("dtend", _day(draft.end) + timedelta(days=1))
    elif isinstance(draft.start, datetime) and isinstance(draft.end, datetime):
        event.add("dtstart", _aware(draft.start, zone).astimezone(UTC))
        event.add("dtend", _aware(draft.end, zone).astimezone(UTC))
    else:
        raise CalendarError("start needs a time")
    if draft.location:
        event.add("location", draft.location)
    if draft.description:
        event.add("description", draft.description)
    calendar.add_component(event)
    raw = calendar.to_ical()
    if not isinstance(raw, bytes):
        raise CalendarError("could not build the event")
    return raw.decode("utf-8")


def _day(value: date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    return value


def _aware(moment: datetime, zone: ZoneInfo) -> datetime:
    if moment.tzinfo is None:
        return moment.replace(tzinfo=zone)
    return moment.astimezone(zone)
