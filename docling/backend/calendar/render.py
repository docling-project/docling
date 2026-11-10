# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Add iCalendar components to a DoclingDocument as headings, fields and lists."""

import re
from datetime import timedelta

from docling_core.types.doc import DocItemLabel, DoclingDocument, NodeItem

from docling.backend.calendar.parser import Component, Property
from docling.backend.calendar.values import (
    describe_rrule,
    format_duration,
    format_property_time,
    format_time,
    parse_duration,
    parse_time,
    property_time,
)

_ADDRESS_SPECIALS = re.compile(r'[][()<>@,:;"\\]')
_ADDRESS_ESCAPES = re.compile(r'[\\"]')
_MAILTO = re.compile(r"^mailto:", re.IGNORECASE)
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n+")

# Apple Calendar writes this trigger for a default alarm that never fires.
_APPLE_PLACEHOLDER_TRIGGER = "19760401T005545Z"

_RENDERED_ENTRIES = ("VEVENT", "VTODO", "VJOURNAL", "VFREEBUSY")
_UNTITLED = {
    "VEVENT": "Untitled event",
    "VTODO": "Untitled task",
    "VJOURNAL": "Untitled journal entry",
    "VFREEBUSY": "Free/busy",
}
_METHODS = {
    "REQUEST": "Invitation",
    "CANCEL": "Cancellation",
    "REPLY": "Reply",
    "ADD": "Added occurrences",
    "REFRESH": "Refresh request",
    "COUNTER": "Counter-proposal",
    "DECLINECOUNTER": "Declined counter-proposal",
}
_ROLES = {
    "CHAIR": "chair",
    "REQ-PARTICIPANT": "required",
    "OPT-PARTICIPANT": "optional",
    "NON-PARTICIPANT": "for information",
}
_PARTICIPATION = {"NEEDS-ACTION": "awaiting response"}
_USER_TYPES = {"GROUP": "group", "RESOURCE": "resource", "ROOM": "room"}
_ALARM_ACTIONS = {
    "AUDIO": "Sound",
    "DISPLAY": "Display",
    "EMAIL": "Email",
    "PROCEDURE": "Procedure",
}
_FREE_BUSY_TYPES = {
    "FREE": "Free",
    "BUSY": "Busy",
    "BUSY-UNAVAILABLE": "Unavailable",
    "BUSY-TENTATIVE": "Tentative",
}
_CONFERENCE_PROPERTIES = (
    "CONFERENCE",
    "X-GOOGLE-CONFERENCE",
    "X-MICROSOFT-SKYPETEAMSMEETINGURL",
)
_FILENAME_PARAMS = ("FILENAME", "X-FILENAME", "X-APPLE-FILENAME", "X-ORACLE-FILENAME")


def add_calendar_components(
    doc: DoclingDocument,
    components: list[Component],
    parent: NodeItem | None = None,
    base_level: int = 1,
) -> None:
    """Add calendars, and entries found outside a calendar, in file order.

    Each entry gets a heading at ``base_level`` and its attendees, reminders
    and attachments get headings one level below.

    Args:
        doc: The document to add the items to.
        components: The top-level components from ``parse_components``.
        parent: The node to add the items under; the body if ``None``.
        base_level: The heading level of each entry.
    """
    writer = _Writer(doc, parent, base_level)
    for component in components:
        if component.name == "VCALENDAR":
            writer.add_calendar(component)
        elif component.name in _RENDERED_ENTRIES:
            writer.add_entry(component)


class _Writer:
    def __init__(self, doc: DoclingDocument, parent: NodeItem | None, level: int):
        self.doc = doc
        self.parent = parent
        self.level = level

    def _add_text(self, text: str) -> None:
        self.doc.add_text(label=DocItemLabel.TEXT, text=text, parent=self.parent)

    def _add_list(self, heading: str, items: list[str]) -> None:
        if not items:
            return
        self.doc.add_heading(text=heading, level=self.level + 1, parent=self.parent)
        group = self.doc.add_list_group(name=heading.lower(), parent=self.parent)
        for item in items:
            self.doc.add_list_item(text=item, parent=group)

    def add_calendar(self, calendar: Component) -> None:
        name = _one_line(calendar.text("X-WR-CALNAME") or calendar.text("NAME"))
        if name:
            self.doc.add_title(text=name, parent=self.parent)
        description = calendar.text("X-WR-CALDESC") or calendar.text("DESCRIPTION")
        for paragraph in _paragraphs(description):
            self._add_text(paragraph)
        method = calendar.text("METHOD").strip().upper()
        if method and method != "PUBLISH":
            self._add_text(f"Message type: {_METHODS.get(method, _words(method))}")
        for component in calendar.components:
            if component.name in _RENDERED_ENTRIES:
                self.add_entry(component)

    def add_entry(self, entry: Component) -> None:
        title = _one_line(entry.text("SUMMARY")) or _UNTITLED[entry.name]
        self.doc.add_heading(text=title, level=self.level, parent=self.parent)
        for field in _entry_fields(entry):
            self._add_text(field)
        # A journal entry may hold several DESCRIPTION properties.
        for description in entry.get_all("DESCRIPTION"):
            for paragraph in _paragraphs(description.value):
                self._add_text(paragraph)
        if entry.name == "VFREEBUSY":
            self._add_list("Free/busy periods", _free_busy_periods(entry))
        self._add_list("Attendees", [_attendee(p) for p in entry.get_all("ATTENDEE")])
        reminders = [
            _alarm(alarm, entry.name)
            for alarm in entry.components
            if alarm.name == "VALARM"
        ]
        self._add_list("Reminders", [r for r in reminders if r])
        self._add_list("Attachments", [_attachment(p) for p in entry.get_all("ATTACH")])


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in _PARAGRAPH_BREAK.split(text.strip()) if p.strip()]


def _words(value: str) -> str:
    return value.replace("-", " ").capitalize()


def _entry_fields(entry: Component) -> list[str]:
    fields = _time_fields(entry)
    for prop in entry.get_all("RRULE"):
        fields.append(f"Repeats: {describe_rrule(prop.value) or prop.value}")
    exceptions = [_list_time(v, p) for p in entry.get_all("EXDATE") for v in p.values]
    if exceptions:
        fields.append(f"Except: {', '.join(exceptions)}")
    additions = [_list_time(v, p) for p in entry.get_all("RDATE") for v in p.values]
    if additions:
        fields.append(f"Also on: {', '.join(additions)}")
    if (recurrence_id := entry.get("RECURRENCE-ID")) is not None:
        label = (
            "Changes occurrences from"
            if (recurrence_id.param("RANGE") or "").upper() == "THISANDFUTURE"
            else "Changes occurrence"
        )
        fields.append(f"{label}: {format_property_time(recurrence_id)}")
    if (completed := entry.get("COMPLETED")) is not None:
        fields.append(f"Completed: {format_property_time(completed)}")
    if percent := entry.text("PERCENT-COMPLETE").strip():
        fields.append(f"Progress: {percent}%")

    if location := entry.text("LOCATION"):
        lines = [line.strip() for line in location.splitlines() if line.strip()]
        fields.append(f"Location: {', '.join(lines)}")
    if (organizer := entry.get("ORGANIZER")) is not None:
        fields.append(f"Organizer: {_address(organizer)}")
    if status := entry.text("STATUS").strip():
        fields.append(f"Status: {_words(status)}")
    visibility = entry.text("CLASS").strip()
    if visibility and visibility.upper() != "PUBLIC":
        fields.append(f"Visibility: {_words(visibility)}")
    if priority := _priority(entry.text("PRIORITY")):
        fields.append(f"Priority: {priority}")
    for name, label in (("CATEGORIES", "Categories"), ("RESOURCES", "Resources")):
        values = [v for p in entry.get_all(name) for v in p.values if v]
        if values:
            fields.append(f"{label}: {', '.join(values)}")
    if url := entry.text("URL").strip():
        fields.append(f"URL: {url}")
    fields.extend(f"Conference: {c}" for c in _conferences(entry))
    for prop in entry.get_all("CONTACT"):
        fields.append(f"Contact: {_one_line(prop.value)}")
    for prop in entry.get_all("COMMENT"):
        fields.append(f"Comment: {_one_line(prop.value)}")
    return fields


def _time_fields(entry: Component) -> list[str]:
    fields: list[str] = []
    start_prop = entry.get("DTSTART")
    end_prop = entry.get("DTEND")
    duration = parse_duration(entry.text("DURATION"))
    start = property_time(start_prop) if start_prop is not None else None
    start_label = "Date" if entry.name == "VJOURNAL" else "Start"

    if start_prop is None:
        pass
    elif start is None:
        fields.append(f"{start_label}: {start_prop.value}")
    elif start.all_day and entry.name == "VEVENT":
        # DTEND of an all-day entry is exclusive: a one-day event ends the next day.
        end = property_time(end_prop) if end_prop is not None else None
        last_day = None
        if end is not None and end.all_day:
            last_day = end.value - timedelta(days=1)
        elif end is None and duration is not None and duration >= timedelta(days=1):
            last_day = start.value + duration - timedelta(days=1)
        if end is not None and not end.all_day:
            fields.append(f"Start: {format_time(start)}")
            fields.append(f"End: {format_time(end)}")
        elif last_day is not None and last_day > start.value:
            fields.append(f"Start: {format_time(start)} (all day)")
            fields.append(f"End: {last_day.isoformat()} (all day)")
        else:
            fields.append(f"Date: {format_time(start)} (all day)")
    else:
        fields.append(f"{start_label}: {format_time(start)}")
        if end_prop is not None:
            fields.append(f"End: {format_property_time(end_prop)}")
        elif duration:
            fields.append(f"Duration: {format_duration(duration)}")

    if (due := entry.get("DUE")) is not None:
        fields.append(f"Due: {format_property_time(due)}")
    return fields


def _list_time(value: str, prop: Property) -> str:
    if (prop.param("VALUE") or "").upper() == "PERIOD" or "/" in value:
        return _period(value, prop.param("TZID"))
    time = parse_time(value, prop.param("TZID"))
    return format_time(time) if time is not None else value


def _period(value: str, tzid: str | None = None) -> str:
    start_raw, sep, end_raw = value.partition("/")
    start = parse_time(start_raw, tzid)
    if not sep or start is None:
        return value
    if (end := parse_time(end_raw, tzid)) is not None:
        return f"{format_time(start)} to {format_time(end)}"
    if duration := parse_duration(end_raw):
        return f"{format_time(start)} for {format_duration(duration)}"
    return value


def _priority(value: str) -> str:
    try:
        number = int(value)
    except ValueError:
        return ""
    if 1 <= number <= 4:
        return "High"
    if number == 5:
        return "Medium"
    if 6 <= number <= 9:
        return "Low"
    return ""


def _conferences(entry: Component) -> list[str]:
    for name in _CONFERENCE_PROPERTIES:
        conferences: list[str] = []
        for prop in entry.get_all(name):
            uri = prop.value.strip()
            label = _one_line(prop.param("LABEL") or "")
            text = f"{uri} ({label})" if label and uri else uri or label
            if text and text not in conferences:
                conferences.append(text)
        if conferences:
            return conferences
    return []


def _quote_display_name(name: str) -> str:
    """Quote a display name as the email backend does, so it parses back as itself."""
    if _ADDRESS_SPECIALS.search(name):
        return '"' + _ADDRESS_ESCAPES.sub(r"\\\g<0>", name) + '"'
    return name


def _address(prop: Property) -> str:
    address = _MAILTO.sub("", prop.value.strip())
    name = _one_line(prop.param("CN") or "")
    if name and address and name.lower() != address.lower():
        return f"{_quote_display_name(name)} <{address}>"
    return name or address


def _attendee(prop: Property) -> str:
    details: list[str] = []
    user_type = (prop.param("CUTYPE") or "").upper()
    if user_type in _USER_TYPES:
        details.append(_USER_TYPES[user_type])
    if role := (prop.param("ROLE") or "").upper():
        details.append(_ROLES.get(role, _words(role).lower()))
    if participation := (prop.param("PARTSTAT") or "").upper():
        details.append(_PARTICIPATION.get(participation, _words(participation).lower()))
    text = _address(prop)
    return f"{text} ({', '.join(details)})" if details else text


def _alarm(alarm: Component, owner: str) -> str | None:
    action = alarm.text("ACTION").strip().upper()
    if action == "NONE":
        return None
    trigger = alarm.get("TRIGGER")
    if trigger is not None and trigger.value.strip() == _APPLE_PLACEHOLDER_TRIGGER:
        return None
    text = _ALARM_ACTIONS.get(action, _words(action) if action else "Reminder")
    if trigger is not None:
        text += f", {_trigger(trigger, owner)}"
    try:
        repeat = int(alarm.text("REPEAT"))
    except ValueError:
        repeat = 0
    interval = parse_duration(alarm.text("DURATION"))
    if repeat > 0 and interval:
        times = "once" if repeat == 1 else f"{repeat} times"
        text += f", repeated {times} every {format_duration(interval)}"
    return text


def _trigger(trigger: Property, owner: str) -> str:
    offset = parse_duration(trigger.value)
    if offset is None:
        return f"at {format_property_time(trigger)}"
    related_to_end = (trigger.param("RELATED") or "").upper() == "END"
    anchor = ("due" if owner == "VTODO" else "end") if related_to_end else "start"
    if not offset:
        return f"at {anchor}"
    direction = "before" if offset < timedelta(0) else "after"
    return f"{format_duration(offset)} {direction} {anchor}"


def _attachment(prop: Property) -> str:
    """Describe an attachment; inline binary content is never decoded or copied."""
    filename = next(
        (_one_line(prop.params[p]) for p in _FILENAME_PARAMS if prop.param(p)), ""
    )
    media_type = (prop.param("FMTTYPE") or "").strip()
    inline = (prop.param("VALUE") or "").upper() == "BINARY" or (
        prop.param("ENCODING") or ""
    ).upper() == "BASE64"
    name = filename or ("Embedded attachment" if inline else "")
    label = f"{name} ({media_type})" if name and media_type else name
    if inline:
        return label
    uri = prop.value.strip()
    if not name:
        return f"{uri} ({media_type})" if media_type else uri
    return f"{label}: {uri}"


def _free_busy_periods(entry: Component) -> list[str]:
    periods: list[str] = []
    for prop in entry.get_all("FREEBUSY"):
        kind = (prop.param("FBTYPE") or "BUSY").upper()
        label = _FREE_BUSY_TYPES.get(kind, _words(kind))
        periods.extend(f"{label}: {_period(value)}" for value in prop.values)
    return periods
