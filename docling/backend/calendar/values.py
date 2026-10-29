# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Turn iCalendar date, duration and recurrence values into readable text.

Names of days and months are spelled out here rather than taken from
``strftime``, whose output depends on the locale and the platform.
"""

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from docling.backend.calendar.parser import Property

_DATE_TIME = re.compile(
    r"(\d{4})-?(\d{2})-?(\d{2})"
    r"(?:T(\d{2}):?(\d{2})(?::?(\d{2}))?(?:\.\d+)?(Z)?)?",
    re.IGNORECASE,
)
_DURATION = re.compile(
    r"([+-])?P(?:(\d+)W)?(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?",
    re.IGNORECASE,
)
_BYDAY = re.compile(r"([+-]?\d{1,2})?(MO|TU|WE|TH|FR|SA|SU)", re.IGNORECASE)

_WEEKDAYS = {
    "MO": "Monday",
    "TU": "Tuesday",
    "WE": "Wednesday",
    "TH": "Thursday",
    "FR": "Friday",
    "SA": "Saturday",
    "SU": "Sunday",
}
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
_ORDINAL_WORDS = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth"}
# FREQ value: (adverb for an interval of 1, unit for longer intervals)
_FREQUENCIES = {
    "SECONDLY": ("Every second", "seconds"),
    "MINUTELY": ("Every minute", "minutes"),
    "HOURLY": ("Hourly", "hours"),
    "DAILY": ("Daily", "days"),
    "WEEKLY": ("Weekly", "weeks"),
    "MONTHLY": ("Monthly", "months"),
    "YEARLY": ("Yearly", "years"),
}
_RRULE_PARTS = frozenset(
    {"FREQ", "INTERVAL", "COUNT", "UNTIL", "BYDAY", "BYMONTHDAY", "BYMONTH", "WKST"}
)


@dataclass
class CalendarTime:
    """A DATE or DATE-TIME value with the time zone label it was written with.

    Attributes:
        value: A ``date`` for a DATE value, a ``datetime`` for a DATE-TIME.
        tz: ``"UTC"``, the TZID as written, or ``None`` for a floating time
            or a date.
    """

    value: date | datetime
    tz: str | None = None

    @property
    def all_day(self) -> bool:
        """Whether the value is a date without a time of day."""
        return not isinstance(self.value, datetime)


def parse_time(raw: str, tzid: str | None = None) -> CalendarTime | None:
    """Read a DATE or DATE-TIME value.

    The shape of the value decides between the two, as producers often leave
    out ``VALUE=DATE``. A trailing ``Z`` wins over a TZID.

    Args:
        raw: The value, for example ``20261020`` or ``20261020T090000Z``.
        tzid: The TZID parameter of the property, if any.

    Returns:
        The parsed value, or ``None`` if the text is not a valid date or
        date-time.
    """
    match = _DATE_TIME.fullmatch(raw.strip())
    if match is None:
        return None
    year, month, day, hour, minute, second, utc = match.groups()
    try:
        if hour is None:
            return CalendarTime(date(int(year), int(month), int(day)))
        value = datetime(
            int(year), int(month), int(day), int(hour), int(minute), int(second or 0)
        )
    except ValueError:
        return None
    return CalendarTime(value, "UTC" if utc else (tzid or None))


def property_time(prop: Property) -> CalendarTime | None:
    """Read the date or date-time value of a property, with its TZID.

    Args:
        prop: A property such as DTSTART, DTEND or DUE.

    Returns:
        The parsed value, or ``None`` if it is not a valid date or date-time.
    """
    return parse_time(prop.value, prop.param("TZID"))


def format_time(time: CalendarTime) -> str:
    """Format a date or date-time for reading, for example ``2026-10-20 09:00 UTC``.

    Seconds are shown only when they are not zero.

    Args:
        time: The value to format.

    Returns:
        The date, the time if any, and the time zone label if any.
    """
    value = time.value
    text = f"{value.year:04d}-{value.month:02d}-{value.day:02d}"
    if isinstance(value, datetime):
        text += f" {value.hour:02d}:{value.minute:02d}"
        if value.second:
            text += f":{value.second:02d}"
    return f"{text} {time.tz}" if time.tz else text


def format_property_time(prop: Property) -> str:
    """Format the date or date-time value of a property for reading.

    Args:
        prop: A property such as DTSTART, DTEND or DUE.

    Returns:
        The formatted value, or the raw value if it cannot be parsed.
    """
    time = property_time(prop)
    return format_time(time) if time is not None else prop.value


def parse_duration(raw: str) -> timedelta | None:
    """Read a DURATION value such as ``PT1H30M``, ``P1W`` or ``-PT15M``.

    Args:
        raw: The value, with an optional sign.

    Returns:
        The duration, negative for a leading ``-``, or ``None`` if the text
        is not a valid duration.
    """
    match = _DURATION.fullmatch(raw.strip())
    if match is None or not any(match.groups()[1:]):
        return None
    sign, weeks, days, hours, minutes, seconds = match.groups()
    delta = timedelta(
        weeks=int(weeks or 0),
        days=int(days or 0),
        hours=int(hours or 0),
        minutes=int(minutes or 0),
        seconds=int(seconds or 0),
    )
    return -delta if sign == "-" else delta


def _plural(count: int, unit: str) -> str:
    return f"{count} {unit}" if count == 1 else f"{count} {unit}s"


def format_duration(delta: timedelta) -> str:
    """Spell out a duration, for example ``1 hour 30 minutes``.

    Args:
        delta: The duration. Its sign is ignored.

    Returns:
        The duration in words, or an empty string for zero.
    """
    total = abs(int(delta.total_seconds()))
    days, rest = divmod(total, 86400)
    hours, rest = divmod(rest, 3600)
    minutes, seconds = divmod(rest, 60)
    parts: list[str] = []
    if days and days % 7 == 0 and not rest:
        parts.append(_plural(days // 7, "week"))
    elif days:
        parts.append(_plural(days, "day"))
    for count, unit in ((hours, "hour"), (minutes, "minute"), (seconds, "second")):
        if count:
            parts.append(_plural(count, unit))
    return " ".join(parts)


def join_words(words: list[str]) -> str:
    """Join words as in a sentence, for example ``A, B and C``.

    Args:
        words: The words to join.

    Returns:
        The joined words, or an empty string for an empty list.
    """
    if len(words) <= 1:
        return "".join(words)
    return ", ".join(words[:-1]) + " and " + words[-1]


def _ordinal(number: int) -> str:
    if number % 100 in (11, 12, 13):
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{number}{suffix}"


def _ordinal_word(number: int) -> str:
    if number == -1:
        return "last"
    if number < 0:
        return f"{_ordinal_word(-number)}-to-last"
    return _ORDINAL_WORDS.get(number, _ordinal(number))


def _describe_byday(value: str) -> str | None:
    days: list[str] = []
    for token in value.split(","):
        match = _BYDAY.fullmatch(token.strip())
        if match is None:
            return None
        position, weekday = match.groups()
        name = _WEEKDAYS[weekday.upper()]
        if position is None:
            days.append(name)
        elif int(position) == 0:
            return None
        else:
            days.append(f"the {_ordinal_word(int(position))} {name}")
    return join_words(days)


def _describe_bymonthday(value: str) -> str | None:
    days: list[str] = []
    for token in value.split(","):
        try:
            number = int(token)
        except ValueError:
            return None
        if number == 0 or not -31 <= number <= 31:
            return None
        if number > 0:
            days.append(_ordinal(number))
        elif number == -1:
            days.append("last day")
        else:
            days.append(f"{_ordinal(-number)}-to-last day")
    return "the " + join_words(days)


def _describe_bymonth(value: str) -> str | None:
    months: list[str] = []
    for token in value.split(","):
        try:
            number = int(token)
        except ValueError:
            return None
        if not 1 <= number <= 12:
            return None
        months.append(_MONTHS[number - 1])
    return join_words(months)


def describe_rrule(raw: str) -> str | None:
    """Describe a recurrence rule in words.

    A rule with a part this does not describe is left raw as a whole, since
    a description that drops BYSETPOS or BYHOUR would state a different
    schedule.

    Args:
        raw: The RRULE value, for example ``FREQ=WEEKLY;BYDAY=MO,WE``.

    Returns:
        A description such as ``Weekly on Monday and Wednesday``, or ``None``
        to keep the raw rule.
    """
    parts: dict[str, str] = {}
    for item in raw.split(";"):
        key, sep, value = item.partition("=")
        if not sep or not value:
            return None
        parts[key.strip().upper()] = value.strip()
    if not parts.keys() <= _RRULE_PARTS:
        return None
    frequency = _FREQUENCIES.get(parts.get("FREQ", "").upper())
    if frequency is None:
        return None
    try:
        interval = int(parts.get("INTERVAL", "1"))
    except ValueError:
        return None
    if interval < 1:
        return None
    # WKST only moves the week boundary that a weekly rule with a longer
    # interval counts from.
    if "WKST" in parts and parts["FREQ"].upper() == "WEEKLY" and interval > 1:
        return None

    text = frequency[0] if interval == 1 else f"Every {interval} {frequency[1]}"
    if "BYMONTH" in parts:
        months = _describe_bymonth(parts["BYMONTH"])
        if months is None:
            return None
        text += f" in {months}"
    if "BYDAY" in parts:
        days = _describe_byday(parts["BYDAY"])
        if days is None:
            return None
        text += f" on {days}"
    if "BYMONTHDAY" in parts:
        month_days = _describe_bymonthday(parts["BYMONTHDAY"])
        if month_days is None:
            return None
        text += f" on {month_days}"
    if "COUNT" in parts:
        try:
            count = int(parts["COUNT"])
        except ValueError:
            return None
        text += ", once" if count == 1 else f", {count} times"
    if "UNTIL" in parts:
        until = parse_time(parts["UNTIL"])
        if until is None:
            return None
        text += f", until {format_time(until)}"
    return text
