# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import mimetypes
from io import BytesIO
from pathlib import Path

import pytest
from docling_core.types.doc import DocItemLabel, DoclingDocument

from docling.backend.calendar.values import describe_rrule
from docling.datamodel.base_models import ConversionStatus, DocumentStream, InputFormat
from docling.document_converter import DocumentConverter

from .test_data_gen_flag import GEN_TEST_DATA
from .verify_utils import verify_document, verify_export

GENERATE = GEN_TEST_DATA
pytestmark = pytest.mark.cross_platform


@pytest.fixture(scope="module")
def converter() -> DocumentConverter:
    return DocumentConverter(allowed_formats=[InputFormat.CALENDAR])


def _convert(converter: DocumentConverter, raw: bytes, name: str = "test.ics"):
    return converter.convert(
        DocumentStream(name=name, stream=BytesIO(raw)), raises_on_error=False
    )


def _texts(doc: DoclingDocument) -> list[str]:
    return [item.text for item in doc.texts]


def _calendar(*lines: str, newline: str = "\r\n") -> bytes:
    body = ["BEGIN:VCALENDAR", "VERSION:2.0", *lines, "END:VCALENDAR"]
    return (newline.join(body) + newline).encode("utf-8")


def test_e2e_calendar_conversions(converter):
    directory = Path("./tests/data/calendar/sources/")
    for source in sorted(directory.glob("*.ics")):
        gt_path = source.parent.parent / "groundtruth" / source.name

        doc = converter.convert(source).document

        pred_md = doc.export_to_markdown(escape_html=False)
        assert verify_export(pred_md, str(gt_path) + ".md", generate=GENERATE)

        pred_itxt = doc._export_to_indented_text(max_text_len=70, explicit_tables=False)
        assert verify_export(pred_itxt, str(gt_path) + ".itxt", generate=GENERATE)

        assert verify_document(doc, str(gt_path) + ".json", GENERATE)


def test_inline_attachment_payload_is_not_copied(converter):
    """An inline ATTACH is listed by name; its base64 content stays out of the document."""
    doc = converter.convert(
        Path("./tests/data/calendar/sources/outlook_invite.ics")
    ).document

    assert "budget-draft.txt (text/plain)" in _texts(doc)
    assert "UTQgYnVkZ2V0" not in doc.model_dump_json()


@pytest.mark.parametrize("other_text", ["", "DESCRIPTION:naïve"])
def test_fold_inside_a_utf8_character(converter, other_text):
    """RFC 5545 lets a fold split a multi-byte character, so folds go before decoding.

    Decoding first broke the character: with other UTF-8 in the file it was
    reported as damaged UTF-8, and without it the cp1252 fallback turned
    "Café" into "CafÃ©".
    """
    lines = ["BEGIN:VEVENT", "SUMMARY:Café au lait", other_text, "END:VEVENT"]
    raw = _calendar(*[line for line in lines if line])
    raw = raw.replace("é".encode(), b"\xc3\r\n \xa9")

    result = _convert(converter, raw)

    assert result.status == ConversionStatus.SUCCESS
    assert "Café au lait" in [
        item.text
        for item in result.document.texts
        if item.label == DocItemLabel.SECTION_HEADER
    ]


@pytest.mark.parametrize("newline", ["\r\n", "\n", "\r"])
def test_line_endings_and_folds(converter, newline):
    """Folds join with a space or a tab, also inside a quoted parameter value."""
    raw = _calendar(
        "BEGIN:VEVENT",
        "SUMMARY:Quarterly",
        "\tplanning",
        'ORGANIZER;CN="Martin, Ca',
        ' rla":mailto:carla@example.com',
        "END:VEVENT",
        newline=newline,
    )

    texts = _texts(_convert(converter, raw).document)

    assert "Quarterlyplanning" in texts
    assert 'Organizer: "Martin, Carla" <carla@example.com>' in texts


def test_utf16_file_is_unfolded_after_decoding(converter):
    raw = _calendar("BEGIN:VEVENT", "SUMMARY:Résumé re", " view", "END:VEVENT")
    raw = raw.decode("utf-8").encode("utf-16")

    texts = _texts(_convert(converter, raw, name="calendar").document)

    assert "Résumé review" in texts


def test_missing_end_does_not_swallow_the_next_entry(converter):
    """A BEGIN that the open component cannot hold closes it first.

    Without that rule, a VTIMEZONE with no END would take the event that
    follows as a child, and the event would not be rendered.
    """
    raw = _calendar(
        "BEGIN:VTIMEZONE",
        "TZID:Europe/Berlin",
        "BEGIN:VEVENT",
        "SUMMARY:Kept event",
        "END:VTODO",
        "END:VEVENT",
        "BEGIN:VEVENT",
        "SUMMARY:Truncated event",
    )
    raw = raw.replace(b"END:VCALENDAR\r\n", b"")

    texts = _texts(_convert(converter, raw).document)

    assert "Kept event" in texts
    assert "Truncated event" in texts


def test_entries_outside_a_calendar_are_rendered(converter):
    raw = b"BEGIN:VEVENT\r\nSUMMARY:Lonely event\r\nEND:VEVENT\r\n"

    assert "Lonely event" in _texts(_convert(converter, raw).document)


def test_empty_calendar_converts_to_an_empty_document(converter):
    result = _convert(converter, _calendar())

    assert result.status == ConversionStatus.SUCCESS
    assert result.document.texts == []


def test_conversion_does_not_depend_on_the_host_mime_table(converter, monkeypatch):
    """CPython's built-in MIME table has no text/calendar; only a system table adds it.

    Without docling-core accepting the type itself, a host with no
    /etc/mime.types could not create the document origin.
    """
    monkeypatch.setattr(mimetypes, "types_map", {})

    result = _convert(converter, _calendar("BEGIN:VEVENT", "SUMMARY:E", "END:VEVENT"))

    assert result.status == ConversionStatus.SUCCESS
    assert result.document.origin.mimetype == "text/calendar"


def test_text_without_calendar_fails(converter):
    result = _convert(converter, b"Meeting on Monday at 10.\r\n")

    assert result.status == ConversionStatus.FAILURE


@pytest.mark.parametrize(
    ("lines", "expected"),
    [
        (["DTSTART;VALUE=DATE:20261020"], ["Date: 2026-10-20 (all day)"]),
        (
            ["DTSTART;VALUE=DATE:20261020", "DTEND;VALUE=DATE:20261021"],
            ["Date: 2026-10-20 (all day)"],
        ),
        (
            ["DTSTART;VALUE=DATE:20261020", "DTEND;VALUE=DATE:20261023"],
            ["Start: 2026-10-20 (all day)", "End: 2026-10-22 (all day)"],
        ),
        (
            ["DTSTART:20261020", "DURATION:P3D"],
            ["Start: 2026-10-20 (all day)", "End: 2026-10-22 (all day)"],
        ),
        (
            ["DTSTART;TZID=Europe/Paris:20261020T090000Z", "DURATION:PT1H30M"],
            ["Start: 2026-10-20 09:00 UTC", "Duration: 1 hour 30 minutes"],
        ),
        (["DTSTART:20261020T090000"], ["Start: 2026-10-20 09:00"]),
        (["DTSTART:2026-13-45"], ["Start: 2026-13-45"]),
    ],
)
def test_event_times(converter, lines, expected):
    """All-day ends are exclusive, Z wins over TZID, and a bad value stays raw."""
    raw = _calendar("BEGIN:VEVENT", "SUMMARY:E", *lines, "END:VEVENT")

    texts = _texts(_convert(converter, raw).document)

    assert texts[1 : 1 + len(expected)] == expected


@pytest.mark.parametrize(
    ("rule", "expected"),
    [
        ("FREQ=WEEKLY;INTERVAL=2;BYDAY=TU", "Every 2 weeks on Tuesday"),
        ("FREQ=MONTHLY;BYDAY=-1FR", "Monthly on the last Friday"),
        ("FREQ=MONTHLY;BYDAY=-2MO", "Monthly on the second-to-last Monday"),
        ("FREQ=YEARLY;BYMONTH=3;BYMONTHDAY=15", "Yearly in March on the 15th"),
        ("FREQ=MONTHLY;BYMONTHDAY=1,-1", "Monthly on the 1st and last day"),
        ("FREQ=DAILY;COUNT=1", "Daily, once"),
        ("FREQ=DAILY;UNTIL=20261231", "Daily, until 2026-12-31"),
        ("FREQ=WEEKLY;WKST=MO;BYDAY=MO", "Weekly on Monday"),
        # Parts a description cannot carry keep the whole rule raw.
        ("FREQ=MONTHLY;BYDAY=MO,TU,WE,TH,FR;BYSETPOS=-1", None),
        ("FREQ=DAILY;BYHOUR=9,17", None),
        ("FREQ=WEEKLY;INTERVAL=2;WKST=SU;BYDAY=MO,SU", None),
        ("FREQ=FORTNIGHTLY", None),
    ],
)
def test_describe_rrule(rule, expected):
    assert describe_rrule(rule) == expected


def test_alarms(converter):
    raw = _calendar(
        "BEGIN:VTODO",
        "SUMMARY:T",
        "DUE:20261031T170000Z",
        "BEGIN:VALARM",
        "ACTION:DISPLAY",
        "TRIGGER;RELATED=END:-PT30M",
        "END:VALARM",
        "BEGIN:VALARM",
        "ACTION:EMAIL",
        "TRIGGER:PT0S",
        "REPEAT:2",
        "DURATION:PT5M",
        "END:VALARM",
        "BEGIN:VALARM",
        "ACTION:NONE",
        "TRIGGER:-PT5M",
        "END:VALARM",
        "END:VTODO",
    )

    texts = _texts(_convert(converter, raw).document)

    assert texts[-2:] == [
        "Display, 30 minutes before due",
        "Email, at start, repeated 2 times every 5 minutes",
    ]
