# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from io import BytesIO
from zipfile import ZipFile

import pytest

from docling.datamodel.base_models import ConversionStatus, DocumentStream, InputFormat
from docling.document_converter import DocumentConverter
from docling.exceptions import ConversionError


def make_zip(*members: tuple[str, bytes]) -> DocumentStream:
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        for name, content in members:
            archive.writestr(name, content)
    return DocumentStream(name="mixed.zip", stream=BytesIO(buffer.getvalue()))


def test_convert_rejects_multiple_results_and_convert_all_yields_them() -> None:
    converter = DocumentConverter(allowed_formats=[InputFormat.MD, InputFormat.HTML])

    with pytest.raises(ConversionError, match="Use convert_all"):
        converter.convert(
            make_zip(
                ("notes.md", b"# Hello"),
                ("page.html", b"<html><body>Hello</body></html>"),
            ),
            raises_on_error=False,
        )

    results = list(
        converter.convert_all(
            [
                make_zip(
                    ("notes.md", b"# Hello"),
                    ("page.html", b"<html><body>Hello</body></html>"),
                )
            ],
            raises_on_error=False,
        )
    )

    assert [result.input.file.as_posix() for result in results] == [
        "mixed.zip/notes.md",
        "mixed.zip/page.html",
    ]
    assert [result.input.format for result in results] == [
        InputFormat.MD,
        InputFormat.HTML,
    ]
    assert all(result.status == ConversionStatus.SUCCESS for result in results)


def test_zip_size_limit_skips_rejected_members_without_charging_the_budget() -> None:
    source = make_zip(
        ("oversized.md", b"x" * 71),
        ("first.md", b"a" * 50),
        ("over_budget.md", b"b" * 30),
        ("last.md", b"c" * 10),
    )
    converter = DocumentConverter(allowed_formats=[InputFormat.MD])

    results = list(
        converter.convert_all([source], max_file_size=70, raises_on_error=False)
    )

    assert [result.input.file.name for result in results] == [
        "oversized.md",
        "first.md",
        "over_budget.md",
        "last.md",
    ]
    assert [result.status for result in results] == [
        ConversionStatus.FAILURE,
        ConversionStatus.SUCCESS,
        ConversionStatus.FAILURE,
        ConversionStatus.SUCCESS,
    ]


def test_unsupported_only_zip_returns_a_failure_result() -> None:
    source = make_zip(("notes.bin", b"unknown format"))
    converter = DocumentConverter(allowed_formats=[InputFormat.MD])

    results = list(converter.convert_all([source], raises_on_error=False))

    assert len(results) == 1
    assert results[0].status == ConversionStatus.FAILURE
    assert results[0].input.file.as_posix() == "mixed.zip/notes.bin"


def test_convert_all_can_continue_after_an_unsupported_member() -> None:
    source = make_zip(
        ("notes.bin", b"unknown format"),
        ("good.md", b"# Good"),
    )
    converter = DocumentConverter(allowed_formats=[InputFormat.MD])

    results = list(converter.convert_all([source], raises_on_error=False))

    assert [result.status for result in results] == [
        ConversionStatus.FAILURE,
        ConversionStatus.SUCCESS,
    ]
    assert results[1].input.file.as_posix() == "mixed.zip/good.md"


def test_convert_all_raises_on_first_failed_member_by_default() -> None:
    source = make_zip(
        ("notes.bin", b"unknown format"),
        ("good.md", b"# Good"),
    )
    converter = DocumentConverter(allowed_formats=[InputFormat.MD])

    with pytest.raises(ConversionError, match=r"notes\.bin"):
        list(converter.convert_all([source]))


def test_macos_archive_metadata_is_skipped() -> None:
    """Finder adds ``__MACOSX/`` AppleDouble files and ``.DS_Store`` to a ZIP.

    They are not documents, so they are not converted and do not fail the batch.
    """
    source = make_zip(
        ("docs/notes.md", b"# Notes"),
        ("docs/.DS_Store", b"\x00\x00\x00\x01Bud1"),
        ("__MACOSX/docs/._notes.md", b"\x00\x05\x16\x07\x00\x02\x00\x00"),
    )
    converter = DocumentConverter(allowed_formats=[InputFormat.MD])

    results = list(converter.convert_all([source]))

    assert [result.input.file.as_posix() for result in results] == [
        "mixed.zip/docs/notes.md"
    ]
    assert results[0].status == ConversionStatus.SUCCESS


def test_member_with_an_empty_name_does_not_abort_the_batch() -> None:
    source = make_zip((".", b"# Dot"), ("good.md", b"# Good"))
    converter = DocumentConverter(allowed_formats=[InputFormat.MD])

    results = list(converter.convert_all([source], raises_on_error=False))

    assert results[-1].input.file.as_posix() == "mixed.zip/good.md"
    assert results[-1].status == ConversionStatus.SUCCESS
