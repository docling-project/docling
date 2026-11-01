# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Decompressed size limits of the ZIP-based DOCX, XLSX and EPUB backends."""

import struct
import tracemalloc
import zipfile
from io import BytesIO
from pathlib import Path

import pytest

from docling.backend.epub_backend import EpubDocumentBackend
from docling.backend.msexcel_backend import MsExcelDocumentBackend
from docling.backend.msword_backend import MsWordDocumentBackend
from docling.backend.utils.archive_limits import (
    ArchiveLimitError,
    ArchiveLimits,
    check_archive_limits,
    read_member,
)
from docling.datamodel.backend_options import (
    BaseBackendOptions,
    EpubBackendOptions,
    MsExcelBackendOptions,
    MsWordBackendOptions,
)
from docling.datamodel.base_models import ConversionStatus, DocumentStream, InputFormat
from docling.datamodel.document import InputDocument, get_input_rejection_cause
from docling.document_converter import (
    DocumentConverter,
    EpubFormatOption,
    ExcelFormatOption,
    WordFormatOption,
)

_DATA = Path("./tests/data")
_MIB = 1024 * 1024
# The inflated member: tens of MiB of zeros that compress to a few tens of KiB.
_INFLATED_BYTES = 32 * _MIB
_LIMIT_BYTES = 1 * _MIB

_DOCX = _DATA / "docx/sources/Transitional.docx"
_STRICT_DOCX = _DATA / "docx/sources/Strict.docx"
_XLSX = _DATA / "xlsx/sources/xlsx_01.xlsx"
_EPUB = _DATA / "epub/sources/epub_purvis_poetry.epub"

_FORMATS = {
    InputFormat.DOCX: (MsWordDocumentBackend, MsWordBackendOptions, WordFormatOption),
    InputFormat.XLSX: (
        MsExcelDocumentBackend,
        MsExcelBackendOptions,
        ExcelFormatOption,
    ),
    InputFormat.EPUB: (EpubDocumentBackend, EpubBackendOptions, EpubFormatOption),
}

_CASES = [
    pytest.param(InputFormat.DOCX, _DOCX, "word/document.xml", id="docx-document"),
    pytest.param(
        InputFormat.DOCX,
        _DOCX,
        "word/_rels/document.xml.rels",
        id="docx-relationships",
    ),
    pytest.param(InputFormat.DOCX, _STRICT_DOCX, "word/document.xml", id="docx-strict"),
    pytest.param(InputFormat.XLSX, _XLSX, "xl/worksheets/sheet1.xml", id="xlsx-sheet"),
    pytest.param(InputFormat.XLSX, _XLSX, "xl/media/padding.bin", id="xlsx-extra"),
    pytest.param(InputFormat.EPUB, _EPUB, "epub/text/poetry.xhtml", id="epub-content"),
    pytest.param(
        InputFormat.EPUB, _EPUB, "META-INF/container.xml", id="epub-container"
    ),
]


def _inflate(source: Path, member: str, target: Path) -> Path:
    """Copy ``source`` and make ``member`` decompress to ``_INFLATED_BYTES`` zeros."""
    zeros = bytes(_MIB)
    with (
        zipfile.ZipFile(source) as original,
        zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as inflated,
    ):
        for info in original.infolist():
            if info.filename != member:
                inflated.writestr(info, original.read(info))
        with inflated.open(member, "w") as stream:
            for _ in range(_INFLATED_BYTES // _MIB):
                stream.write(zeros)
    return target


def _limits(input_format: InputFormat) -> BaseBackendOptions:
    options_cls = _FORMATS[input_format][1]
    return options_cls(max_file_bytes=_LIMIT_BYTES)


@pytest.mark.parametrize(("input_format", "source", "member"), _CASES)
@pytest.mark.parametrize("as_stream", [False, True], ids=["path", "stream"])
def test_oversized_member_is_rejected_before_it_is_decompressed(
    tmp_path: Path,
    input_format: InputFormat,
    source: Path,
    member: str,
    as_stream: bool,
) -> None:
    inflated = _inflate(source, member, tmp_path / f"inflated{source.suffix}")
    path_or_stream = BytesIO(inflated.read_bytes()) if as_stream else inflated
    backend_cls = _FORMATS[input_format][0]

    tracemalloc.start()
    try:
        in_doc = InputDocument(
            path_or_stream=path_or_stream,
            format=input_format,
            backend=backend_cls,
            backend_options=_limits(input_format),
            filename=inflated.name,
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert not in_doc.valid
    cause = get_input_rejection_cause(in_doc)
    assert isinstance(cause, ArchiveLimitError)
    assert member in str(cause)
    assert peak < _INFLATED_BYTES // 4


@pytest.mark.parametrize("input_format", list(_FORMATS))
def test_converter_reports_the_exceeded_limit(
    tmp_path: Path, input_format: InputFormat
) -> None:
    source, member = {
        InputFormat.DOCX: (_DOCX, "word/document.xml"),
        InputFormat.XLSX: (_XLSX, "xl/worksheets/sheet1.xml"),
        InputFormat.EPUB: (_EPUB, "epub/text/poetry.xhtml"),
    }[input_format]
    inflated = _inflate(source, member, tmp_path / f"inflated{source.suffix}")
    format_option_cls = _FORMATS[input_format][2]
    converter = DocumentConverter(
        allowed_formats=[input_format],
        format_options={
            input_format: format_option_cls(backend_options=_limits(input_format))
        },
    )

    result = converter.convert(
        DocumentStream(name=inflated.name, stream=BytesIO(inflated.read_bytes())),
        raises_on_error=False,
    )

    assert result.status == ConversionStatus.FAILURE
    assert any("max_file_bytes" in error.error_message for error in result.errors)


def test_ooxml_total_and_member_count_limits_apply_to_the_whole_package() -> None:
    with zipfile.ZipFile(_DOCX) as archive:
        infos = archive.infolist()
        total = sum(info.file_size for info in infos)
        largest = max(info.file_size for info in infos)

    with pytest.raises(ArchiveLimitError, match="max_total_bytes"):
        MsWordDocumentBackend.load_msword_file(
            _DOCX, "hash", ArchiveLimits(largest, total - 1, len(infos))
        )
    with pytest.raises(ArchiveLimitError, match="max_member_count"):
        MsWordDocumentBackend.load_msword_file(
            _DOCX, "hash", ArchiveLimits(largest, total, len(infos) - 1)
        )
    assert (
        MsWordDocumentBackend.load_msword_file(
            _DOCX, "hash", ArchiveLimits(largest, total, len(infos))
        )
        is not None
    )


def _understate_size(archive_bytes: bytes, member: str, declared: int) -> bytes:
    """Rewrite the central directory so ``member`` declares ``declared`` bytes."""
    data = bytearray(archive_bytes)
    offset = 0
    while True:
        offset = data.index(b"PK\x01\x02", offset)
        name_length = struct.unpack_from("<H", data, offset + 28)[0]
        name = bytes(data[offset + 46 : offset + 46 + name_length]).decode()
        if name == member:
            struct.pack_into("<I", data, offset + 24, declared)
            return bytes(data)
        offset += 46


def test_read_never_returns_more_than_the_declared_size() -> None:
    """A member that declares fewer bytes than it holds cannot be read in full.

    The declared sizes are what ``check_archive_limits`` adds up, so this is the
    property that makes the declared-size check bound the actual reads.
    """
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("part.xml", bytes(4 * _MIB))
    patched = _understate_size(buffer.getvalue(), "part.xml", 1024)

    with zipfile.ZipFile(BytesIO(patched)) as archive:
        limits = ArchiveLimits(
            max_file_bytes=4096, max_total_bytes=4096, max_member_count=1
        )
        check_archive_limits(archive, limits)
        with pytest.raises(zipfile.BadZipFile):
            read_member(archive, "part.xml", limits.max_file_bytes)


def test_read_member_rejects_a_declared_size_over_the_limit() -> None:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("part.xml", bytes(4097))

    with zipfile.ZipFile(buffer) as archive:
        assert read_member(archive, "part.xml", 4097) == bytes(4097)
        with pytest.raises(ArchiveLimitError, match=r"part\.xml"):
            read_member(archive, "part.xml", 4096)
