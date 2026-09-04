# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tests use a tiny AFP stream generated here from architectural constants.

The fixture contains no third-party document content and can be distributed
under the repository license.
"""

from io import BytesIO

import pytest

from docling.backend.afp_backend import AfpDocumentBackend, AfpParseError
from docling.datamodel.backend_options import AfpBackendOptions
from docling.datamodel.base_models import ConversionStatus, DocumentStream, InputFormat
from docling.datamodel.document import InputDocument
from docling.datamodel.settings import DocumentLimits
from docling.document_converter import AfpFormatOption, DocumentConverter
from docling.exceptions import DocumentLoadError

BDT = b"\xd3\xa8\xa8"
EDT = b"\xd3\xa9\xa8"
BPG = b"\xd3\xa8\xaf"
EPG = b"\xd3\xa9\xaf"
BPT = b"\xd3\xa8\x9b"
EPT = b"\xd3\xa9\x9b"
PTX = b"\xd3\xee\x9b"
IPD = b"\xd3\xee\xfb"
BPS = b"\xd3\xa8\x5f"
EPS = b"\xd3\xa9\x5f"


def _structured_field(
    identifier: bytes,
    data: bytes = b"",
    *,
    extension: bytes = b"",
    padding: int = 0,
) -> bytes:
    """Build one synthetic MO:DCA structured field."""
    flags = 0
    extension_data = b""
    if extension:
        flags |= 0x01
        extension_data = bytes((len(extension) + 1,)) + extension
    padding_data = b""
    if padding:
        flags |= 0x10
        padding_data = bytes(padding - 1) + bytes((padding,))
    payload = extension_data + data + padding_data
    length = 8 + len(payload)  # X'5A' carriage control is not included.
    return (
        b"\x5a"
        + length.to_bytes(2, "big")
        + identifier
        + bytes((flags, 0, 0))
        + payload
    )


def _trn(text: str, encoding: str = "cp500", chained: bool = False) -> bytes:
    encoded = text.encode(encoding)
    introducer = b"" if chained else b"\x2b\xd3"
    function_type = 0xDB if chained else 0xDA
    return introducer + bytes((len(encoded) + 2, function_type)) + encoded


def _page(*ptoca_parts: bytes, include_image: bool = False) -> bytes:
    fields = [_structured_field(BPG), _structured_field(BPT)]
    fields.extend(_structured_field(PTX, part) for part in ptoca_parts)
    if include_image:
        fields.append(_structured_field(IPD, b"synthetic-image-payload"))
    fields.extend((_structured_field(EPT), _structured_field(EPG)))
    return b"".join(fields)


@pytest.fixture
def synthetic_afp() -> bytes:
    first_control = _trn("Hello AFP")
    first_page = _page(
        first_control[:6],
        first_control[6:] + b"\x2b\xd3\x02\xd8" + _trn("Second line", chained=True),
    )
    second_page = _page(_trn("Page two"))
    return b"".join(
        (
            _structured_field(BDT, "SYNTHAFP".encode("cp500")),
            first_page,
            second_page,
            _structured_field(EDT),
        )
    )


def _backend(
    data: bytes,
    options: AfpBackendOptions | None = None,
    limits: DocumentLimits | None = None,
) -> AfpDocumentBackend:
    options = options or AfpBackendOptions()
    in_doc = InputDocument(
        path_or_stream=BytesIO(data),
        format=InputFormat.AFP,
        filename="synthetic.afp",
        backend=AfpDocumentBackend,
        backend_options=options,
        limits=limits,
    )
    return AfpDocumentBackend(in_doc, BytesIO(data), options)


def test_afp_conversion_preserves_pages_and_extracts_ptoca_text(synthetic_afp: bytes):
    result = DocumentConverter(allowed_formats=[InputFormat.AFP]).convert(
        DocumentStream(name="synthetic.afp", stream=BytesIO(synthetic_afp))
    )

    assert result.status is ConversionStatus.SUCCESS
    assert result.input.format is InputFormat.AFP
    assert result.input.page_count == 2
    assert result.document.origin.mimetype == "application/vnd.ibm.modcap"
    assert sorted(result.document.pages) == [1, 2]
    assert [item.text for item in result.document.texts] == [
        "Hello AFP",
        "Second line",
        "Page two",
    ]
    assert [item.prov[0].page_no for item in result.document.texts] == [1, 1, 2]


def test_afp_is_detected_from_signature_without_extension(synthetic_afp: bytes):
    result = DocumentConverter(allowed_formats=[InputFormat.AFP]).convert(
        DocumentStream(name="print-stream.bin", stream=BytesIO(synthetic_afp))
    )

    assert result.input.format is InputFormat.AFP


def test_afp_page_range_keeps_original_page_number(synthetic_afp: bytes):
    doc = _backend(synthetic_afp, limits=DocumentLimits(page_range=(2, 2))).convert()

    assert sorted(doc.pages) == [2]
    assert [item.text for item in doc.texts] == ["Page two"]
    assert doc.texts[0].prov[0].page_no == 2


def test_afp_page_count_limit_is_enforced(synthetic_afp: bytes):
    result = DocumentConverter(allowed_formats=[InputFormat.AFP]).convert(
        DocumentStream(name="synthetic.afp", stream=BytesIO(synthetic_afp)),
        max_num_pages=1,
        raises_on_error=False,
    )

    assert result.status is ConversionStatus.FAILURE
    assert result.input.page_count == 2
    assert "exceeding the max_num_pages limit of 1" in result.errors[0].error_message


def test_afp_encoding_is_configurable():
    data = b"".join(
        (
            _structured_field(BDT),
            _page(_trn("Olá", encoding="cp037")),
            _structured_field(EDT),
        )
    )
    result = DocumentConverter(
        allowed_formats=[InputFormat.AFP],
        format_options={
            InputFormat.AFP: AfpFormatOption(
                backend_options=AfpBackendOptions(encoding="cp037")
            )
        },
    ).convert(DocumentStream(name="localized.afp", stream=BytesIO(data)))

    assert [item.text for item in result.document.texts] == ["Olá"]


def test_afp_structured_field_extension_and_padding_are_removed():
    data = b"".join(
        (
            _structured_field(BDT),
            _structured_field(BPG),
            _structured_field(BPT),
            _structured_field(PTX, _trn("Extended"), extension=b"\xaa\xbb", padding=3),
            _structured_field(EPT),
            _structured_field(EPG),
            _structured_field(EDT),
        )
    )

    assert [item.text for item in _backend(data).convert().texts] == ["Extended"]


def test_unsupported_afp_image_data_emits_clear_warning():
    data = b"".join(
        (
            _structured_field(BDT),
            _page(_trn("Text remains"), include_image=True),
            _structured_field(EDT),
        )
    )

    with pytest.warns(UserWarning, match=r"Skipped 1 AFP image data .*does not render"):
        doc = _backend(data).convert()

    assert [item.text for item in doc.texts] == ["Text remains"]


def test_unsupported_afp_resource_emits_clear_warning():
    data = b"".join(
        (
            _structured_field(BDT),
            _structured_field(BPS),
            _structured_field(BPT),
            _structured_field(PTX, _trn("Resource text")),
            _structured_field(EPT),
            _structured_field(EPS),
            _page(_trn("Page text")),
            _structured_field(EDT),
        )
    )

    with pytest.warns(UserWarning, match=r"Skipped 1 AFP page-segment resource"):
        doc = _backend(data).convert()

    assert [item.text for item in doc.texts] == ["Page text"]


def test_malformed_structured_field_reports_offset():
    malformed = b"\x5a\x00\x20\xd3\xa8\xa8\x00\x00\x00"

    with pytest.raises(AfpParseError, match=r"byte 0 declares 32 bytes"):
        _backend(malformed)


def test_unknown_afp_codec_is_reported(synthetic_afp: bytes):
    with pytest.raises(DocumentLoadError, match=r"check AfpBackendOptions\.encoding"):
        _backend(synthetic_afp, AfpBackendOptions(encoding="not-a-codec"))
