# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pytest
from docling_core.types.doc import ContentLayer
from docx import Document

from docling.datamodel.base_models import ConversionStatus, DocumentStream, InputFormat
from docling.document_converter import DocumentConverter


@pytest.mark.parametrize("as_stream", [False, True], ids=["path", "stream"])
@pytest.mark.parametrize(
    ("extension", "main_content_type"),
    [
        ("docm", b"application/vnd.ms-word.document.macroEnabled.main+xml"),
        ("docm", b"application/vnd.ms-word.document.macroenabled.main+xml"),
        (
            "dotx",
            b"application/vnd.openxmlformats-officedocument.wordprocessingml.template.main+xml",
        ),
        ("dotm", b"application/vnd.ms-word.template.macroEnabledTemplate.main+xml"),
    ],
    ids=["docm", "docm-lowercase", "dotx", "dotm"],
)
def test_convert_word_package_variants(
    tmp_path: Path, extension: str, main_content_type: bytes, as_stream: bool
) -> None:
    source = Document()
    source.add_paragraph("Quarterly inspection report")
    source.sections[0].header.paragraphs[0].text = "Operations department"
    docx = BytesIO()
    source.save(docx)

    # These variants use the same WordprocessingML body and relationships.
    path = tmp_path / f"report.{extension}"
    with ZipFile(docx) as original, ZipFile(path, "w") as variant:
        for member in original.infolist():
            content = original.read(member)
            if member.filename == "[Content_Types].xml":
                content = content.replace(
                    b"application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml",
                    main_content_type,
                )
            variant.writestr(member, content)

    input_doc = (
        DocumentStream(name=path.name, stream=BytesIO(path.read_bytes()))
        if as_stream
        else path
    )
    result = DocumentConverter(allowed_formats=[InputFormat.DOCX]).convert(
        input_doc, raises_on_error=False
    )

    assert result.status == ConversionStatus.SUCCESS, result.errors
    assert result.document.export_to_markdown() == "Quarterly inspection report"
    assert [
        item.text
        for item in result.document.texts
        if item.content_layer == ContentLayer.FURNITURE
    ] == ["Operations department"]
