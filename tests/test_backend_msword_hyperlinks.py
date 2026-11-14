# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest
from docx import Document
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.document_converter import DocumentConverter


def _write_link_document(path: Path, address: str | None) -> Path:
    document = Document()
    paragraph = document.add_paragraph("Before ")
    hyperlink = OxmlElement("w:hyperlink")
    if address is not None:
        relationship = document.part.relate_to(address, RT.HYPERLINK, is_external=True)
        hyperlink.set(qn("r:id"), relationship)
    hyperlink.append(paragraph.add_run("Link text")._r)
    paragraph._p.append(hyperlink)
    document.add_paragraph("After paragraph")
    document.save(path)
    return path


@pytest.mark.parametrize(
    "address, expected_target",
    [
        ("https://[broken", None),
        ("https://broken]", None),
        ("https://[not-ipv6]/", None),
        ("http://exa mple.com/bad url", None),
        ("https://example.org/valid", "https://example.org/valid"),
        ("https://[::1]/guide", "https://[::1]/guide"),
        ("mailto:a@example.org", "mailto:a@example.org"),
        ("https://example.org/guide#section", "https://example.org/guide#section"),
        ("manual[1].docx", "manual[1].docx"),
        ("manual.docx#section", "manual.docx#section"),
        ("", None),
        (None, None),
    ],
)
def test_hyperlink_target_keeps_document_content(
    tmp_path: Path, address: str | None, expected_target: str | None
) -> None:
    source = _write_link_document(tmp_path / "link.docx", address)
    converter = DocumentConverter(allowed_formats=[InputFormat.DOCX])

    result = converter.convert(source)

    assert result.status == ConversionStatus.SUCCESS
    document = result.document
    document.validate_document()
    assert len(document.body.children) == 2
    following = document.body.children[-1].resolve(document)
    assert following.text == "After paragraph"
    assert following.parent.cref == "#/body"
    targets = [str(item.hyperlink) for item in document.texts if item.hyperlink]
    assert targets == ([expected_target] if expected_target is not None else [])
    if expected_target is None:
        assert document.export_to_markdown() == "Before Link text\n\nAfter paragraph"
    else:
        assert document.export_to_markdown() == (
            f"Before [Link text]({expected_target})\n\nAfter paragraph"
        )


def test_malformed_hyperlink_keeps_following_document(tmp_path: Path) -> None:
    malformed = _write_link_document(tmp_path / "malformed.docx", "https://[broken")
    valid = _write_link_document(tmp_path / "valid.docx", "https://example.org/valid")
    converter = DocumentConverter(allowed_formats=[InputFormat.DOCX])

    results = list(converter.convert_all([malformed, valid], raises_on_error=False))

    assert [result.status for result in results] == [
        ConversionStatus.SUCCESS,
        ConversionStatus.SUCCESS,
    ]
    assert [result.document.export_to_markdown() for result in results] == [
        "Before Link text\n\nAfter paragraph",
        "Before [Link text](https://example.org/valid)\n\nAfter paragraph",
    ]
