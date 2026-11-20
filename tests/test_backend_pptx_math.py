# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from io import BytesIO

import pytest
from docling_core.types.doc import DocItemLabel, DoclingDocument, ListItem
from lxml import etree
from pptx import Presentation
from pptx.util import Inches

from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.document_converter import DocumentConverter

_NAMESPACES = (
    'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
    'xmlns:a14="http://schemas.microsoft.com/office/drawing/2010/main" '
    'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" '
    'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006"'
)
_EQUATION = "<a14:m><m:oMath><m:r><m:t>E=mc^2</m:t></m:r></m:oMath></a14:m>"


def _convert_paragraph(xml: str, *, title: bool = False) -> DoclingDocument:
    presentation = Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[0 if title else 6])
    shape = (
        slide.shapes.title
        if title
        else slide.shapes.add_textbox(Inches(1), Inches(1), Inches(5), Inches(2))
    )
    paragraph = shape.text_frame.paragraphs[0]._element
    paragraph.getparent().replace(
        paragraph, etree.fromstring(f"<a:p {_NAMESPACES}>{xml}</a:p>")
    )
    stream = BytesIO()
    presentation.save(stream)
    stream.seek(0)
    return (
        DocumentConverter(allowed_formats=[InputFormat.PPTX])
        .convert(DocumentStream(name="equations.pptx", stream=stream))
        .document
    )


@pytest.mark.parametrize("wrapped", [False, True])
def test_equation_only_shape_is_converted_once(wrapped: bool) -> None:
    xml = _EQUATION
    if wrapped:
        xml = (
            '<mc:AlternateContent><mc:Choice Requires="a14">'
            f"{xml}</mc:Choice><mc:Fallback>"
            "<a:r><a:t>Fallback equation</a:t></a:r>"
            "</mc:Fallback></mc:AlternateContent>"
        )

    doc = _convert_paragraph(xml)

    assert [(item.label, item.text) for item in doc.texts] == [
        (DocItemLabel.FORMULA, "E=mc^2")
    ]
    assert "E=mc^2" in doc.export_to_markdown()
    assert "Fallback" not in doc.export_to_markdown()
    assert doc.texts[0].prov[0].page_no == 1


@pytest.mark.parametrize("listed", [False, True])
def test_inline_equation_preserves_text_order_and_list_parent(listed: bool) -> None:
    properties = '<a:pPr><a:buChar char="•"/></a:pPr>' if listed else ""
    doc = _convert_paragraph(
        properties
        + "<a:r><a:t>Energy is </a:t></a:r>"
        + _EQUATION
        + "<a:r><a:t> in this model.</a:t></a:r>"
    )

    formulas = [item for item in doc.texts if item.label == DocItemLabel.FORMULA]
    assert [item.text for item in formulas] == ["E=mc^2"]
    markdown = doc.export_to_markdown()
    assert markdown.index("Energy") < markdown.index("E=mc^2") < markdown.index("model")
    if listed:
        assert any(isinstance(item, ListItem) for item in doc.texts)
        assert markdown.startswith("- ")


def test_unsupported_compatibility_choice_uses_fallback_text() -> None:
    doc = _convert_paragraph(
        '<mc:AlternateContent xmlns:future="https://example.com/unsupported">'
        '<mc:Choice Requires="future">'
        + _EQUATION
        + "</mc:Choice><mc:Fallback><a:r><a:t>Fallback text</a:t></a:r>"
        "</mc:Fallback></mc:AlternateContent>"
    )
    assert [item.text for item in doc.texts] == ["Fallback text"]


def test_fraction_uses_the_shared_omml_converter() -> None:
    doc = _convert_paragraph(
        "<a14:m><m:oMathPara><m:oMath><m:f>"
        "<m:num><m:r><m:t>a</m:t></m:r></m:num>"
        "<m:den><m:r><m:t>b</m:t></m:r></m:den>"
        "</m:f></m:oMath></m:oMathPara></a14:m>"
    )
    assert [(item.label, item.text) for item in doc.texts] == [
        (DocItemLabel.FORMULA, r"\frac{a}{b}")
    ]


def test_inline_equation_retains_the_title_label() -> None:
    doc = _convert_paragraph("<a:r><a:t>Energy </a:t></a:r>" + _EQUATION, title=True)
    title = next(item for item in doc.texts if item.label == DocItemLabel.TITLE)
    assert title.children
    assert "Energy" in doc.export_to_markdown()
    assert "E=mc^2" in doc.export_to_markdown()


def _equation(text: str) -> str:
    return f"<a14:m><m:oMath><m:r><m:t>{text}</m:t></m:r></m:oMath></a14:m>"


@pytest.mark.parametrize("listed", [False, True])
def test_whitespace_between_inline_equations_is_kept(listed: bool) -> None:
    """Only the block edges are trimmed; the space between two formulas is text."""
    properties = '<a:pPr><a:buChar char="•"/></a:pPr>' if listed else ""
    doc = _convert_paragraph(
        properties
        + "<a:r><a:t> Let </a:t></a:r>"
        + _equation("a")
        + "<a:r><a:t> </a:t></a:r>"
        + _equation("b")
        + "<a:r><a:t> hold. </a:t></a:r>"
    )

    runs = [
        item.text
        for item in doc.texts
        if item.label in (DocItemLabel.TEXT, DocItemLabel.FORMULA)
    ]
    assert runs == ["Let ", "a", " ", "b", " hold."]
    assert ("- " if listed else "") + "Let $a$ $b$ hold." in doc.export_to_markdown()


def _text_box(paragraphs: str) -> str:
    return (
        '<p:sp xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
        '<p:nvSpPr><p:cNvPr id="9" name="TextBox 8"/><p:cNvSpPr txBox="1"/><p:nvPr/>'
        "</p:nvSpPr><p:spPr><a:xfrm>"
        '<a:off x="914400" y="914400"/><a:ext cx="4572000" cy="914400"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr>'
        f"<p:txBody><a:bodyPr/><a:lstStyle/>{paragraphs}</p:txBody></p:sp>"
    )


@pytest.mark.parametrize("grouped", [False, True])
def test_equation_shape_wrapped_in_alternate_content(grouped: bool) -> None:
    """PowerPoint wraps a whole shape that holds an equation in mc:AlternateContent.

    The mc:Choice holds the text box and mc:Fallback a copy of it filled with a
    picture of the equation (MS-ODRAWXML, Math). python-pptx does not list such
    a shape, so the text box was dropped with all of its text.
    """
    choice = _text_box(
        "<a:p><a:r><a:t>Energy is </a:t></a:r>"
        + _EQUATION
        + "<a:r><a:t> in this model.</a:t></a:r></a:p>"
        "<a:p><a:r><a:t>Second paragraph</a:t></a:r></a:p>"
    )
    fallback = _text_box("<a:p><a:r><a:t>Fallback picture</a:t></a:r></a:p>")
    alternate = etree.fromstring(
        f"<mc:AlternateContent {_NAMESPACES}>"
        f'<mc:Choice Requires="a14">{choice}</mc:Choice>'
        f"<mc:Fallback>{fallback}</mc:Fallback></mc:AlternateContent>"
    )
    presentation = Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[6])
    container = (
        slide.shapes.add_group_shape()._element if grouped else slide.shapes._spTree
    )
    container.append(alternate)
    stream = BytesIO()
    presentation.save(stream)
    stream.seek(0)

    doc = (
        DocumentConverter(allowed_formats=[InputFormat.PPTX])
        .convert(DocumentStream(name="equations.pptx", stream=stream))
        .document
    )

    assert doc.export_to_markdown() == (
        "Energy is $E=mc^2$ in this model.\n\nSecond paragraph"
    )
