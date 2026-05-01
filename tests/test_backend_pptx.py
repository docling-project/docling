from pathlib import Path

import pytest
from pptx import Presentation
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Inches, Pt

from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult, DoclingDocument
from docling.document_converter import DocumentConverter

from .test_data_gen_flag import GEN_TEST_DATA
from .verify_utils import verify_document, verify_export

GENERATE = GEN_TEST_DATA


def get_pptx_paths():
    # Define the directory you want to search
    directory = Path("./tests/data/pptx/")

    # List all PPTX files in the directory and its subdirectories
    pptx_files = sorted(directory.rglob("*.pptx"))
    return pptx_files


def get_converter():
    converter = DocumentConverter(allowed_formats=[InputFormat.PPTX])

    return converter


def test_e2e_pptx_conversions():
    pptx_paths = get_pptx_paths()
    converter = get_converter()

    for pptx_path in pptx_paths:
        # print(f"converting {pptx_path}")

        gt_path = (
            pptx_path.parent.parent / "groundtruth" / "docling_v2" / pptx_path.name
        )

        conv_result: ConversionResult = converter.convert(pptx_path)

        doc: DoclingDocument = conv_result.document

        pred_md: str = doc.export_to_markdown()
        assert verify_export(pred_md, str(gt_path) + ".md", GENERATE), "export to md"

        pred_itxt: str = doc._export_to_indented_text(
            max_text_len=70, explicit_tables=False
        )
        assert verify_export(pred_itxt, str(gt_path) + ".itxt", GENERATE), (
            "export to indented-text"
        )

        assert verify_document(doc, str(gt_path) + ".json", GENERATE), (
            "document document"
        )


def test_pptx_unrecognized_shape_type():
    """PPTX with a <p:sp> that has no geometry should not crash.

    python-pptx raises NotImplementedError from Shape.shape_type for shapes
    that aren't placeholders, autoshapes, textboxes, or freeforms. The
    backend should skip the unrecognized shape gracefully and still extract
    text from the rest of the presentation.

    Ref: https://github.com/docling-project/docling/issues/3308
    """
    converter = get_converter()
    pptx_path = Path("./tests/data/pptx/powerpoint_unrecognized_shape.pptx")

    conv_result: ConversionResult = converter.convert(pptx_path)
    doc: DoclingDocument = conv_result.document

    pred_md = doc.export_to_markdown()

    # Normal slide content should still be extracted
    assert "Q3 Revenue Summary" in pred_md
    assert "Enterprise segment" in pred_md
    assert "Key Metrics" in pred_md
    assert "Next Steps" in pred_md


def test_pptx_malformed_picture_shapes():
    """PPTX with malformed <p:pic> shapes should not crash conversion.

    python-pptx's shape.image accessor raises three distinct exceptions on
    picture shapes that slip past other tools' parsers (Keynote/Google Drive
    open these files fine): InvalidXmlError when <p:blipFill> is missing,
    KeyError when <a:blip r:embed> points at an unknown relationship, and
    AttributeError when the embedded part's content-type isn't an image.

    The backend should skip each malformed picture with a warning and still
    extract text from the slides.
    """
    converter = get_converter()
    pptx_path = Path("./tests/data/pptx/powerpoint_malformed_pictures.pptx")

    with pytest.warns(UserWarning, match="Skipping malformed picture shape"):
        conv_result: ConversionResult = converter.convert(pptx_path)

    doc: DoclingDocument = conv_result.document

    pred_md = doc.export_to_markdown()
    assert "Slide With Missing BlipFill" in pred_md
    assert "Slide With Dangling Rel" in pred_md
    assert "Slide With Wrong Content Type" in pred_md


def test_pptx_page_range():
    converter = get_converter()
    pptx_path = Path("./tests/data/pptx/powerpoint_sample.pptx")

    conv_result: ConversionResult = converter.convert(pptx_path, page_range=(2, 2))

    assert conv_result.input.page_count == 3
    assert conv_result.document.num_pages() == 1
    assert list(conv_result.document.pages.keys()) == [2]

    pred_md = conv_result.document.export_to_markdown()
    assert "Second slide title" in pred_md
    assert "Test Table Slide" not in pred_md
    assert "List item4" not in pred_md


def test_pptx_split_list_textboxes_follow_visual_order(tmp_path):
    """Visually ordered subheadings should keep their own following bullets."""

    def add_textbox(slide, left, top, width, height, text, font_size=24):
        textbox = slide.shapes.add_textbox(
            Inches(left), Inches(top), Inches(width), Inches(height)
        )
        text_frame = textbox.text_frame
        text_frame.clear()
        paragraph = text_frame.paragraphs[0]
        paragraph.text = text
        paragraph.font.size = Pt(font_size)
        return textbox

    def mark_as_bullet(paragraph):
        paragraph_properties = paragraph._p.get_or_add_pPr()
        bullet = OxmlElement("a:buChar")
        bullet.set("char", "\u2022")
        paragraph_properties.insert(0, bullet)

    def add_bullet_textbox(slide, left, top, width, height, items):
        textbox = slide.shapes.add_textbox(
            Inches(left), Inches(top), Inches(width), Inches(height)
        )
        text_frame = textbox.text_frame
        text_frame.clear()

        for index, item in enumerate(items):
            paragraph = (
                text_frame.paragraphs[0] if index == 0 else text_frame.add_paragraph()
            )
            paragraph.text = item
            paragraph.font.size = Pt(18)
            mark_as_bullet(paragraph)

        return textbox

    presentation = Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[6])

    add_textbox(slide, 3.0, 0.4, 4.0, 0.5, "Open-Source Software", 32)
    add_textbox(slide, 4.6, 1.4, 2.5, 0.4, "Introduction", 20)
    add_textbox(slide, 0.9, 1.5, 3.0, 0.4, "Key Benefits:", 22)
    add_bullet_textbox(
        slide,
        1.2,
        2.1,
        8.0,
        1.6,
        [
            "Cost effective",
            "Transparent community",
        ],
    )
    # Add this textbox before its subheading to mimic PPTX creation/z-order that
    # does not match the visual reading order.
    add_bullet_textbox(
        slide,
        1.2,
        5.2,
        8.0,
        1.2,
        [
            "Community support can vary",
            "Maintenance requires expertise",
        ],
    )
    add_textbox(slide, 0.9, 4.6, 6.0, 0.4, "Considerations:", 22)

    pptx_path = tmp_path / "split_list_textboxes.pptx"
    presentation.save(pptx_path)

    converter = get_converter()
    conv_result: ConversionResult = converter.convert(pptx_path)
    pred_md = conv_result.document.export_to_markdown()

    assert pred_md.index("Key Benefits:") < pred_md.index("Cost effective")
    assert pred_md.index("Transparent community") < pred_md.index("Considerations:")
    assert pred_md.index("Considerations:") < pred_md.index("Community support")
    assert pred_md.index("Community support") < pred_md.index(
        "Maintenance requires expertise"
    )
