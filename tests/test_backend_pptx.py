from pathlib import Path
from types import SimpleNamespace

import pytest
from docling_core.types.doc import (
    ContentLayer,
    GroupItem,
    PictureClassificationLabel,
    TextItem,
)
from pptx import Presentation
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Inches, Pt

from docling.backend.docx.drawingml.utils import get_libreoffice_cmd
from docling.backend.mspowerpoint_backend import MsPowerpointDocumentBackend
from docling.datamodel.backend_options import MsPowerpointBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult, DoclingDocument
from docling.document_converter import DocumentConverter, PowerpointFormatOption

from .test_data_gen_flag import GEN_TEST_DATA
from .verify_utils import verify_document, verify_export

GENERATE = GEN_TEST_DATA

CHART_PPTX = Path("./tests/data/pptx/sources/pptx_chart.pptx")


@pytest.fixture(scope="module")
def libreoffice_available() -> bool:
    """Return True when a working LibreOffice installation is detected."""
    try:
        return get_libreoffice_cmd(raise_if_unavailable=True) is not None
    except Exception:
        return False


def get_pptx_paths():
    # Define the directory you want to search
    directory = Path("./tests/data/pptx/sources/")

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

        gt_path = pptx_path.parent.parent / "groundtruth" / pptx_path.name

        conv_result: ConversionResult = converter.convert(pptx_path)

        doc: DoclingDocument = conv_result.document

        included_content_layers = (
            set(ContentLayer) if gt_path.stem in "powerpoint_comments" else None
        )
        pred_md: str = doc.export_to_markdown(
            compact_tables=True,
            included_content_layers=included_content_layers,
        )
        assert verify_export(
            pred_md,
            str(gt_path) + ".md",
            GENERATE,
        ), "export to md"

        pred_itxt: str = doc._export_to_indented_text(
            max_text_len=70, explicit_tables=False
        )
        assert verify_export(pred_itxt, str(gt_path) + ".itxt", GENERATE), (
            "export to indented-text"
        )

        assert verify_document(doc, str(gt_path) + ".json", GENERATE), (
            "document document"
        )


def test_comments_extraction() -> None:
    """Test comprehensive comment extraction including metadata, authors, and slide distribution."""

    converter = get_converter()
    path = Path("./tests/data/pptx/sources/powerpoint_comments.pptx")
    doc: DoclingDocument = converter.convert(path).document

    assert doc.num_pages() == 3, f"Expected 3 slides, got {doc.num_pages()}"

    # Comment groups: 4 total (2 on slide 1, 0 on slide 2, 2 on slide 3)
    comment_groups = [
        g
        for g in doc.groups
        if isinstance(g, GroupItem) and g.name.startswith("comment-")
    ]
    assert len(comment_groups) == 4, (
        f"Expected 4 comment groups, got {len(comment_groups)}"
    )

    assert all(g.content_layer == ContentLayer.NOTES for g in comment_groups), (
        "All comment groups should be in NOTES content layer"
    )

    slide1_comments = [g for g in comment_groups if "slide1" in g.name]
    slide2_comments = [g for g in comment_groups if "slide2" in g.name]
    slide3_comments = [g for g in comment_groups if "slide3" in g.name]
    assert len(slide1_comments) == 2, (
        f"Expected 2 comments on slide 1, got {len(slide1_comments)}"
    )
    assert len(slide2_comments) == 0, (
        f"Expected 0 comments on slide 2, got {len(slide2_comments)}"
    )
    assert len(slide3_comments) == 2, (
        f"Expected 2 comments on slide 3, got {len(slide3_comments)}"
    )

    comment_texts = [
        t.text
        for t in doc.texts
        if isinstance(t, TextItem) and t.content_layer == ContentLayer.NOTES
    ]
    assert len(comment_texts) == 4, (
        f"Expected 4 comment texts, got {len(comment_texts)}"
    )

    assert all("[author:" in text for text in comment_texts), (
        "All comments should have author metadata"
    )

    all_text = " ".join(comment_texts)
    assert "John Reviewer (JR)" in all_text, "Expected John Reviewer (JR) in comments"
    assert "Jane Smith (JS)" in all_text, "Expected Jane Smith (JS) in comments"
    assert "sample reviewer comment" in all_text, "Expected original comment text"
    assert "sample response" in all_text, "Expected reply comment text"

    jr_comments = [t for t in comment_texts if "John Reviewer (JR)" in t]
    js_comments = [t for t in comment_texts if "Jane Smith (JS)" in t]
    assert len(jr_comments) == 1, f"Expected 1 comment from JR, got {len(jr_comments)}"
    assert len(js_comments) == 3, f"Expected 3 comments from JS, got {len(js_comments)}"


def test_comments_respect_page_range() -> None:
    """Test that comments are only extracted for slides within page_range."""
    path = Path("./tests/data/pptx/sources/powerpoint_comments.pptx")
    converter = get_converter()

    doc: DoclingDocument = converter.convert(path, page_range=(1, 1)).document

    comment_groups = [g for g in doc.groups if g.name.startswith("comment-")]
    assert len(comment_groups) == 2, (
        f"Expected 2 comment groups from slide 1, got {len(comment_groups)}"
    )

    assert all("slide1" in g.name for g in comment_groups), (
        "Comments should only be from slide 1 when page_range is (1,1)"
    )

    doc3: DoclingDocument = converter.convert(path, page_range=(3, 3)).document

    comment_groups3 = [g for g in doc3.groups if g.name.startswith("comment-")]
    assert len(comment_groups3) == 2, (
        f"Expected 2 comment groups from slide 3, got {len(comment_groups3)}"
    )

    assert all("slide3" in g.name for g in comment_groups3), (
        "Comments should only be from slide 3 when page_range is (3,3)"
    )

    doc2: DoclingDocument = converter.convert(path, page_range=(2, 2)).document
    comment_groups2 = [g for g in doc2.groups if g.name.startswith("comment-")]
    assert len(comment_groups2) == 0, (
        f"Expected 0 comment groups from slide 2, got {len(comment_groups2)}"
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
    pptx_path = Path("./tests/data/pptx/sources/powerpoint_unrecognized_shape.pptx")

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
    pptx_path = Path("./tests/data/pptx/sources/powerpoint_malformed_pictures.pptx")

    with pytest.warns(UserWarning, match="Skipping malformed picture shape"):
        conv_result: ConversionResult = converter.convert(pptx_path)

    doc: DoclingDocument = conv_result.document

    pred_md = doc.export_to_markdown()
    assert "Slide With Missing BlipFill" in pred_md
    assert "Slide With Dangling Rel" in pred_md
    assert "Slide With Wrong Content Type" in pred_md


def test_pptx_page_range():
    converter = get_converter()
    pptx_path = Path("./tests/data/pptx/sources/powerpoint_sample.pptx")

    conv_result: ConversionResult = converter.convert(pptx_path, page_range=(2, 2))

    assert conv_result.input.page_count == 3
    assert conv_result.document.num_pages() == 1
    assert list(conv_result.document.pages.keys()) == [2]

    pred_md = conv_result.document.export_to_markdown()
    assert "Second slide title" in pred_md
    assert "Test Table Slide" not in pred_md
    assert "List item4" not in pred_md


def test_chart_parsed_as_classified_picture_with_data():
    """A native PPTX chart becomes one classified picture carrying its data.

    ``pptx_chart.pptx`` holds a single clustered-column chart titled "Wild Duck
    Observations by Year" with two series over four years. It should convert to
    exactly one PictureItem classified as a bar chart, captioned with the chart
    title, and carrying the chart's plotted numbers reconstructed as a table:

        | <blank> | Freshwater Ducks | Saltwater Ducks |
        | 2019    | 120              | 80              |
        ...
        | 2022    | 175              | 130             |
    """
    converter = get_converter()
    doc = converter.convert(CHART_PPTX).document

    pictures = list(doc.pictures)
    assert len(pictures) == 1, f"Expected one chart picture, got {len(pictures)}"

    picture = pictures[0]
    assert (
        picture.meta.classification.predictions[0].class_name
        == PictureClassificationLabel.BAR_CHART
    )
    assert picture.caption_text(doc) == "Wild Duck Observations by Year"

    chart_data = picture.meta.tabular_chart.chart_data
    assert (chart_data.num_rows, chart_data.num_cols) == (5, 3)
    grid = {
        (cell.start_row_offset_idx, cell.start_col_offset_idx): cell.text
        for cell in chart_data.table_cells
    }
    assert grid[(0, 1)] == "Freshwater Ducks"
    assert grid[(0, 2)] == "Saltwater Ducks"
    assert grid[(1, 0)] == "2019"
    assert grid[(4, 0)] == "2022"
    assert grid[(4, 1)] == "175"
    assert grid[(4, 2)] == "130"


def test_chart_image_not_rendered_by_default():
    """Charts carry classification and data but no image unless opted in.

    render_chart_images defaults to False, so the chart picture keeps its
    classification and reconstructed data but no pixels. This guards the promise
    that the feature does not change default output size for existing users.
    """
    converter = get_converter()
    doc = converter.convert(CHART_PPTX).document

    picture = next(iter(doc.pictures))
    assert picture.meta.tabular_chart is not None
    assert picture.image is None, (
        "chart picture should have no image when render_chart_images is off"
    )


def test_chart_image_rendering(libreoffice_available):
    """render_chart_images=True attaches a LibreOffice-rendered image.

    LibreOffice output is not byte-stable and the cropped image size depends on
    the LibreOffice version, so pixels are not compared against groundtruth. We
    assert the picture gains a non-trivial image while keeping the classification
    and tabular data. Requires LibreOffice; skipped when it is not installed.
    """
    if not libreoffice_available:
        pytest.skip("LibreOffice is not installed — chart rendering cannot be tested")

    options = MsPowerpointBackendOptions(render_chart_images=True)
    format_options = {InputFormat.PPTX: PowerpointFormatOption(backend_options=options)}
    converter = DocumentConverter(
        allowed_formats=[InputFormat.PPTX], format_options=format_options
    )
    doc = converter.convert(CHART_PPTX).document

    pictures = list(doc.pictures)
    assert len(pictures) == 1, f"Expected one chart picture, got {len(pictures)}"

    picture = pictures[0]
    assert (
        picture.meta.classification.predictions[0].class_name
        == PictureClassificationLabel.BAR_CHART
    )
    assert picture.meta.tabular_chart is not None

    image = picture.get_image(doc=doc)
    assert image is not None, "chart picture should carry a rendered image"
    assert image.width > 50 and image.height > 50, (
        f"rendered chart image is implausibly small: {image.size}"
    )


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
