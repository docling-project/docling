from pathlib import Path
from types import SimpleNamespace

import pytest
from docling_core.types.doc import ContentLayer, GroupItem, TextItem

from docling.backend.mspowerpoint_backend import MsPowerpointDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult, DoclingDocument
from docling.document_converter import DocumentConverter

from .test_data_gen_flag import GEN_TEST_DATA
from .verify_utils import verify_document, verify_export

GENERATE = GEN_TEST_DATA


def get_pptx_paths():
    # Define the directory you want to search
    directory = Path("./tests/data/pptx/regression/")

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
            pptx_path.parent.parent / "groundtruth" / pptx_path.name
        )

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
    path = Path("./tests/data/pptx/regression/powerpoint_comments.pptx")
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
    path = Path("./tests/data/pptx/regression/powerpoint_comments.pptx")
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
    pptx_path = Path("./tests/data/pptx/regression/powerpoint_unrecognized_shape.pptx")

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
    pptx_path = Path("./tests/data/pptx/regression/powerpoint_malformed_pictures.pptx")

    with pytest.warns(UserWarning, match="Skipping malformed picture shape"):
        conv_result: ConversionResult = converter.convert(pptx_path)

    doc: DoclingDocument = conv_result.document

    pred_md = doc.export_to_markdown()
    assert "Slide With Missing BlipFill" in pred_md
    assert "Slide With Dangling Rel" in pred_md
    assert "Slide With Wrong Content Type" in pred_md


def test_pptx_page_range():
    converter = get_converter()
    pptx_path = Path("./tests/data/pptx/regression/powerpoint_sample.pptx")

    conv_result: ConversionResult = converter.convert(pptx_path, page_range=(2, 2))

    assert conv_result.input.page_count == 3
    assert conv_result.document.num_pages() == 1
    assert list(conv_result.document.pages.keys()) == [2]

    pred_md = conv_result.document.export_to_markdown()
    assert "Second slide title" in pred_md
    assert "Test Table Slide" not in pred_md
    assert "List item4" not in pred_md
