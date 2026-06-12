from pathlib import Path
from types import SimpleNamespace

import pytest
from docling_core.types.doc import ContentLayer

from docling.backend.mspowerpoint_backend import MsPowerpointDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult, DoclingDocument
from docling.document_converter import DocumentConverter

from .test_data_gen_flag import GEN_TEST_DATA
from .verify_utils import verify_document, verify_export

GENERATE = GEN_TEST_DATA
PPTX_NAMESPACES = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "c": "http://schemas.openxmlformats.org/drawingml/2006/chart",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
}
COMMENT_FIXTURES = {"powerpoint_comments"}


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

        pred_md: str = doc.export_to_markdown(compact_tables=True)
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


def test_comments_extraction() -> None:
    """Test that slide comments are extracted into the NOTES content layer."""
    from docling_core.types.doc import GroupItem

    converter = get_converter()
    path = Path("./tests/data/pptx/powerpoint_comments.pptx")
    doc: DoclingDocument = converter.convert(path).document

    comment_groups = [
        g
        for g in doc.groups
        if isinstance(g, GroupItem) and g.name.startswith("comment-")
    ]
    assert len(comment_groups) >= 1, (
        f"Expected ≥1 comment group, got {len(comment_groups)}"
    )

    comment_texts = [
        t.text
        for t in doc.texts
        if hasattr(t, "content_layer") and t.content_layer == "notes"
    ]
    assert any("John Reviewer" in t for t in comment_texts), (
        "Expected 'John Reviewer' in comment texts"
    )
    assert any("sample reviewer comment" in t for t in comment_texts), (
        "Expected comment body text content"
    )
    for group in comment_groups:
        assert group.content_layer == "notes", (
            "Comments should be in NOTES content layer"
        )

def test_add_comments_handles_missing_metadata_and_parse_failures() -> None:
    """_add_comments should skip broken slides/comments and keep valid ones."""

    class FakeTargetPart:
        def __init__(self, blob: bytes):
            self.blob = blob

    class FakeRel:
        def __init__(self, reltype: str, blob: bytes = b""):
            self.reltype = reltype
            self.target_part = FakeTargetPart(blob)

    class FakeSlide:
        def __init__(self, rels):
            self._part = SimpleNamespace(rels=rels)

        @property
        def part(self):
            return self._part

    class BrokenSlide:
        @property
        def part(self):
            raise RuntimeError("broken slide")

    backend = MsPowerpointDocumentBackend.__new__(MsPowerpointDocumentBackend)
    backend.namespaces = PPTX_NAMESPACES
    author_reltype = (
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/"
        "commentAuthors"
    )
    comment_reltype = (
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments"
    )
    valid_comments = b"""
    <p:cmLst xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
      <p:cm idx="7" authorId="1" dt="2026-05-29T10:00:00Z">
        <p:text>  first note  </p:text>
      </p:cm>
      <p:cm idx="8" authorId="99">
        <p:text>authorless note</p:text>
      </p:cm>
      <p:cm idx="9" authorId="1">
        <p:text>   </p:text>
      </p:cm>
    </p:cmLst>
    """
    invalid_comments = b"<p:cmLst"
    authors = b"""
    <p:cmAuthorLst xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
      <p:cmAuthor id="1" name="John Reviewer" initials="JR" />
    </p:cmAuthorLst>
    """
    pptx_obj = SimpleNamespace(
        part=SimpleNamespace(rels={"authors": FakeRel(author_reltype, authors)}),
        slides=[
            BrokenSlide(),
            FakeSlide(
                {
                    "noop": FakeRel("urn:ignored"),
                    "comments": FakeRel(comment_reltype, valid_comments),
                }
            ),
            FakeSlide({"comments": FakeRel(comment_reltype, invalid_comments)}),
        ],
    )
    doc = DoclingDocument(name="comments")

    backend._add_comments(pptx_obj, doc)

    comment_groups = [
        group for group in doc.groups if group.name.startswith("comment-")
    ]
    assert [group.name for group in comment_groups] == [
        "comment-slide2-7",
        "comment-slide2-8",
    ]
    assert [text.text for text in doc.texts] == [
        "[author: John Reviewer (JR), time: 2026-05-29T10:00:00Z]: first note",
        "authorless note",
    ]
    assert all(group.content_layer == "notes" for group in comment_groups)


def test_add_comments_continues_when_comment_authors_cannot_be_parsed() -> None:
    """_add_comments should still emit raw comment text when author parsing fails."""

    class FakeTargetPart:
        def __init__(self, blob: bytes):
            self.blob = blob

    class FakeRel:
        def __init__(self, reltype: str, blob: bytes):
            self.reltype = reltype
            self.target_part = FakeTargetPart(blob)

    class FakeSlide:
        def __init__(self, rels):
            self._part = SimpleNamespace(rels=rels)

        @property
        def part(self):
            return self._part

    backend = MsPowerpointDocumentBackend.__new__(MsPowerpointDocumentBackend)
    backend.namespaces = PPTX_NAMESPACES
    author_reltype = (
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/"
        "commentAuthors"
    )
    comment_reltype = (
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments"
    )
    comments = b"""
    <p:cmLst xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
      <p:cm idx="3" authorId="1">
        <p:text>kept without metadata</p:text>
      </p:cm>
    </p:cmLst>
    """
    pptx_obj = SimpleNamespace(
        part=SimpleNamespace(
            rels={"authors": FakeRel(author_reltype, b"<p:cmAuthorLst")}
        ),
        slides=[FakeSlide({"comments": FakeRel(comment_reltype, comments)})],
    )
    doc = DoclingDocument(name="comments")

    backend._add_comments(pptx_obj, doc)

    assert [group.name for group in doc.groups] == ["comment-slide1-3"]
    assert [text.text for text in doc.texts] == ["kept without metadata"]


def test_add_comments_returns_when_pptx_object_is_missing() -> None:
    """_add_comments should be a no-op when no presentation object is available."""
    backend = MsPowerpointDocumentBackend.__new__(MsPowerpointDocumentBackend)
    doc = DoclingDocument(name="comments")

    backend._add_comments(None, doc)

    assert doc.groups == []
    assert doc.texts == []
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
