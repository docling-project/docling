"""Test chandra-ocr-2 HTML parsing in VLM pipeline."""

from pathlib import Path

from docling_core.types.doc import DoclingDocument, Size

from docling.utils.chandra_utils import parse_chandra_html


def get_chandra_test_paths():
    """Get all chandra HTML test files."""
    directory = Path("./tests/data/html_chandra/")
    return sorted(directory.glob("*.html"))


def test_chandra_simple_parsing():
    """Test chandra HTML parsing produces expected document structure."""
    path = Path("./tests/data/html_chandra/chandra_simple.html")
    content = path.read_text()

    doc = parse_chandra_html(
        content=content,
        original_page_size=Size(width=612, height=792),
        page_no=1,
        filename="chandra_simple.html",
    )

    assert isinstance(doc, DoclingDocument)
    assert len(doc.texts) > 0, "Should have text elements"

    labels = [t.label.value if hasattr(t.label, "value") else str(t.label) for t in doc.texts]
    assert "section_header" in labels, "Should have section headers"
    assert "caption" in labels, "Should have caption"
    assert "page_footer" in labels, "Should have page footer"

    assert len(doc.tables) > 0, "Should have table elements"

    for item in doc.texts:
        assert len(item.prov) > 0, f"Text item should have provenance"
        bbox = item.prov[0].bbox
        assert bbox is not None, "Should have bbox"
        assert bbox.l >= 0 and bbox.t >= 0, "Bbox coords should be non-negative"


def test_chandra_multiblock_parsing():
    """Test chandra parsing with images, equations, and footnotes."""
    path = Path("./tests/data/html_chandra/chandra_multiblock.html")
    content = path.read_text()

    doc = parse_chandra_html(
        content=content,
        original_page_size=Size(width=612, height=792),
        page_no=1,
        filename="chandra_multiblock.html",
    )

    labels = [t.label.value if hasattr(t.label, "value") else str(t.label) for t in doc.texts]
    assert "page_header" in labels, "Should have page header"
    assert "footnote" in labels, "Should have footnote"

    assert len(doc.pictures) > 0, "Should have picture/image elements"


def test_chandra_bbox_normalization():
    """Test that chandra bboxes (normalized 0-1000) map to page coordinates."""
    content = '<div data-bbox="0 0 1000 1000" data-label="Text"><p>full page</p></div>'

    doc = parse_chandra_html(
        content=content,
        original_page_size=Size(width=612, height=792),
        page_no=1,
        filename="test.html",
    )

    assert len(doc.texts) == 1
    bbox = doc.texts[0].prov[0].bbox
    assert abs(bbox.r - 612) < 1, f"Right edge should map to page width, got {bbox.r}"
    assert abs(bbox.b - 792) < 1, f"Bottom edge should map to page height, got {bbox.b}"


def test_chandra_all_files_parse():
    """Ensure all chandra test files parse without errors."""
    for path in get_chandra_test_paths():
        content = path.read_text()
        doc = parse_chandra_html(
            content=content,
            original_page_size=Size(width=612, height=792),
            page_no=1,
            filename=path.name,
        )
        assert isinstance(doc, DoclingDocument), f"Failed to parse {path.name}"
        assert len(doc.texts) + len(doc.tables) + len(doc.pictures) > 0, (
            f"No elements parsed from {path.name}"
        )
