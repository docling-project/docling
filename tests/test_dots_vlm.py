"""Test dots.ocr / dots.mocr JSON parsing in VLM pipeline."""

from pathlib import Path

from docling_core.types.doc import DoclingDocument, Size

from docling.utils.dots_utils import parse_dots_json


def get_dots_test_paths():
    """Get all dots JSON test files."""
    directory = Path("./tests/data/json_dots/")
    return sorted(directory.glob("*.json"))


def test_dots_simple_parsing():
    """Test dots JSON parsing produces expected document structure."""
    path = Path("./tests/data/json_dots/dots_simple.json")
    content = path.read_text()

    doc = parse_dots_json(
        content=content,
        original_page_size=Size(width=612, height=792),
        page_no=1,
        filename="dots_simple.json",
    )

    assert isinstance(doc, DoclingDocument)
    assert len(doc.texts) > 0, "Should have text elements"

    labels = [
        t.label.value if hasattr(t.label, "value") else str(t.label) for t in doc.texts
    ]
    assert "title" in labels, "Should have a title element"
    assert "section_header" in labels, "Should have section headers"

    assert len(doc.tables) > 0, "Should have table elements"
    assert len(doc.pictures) > 0, "Should have picture elements"

    for item in doc.texts:
        assert len(item.prov) > 0, f"Text item should have provenance: {item.text[:30]}"
        bbox = item.prov[0].bbox
        assert bbox is not None, f"Should have bbox: {item.text[:30]}"
        assert bbox.l >= 0 and bbox.t >= 0, "Bbox coords should be non-negative"


def test_dots_formula_parsing():
    """Test dots JSON parsing handles formulas and list items."""
    path = Path("./tests/data/json_dots/dots_formula.json")
    content = path.read_text()

    doc = parse_dots_json(
        content=content,
        original_page_size=Size(width=612, height=792),
        page_no=1,
        filename="dots_formula.json",
    )

    labels = [
        t.label.value if hasattr(t.label, "value") else str(t.label) for t in doc.texts
    ]
    assert "list_item" in labels, "Should have list items"

    assert any("\\mathcal{L}" in (t.text or "") for t in doc.texts), (
        "Should preserve LaTeX formula content"
    )


def test_dots_model_image_size_rescaling():
    """Test that model_image_size rescales bboxes correctly."""
    content = '[{"bbox": [0, 0, 560, 560], "category": "Text", "text": "hello"}]'

    doc = parse_dots_json(
        content=content,
        original_page_size=Size(width=612, height=792),
        page_no=1,
        filename="test.json",
        model_image_size=Size(width=560, height=560),
    )

    assert len(doc.texts) == 1
    bbox = doc.texts[0].prov[0].bbox
    assert abs(bbox.r - 612) < 1, f"Right edge should map to page width, got {bbox.r}"
    assert abs(bbox.b - 792) < 1, f"Bottom edge should map to page height, got {bbox.b}"


def test_dots_all_files_parse():
    """Ensure all dots test files parse without errors."""
    for path in get_dots_test_paths():
        content = path.read_text()
        doc = parse_dots_json(
            content=content,
            original_page_size=Size(width=612, height=792),
            page_no=1,
            filename=path.name,
        )
        assert isinstance(doc, DoclingDocument), f"Failed to parse {path.name}"
        assert len(doc.texts) + len(doc.tables) + len(doc.pictures) > 0, (
            f"No elements parsed from {path.name}"
        )
