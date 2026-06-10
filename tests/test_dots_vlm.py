"""Test dots.ocr / dots.mocr JSON parsing in VLM pipeline."""

from pathlib import Path

from docling_core.types.doc import DocItemLabel, DoclingDocument, Size

from docling.utils.dots_utils import parse_dots_json


def get_dots_test_paths():
    """Get all dots JSON test files."""
    directory = Path("./tests/data/json_dots/")
    return sorted(directory.glob("*.json"))


def test_dots_simple_parsing():
    """Test dots JSON parsing produces expected document structure."""
    path = Path("./tests/data/json_dots/dots_simple.json")
    content = path.read_text()
    source = path.with_suffix(".source.txt").read_text()

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
    assert "caption" in labels, "Should have captions"
    assert "footnote" in labels, "Should have footnotes"

    assert "tests/data/pdf/2206.01062.pdf, page 1" in source
    assert any("DocLayNet" in (t.text or "") for t in doc.texts)
    assert len(doc.pictures) > 0, "Should have picture elements"

    for item in doc.texts:
        assert len(item.prov) > 0, f"Text item should have provenance: {item.text[:30]}"
        bbox = item.prov[0].bbox
        assert bbox is not None, f"Should have bbox: {item.text[:30]}"
        assert bbox.l >= 0 and bbox.t >= 0, "Bbox coords should be non-negative"


def test_dots_list_parsing():
    """Test dots JSON parsing handles real list-item predictions."""
    path = Path("./tests/data/json_dots/dots_list.json")
    content = path.read_text()
    source = path.with_suffix(".source.txt").read_text()

    doc = parse_dots_json(
        content=content,
        original_page_size=Size(width=612, height=792),
        page_no=1,
        filename=path.name,
    )

    labels = [
        t.label.value if hasattr(t.label, "value") else str(t.label) for t in doc.texts
    ]
    list_items = [item for item in doc.texts if item.label == DocItemLabel.LIST_ITEM]

    assert "tests/data/pdf/multi_page.pdf, page 1" in source
    assert "list_item" in labels, "Should have list items"
    assert len(list_items) == 2
    assert "IBM MT/ST" in list_items[0].text
    assert "Microsoft Word" in list_items[1].text


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


def test_dots_empty_content():
    """Test that empty/whitespace content returns empty doc."""
    for content in ["", "   ", "\n"]:
        doc = parse_dots_json(
            content=content,
            original_page_size=Size(width=612, height=792),
            page_no=1,
            filename="empty.json",
        )
        assert isinstance(doc, DoclingDocument)
        assert len(doc.texts) == 0


def test_dots_malformed_json():
    """Test graceful handling of invalid JSON."""
    doc = parse_dots_json(
        content="this is not json at all",
        original_page_size=Size(width=612, height=792),
        page_no=1,
        filename="bad.json",
    )
    assert isinstance(doc, DoclingDocument)
    assert len(doc.texts) == 0


def test_dots_truncated_json():
    """Test that truncated JSON (common in model output) is recovered."""
    content = '[{"bbox": [0, 0, 100, 100], "category": "Text", "text": "hello"}, {"bbox": [0, 100, 200, 200], "category": "Tex'
    doc = parse_dots_json(
        content=content,
        original_page_size=Size(width=612, height=792),
        page_no=1,
        filename="truncated.json",
    )
    assert len(doc.texts) >= 1


def test_dots_bad_bbox_elements():
    """Test that elements with invalid bbox are skipped."""
    content = (
        "["
        '{"bbox": "not a list", "category": "Text", "text": "bad"},'
        '{"bbox": [0, 0], "category": "Text", "text": "short"},'
        '{"bbox": [0, 0, 100, 100], "category": "Text", "text": "good"}'
        "]"
    )
    doc = parse_dots_json(
        content=content,
        original_page_size=Size(width=612, height=792),
        page_no=1,
        filename="bad_bbox.json",
    )
    assert len(doc.texts) == 1
    assert doc.texts[0].text == "good"


def test_dots_non_dict_elements():
    """Test that non-dict elements in array are skipped."""
    content = '[42, "string", {"bbox": [0, 0, 100, 100], "category": "Text", "text": "valid"}]'
    doc = parse_dots_json(
        content=content,
        original_page_size=Size(width=612, height=792),
        page_no=1,
        filename="mixed.json",
    )
    assert len(doc.texts) == 1


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
