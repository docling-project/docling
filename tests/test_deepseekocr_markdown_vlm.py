"""Test DeepSeek OCR markdown parsing in VLM pipeline."""

from pathlib import Path

from docling_core.types.doc import DocItem, DocItemLabel, DoclingDocument

from docling.datamodel.base_models import InputFormat


def mock_parsing(content: str) -> DoclingDocument:
    # Create a mock conversion result with the DeepSeek OCR markdown as VLM response
    from PIL import Image as PILImage

    from docling.datamodel.base_models import Page, PagePredictions, VlmPrediction
    from docling.datamodel.document import ConversionResult, InputDocument
    from docling.pipeline.vlm_pipeline import VlmPipeline

    # Create a mock InputDocument class that bypasses custom __init__
    class MockInputDocument(InputDocument):
        def __init__(self, **data):
            super(InputDocument, self).__init__(**data)

    in_doc = MockInputDocument(
        file=Path("test.md"), format=InputFormat.MD, document_hash="0"
    )

    conv_res = ConversionResult(
        input=in_doc,
    )

    # Create a page with the DeepSeek OCR markdown as VLM response
    page = Page(page_no=1)
    page._image_cache[1.0] = PILImage.new("RGB", (800, 1000), color="white")
    page.predictions = PagePredictions()
    page.predictions.vlm_response = VlmPrediction(text=content)

    conv_res.pages = [page]

    # Call the parser method directly without initializing the full pipeline
    # We just need to test the parsing logic, not the full pipeline
    from docling.pipeline.vlm_pipeline import VlmPipeline

    # Parse the DeepSeek OCR markdown by calling the method directly
    doc = VlmPipeline._parse_deepseekocr_markdown(None, conv_res)

    return doc


def test_parse_deepseekocr_markdown_simple():
    """Test that DeepSeek OCR markdown files can be parsed."""
    # Test with first file
    test_file = Path("tests/data/md_deepseek/annotated_simple.md")

    # Read the DeepSeek OCR markdown content
    with open(test_file, encoding="utf-8") as f:
        annotated_content = f.read()

    doc = mock_parsing(annotated_content)

    doc.save_as_html(Path(f"test_{test_file.stem}.html"))
    doc.save_as_markdown(Path(f"test_{test_file.stem}.md"))
    doc.save_as_json(Path(f"test_{test_file.stem}.json"))

    # Verify items were created
    items = list(doc.iterate_items())
    assert len(items) > 0, "Expected at least one item in the document"

    # Verify at least some items have provenance with bounding boxes
    items_with_bbox = []
    for item, _ in items:
        assert isinstance(item, DocItem)
        if item.prov and len(item.prov) > 0:
            prov = item.prov[0]
            bbox = prov.bbox
            # Check that bbox has valid coordinates (not all zeros)
            if bbox.area() > 0:
                items_with_bbox.append(item)

    assert len(items_with_bbox) > 0, (
        "Expected at least one item with valid bounding box"
    )

    # Verify labels are correctly assigned
    labels_found = set()
    heading_levels = []
    for item, _ in items:
        assert isinstance(item, DocItem)
        labels_found.add(item.label)
        # Check heading levels for section headers
        if item.label == DocItemLabel.SECTION_HEADER:
            if hasattr(item, "level"):
                heading_levels.append(item.level)

    # Should have at least some of the expected labels
    expected_labels = {DocItemLabel.TEXT, DocItemLabel.SECTION_HEADER}
    assert len(labels_found & expected_labels) > 0, (
        f"Expected some of {expected_labels}, found {labels_found}"
    )

    # Verify heading levels are detected (if any section headers exist)
    if DocItemLabel.SECTION_HEADER in labels_found:
        assert len(heading_levels) > 0, (
            "Expected heading levels to be set for section headers"
        )
        # Verify levels are reasonable (1-6 for markdown)
        assert all(1 <= level <= 6 for level in heading_levels), (
            f"Heading levels should be 1-6, found {heading_levels}"
        )

    print(f"✓ Successfully parsed {test_file.name}")
    print(f"  - Total items: {len(items)}")
    print(f"  - Items with bbox: {len(items_with_bbox)}")
    print(f"  - Labels found: {labels_found}")
    if heading_levels:
        print(f"  - Heading levels: {heading_levels}")


def test_parse_deepseekocr_markdown_example():
    # Test with first file
    test_file = Path("tests/data/md_deepseek/annotated_example.md")

    # Read the annotated markdown content
    with open(test_file, encoding="utf-8") as f:
        annotated_content = f.read()

    doc = mock_parsing(annotated_content)

    print()
    doc.save_as_html(Path(f"test_{test_file.stem}.html"))
    doc.save_as_markdown(Path(f"test_{test_file.stem}.md"))
    doc.save_as_json(Path(f"test_{test_file.stem}.json"))


if __name__ == "__main__":
    test_parse_deepseekocr_markdown_simple()
    test_parse_deepseekocr_markdown_example()
