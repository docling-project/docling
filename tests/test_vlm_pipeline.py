"""
Tests for VLM pipeline functionality.

Includes tests for handling nested lists in Markdown responses,
which previously caused: ValueError: Can not append a child with children
See: https://github.com/docling-project/docling/issues/2301

Test structure based on reproducer code contributed by @amomra in issue #2301.
"""

import time

import pytest
import requests_mock
from docling_core.types.doc import GroupItem, ListItem

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import VlmPipelineOptions
from docling.datamodel.pipeline_options_vlm_model import (
    ApiVlmOptions,
    ResponseFormat,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline


@pytest.fixture
def mock_api_endpoint():
    """Create a mock API endpoint for VLM responses."""
    with requests_mock.Mocker() as m:
        yield m


# Dummy file needs to exist even though its not processed by the VLM
TEST_PDF = "tests/data/pdf/code_and_formula.pdf"


def create_vlm_converter(mock_endpoint, markdown_response):
    """Helper to create a DocumentConverter with mocked VLM API."""
    test_url = "http://test-vlm-api.com"

    mock_endpoint.post(
        test_url,
        json={
            "id": "test-123",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": markdown_response},
                    "finish_reason": "stop",
                }
            ],
            "created": int(time.time()),
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )

    pipeline_options = VlmPipelineOptions(enable_remote_services=True)
    pipeline_options.vlm_options = ApiVlmOptions(
        url=test_url,
        headers={"Authorization": "Bearer test"},
        params=dict(model="test-model"),
        prompt="Convert to markdown",
        timeout=90,
        scale=1.0,
        response_format=ResponseFormat.MARKDOWN,
    )

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                pipeline_cls=VlmPipeline,
            ),
        }
    )


def test_nested_list_with_html_tag(mock_api_endpoint):
    """Test the original failing case: nested list with HTML tag."""
    markdown = """- item 1
- item 2
  - sub item 1 <text>
  - sub item 2"""

    converter = create_vlm_converter(mock_api_endpoint, markdown)

    # Should not raise ValueError
    result = converter.convert(TEST_PDF)

    assert result.document is not None
    output = result.document.export_to_markdown()

    # Verify content is present (order may vary due to flattening)
    assert "item 1" in output
    assert "item 2" in output
    assert "sub item 1" in output
    assert "sub item 2" in output


def test_simple_nested_list(mock_api_endpoint):
    """Test simple nested list without special characters."""
    markdown = """- item 1
- item 2
  - sub item 1
  - sub item 2"""

    converter = create_vlm_converter(mock_api_endpoint, markdown)

    result = converter.convert(TEST_PDF)

    assert result.document is not None
    output = result.document.export_to_markdown()

    assert "item 1" in output
    assert "item 2" in output
    assert "sub item 1" in output
    assert "sub item 2" in output


def test_parent_item_with_text_and_children(mock_api_endpoint):
    """Test that parent item text is preserved when it has children."""
    markdown = """- item 1
- item 2 has some text
  - sub item 1
  - sub item 2
- item 3"""

    converter = create_vlm_converter(mock_api_endpoint, markdown)

    result = converter.convert(TEST_PDF)

    assert result.document is not None
    output = result.document.export_to_markdown()

    # Verify parent text is preserved
    assert "item 1" in output
    assert "item 2 has some text" in output  # Parent text must not be lost
    assert "sub item 1" in output
    assert "sub item 2" in output
    assert "item 3" in output


def test_deeply_nested_list(mock_api_endpoint):
    """Test deeply nested lists (3+ levels)."""
    markdown = """- level 1 item 1
  - level 2 item 1
    - level 3 item 1
  - level 2 item 2"""

    converter = create_vlm_converter(mock_api_endpoint, markdown)

    result = converter.convert(TEST_PDF)

    assert result.document is not None
    output = result.document.export_to_markdown()

    assert "level 1 item 1" in output
    assert "level 2 item 1" in output
    assert "level 3 item 1" in output
    assert "level 2 item 2" in output


def test_flat_list_still_works(mock_api_endpoint):
    """Ensure flat lists (no nesting) continue to work correctly."""
    markdown = """- item 1
- item 2
- item 3"""

    converter = create_vlm_converter(mock_api_endpoint, markdown)

    result = converter.convert(TEST_PDF)

    assert result.document is not None
    output = result.document.export_to_markdown()

    assert "item 1" in output
    assert "item 2" in output
    assert "item 3" in output


# Structure Preservation Tests (added for comprehensive fix of issue #2301)
def test_nested_list_structure_preserved(mock_api_endpoint):
    """Verify nested list structure is correctly preserved with proper levels."""
    markdown = """- item 1
- item 2
  - sub item 1
  - sub item 2
- item 3"""

    converter = create_vlm_converter(mock_api_endpoint, markdown)
    result = converter.convert(TEST_PDF)

    assert result.document is not None

    # Verify structure using iterate_items without groups (user view)
    # Note: PDF may have multiple pages, so get items from first page only
    all_items = list(result.document.iterate_items(with_groups=False))

    # Filter to first page's items (page_no=1 in prov)
    page1_items = [
        (item, level)
        for item, level in all_items
        if hasattr(item, "prov") and len(item.prov) > 0 and item.prov[0].page_no == 1
    ]

    # Expected: at least 5 items (item 1, item 2, sub item 1, sub item 2, item 3)
    assert len(page1_items) >= 5, (
        f"Expected at least 5 items on page 1, got {len(page1_items)}"
    )

    # Check first 5 items (our list)
    first_5_items = page1_items[:5]
    levels = [level for _, level in first_5_items]

    # Verify relative structure: sub-items should be deeper than parents, siblings same level
    assert levels[0] == levels[1], (
        "item 1 and item 2 should be at same level (siblings)"
    )
    assert levels[2] > levels[1], "sub item 1 should be deeper than item 2 (nested)"
    assert levels[2] == levels[3], (
        "sub item 1 and sub item 2 should be at same level (siblings)"
    )
    assert levels[4] == levels[0], "item 3 should be at same level as item 1 (siblings)"
    assert levels[2] == levels[1] + 2, (
        "Nested items should be 2 levels deeper (without groups: 2→4)"
    )

    # Verify text content
    item1, _ = first_5_items[0]
    item2, _ = first_5_items[1]
    sub1, _ = first_5_items[2]
    sub2, _ = first_5_items[3]
    item3, _ = first_5_items[4]

    assert "item 1" in getattr(item1, "text", "")
    assert "item 2" in getattr(item2, "text", "")
    assert "sub item 1" in getattr(sub1, "text", "")
    assert "sub item 2" in getattr(sub2, "text", "")
    assert "item 3" in getattr(item3, "text", "")
