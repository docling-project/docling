"""Tests for reading order correction utilities."""

from __future__ import annotations

from unittest.mock import MagicMock, Mock, patch

import pytest


def test_get_first_prov_circular_reference():
    """Test that a circular reference does not cause infinite recursion."""
    from docling.pipeline.reading_order_utils import _get_first_prov

    # Create a mock item
    mock_item = Mock()
    mock_resolved = Mock()
    mock_resolved.prov = None

    # Create a dummy object to represent the cycle, avoiding Mock-to-Mock loops
    # which can cause pytest/logging to crash during __repr__
    class DummyItem:
        pass

    cycle_item = DummyItem()
    mock_resolved.children = [cycle_item]

    # Mock the resolve method to return the resolved object
    mock_item.resolve = Mock(return_value=mock_resolved)
    mock_doc = Mock()

    # Pre-populate the _visited set to simulate that we have already seen this item
    # This directly triggers the early return we are trying to cover
    visited = {id(mock_item)}

    # Pass the populated _visited set to trigger the `if id(ref_item) in _visited:` branch
    result = _get_first_prov(mock_item, mock_doc, _visited=visited)

    assert result is None, "Should return None when item is in _visited set"


def test_get_first_prov_without_provenance():
    """Test when item has no provenance."""
    from docling.pipeline.reading_order_utils import _get_first_prov

    # Create a mock item without provenance
    mock_item = Mock()
    mock_resolved = Mock()
    mock_resolved.prov = None
    mock_resolved.children = []

    mock_item.resolve = Mock(return_value=mock_resolved)
    mock_doc = Mock()

    result = _get_first_prov(mock_item, mock_doc)

    assert result is None, "Should return None when no provenance"


def test_get_first_prov_with_child_provenance():
    """Test extraction of provenance from child when parent has none."""
    from docling.pipeline.reading_order_utils import _get_first_prov

    # Create mock child with provenance
    mock_child = Mock()
    mock_child_prov = Mock()
    mock_child_bbox = Mock()
    mock_child_bbox.t = 600.0
    mock_child_prov.page_no = 2
    mock_child_prov.bbox = mock_child_bbox

    mock_child_resolved = Mock()
    mock_child_resolved.prov = [mock_child_prov]
    mock_child_resolved.children = []

    mock_child.resolve = Mock(return_value=mock_child_resolved)

    # Create parent without provenance but with child
    mock_parent = Mock()
    mock_parent_resolved = Mock()
    mock_parent_resolved.prov = None
    mock_parent_resolved.children = [mock_child]

    mock_parent.resolve = Mock(return_value=mock_parent_resolved)
    mock_doc = Mock()

    result = _get_first_prov(mock_parent, mock_doc)

    assert result is not None
    assert result == (2, -600.0), "Should return child's provenance"


def test_correct_reading_order_basic_sort():
    """Test basic sorting of items by page and bbox."""
    from docling.pipeline.reading_order_utils import correct_reading_order_on_page

    # Create mock items with different bbox.t values
    # Item 1: page 1, bbox.t = 400 (lower on page)
    mock_item1 = Mock()
    mock_item1_prov = Mock()
    mock_item1_bbox = Mock()
    mock_item1_bbox.t = 400.0
    mock_item1_prov.page_no = 1
    mock_item1_prov.bbox = mock_item1_bbox
    mock_item1_resolved = Mock()
    mock_item1_resolved.prov = [mock_item1_prov]
    mock_item1_resolved.children = []
    mock_item1.resolve = Mock(return_value=mock_item1_resolved)

    # Item 2: page 1, bbox.t = 600 (higher on page)
    mock_item2 = Mock()
    mock_item2_prov = Mock()
    mock_item2_bbox = Mock()
    mock_item2_bbox.t = 600.0
    mock_item2_prov.page_no = 1
    mock_item2_prov.bbox = mock_item2_bbox
    mock_item2_resolved = Mock()
    mock_item2_resolved.prov = [mock_item2_prov]
    mock_item2_resolved.children = []
    mock_item2.resolve = Mock(return_value=mock_item2_resolved)

    # Create mock document with body
    mock_doc = Mock()
    mock_body = Mock()
    mock_doc.body = mock_body

    # Add items in wrong order (lower bbox.t first)
    mock_body.children = [mock_item1, mock_item2]

    # Apply correction
    moved = correct_reading_order_on_page(mock_doc)

    # Item 2 (higher bbox.t) should be first
    assert moved > 0, "Should have moved items"
    assert mock_body.children[0] == mock_item2, "Higher bbox.t item should be first"
    assert mock_body.children[1] == mock_item1, "Lower bbox.t item should be second"


def test_correct_reading_order_no_movement_if_already_sorted():
    """Test that already sorted items are not moved."""
    from docling.pipeline.reading_order_utils import correct_reading_order_on_page

    # Item 1: page 1, bbox.t = 600 (higher)
    mock_item1 = Mock()
    mock_item1_prov = Mock()
    mock_item1_bbox = Mock()
    mock_item1_bbox.t = 600.0
    mock_item1_prov.page_no = 1
    mock_item1_prov.bbox = mock_item1_bbox
    mock_item1_resolved = Mock()
    mock_item1_resolved.prov = [mock_item1_prov]
    mock_item1_resolved.children = []
    mock_item1.resolve = Mock(return_value=mock_item1_resolved)

    # Item 2: page 1, bbox.t = 400 (lower)
    mock_item2 = Mock()
    mock_item2_prov = Mock()
    mock_item2_bbox = Mock()
    mock_item2_bbox.t = 400.0
    mock_item2_prov.page_no = 1
    mock_item2_prov.bbox = mock_item2_bbox
    mock_item2_resolved = Mock()
    mock_item2_resolved.prov = [mock_item2_prov]
    mock_item2_resolved.children = []
    mock_item2.resolve = Mock(return_value=mock_item2_resolved)

    mock_doc = Mock()
    mock_body = Mock()
    mock_doc.body = mock_body

    # Items already in correct order
    mock_body.children = [mock_item1, mock_item2]

    moved = correct_reading_order_on_page(mock_doc)

    assert moved == 0, "Should not move already sorted items"
    assert mock_body.children == [mock_item1, mock_item2], (
        "Order should remain unchanged"
    )


def test_correct_reading_order_multipage():
    """Test that cross-page ordering is preserved (no cross-page reordering).

    The algorithm only reorders within pages. Page boundaries act as barriers
    that prevent items from being moved across them.
    """
    from docling.pipeline.reading_order_utils import correct_reading_order_on_page

    # Page 1, bbox.t = 600 (higher on page)
    mock_item1_high = Mock()
    mock_prov1h = Mock()
    mock_bbox1h = Mock()
    mock_bbox1h.t = 600.0
    mock_prov1h.page_no = 1
    mock_prov1h.bbox = mock_bbox1h
    mock_resolved1h = Mock()
    mock_resolved1h.prov = [mock_prov1h]
    mock_resolved1h.children = []
    mock_item1_high.resolve = Mock(return_value=mock_resolved1h)

    # Page 1, bbox.t = 400 (lower on page)
    mock_item1_low = Mock()
    mock_prov1l = Mock()
    mock_bbox1l = Mock()
    mock_bbox1l.t = 400.0
    mock_prov1l.page_no = 1
    mock_prov1l.bbox = mock_bbox1l
    mock_resolved1l = Mock()
    mock_resolved1l.prov = [mock_prov1l]
    mock_resolved1l.children = []
    mock_item1_low.resolve = Mock(return_value=mock_resolved1l)

    # Page 2, bbox.t = 500 (irrelevant for this test)
    mock_item2 = Mock()
    mock_prov2 = Mock()
    mock_bbox2 = Mock()
    mock_bbox2.t = 500.0
    mock_prov2.page_no = 2
    mock_prov2.bbox = mock_bbox2
    mock_resolved2 = Mock()
    mock_resolved2.prov = [mock_prov2]
    mock_resolved2.children = []
    mock_item2.resolve = Mock(return_value=mock_resolved2)

    mock_doc = Mock()
    mock_body = Mock()
    mock_doc.body = mock_body

    # Items in order: page1-low, page1-high (wrong order within page 1, before page 2)
    # Should reorder to: page1-high, page1-low, page2
    mock_body.children = [mock_item1_low, mock_item1_high, mock_item2]

    moved = correct_reading_order_on_page(mock_doc)

    # Page 1 items should be reordered by bbox within their page
    assert mock_body.children[0] == mock_item1_high, "Page 1 high bbox should be first"
    assert mock_body.children[1] == mock_item1_low, "Page 1 low bbox should be second"
    assert mock_body.children[2] == mock_item2, "Page 2 should remain last"
    assert moved > 0, "Should have repositioned items"


def test_correct_reading_order_empty_document():
    """Test handling of empty document."""
    from docling.pipeline.reading_order_utils import correct_reading_order_on_page

    mock_doc = Mock()
    mock_body = Mock()
    mock_body.children = []
    mock_doc.body = mock_body

    moved = correct_reading_order_on_page(mock_doc)

    assert moved == 0, "Empty document should return 0"
    assert mock_body.children == [], "Children should remain empty"


def test_correct_reading_order_single_item():
    """Test handling of single item document."""
    from docling.pipeline.reading_order_utils import correct_reading_order_on_page

    mock_item = Mock()
    mock_item_prov = Mock()
    mock_item_bbox = Mock()
    mock_item_bbox.t = 500.0
    mock_item_prov.page_no = 1
    mock_item_prov.bbox = mock_item_bbox
    mock_item_resolved = Mock()
    mock_item_resolved.prov = [mock_item_prov]
    mock_item_resolved.children = []
    mock_item.resolve = Mock(return_value=mock_item_resolved)

    mock_doc = Mock()
    mock_body = Mock()
    mock_body.children = [mock_item]
    mock_doc.body = mock_body

    moved = correct_reading_order_on_page(mock_doc)

    assert moved == 0, "Single item should not move"
    assert mock_body.children == [mock_item]


def test_correct_reading_order_with_no_provenance_item():
    """Test handling of items with no provenance (e.g., empty groups)."""
    from docling.pipeline.reading_order_utils import correct_reading_order_on_page

    # Item with provenance: page 1, bbox.t = 500
    mock_item_with_prov = Mock()
    mock_prov = Mock()
    mock_bbox = Mock()
    mock_bbox.t = 500.0
    mock_prov.page_no = 1
    mock_prov.bbox = mock_bbox
    mock_resolved = Mock()
    mock_resolved.prov = [mock_prov]
    mock_resolved.children = []
    mock_item_with_prov.resolve = Mock(return_value=mock_resolved)

    # Item without provenance (should be anchored)
    mock_item_no_prov = Mock()
    mock_resolved_no_prov = Mock()
    mock_resolved_no_prov.prov = None
    mock_resolved_no_prov.children = []
    mock_item_no_prov.resolve = Mock(return_value=mock_resolved_no_prov)

    mock_doc = Mock()
    mock_body = Mock()
    mock_body.children = [mock_item_no_prov, mock_item_with_prov]
    mock_doc.body = mock_body

    correct_reading_order_on_page(mock_doc)

    # Item without provenance should stay in place (anchored)
    assert mock_body.children[0] == mock_item_no_prov, "No-prov item should stay first"
    assert mock_body.children[1] == mock_item_with_prov, "Prov item should stay second"


def test_get_first_prov_circular_reference():
    """Test that a circular reference does not cause infinite recursion."""
    from docling.pipeline.reading_order_utils import _get_first_prov

    # Create a mock item that has itself as a child (circular reference)
    mock_item = Mock()
    mock_resolved = Mock()
    mock_resolved.prov = None
    mock_resolved.children = [mock_item]  # Cycle created here

    mock_item.resolve = Mock(return_value=mock_resolved)
    mock_doc = Mock()

    # The function should hit the `if id(ref_item) in _visited:` check and return None safely
    result = _get_first_prov(mock_item, mock_doc)

    assert result is None, "Should return None on circular reference"


def test_get_first_prov_resolve_exception():
    """Test handling of an exception raised during item resolution."""
    from docling.pipeline.reading_order_utils import _get_first_prov

    mock_item = Mock()
    # Force the resolve method to crash
    mock_item.resolve.side_effect = Exception("Resolution failed internally")

    mock_doc = Mock()

    # The function should catch the Exception and return None
    result = _get_first_prov(mock_item, mock_doc)

    assert result is None, "Should return None when resolve() raises an exception"


def test_get_first_prov_corrupted_prov_data():
    """Test handling of corrupted provenance data (e.g., missing bbox)."""
    from docling.pipeline.reading_order_utils import _get_first_prov

    mock_item = Mock()
    mock_resolved = Mock()

    # Create a bad provenance object that is missing the bbox attribute
    mock_bad_prov = Mock()
    mock_bad_prov.page_no = 1
    del mock_bad_prov.bbox  # This will force an AttributeError when bbox.t is accessed

    mock_resolved.prov = [mock_bad_prov]
    mock_resolved.children = []

    mock_item.resolve = Mock(return_value=mock_resolved)
    mock_doc = Mock()

    # The function should catch the AttributeError and return None
    result = _get_first_prov(mock_item, mock_doc)

    assert result is None, "Should return None when prov data is corrupted"
