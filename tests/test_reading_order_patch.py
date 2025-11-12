"""
Unit tests for the reading order patch.

Tests that the monkey patch for docling_ibm_models.reading_order.reading_order_rb
is correctly applied and handles edge cases that could cause KeyError.
"""

import pytest
from docling_core.types.doc import DocItemLabel
from docling_ibm_models.reading_order.reading_order_rb import ReadingOrderPredictor


def test_reading_order_patch_applied():
    """Test that the monkey patch was successfully applied."""
    # Import the readingorder_model to trigger the patch
    from docling.models.readingorder_model import ReadingOrderModel  # noqa: F401

    # Verify the patch was applied
    assert hasattr(
        ReadingOrderPredictor, "_original_init_ud_maps"
    ), "Patch was not applied"
    assert (
        ReadingOrderPredictor._init_ud_maps.__name__ == "_patched_init_ud_maps"
    ), "Patched method name doesn't match"


def test_reading_order_model_init():
    """Test that ReadingOrderModel can be initialized with the patch."""
    from docling.models.readingorder_model import ReadingOrderModel, ReadingOrderOptions

    options = ReadingOrderOptions()
    model = ReadingOrderModel(options)
    assert model is not None
    assert model.ro_model is not None
    assert isinstance(model.ro_model, ReadingOrderPredictor)


def test_patched_method_defensive_checks():
    """Test that the patched method handles edge cases gracefully."""
    from dataclasses import dataclass, field
    from typing import Dict, List

    from docling.models.readingorder_model import ReadingOrderModel, ReadingOrderOptions
    from docling_core.types.doc import Size
    from docling_core.types.doc.base import CoordOrigin
    from docling_ibm_models.reading_order.reading_order_rb import PageElement

    options = ReadingOrderOptions()
    model = ReadingOrderModel(options)

    # Create a simple test case with page elements
    @dataclass
    class MockState:
        l2r_map: Dict[int, int] = field(default_factory=dict)
        r2l_map: Dict[int, int] = field(default_factory=dict)
        up_map: Dict[int, List[int]] = field(default_factory=dict)
        dn_map: Dict[int, List[int]] = field(default_factory=dict)

    state = MockState()

    # Create some page elements
    page_elements = [
        PageElement(
            cid=0,
            text="Element 0",
            page_no=0,
            page_size=Size(width=612, height=792),
            l=100,
            r=200,
            b=600,
            t=700,
            coord_origin=CoordOrigin.BOTTOMLEFT,
            label=DocItemLabel.TEXT,
        ),
        PageElement(
            cid=1,
            text="Element 1",
            page_no=0,
            page_size=Size(width=612, height=792),
            l=100,
            r=200,
            b=500,
            t=600,
            coord_origin=CoordOrigin.BOTTOMLEFT,
            label=DocItemLabel.TEXT,
        ),
    ]

    # Test that the patched method can handle the page elements
    # without raising KeyError
    try:
        model.ro_model._init_ud_maps(page_elements, state)
    except KeyError as e:
        pytest.fail(f"Patched method raised KeyError: {e}")

    # Verify that the maps were initialized
    assert len(state.up_map) == 2, "up_map should have 2 entries"
    assert len(state.dn_map) == 2, "dn_map should have 2 entries"
    assert 0 in state.up_map, "Element 0 should be in up_map"
    assert 1 in state.up_map, "Element 1 should be in up_map"
    assert 0 in state.dn_map, "Element 0 should be in dn_map"
    assert 1 in state.dn_map, "Element 1 should be in dn_map"


def test_patched_method_with_l2r_map():
    """Test that the patched method handles l2r_map chains correctly."""
    from dataclasses import dataclass, field
    from typing import Dict, List

    from docling.models.readingorder_model import ReadingOrderModel, ReadingOrderOptions
    from docling_core.types.doc import Size
    from docling_core.types.doc.base import CoordOrigin
    from docling_ibm_models.reading_order.reading_order_rb import PageElement

    options = ReadingOrderOptions()
    model = ReadingOrderModel(options)

    # Create a simple test case with page elements
    @dataclass
    class MockState:
        l2r_map: Dict[int, int] = field(default_factory=dict)
        r2l_map: Dict[int, int] = field(default_factory=dict)
        up_map: Dict[int, List[int]] = field(default_factory=dict)
        dn_map: Dict[int, List[int]] = field(default_factory=dict)

    state = MockState()

    # Create page elements
    page_elements = [
        PageElement(
            cid=0,
            text="Element 0",
            page_no=0,
            page_size=Size(width=612, height=792),
            l=100,
            r=200,
            b=600,
            t=700,
            coord_origin=CoordOrigin.BOTTOMLEFT,
            label=DocItemLabel.TEXT,
        ),
        PageElement(
            cid=1,
            text="Element 1",
            page_no=0,
            page_size=Size(width=612, height=792),
            l=250,
            r=350,
            b=600,
            t=700,
            coord_origin=CoordOrigin.BOTTOMLEFT,
            label=DocItemLabel.TEXT,
        ),
        PageElement(
            cid=2,
            text="Element 2",
            page_no=0,
            page_size=Size(width=612, height=792),
            l=150,
            r=250,
            b=500,
            t=600,
            coord_origin=CoordOrigin.BOTTOMLEFT,
            label=DocItemLabel.TEXT,
        ),
    ]

    # Set up l2r_map with a chain: 0 -> 1
    # Note: This is normally set by _init_l2r_map but we set it manually for testing
    state.l2r_map = {0: 1}

    # Test that the patched method can handle l2r_map chains
    # without raising KeyError even if the chain points to indices
    try:
        model.ro_model._init_ud_maps(page_elements, state)
    except KeyError as e:
        pytest.fail(f"Patched method raised KeyError with l2r_map: {e}")

    # Verify maps were initialized
    assert len(state.up_map) == 3, "up_map should have 3 entries"
    assert len(state.dn_map) == 3, "dn_map should have 3 entries"


def test_patched_method_with_invalid_r2l_map():
    """Test that the patched method handles invalid r2l_map gracefully."""
    from dataclasses import dataclass, field
    from typing import Dict, List

    from docling.models.readingorder_model import ReadingOrderModel, ReadingOrderOptions
    from docling_core.types.doc import Size
    from docling_core.types.doc.base import CoordOrigin
    from docling_ibm_models.reading_order.reading_order_rb import PageElement

    options = ReadingOrderOptions()
    model = ReadingOrderModel(options)

    # Create a simple test case with page elements
    @dataclass
    class MockState:
        l2r_map: Dict[int, int] = field(default_factory=dict)
        r2l_map: Dict[int, int] = field(default_factory=dict)
        up_map: Dict[int, List[int]] = field(default_factory=dict)
        dn_map: Dict[int, List[int]] = field(default_factory=dict)

    state = MockState()

    # Create page elements
    page_elements = [
        PageElement(
            cid=0,
            text="Element 0",
            page_no=0,
            page_size=Size(width=612, height=792),
            l=100,
            r=200,
            b=600,
            t=700,
            coord_origin=CoordOrigin.BOTTOMLEFT,
            label=DocItemLabel.TEXT,
        ),
    ]

    # Set up r2l_map with an invalid reference (index 99 doesn't exist)
    # This simulates the edge case that could cause KeyError
    state.r2l_map = {0: 99}

    # Test that the patched method handles invalid r2l_map gracefully
    try:
        model.ro_model._init_ud_maps(page_elements, state)
    except KeyError as e:
        pytest.fail(f"Patched method raised KeyError with invalid r2l_map: {e}")

    # The patched method should skip the invalid mapping
    # and still initialize the maps correctly
    assert len(state.up_map) == 1, "up_map should have 1 entry"
    assert len(state.dn_map) == 1, "dn_map should have 1 entry"
