"""Tests for layout object detection model (label mapping, Egret/Heron compatibility)."""

from docling_core.types.doc import DocItemLabel

from docling.models.stages.layout.layout_object_detection_model import (
    LayoutObjectDetectionModel,
)


def test_build_label_map_egret_style_id2label():
    """Egret layout models use id2label with hyphens/spaces (e.g. 'List-item', 'Key-Value Region').

    _build_label_map_from_mapping normalizes these to DocItemLabel enum names (LIST_ITEM, etc.).
    """
    id2label = {
        3: "List-item",
        4: "Page-footer",
        11: "Document Index",
        16: "Key-Value Region",
    }
    label_map = LayoutObjectDetectionModel._build_label_map_from_mapping(id2label)
    assert label_map[3] == DocItemLabel.LIST_ITEM
    assert label_map[4] == DocItemLabel.PAGE_FOOTER
    assert label_map[11] == DocItemLabel.DOCUMENT_INDEX
    assert label_map[16] == DocItemLabel.KEY_VALUE_REGION


def test_build_label_map_heron_style_unchanged():
    """Heron models use underscored labels; normalization should not break them."""
    id2label = {
        0: "list_item",
        1: "page_footer",
        2: "document_index",
    }
    label_map = LayoutObjectDetectionModel._build_label_map_from_mapping(id2label)
    assert label_map[0] == DocItemLabel.LIST_ITEM
    assert label_map[1] == DocItemLabel.PAGE_FOOTER
    assert label_map[2] == DocItemLabel.DOCUMENT_INDEX
