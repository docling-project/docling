"""Tests for layout label mapping normalization."""

from unittest.mock import MagicMock

import pytest
from docling_core.types.doc import DocItemLabel

from docling.models.stages.layout.layout_object_detection_model import (
    LayoutObjectDetectionModel,
)


def _make_model_with_labels(id2label: dict) -> LayoutObjectDetectionModel:
    """Create a LayoutObjectDetectionModel with a mocked engine returning the given labels."""
    model = object.__new__(LayoutObjectDetectionModel)
    model.engine = MagicMock()
    model.engine.get_label_mapping.return_value = id2label
    return model


class TestBuildLabelMap:
    """Tests for _build_label_map label normalization."""

    def test_underscore_labels(self):
        """Heron-style labels with underscores should map correctly."""
        labels = {0: "list_item", 1: "page_footer", 2: "section_header"}
        model = _make_model_with_labels(labels)
        result = model._build_label_map()
        assert result == {
            0: DocItemLabel.LIST_ITEM,
            1: DocItemLabel.PAGE_FOOTER,
            2: DocItemLabel.SECTION_HEADER,
        }

    def test_hyphenated_labels(self):
        """Egret-style labels with hyphens should map correctly."""
        labels = {
            3: "List-item",
            4: "Page-footer",
            5: "Page-header",
            7: "Section-header",
            13: "Checkbox-Selected",
            14: "Checkbox-Unselected",
            16: "Key-Value Region",
        }
        model = _make_model_with_labels(labels)
        result = model._build_label_map()
        assert result[3] == DocItemLabel.LIST_ITEM
        assert result[4] == DocItemLabel.PAGE_FOOTER
        assert result[5] == DocItemLabel.PAGE_HEADER
        assert result[7] == DocItemLabel.SECTION_HEADER
        assert result[13] == DocItemLabel.CHECKBOX_SELECTED
        assert result[14] == DocItemLabel.CHECKBOX_UNSELECTED
        assert result[16] == DocItemLabel.KEY_VALUE_REGION

    def test_space_separated_labels(self):
        """Labels with spaces (e.g. 'Document Index') should map correctly."""
        labels = {11: "Document Index"}
        model = _make_model_with_labels(labels)
        result = model._build_label_map()
        assert result[11] == DocItemLabel.DOCUMENT_INDEX

    def test_invalid_label_raises(self):
        """Unknown labels should raise RuntimeError."""
        labels = {0: "nonexistent_label"}
        model = _make_model_with_labels(labels)
        with pytest.raises(RuntimeError, match="does not match any DocItemLabel"):
            model._build_label_map()
