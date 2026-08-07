"""Regression coverage for LayoutObjectDetectionModel label normalization."""

from docling_core.types.doc import DocItemLabel

from docling.models.stages.layout.layout_object_detection_model import (
    LayoutObjectDetectionModel,
)


class _HyphenatedLabelEngine:
    """Engine stub whose config id2label uses hyphenated / spaced names."""

    def get_label_mapping(self):
        return {0: "List-item", 1: "Section-header", 2: "Key value region"}


def test_build_label_map_normalizes_hyphenated_and_spaced_labels():
    # Model configs expose labels like "List-item"; .upper() alone yields
    # "LIST-ITEM", which is not a DocItemLabel member name (LIST_ITEM is), so
    # the enum lookup used to raise. Hyphens and spaces must become underscores.
    model = LayoutObjectDetectionModel.__new__(LayoutObjectDetectionModel)
    model.engine = _HyphenatedLabelEngine()

    label_map = model._build_label_map()

    assert label_map == {
        0: DocItemLabel.LIST_ITEM,
        1: DocItemLabel.SECTION_HEADER,
        2: DocItemLabel.KEY_VALUE_REGION,
    }
