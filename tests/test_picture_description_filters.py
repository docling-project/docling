from docling_core.types.doc import (
    PictureClassificationLabel,
    PictureClassificationMetaField,
    PictureMeta,
)

from docling.models.picture_description_base_model import _passes_classification


def _meta_with_predictions(predictions):
    return PictureMeta(
        classification=PictureClassificationMetaField(predictions=predictions)
    )


def test_passes_with_no_filters():
    assert _passes_classification(None, None, None, 0.5)


def test_allow_without_predictions_fails():
    assert not _passes_classification(
        None,
        [PictureClassificationLabel.BAR_CHART],
        None,
        0.0,
    )


def test_deny_without_predictions_passes():
    assert _passes_classification(
        None,
        None,
        [PictureClassificationLabel.BAR_CHART],
        0.0,
    )


def test_deny_blocks_matching_prediction():
    meta = _meta_with_predictions([{"class_name": "bar_chart", "confidence": 0.9}])
    assert not _passes_classification(
        meta,
        None,
        [PictureClassificationLabel.BAR_CHART],
        0.0,
    )


def test_allow_accepts_matching_prediction():
    meta = _meta_with_predictions([{"class_name": "bar_chart", "confidence": 0.9}])
    assert _passes_classification(
        meta,
        [PictureClassificationLabel.BAR_CHART],
        None,
        0.0,
    )


def test_allow_respects_min_confidence():
    meta = _meta_with_predictions([{"class_name": "bar_chart", "confidence": 0.1}])
    assert not _passes_classification(
        meta,
        [PictureClassificationLabel.BAR_CHART],
        None,
        0.5,
    )
