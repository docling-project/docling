"""Tests for API KServe v2 remote engine scaffolding."""

from __future__ import annotations

import pytest

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.image_classification_engine_options import (
    ApiKserveV2ImageClassificationEngineOptions,
)
from docling.datamodel.object_detection_engine_options import (
    ApiKserveV2ObjectDetectionEngineOptions,
)
from docling.datamodel.stage_model_specs import (
    ImageClassificationModelSpec,
    ObjectDetectionModelSpec,
)
from docling.exceptions import OperationNotAllowed
from docling.models.inference_engines.image_classification import (
    create_image_classification_engine,
)
from docling.models.inference_engines.object_detection import (
    create_object_detection_engine,
)

pytestmark = pytest.mark.cross_platform


def test_object_detection_factory_requires_remote_enablement() -> None:
    options = ApiKserveV2ObjectDetectionEngineOptions(
        url="http://localhost:8000",
        model_name="od_model",
        transport="http",
    )
    spec = ObjectDetectionModelSpec(name="od", repo_id="org/od")

    with pytest.raises(OperationNotAllowed):
        create_object_detection_engine(
            options=options,
            model_spec=spec,
            enable_remote_services=False,
            accelerator_options=AcceleratorOptions(),
        )


def test_image_classification_factory_requires_remote_enablement() -> None:
    options = ApiKserveV2ImageClassificationEngineOptions(
        url="http://localhost:8000",
        model_name="ic_model",
        transport="http",
    )
    spec = ImageClassificationModelSpec(name="ic", repo_id="org/ic")

    with pytest.raises(OperationNotAllowed):
        create_image_classification_engine(
            options=options,
            model_spec=spec,
            enable_remote_services=False,
            accelerator_options=AcceleratorOptions(),
        )
