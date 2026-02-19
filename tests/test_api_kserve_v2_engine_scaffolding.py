"""Tests for API KServe v2 remote engine scaffolding."""

from __future__ import annotations

from typing import ClassVar

import pytest
from pydantic import BaseModel

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.image_classification_engine_options import (
    ApiKserveV2ImageClassificationEngineOptions,
    BaseImageClassificationEngineOptions,
)
from docling.datamodel.object_detection_engine_options import (
    ApiKserveV2ObjectDetectionEngineOptions,
    BaseObjectDetectionEngineOptions,
)
from docling.datamodel.stage_model_specs import (
    ImageClassificationModelSpec,
    ImageClassificationStagePreset,
    ImageClassificationStagePresetMixin,
    ObjectDetectionModelSpec,
    ObjectDetectionStagePreset,
    ObjectDetectionStagePresetMixin,
)
from docling.exceptions import OperationNotAllowed
from docling.models.inference_engines.common.kserve_v2_http import KserveV2HttpClient
from docling.models.inference_engines.image_classification import (
    ImageClassificationEngineType,
    create_image_classification_engine,
)
from docling.models.inference_engines.object_detection import (
    ObjectDetectionEngineType,
    create_object_detection_engine,
)


class _DummyObjectDetectionStageOptions(ObjectDetectionStagePresetMixin, BaseModel):
    kind: ClassVar[str] = "dummy_object_detection_stage_options"
    model_spec: ObjectDetectionModelSpec
    engine_options: BaseObjectDetectionEngineOptions


class _DummyImageClassificationStageOptions(
    ImageClassificationStagePresetMixin, BaseModel
):
    kind: ClassVar[str] = "dummy_image_classification_stage_options"
    model_spec: ImageClassificationModelSpec
    engine_options: BaseImageClassificationEngineOptions


def test_object_detection_preset_supports_api_kserve_v2_engine_default() -> None:
    preset = ObjectDetectionStagePreset(
        preset_id="od_api_kserve_v2_test",
        name="OD API KServe v2 Test",
        description="OD API KServe v2 preset",
        model_spec=ObjectDetectionModelSpec(name="od", repo_id="org/od"),
        default_engine_type=ObjectDetectionEngineType.API_KSERVE_V2,
    )
    _DummyObjectDetectionStageOptions.register_preset(preset)

    # API_KSERVE_V2 presets require explicit engine_options with URL
    engine_opts = ApiKserveV2ObjectDetectionEngineOptions(url="http://localhost:8000")
    options = _DummyObjectDetectionStageOptions.from_preset(
        "od_api_kserve_v2_test", engine_options=engine_opts
    )
    assert isinstance(options.engine_options, ApiKserveV2ObjectDetectionEngineOptions)


def test_image_classification_preset_supports_api_kserve_v2_engine_default() -> None:
    preset = ImageClassificationStagePreset(
        preset_id="ic_api_kserve_v2_test",
        name="IC API KServe v2 Test",
        description="IC API KServe v2 preset",
        model_spec=ImageClassificationModelSpec(name="ic", repo_id="org/ic"),
        default_engine_type=ImageClassificationEngineType.API_KSERVE_V2,
    )
    _DummyImageClassificationStageOptions.register_preset(preset)

    # API_KSERVE_V2 presets require explicit engine_options with URL
    engine_opts = ApiKserveV2ImageClassificationEngineOptions(
        url="http://localhost:8000"
    )
    options = _DummyImageClassificationStageOptions.from_preset(
        "ic_api_kserve_v2_test", engine_options=engine_opts
    )
    assert isinstance(
        options.engine_options, ApiKserveV2ImageClassificationEngineOptions
    )


def test_object_detection_factory_requires_remote_enablement() -> None:
    options = ApiKserveV2ObjectDetectionEngineOptions(
        url="http://localhost:8000",
        model_name="od_model",
    )
    spec = ObjectDetectionModelSpec(name="od", repo_id="org/od")

    with pytest.raises(OperationNotAllowed):
        create_object_detection_engine(
            options=options,
            model_spec=spec,
            enable_remote_services=False,
            accelerator_options=AcceleratorOptions(),
        )

    engine = create_object_detection_engine(
        options=options,
        model_spec=spec,
        enable_remote_services=True,
        accelerator_options=AcceleratorOptions(),
    )
    assert engine.options.engine_type == ObjectDetectionEngineType.API_KSERVE_V2


def test_image_classification_factory_requires_remote_enablement() -> None:
    options = ApiKserveV2ImageClassificationEngineOptions(
        url="http://localhost:8000",
        model_name="ic_model",
    )
    spec = ImageClassificationModelSpec(name="ic", repo_id="org/ic")

    with pytest.raises(OperationNotAllowed):
        create_image_classification_engine(
            options=options,
            model_spec=spec,
            enable_remote_services=False,
            accelerator_options=AcceleratorOptions(),
        )

    engine = create_image_classification_engine(
        options=options,
        model_spec=spec,
        enable_remote_services=True,
        accelerator_options=AcceleratorOptions(),
    )
    assert engine.options.engine_type == ImageClassificationEngineType.API_KSERVE_V2


def test_kserve_v2_client_infer_url_with_version() -> None:
    client = KserveV2HttpClient(
        base_url="http://localhost:8000",
        model_name="layout_model",
        model_version="1",
        timeout=30.0,
        headers={},
    )
    assert (
        client.infer_url
        == "http://localhost:8000/v2/models/layout_model/versions/1/infer"
    )


def test_kserve_v2_client_infer_url_without_version() -> None:
    client = KserveV2HttpClient(
        base_url="http://localhost:8000",
        model_name="layout_model",
        model_version=None,
        timeout=30.0,
        headers={},
    )
    assert client.infer_url == "http://localhost:8000/v2/models/layout_model/infer"
