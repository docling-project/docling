from unittest.mock import patch

import pytest
from PIL import Image

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import (
    ApiImageRequestResult,
    ApiImageStreamingRequestResult,
    VlmStopReason,
)
from docling.datamodel.pipeline_options import PictureDescriptionApiOptions
from docling.datamodel.pipeline_options_vlm_model import ApiVlmOptions, ResponseFormat
from docling.datamodel.vlm_engine_options import ApiVlmEngineOptions
from docling.models.inference_engines.vlm.api_openai_compatible_engine import (
    ApiVlmEngine,
)
from docling.models.inference_engines.vlm.base import VlmEngineInput
from docling.models.stages.picture_description.picture_description_api_model import (
    PictureDescriptionApiModel,
)
from docling.models.utils.generation_utils import GenerationStopper
from docling.models.vlm_pipeline_models.api_vlm_model import ApiVlmModel

pytestmark = pytest.mark.cross_platform


class _StopOnDone(GenerationStopper):
    def should_stop(self, s: str) -> bool:
        return "done" in s


def test_api_vlm_model_preserves_usage_on_prediction() -> None:
    image = Image.new("RGB", (8, 8), "red")
    options = ApiVlmOptions(
        prompt="Describe",
        url="http://test.api/v1/chat/completions",
        response_format=ResponseFormat.PLAINTEXT,
    )
    model = ApiVlmModel(
        enabled=True,
        enable_remote_services=True,
        vlm_options=options,
    )

    with patch(
        "docling.models.vlm_pipeline_models.api_vlm_model.api_image_request",
        return_value=ApiImageRequestResult(
            text="description",
            num_tokens=7,
            stop_reason=VlmStopReason.END_OF_SEQUENCE,
            usage={"total_tokens": 7},
        ),
    ):
        predictions = list(model.process_images([image], "Describe"))

    assert predictions[0].text == "description"
    assert predictions[0].num_tokens == 7
    assert predictions[0].usage == {"total_tokens": 7}


def test_api_vlm_model_preserves_streaming_usage_on_prediction() -> None:
    image = Image.new("RGB", (8, 8), "red")
    options = ApiVlmOptions(
        prompt="Describe",
        url="http://test.api/v1/chat/completions",
        response_format=ResponseFormat.PLAINTEXT,
        custom_stopping_criteria=[_StopOnDone()],
    )
    model = ApiVlmModel(
        enabled=True,
        enable_remote_services=True,
        vlm_options=options,
    )

    with patch(
        "docling.models.vlm_pipeline_models.api_vlm_model.api_image_request_streaming",
        return_value=ApiImageStreamingRequestResult(
            text="done",
            num_tokens=8,
            usage={"total_tokens": 8},
        ),
    ):
        predictions = list(model.process_images([image], "Describe"))

    assert predictions[0].text == "done"
    assert predictions[0].num_tokens == 8
    assert predictions[0].usage == {"total_tokens": 8}


def test_api_vlm_engine_preserves_usage_on_output_metadata() -> None:
    image = Image.new("RGB", (8, 8), "red")
    engine = ApiVlmEngine(
        enable_remote_services=True,
        options=ApiVlmEngineOptions(url="http://test.api/v1/chat/completions"),
    )

    with patch(
        "docling.models.inference_engines.vlm.api_openai_compatible_engine.api_image_request",
        return_value=ApiImageRequestResult(
            text="description",
            num_tokens=9,
            stop_reason=VlmStopReason.END_OF_SEQUENCE,
            usage={"total_tokens": 9},
        ),
    ):
        outputs = engine.predict_batch([VlmEngineInput(image=image, prompt="Describe")])

    assert outputs[0].text == "description"
    assert outputs[0].metadata["num_tokens"] == 9
    assert outputs[0].metadata["usage"] == {"total_tokens": 9}


def test_api_vlm_engine_preserves_streaming_usage_on_output_metadata() -> None:
    image = Image.new("RGB", (8, 8), "red")
    engine = ApiVlmEngine(
        enable_remote_services=True,
        options=ApiVlmEngineOptions(url="http://test.api/v1/chat/completions"),
    )

    with patch(
        "docling.models.inference_engines.vlm.api_openai_compatible_engine.api_image_request_streaming",
        return_value=ApiImageStreamingRequestResult(
            text="done",
            num_tokens=10,
            usage={"total_tokens": 10},
        ),
    ):
        outputs = engine.predict_batch(
            [
                VlmEngineInput(
                    image=image,
                    prompt="Describe",
                    extra_generation_config={
                        "custom_stopping_criteria": [_StopOnDone()],
                    },
                )
            ]
        )

    assert outputs[0].text == "done"
    assert outputs[0].stop_reason == "custom_criteria"
    assert outputs[0].metadata["num_tokens"] == 10
    assert outputs[0].metadata["usage"] == {"total_tokens": 10}


def test_picture_description_api_model_forwards_usage_response_key() -> None:
    image = Image.new("RGB", (8, 8), "red")
    model = PictureDescriptionApiModel(
        enabled=True,
        enable_remote_services=True,
        artifacts_path=None,
        options=PictureDescriptionApiOptions(
            url="http://test.api/v1/chat/completions",
            usage_response_key="providerUsage",
        ),
        accelerator_options=AcceleratorOptions(),
    )

    def _api_image_request(**kwargs):
        assert kwargs["usage_response_key"] == "providerUsage"
        return ApiImageRequestResult(
            text="description",
            num_tokens=11,
            stop_reason=VlmStopReason.END_OF_SEQUENCE,
            usage={"provider_tokens": 11},
        )

    with patch(
        "docling.models.stages.picture_description.picture_description_api_model.api_image_request",
        side_effect=_api_image_request,
    ):
        results = list(model._annotate_images([image]))

    assert results[0].text == "description"
    assert results[0].usage == {"provider_tokens": 11}
