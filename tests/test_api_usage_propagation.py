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


@pytest.mark.parametrize(
    ("streaming", "request_name", "api_result", "expected_stop_reason"),
    [
        (
            False,
            "api_image_request",
            ApiImageRequestResult(
                "description", 7, VlmStopReason.END_OF_SEQUENCE, {"total_tokens": 7}
            ),
            VlmStopReason.END_OF_SEQUENCE,
        ),
        (
            True,
            "api_image_request_streaming",
            ApiImageStreamingRequestResult("done", 8, {"total_tokens": 8}),
            VlmStopReason.UNSPECIFIED,
        ),
    ],
)
def test_api_vlm_model_preserves_usage_on_prediction(
    streaming, request_name, api_result, expected_stop_reason
) -> None:
    options = ApiVlmOptions(
        prompt="Describe",
        url="http://test.api/v1/chat/completions",
        response_format=ResponseFormat.PLAINTEXT,
        custom_stopping_criteria=[_StopOnDone()] if streaming else [],
    )
    model = ApiVlmModel(True, True, options)

    with patch(
        f"docling.models.vlm_pipeline_models.api_vlm_model.{request_name}",
        return_value=api_result,
    ):
        prediction = next(model.process_images([Image.new("RGB", (8, 8))], "Describe"))

    assert prediction.text == api_result.text
    assert prediction.num_tokens == api_result.num_tokens
    assert prediction.usage == api_result.usage
    assert prediction.stop_reason == expected_stop_reason


@pytest.mark.parametrize(
    ("input_data", "request_name", "api_result", "expected_stop_reason"),
    [
        (
            VlmEngineInput(image=Image.new("RGB", (8, 8)), prompt="Describe"),
            "api_image_request",
            ApiImageRequestResult(
                "description", 9, VlmStopReason.END_OF_SEQUENCE, {"total_tokens": 9}
            ),
            VlmStopReason.END_OF_SEQUENCE,
        ),
        (
            VlmEngineInput(
                image=Image.new("RGB", (8, 8)),
                prompt="Describe",
                extra_generation_config={"custom_stopping_criteria": [_StopOnDone()]},
            ),
            "api_image_request_streaming",
            ApiImageStreamingRequestResult("done", 10, {"total_tokens": 10}),
            "custom_criteria",
        ),
    ],
)
def test_api_vlm_engine_preserves_usage_on_output_metadata(
    input_data, request_name, api_result, expected_stop_reason
) -> None:
    engine = ApiVlmEngine(
        enable_remote_services=True,
        options=ApiVlmEngineOptions(url="http://test.api/v1/chat/completions"),
    )

    with patch(
        f"docling.models.inference_engines.vlm.api_openai_compatible_engine.{request_name}",
        return_value=api_result,
    ):
        output = engine.predict_batch([input_data])[0]

    assert output.text == api_result.text
    assert output.stop_reason == expected_stop_reason
    assert output.metadata["num_tokens"] == api_result.num_tokens
    assert output.metadata["usage"] == api_result.usage


def test_picture_description_api_model_forwards_usage_response_key() -> None:
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
        return ApiImageRequestResult("description", 11, VlmStopReason.END_OF_SEQUENCE)

    with patch(
        "docling.models.stages.picture_description.picture_description_api_model.api_image_request",
        side_effect=_api_image_request,
    ):
        result = next(model._annotate_images([Image.new("RGB", (8, 8))]))

    assert result.text == "description"
