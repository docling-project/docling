# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""The engine output -> VlmPrediction mapping keeps the API failure marker."""

import pytest

from docling.datamodel.base_models import VlmStopReason
from docling.models.inference_engines.vlm.base import VlmEngineOutput
from docling.models.stages.vlm_convert.vlm_convert_model import (
    _prediction_from_engine_output,
)

pytestmark = pytest.mark.ml_vlm


def test_inference_error_output_maps_to_prediction_with_reason() -> None:
    output = VlmEngineOutput(
        text="",
        stop_reason=VlmStopReason.INFERENCE_ERROR.value,
        metadata={"error": "HTTP 400: unsupported parameter"},
    )

    prediction = _prediction_from_engine_output(output)

    assert prediction.text == ""
    assert prediction.stop_reason == VlmStopReason.INFERENCE_ERROR
    assert prediction.error_message == "HTTP 400: unsupported parameter"


def test_regular_output_has_no_error_message() -> None:
    output = VlmEngineOutput(text="hello", stop_reason="end_of_sequence")

    prediction = _prediction_from_engine_output(output)

    assert prediction.stop_reason == VlmStopReason.END_OF_SEQUENCE
    assert prediction.error_message is None
