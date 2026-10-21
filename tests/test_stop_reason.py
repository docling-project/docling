# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""The stop reason values that the local VLM engines hand to the normaliser."""

import pytest

from docling.datamodel.base_models import VlmStopReason
from docling.utils.stop_reason import normalize_stop_reason


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # MLX reports custom_criteria when a GenerationStopper ended the
        # generation, which is a stopped sequence and not a natural end.
        ("custom_criteria", VlmStopReason.STOP_SEQUENCE),
        # The configured stop strings are the proper end of the output.
        ("stop_string", VlmStopReason.END_OF_SEQUENCE),
        # An unknown value must never be read as a finished sequence.
        ("made_up", VlmStopReason.UNSPECIFIED),
    ],
)
def test_the_mlx_stop_reasons_are_mapped(raw, expected) -> None:
    assert normalize_stop_reason(raw) == expected


def test_a_vllm_stop_string_end_is_a_completed_call() -> None:
    assert normalize_stop_reason("stop") == VlmStopReason.END_OF_SEQUENCE


@pytest.mark.parametrize("raw", ["abort", "error"])
def test_a_request_that_did_not_finish_is_an_inference_error(raw) -> None:
    assert normalize_stop_reason(raw) == VlmStopReason.INFERENCE_ERROR
