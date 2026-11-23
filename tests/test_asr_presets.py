# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tests for the built-in ASR preset registry and the ASR request options."""

import json

import pytest
from pydantic import ValidationError

from docling.datamodel.accelerator_options import AcceleratorDevice
from docling.datamodel.asr_model_specs import (
    WHISPER_TINY,
    WHISPER_TURBO_NATIVE,
    AsrModelType,
)
from docling.datamodel.pipeline_options import AsrPipelineOptions
from docling.datamodel.pipeline_options_asr_model import (
    ApiAsrOptions,
    InlineAsrMlxAudioOptions,
    InlineAsrOptions,
    InlineAsrTransformersOptions,
)
from docling.datamodel.service.options import ConvertDocumentsOptions


def test_preset_ids_are_the_asr_model_types():
    assert InlineAsrOptions.list_preset_ids() == [model.value for model in AsrModelType]


def test_whisper_tiny_preset_is_the_pipeline_default():
    options = InlineAsrOptions.from_preset("whisper_tiny")

    assert options == AsrPipelineOptions().asr_options
    assert options == WHISPER_TINY


@pytest.mark.parametrize("preset_id", InlineAsrOptions.list_preset_ids())
def test_every_preset_resolves_to_inline_options(preset_id: str):
    options = InlineAsrOptions.from_preset(preset_id)

    assert isinstance(options, InlineAsrOptions)


def test_preset_options_do_not_change_the_module_constant():
    options = InlineAsrOptions.from_preset("whisper_turbo_native")
    options.temperature = 0.5

    assert WHISPER_TURBO_NATIVE.temperature == 0.0


def test_model_family_preset_selects_the_engine_for_the_device(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "docling.datamodel.asr_presets._mlx_audio_is_usable", lambda: True
    )

    auto = InlineAsrOptions.from_preset("parakeet_ctc_1_1b")
    on_cpu = InlineAsrOptions.from_preset("parakeet_ctc_1_1b", AcceleratorDevice.CPU)

    assert isinstance(auto, InlineAsrMlxAudioOptions)
    assert isinstance(on_cpu, InlineAsrTransformersOptions)


def test_unknown_preset_lists_the_available_presets():
    with pytest.raises(KeyError, match=r"Available presets:.*whisper_tiny"):
        InlineAsrOptions.from_preset("whisper_nonexistent")


def test_request_options_accept_asr_preset_or_custom_config():
    assert ConvertDocumentsOptions(asr_preset="whisper_turbo").asr_preset == (
        "whisper_turbo"
    )
    custom = ConvertDocumentsOptions(asr_custom_config=ApiAsrOptions())
    assert isinstance(custom.asr_custom_config, ApiAsrOptions)


def test_request_options_reject_asr_preset_with_custom_config():
    with pytest.raises(ValidationError, match="asr_preset and asr_custom_config"):
        ConvertDocumentsOptions(
            asr_preset="whisper_turbo",
            asr_custom_config=InlineAsrOptions.from_preset("whisper_tiny_native"),
        )


def test_asr_custom_config_keeps_the_engine_fields():
    """A custom config must keep the fields of its subclass in a request."""
    config = InlineAsrOptions.from_preset("whisper_tiny_transformers")
    sent = ConvertDocumentsOptions(asr_custom_config=config).model_dump(mode="json")

    # Form uploads send nested options as JSON strings.
    received = ConvertDocumentsOptions(
        asr_custom_config=json.dumps(sent["asr_custom_config"])
    )

    assert received.asr_custom_config == config.model_dump(mode="json")
    assert sent["asr_custom_config"]["inference_framework"] == "transformers"
