# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Regression tests at the model processor and generation boundaries."""

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from docling.cli.main import _resolve_asr_options
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.asr_model_specs import ADDITIONAL_ASR_PRESETS, AsrModelType
from docling.datamodel.pipeline_options_asr_model import InlineAsrTransformersOptions
from docling.pipeline.asr_transcriber import _AsrModelFactory
from docling.pipeline.asr_transformers import (
    TransformersAsrDecoder,
    _expand_granite_queries,
)


class _Inputs(dict):
    def to(self, device):
        return self


def test_granite_projector_keeps_every_audio_window():
    from transformers import GraniteSpeechConfig
    from transformers.models.granite_speech.modeling_granite_speech import (
        GraniteSpeechEncoderProjector,
    )

    config = GraniteSpeechConfig(
        projector_config={
            "model_type": "blip_2_qformer",
            "hidden_size": 16,
            "num_attention_heads": 4,
            "num_hidden_layers": 2,
            "intermediate_size": 32,
            "encoder_hidden_size": 16,
        },
        text_config={"model_type": "granite", "hidden_size": 16},
        window_size=15,
        downsample_rate=5,
    )
    projector = GraniteSpeechEncoderProjector(config).eval()
    projector.qformer.register_forward_pre_hook(
        _expand_granite_queries, with_kwargs=True
    )
    audio = torch.randn(1, 45, 16)
    with torch.inference_mode():
        output = projector(audio)
        # Decode each window separately as the reference. This checks both the
        # number of returned windows and their content without model downloads.
        expected = torch.cat(
            [projector(audio[:, start : start + 15]) for start in range(0, 45, 15)],
            dim=1,
        )
    assert output.shape == (1, 9, 16)
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize("family", ["granite", "granite_ctc"])
@pytest.mark.parametrize("device", ["cpu", "cuda:0", "mps"])
def test_granite_uses_safe_loading_dtype_on_mps(family, device, model_boundary):
    transformers, _, _ = model_boundary
    options = InlineAsrTransformersOptions(
        repo_id="test/model",
        model_family=family,
        torch_dtype="bfloat16",
    )
    TransformersAsrDecoder(options, device, options.repo_id, False)
    factory = (
        transformers.AutoModelForCTC
        if family == "granite_ctc"
        else transformers.AutoModelForSpeechSeq2Seq
    )
    assert factory.from_pretrained.call_args.kwargs["dtype"] == (
        "float32" if device == "mps" else "bfloat16"
    )


@pytest.fixture
def model_boundary(monkeypatch):
    """Replace downloaded models while keeping tensor slicing and audio shapes real."""
    model = Mock()
    model.generate.return_value = torch.tensor([[10, 11, 42, 43]])
    model.to.return_value = model
    model.eval.return_value = model
    processor = Mock()
    processor.return_value = _Inputs(input_ids=torch.tensor([[10, 11]]))
    processor.tokenizer.apply_chat_template.return_value = "formatted audio prompt"
    processor.tokenizer.batch_decode.return_value = [" decoded speech "]
    processor.batch_decode.return_value = [" decoded speech "]
    transformers = ModuleType("transformers")
    for name in (
        "AutoModelForCTC",
        "AutoModelForSpeechSeq2Seq",
    ):
        setattr(transformers, name, Mock(from_pretrained=Mock(return_value=model)))
    transformers.AutoProcessor = Mock(from_pretrained=Mock(return_value=processor))
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    audio_utils = ModuleType("transformers.pipelines.audio_utils")
    audio_utils.ffmpeg_read = Mock(return_value=np.zeros(16000, dtype=np.float32))
    monkeypatch.setitem(sys.modules, "transformers.pipelines.audio_utils", audio_utils)
    return transformers, model, processor


def test_granite_crops_prompt_and_preserves_revision(model_boundary, tmp_path):
    transformers, model, processor = model_boundary
    path = tmp_path / "sample.wav"
    path.write_bytes(b"audio")
    options = InlineAsrTransformersOptions(
        repo_id="ibm-granite/granite-speech-5.0-2b",
        model_family="granite",
        revision="pinned-model-revision",
        prompt="transcribe these technical terms",
    )
    decoder = TransformersAsrDecoder(options, "cpu", options.repo_id, False)
    assert decoder.transcribe(path) == "decoded speech"
    for factory in (transformers.AutoProcessor, transformers.AutoModelForSpeechSeq2Seq):
        assert factory.from_pretrained.call_args.kwargs["revision"] == options.revision
    assert processor.tokenizer.apply_chat_template.call_args.args[0] == [
        {"role": "user", "content": "<|audio|>transcribe these technical terms"}
    ]
    assert torch.equal(
        processor.tokenizer.batch_decode.call_args.args[0], torch.tensor([[42, 43]])
    )
    assert model.generate.call_args.kwargs["do_sample"] is False


def test_granite_ctc_output_becomes_document_segment(
    model_boundary, tmp_path, monkeypatch
):
    transformers, model, processor = model_boundary
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"audio")
    monkeypatch.setattr(
        "docling.pipeline.asr_transcriber.decide_device",
        lambda *args, **kwargs: "cuda:0",
    )
    monkeypatch.setattr(
        "docling.pipeline.asr_transcriber._audio_duration_seconds", lambda path: 1.0
    )
    options = InlineAsrTransformersOptions(
        repo_id="test/model",
        model_family="granite_ctc",
        revision="must-not-fetch",
    )
    transcriber = _AsrModelFactory.create(options, checkpoint, AcceleratorOptions())
    conversation = transcriber.transcribe(audio)
    assert [(item.text, item.start_time, item.end_time) for item in conversation] == [
        ("decoded speech", 0.0, 1.0)
    ]
    assert (
        transformers.AutoProcessor.from_pretrained.call_args.kwargs["local_files_only"]
        is True
    )
    assert "revision" not in transformers.AutoProcessor.from_pretrained.call_args.kwargs
    assert processor.batch_decode.call_args.kwargs["skip_special_tokens"] is True


@pytest.mark.parametrize(
    "preset", list(ADDITIONAL_ASR_PRESETS.values()), ids=lambda preset: preset.preset_id
)
def test_added_models_resolve_through_the_cli(preset):
    options = _resolve_asr_options(AsrModelType(preset.preset_id))
    assert options.repo_id == preset.model_spec.default_repo_id
    assert options.model_family == preset.model_spec.model_family
