# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Regression tests at the model processor and generation boundaries."""

import importlib.metadata
import subprocess
import sys
from types import ModuleType
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from docling.cli.main import _resolve_asr_options
from docling.datamodel import asr_presets
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.asr_model_specs import (
    ADDITIONAL_ASR_PRESETS,
    GRANITE_SPEECH_5_0_2B_PRESET,
    GRANITE_SPEECH_5_0_470M_TURBOCTC_PRESET,
    GRANITE_SPEECH_5_0_470M_TURBOCTC_TRANSFORMERS,
    AsrModelType,
)
from docling.datamodel.pipeline_options_asr_model import (
    InferenceAsrFramework,
    InlineAsrTransformersOptions,
)
from docling.pipeline import asr_mlx_audio
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
    assert [segment.text for segment in decoder.transcribe(path)] == ["decoded speech"]
    for factory in (transformers.AutoProcessor, transformers.AutoModelForSpeechSeq2Seq):
        assert factory.from_pretrained.call_args.kwargs["revision"] == options.revision
    assert processor.tokenizer.apply_chat_template.call_args.args[0] == [
        {"role": "user", "content": "<|audio|>transcribe these technical terms"}
    ]
    assert torch.equal(
        processor.tokenizer.batch_decode.call_args.args[0], torch.tensor([[42, 43]])
    )
    assert model.generate.call_args.kwargs["do_sample"] is False


@pytest.mark.parametrize("family", ["granite", "granite_ctc"])
def test_long_granite_audio_gives_one_timed_item_per_window(
    family, model_boundary, tmp_path, monkeypatch
):
    transformers, model, processor = model_boundary
    # 60 s of speech with a pause at 33 s: a cut there leaves two windows.
    waveform = np.full(60 * 16000, 0.2, dtype=np.float32)
    waveform[33 * 16000 : 34 * 16000] = 0.0
    sys.modules[
        "transformers.pipelines.audio_utils"
    ].ffmpeg_read.return_value = waveform
    decode = (
        processor.batch_decode
        if family == "granite_ctc"
        else processor.tokenizer.batch_decode
    )
    decode.side_effect = [[" first "], [" second "]]
    monkeypatch.setattr(
        "docling.pipeline.asr_transcriber.decide_device", lambda *args, **kwargs: "cpu"
    )
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"audio")
    options = InlineAsrTransformersOptions(repo_id="test/model", model_family=family)

    items = _AsrModelFactory.create(options, None, AcceleratorOptions()).transcribe(
        audio
    )

    assert [item.text for item in items] == ["first", "second"]
    assert items[0].start_time == 0.0
    assert items[0].end_time == pytest.approx(33.5, abs=0.05)
    assert items[1].start_time == items[0].end_time
    assert items[1].end_time == pytest.approx(60.0)
    assert model.generate.call_count == 2


def test_long_granite_audio_does_not_lose_the_end_of_the_text(model_boundary, tmp_path):
    # A 3-minute clip stopped in mid-sentence, because the limit of 256 tokens applied
    # to the full recording. The stand-in model stops at the limit in the same way.
    _, model, processor = model_boundary
    tokens_per_second = 10
    seconds = 180
    waveform = np.full(seconds * 16000, 0.2, dtype=np.float32)
    sys.modules[
        "transformers.pipelines.audio_utils"
    ].ffmpeg_read.return_value = waveform
    window_seconds: list[float] = []

    def process(_prompt, audio, **_kwargs):
        window_seconds.append(audio.shape[-1] / 16000)
        return _Inputs(input_ids=torch.tensor([[10, 11]]))

    def generate(input_ids, max_new_tokens, **_kwargs):
        spoken = round(window_seconds[-1] * tokens_per_second)
        return torch.cat(
            [input_ids, torch.ones(1, min(spoken, max_new_tokens), dtype=torch.long)],
            dim=1,
        )

    processor.side_effect = process
    model.generate.side_effect = generate
    processor.tokenizer.batch_decode.side_effect = lambda ids, **_kwargs: [
        " ".join(["word"] * ids.shape[-1])
    ]
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"audio")
    options = InlineAsrTransformersOptions(repo_id="test/model", model_family="granite")

    items = TransformersAsrDecoder(options, "cpu", options.repo_id, False).transcribe(
        audio
    )

    assert sum(len(item.text.split()) for item in items) == seconds * tokens_per_second


def _auto_selected_mlx_audio(monkeypatch, probe_exit_code: int = 0) -> Mock:
    """Report Apple Silicon with mlx-audio. Returns the Metal probe."""
    monkeypatch.setattr(asr_presets, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(asr_mlx_audio, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(importlib.metadata, "version", lambda package: "0.6.4")
    monkeypatch.setattr(
        "docling.pipeline.asr_transcriber.decide_device", lambda *args, **kwargs: "cpu"
    )
    probe = Mock(return_value=subprocess.CompletedProcess([], probe_exit_code))
    monkeypatch.setattr(subprocess, "run", probe)
    return probe


# -6 is SIGABRT: the import of `mlx` can abort a restricted process.
@pytest.mark.parametrize("probe_exit_code", [1, -6])
def test_auto_selected_mlx_audio_uses_transformers_without_metal(
    model_boundary, tmp_path, monkeypatch, probe_exit_code
):
    transformers, _, _ = model_boundary
    _auto_selected_mlx_audio(monkeypatch, probe_exit_code)
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"audio")
    options = _resolve_asr_options(AsrModelType.GRANITE_SPEECH_5_0_470M_TURBOCTC)
    assert options.inference_framework == InferenceAsrFramework.MLX_AUDIO

    items = _AsrModelFactory.create(options, None, AcceleratorOptions()).transcribe(
        audio
    )

    assert [item.text for item in items] == ["decoded speech"]
    assert (
        transformers.AutoModelForCTC.from_pretrained.call_args.args[0]
        == GRANITE_SPEECH_5_0_470M_TURBOCTC_TRANSFORMERS.repo_id
    )


def test_sdk_constant_follows_the_device_of_the_accelerator_options(
    model_boundary, tmp_path, monkeypatch
):
    # The SDK constants select at import, before the device is known.
    transformers, _, _ = model_boundary
    probe = _auto_selected_mlx_audio(monkeypatch)
    options = GRANITE_SPEECH_5_0_470M_TURBOCTC_PRESET.to_auto_options()

    _AsrModelFactory.create(options, None, AcceleratorOptions(device="cpu"))

    transformers.AutoModelForCTC.from_pretrained.assert_called_once()
    probe.assert_not_called()


def test_fallback_keeps_the_settings_of_the_user(model_boundary, tmp_path, monkeypatch):
    _, model, _ = model_boundary
    _auto_selected_mlx_audio(monkeypatch, probe_exit_code=1)
    sys.modules[
        "transformers.pipelines.audio_utils"
    ].ffmpeg_read.return_value = np.full(60 * 16000, 0.2, dtype=np.float32)
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"audio")
    options = GRANITE_SPEECH_5_0_2B_PRESET.to_auto_options().model_copy(
        update={"language": "fr", "max_time_chunk": 15.0, "chunk_search_window": 5.0}
    )

    transcriber = _AsrModelFactory.create(options, None, AcceleratorOptions())
    transcriber.transcribe(audio)

    assert transcriber.asr_options.language == "fr"
    assert model.generate.call_count == 4


def test_silent_window_gives_no_item(model_boundary, tmp_path):
    _, _, processor = model_boundary
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"audio")
    processor.tokenizer.batch_decode.return_value = ["  "]
    options = InlineAsrTransformersOptions(repo_id="test/model", model_family="granite")

    decoder = TransformersAsrDecoder(options, "cpu", options.repo_id, False)

    assert decoder.transcribe(audio) == []


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
    config = preset.model_spec.get_engine_config(options.inference_framework)
    assert options.repo_id == config.repo_id
    assert options.revision == config.revision
    assert options.model_family == preset.model_spec.model_family
