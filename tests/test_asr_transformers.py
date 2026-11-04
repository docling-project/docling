# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Local Whisper checkpoint and Transformers ASR boundary tests."""

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.asr_model_specs import (
    NEMOTRON_3_5_ASR_STREAMING,
    PARAKEET_CTC_1_1B,
    WHISPER_ASR_PRESETS,
)
from docling.datamodel.pipeline_options_asr_model import (
    InferenceAsrFramework,
    InlineAsrMlxWhisperOptions,
    InlineAsrTransformersOptions,
)
from docling.pipeline.asr_transcriber import (
    _AsrModelFactory,
    _MlxWhisperModel,
    _TransformersAsrModel,
)


def test_whisper_preset_engine_exports():
    preset = WHISPER_ASR_PRESETS["tiny"]
    assert preset.to_options().repo_id == "openai/whisper-tiny"
    assert (
        preset.to_options(InferenceAsrFramework.MLX).repo_id
        == "mlx-community/whisper-tiny-mlx"
    )
    with pytest.raises(ValueError, match="does not support"):
        preset.to_options(InferenceAsrFramework.WHISPER_S2T)


def test_transformers_uses_local_checkpoint_and_keeps_timestamps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkpoint = tmp_path / "openai--whisper-tiny"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    asr_pipeline = Mock(
        return_value={
            "chunks": [
                {"text": " hello", "timestamp": (0.2, 1.4)},
                {"text": " world", "timestamp": (1.4, 2.0)},
            ]
        }
    )
    make_pipeline = Mock(return_value=asr_pipeline)
    transformers = ModuleType("transformers")
    transformers.pipeline = make_pipeline
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(
        "docling.pipeline.asr_transcriber.decide_device", lambda *args, **kwargs: "cpu"
    )

    model = _AsrModelFactory.create(
        InlineAsrTransformersOptions(repo_id="openai/whisper-tiny"),
        tmp_path,
        AcceleratorOptions(),
    )
    assert isinstance(model, _TransformersAsrModel)
    assert make_pipeline.call_args.kwargs["model"] == str(checkpoint)
    assert make_pipeline.call_args.kwargs["model_kwargs"] == {"local_files_only": True}
    conversation = model.transcribe(Path("sample.wav"))
    assert [(item.text, item.start_time, item.end_time) for item in conversation] == [
        ("hello", 0.2, 1.4),
        ("world", 1.4, 2.0),
    ]
    assert asr_pipeline.call_args.kwargs["return_timestamps"] is True


def test_local_checkpoint_is_required_without_hub_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    transformers = ModuleType("transformers")
    transformers.pipeline = Mock()
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(
        "docling.pipeline.asr_transcriber.decide_device", lambda *args, **kwargs: "cpu"
    )
    with pytest.raises(FileNotFoundError, match="openai/whisper-tiny"):
        _AsrModelFactory.create(
            InlineAsrTransformersOptions(repo_id="openai/whisper-tiny"),
            tmp_path,
            AcceleratorOptions(),
        )
    transformers.pipeline.assert_not_called()


def test_mlx_uses_local_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    checkpoint = tmp_path / "mlx-community--whisper-tiny-mlx"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "weights.safetensors").write_bytes(b"checkpoint")
    mlx_whisper = ModuleType("mlx_whisper")
    mlx_whisper.transcribe = Mock(return_value={"segments": []})
    monkeypatch.setitem(sys.modules, "mlx_whisper", mlx_whisper)
    monkeypatch.setattr(
        "docling.pipeline.asr_transcriber.decide_device", lambda *args, **kwargs: "mps"
    )
    model = _MlxWhisperModel(
        True,
        tmp_path,
        AcceleratorOptions(),
        InlineAsrMlxWhisperOptions(repo_id="mlx-community/whisper-tiny-mlx"),
    )
    model.transcribe(Path("sample.wav"))
    assert mlx_whisper.transcribe.call_args.kwargs["path_or_hf_repo"] == str(checkpoint)


@pytest.mark.parametrize(
    ("options", "family"),
    [
        (PARAKEET_CTC_1_1B, "parakeet"),
        (NEMOTRON_3_5_ASR_STREAMING, "nemotron"),
    ],
)
def test_other_transformers_models_keep_text_without_segment_timestamps(
    options: InlineAsrTransformersOptions,
    family: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    checkpoint = tmp_path / options.repo_cache_folder
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    asr_pipeline = Mock(return_value={"text": " recognized speech "})
    transformers = ModuleType("transformers")
    transformers.pipeline = Mock(return_value=asr_pipeline)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(
        "docling.pipeline.asr_transcriber.decide_device", lambda *args, **kwargs: "cpu"
    )
    monkeypatch.setattr(
        "docling.pipeline.asr_transcriber._audio_duration_seconds", lambda path: 2.5
    )
    model = _AsrModelFactory.create(options, tmp_path, AcceleratorOptions())
    conversation = model.transcribe(Path("sample.wav"))
    assert options.model_family == family
    assert [(item.text, item.start_time, item.end_time) for item in conversation] == [
        ("recognized speech", 0.0, 2.5)
    ]
    asr_pipeline.assert_called_once_with("sample.wav")
