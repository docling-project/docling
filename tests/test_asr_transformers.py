# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Local Whisper checkpoint and Transformers ASR boundary tests."""

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import numpy as np
import pytest

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.asr_model_specs import (
    NEMOTRON_3_5_ASR_STREAMING_TRANSFORMERS,
    PARAKEET_CTC_1_1B_TRANSFORMERS,
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


@pytest.mark.parametrize("temperature", [0.0, 0.2])
def test_transformers_whisper_forwards_temperature(
    temperature: float, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkpoint = tmp_path / "openai--whisper-tiny"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    asr_pipeline = Mock(
        return_value={"chunks": [{"text": " hello", "timestamp": (0.2, 1.4)}]}
    )
    transformers = ModuleType("transformers")
    transformers.pipeline = Mock(return_value=asr_pipeline)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(
        "docling.pipeline.asr_transcriber.decide_device", lambda *args, **kwargs: "cpu"
    )

    model = _AsrModelFactory.create(
        InlineAsrTransformersOptions(
            repo_id="openai/whisper-tiny", temperature=temperature
        ),
        tmp_path,
        AcceleratorOptions(),
    )
    model.transcribe(Path("sample.wav"))
    assert asr_pipeline.call_args.kwargs["generate_kwargs"] == {
        "task": "transcribe",
        "temperature": temperature,
    }


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


def _setup_non_whisper_pipeline(
    options: InlineAsrTransformersOptions,
    waveform: np.ndarray,
    responses: list[dict[str, str]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Mock, Path]:
    checkpoint = tmp_path / options.repo_cache_folder
    checkpoint.mkdir(exist_ok=True)
    (checkpoint / "config.json").write_text("{}")
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"audio")
    asr_pipeline = Mock(side_effect=responses)
    transformers = ModuleType("transformers")
    transformers.pipeline = Mock(return_value=asr_pipeline)
    audio_utils = ModuleType("transformers.pipelines.audio_utils")
    audio_utils.ffmpeg_read = Mock(return_value=waveform)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "transformers.pipelines.audio_utils", audio_utils)
    monkeypatch.setattr(
        "docling.pipeline.asr_transcriber.decide_device", lambda *args, **kwargs: "cpu"
    )
    return asr_pipeline, audio


@pytest.mark.parametrize(
    ("options", "family"),
    [
        (PARAKEET_CTC_1_1B_TRANSFORMERS, "parakeet"),
        (NEMOTRON_3_5_ASR_STREAMING_TRANSFORMERS, "nemotron"),
    ],
)
def test_other_transformers_models_transcribe_short_audio_into_single_segment(
    options: InlineAsrTransformersOptions,
    family: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    waveform = np.full(int(2.5 * 16000), 0.2, dtype=np.float32)
    asr_pipeline, audio = _setup_non_whisper_pipeline(
        options, waveform, [{"text": " recognized speech "}], tmp_path, monkeypatch
    )
    model = _AsrModelFactory.create(options, tmp_path, AcceleratorOptions())
    conversation = model.transcribe(audio)
    assert options.model_family == family
    assert [(item.text, item.start_time, item.end_time) for item in conversation] == [
        ("recognized speech", 0.0, 2.5)
    ]
    asr_pipeline.assert_called_once()
    np.testing.assert_array_equal(asr_pipeline.call_args.args[0], waveform)


@pytest.mark.parametrize(
    ("options", "family"),
    [
        (PARAKEET_CTC_1_1B_TRANSFORMERS, "parakeet"),
        (NEMOTRON_3_5_ASR_STREAMING_TRANSFORMERS, "nemotron"),
    ],
)
def test_other_transformers_models_split_long_audio_on_pauses(
    options: InlineAsrTransformersOptions,
    family: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    # 60 s of speech with a 1 s pause at 33 s: splits into two timed windows.
    waveform = np.full(60 * 16000, 0.2, dtype=np.float32)
    waveform[33 * 16000 : 34 * 16000] = 0.0
    asr_pipeline, audio = _setup_non_whisper_pipeline(
        options,
        waveform,
        [{"text": " first "}, {"text": " second "}],
        tmp_path,
        monkeypatch,
    )
    model = _AsrModelFactory.create(options, tmp_path, AcceleratorOptions())
    conversation = model.transcribe(audio)
    assert options.model_family == family
    assert [item.text for item in conversation] == ["first", "second"]
    assert conversation[0].start_time == 0.0
    assert conversation[0].end_time == pytest.approx(33.5, abs=0.05)
    assert conversation[1].start_time == conversation[0].end_time
    assert conversation[1].end_time == pytest.approx(60.0)
    assert asr_pipeline.call_count == 2


def test_other_transformers_models_handle_low_max_time_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    options = PARAKEET_CTC_1_1B_TRANSFORMERS.model_copy(update={"max_time_chunk": 5.0})
    waveform = np.full(12 * 16000, 0.2, dtype=np.float32)
    asr_pipeline, audio = _setup_non_whisper_pipeline(
        options,
        waveform,
        [{"text": " part one "}, {"text": " part two "}],
        tmp_path,
        monkeypatch,
    )
    model = _AsrModelFactory.create(options, tmp_path, AcceleratorOptions())
    conversation = model.transcribe(audio)
    assert [(item.text, item.start_time, item.end_time) for item in conversation] == [
        ("part one", 0.0, 5.0),
        ("part two", 5.0, 12.0),
    ]
    assert asr_pipeline.call_count == 2


def test_other_transformers_models_skip_silent_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    waveform = np.full(60 * 16000, 0.2, dtype=np.float32)
    waveform[33 * 16000 : 34 * 16000] = 0.0
    _, audio = _setup_non_whisper_pipeline(
        PARAKEET_CTC_1_1B_TRANSFORMERS,
        waveform,
        [{"text": "   "}, {"text": " spoken "}],
        tmp_path,
        monkeypatch,
    )
    model = _AsrModelFactory.create(
        PARAKEET_CTC_1_1B_TRANSFORMERS, tmp_path, AcceleratorOptions()
    )
    conversation = model.transcribe(audio)
    assert [item.text for item in conversation] == ["spoken"]
    assert conversation[0].start_time == pytest.approx(33.5, abs=0.05)
    assert conversation[0].end_time == pytest.approx(60.0)
