import shutil
from pathlib import Path
from unittest.mock import Mock

from docling.backend.noop_backend import NoOpBackend
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.document import ConversionResult, InputDocument
from docling.datamodel.pipeline_options_asr_model import (
    InferenceAsrFramework,
    InlineAsrMlxWhisperOptions,
    InlineAsrNativeWhisperOptions,
)
from docling.pipeline.asr_pipeline import _MlxWhisperModel, _NativeWhisperModel


def test_native_whisper_reports_missing_ffmpeg_before_transcription(
    monkeypatch, tmp_path: Path
) -> None:
    audio_path = tmp_path / "sample.mp3"
    audio_path.write_bytes(b"not real mp3 data")
    input_doc = InputDocument(
        path_or_stream=audio_path,
        format=InputFormat.AUDIO,
        backend=NoOpBackend,
    )
    conv_res = ConversionResult(input=input_doc)

    options = InlineAsrNativeWhisperOptions(
        repo_id="tiny",
        inference_framework=InferenceAsrFramework.WHISPER,
        verbose=False,
        timestamps=False,
        word_timestamps=False,
        temperature=0.0,
        max_new_tokens=1,
        max_time_chunk=1.0,
        language="en",
    )
    model = _NativeWhisperModel(
        enabled=False,
        artifacts_path=None,
        accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CPU),
        asr_options=options,
    )
    model.model = Mock()
    model.model.transcribe.return_value = {"segments": []}
    model.verbose = False
    model.word_timestamps = False

    monkeypatch.setattr(shutil, "which", lambda executable: None)

    out = model.run(conv_res)

    assert out.status == ConversionStatus.FAILURE
    assert len(out.errors) == 1
    assert "FFmpeg is required" in out.errors[0].error_message
    assert "PATH" in out.errors[0].error_message
    model.model.transcribe.assert_not_called()


def test_native_whisper_maps_late_ffmpeg_file_not_found(
    monkeypatch, tmp_path: Path
) -> None:
    audio_path = tmp_path / "sample.mp3"
    audio_path.write_bytes(b"not real mp3 data")
    input_doc = InputDocument(
        path_or_stream=audio_path,
        format=InputFormat.AUDIO,
        backend=NoOpBackend,
    )
    conv_res = ConversionResult(input=input_doc)

    options = InlineAsrNativeWhisperOptions(
        repo_id="tiny",
        inference_framework=InferenceAsrFramework.WHISPER,
        verbose=False,
        timestamps=False,
        word_timestamps=False,
        temperature=0.0,
        max_new_tokens=1,
        max_time_chunk=1.0,
        language="en",
    )
    model = _NativeWhisperModel(
        enabled=False,
        artifacts_path=None,
        accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CPU),
        asr_options=options,
    )
    model.model = Mock()
    model.model.transcribe.side_effect = FileNotFoundError(
        2, "No such file or directory", "ffmpeg"
    )
    model.verbose = False
    model.word_timestamps = False

    monkeypatch.setattr(shutil, "which", lambda executable: "/usr/bin/ffmpeg")

    out = model.run(conv_res)

    assert out.status == ConversionStatus.FAILURE
    assert len(out.errors) == 1
    assert "FFmpeg is required" in out.errors[0].error_message


def test_mlx_whisper_reports_missing_ffmpeg_before_transcription(
    monkeypatch, tmp_path: Path
) -> None:
    audio_path = tmp_path / "sample.mp3"
    audio_path.write_bytes(b"not real mp3 data")
    input_doc = InputDocument(
        path_or_stream=audio_path,
        format=InputFormat.AUDIO,
        backend=NoOpBackend,
    )
    conv_res = ConversionResult(input=input_doc)

    options = InlineAsrMlxWhisperOptions(
        repo_id="mlx-community/whisper-tiny",
        inference_framework=InferenceAsrFramework.MLX,
        language="en",
        task="transcribe",
        word_timestamps=False,
        no_speech_threshold=0.6,
        logprob_threshold=-1.0,
        compression_ratio_threshold=2.4,
    )
    model = _MlxWhisperModel(
        enabled=False,
        artifacts_path=None,
        accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CPU),
        asr_options=options,
    )
    model.transcribe = Mock(return_value=[])  # type: ignore[method-assign]

    monkeypatch.setattr(shutil, "which", lambda executable: None)

    out = model.run(conv_res)

    assert out.status == ConversionStatus.FAILURE
    assert len(out.errors) == 1
    assert "FFmpeg is required" in out.errors[0].error_message
    model.transcribe.assert_not_called()
