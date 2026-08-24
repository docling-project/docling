"""Tests for docling.pipeline.video_pipeline.

Unit tests mock ffmpeg, ASR, frame sampling, and diarization so they stay in
the core CI lane without downloading models or requiring video extras.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from docling_core.types.doc import DocItemLabel, DoclingDocument, PictureItem, TextItem
from PIL import Image

from docling.backend.noop_backend import NoOpBackend
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import (
    ConversionStatus,
    DoclingComponentType,
    ErrorItem,
    InputFormat,
)
from docling.datamodel.document import ConversionResult, InputDocument
from docling.datamodel.pipeline_options import VideoPipelineOptions
from docling.pipeline.asr_transcriber import _ConversationItem
from docling.utils.speaker_diarization import DiarizationResult, SpeakerSegment
from docling.utils.video_frame_sampling import VideoFrame, VideoFrameSamplingMode


@pytest.fixture(scope="module")
def video_pipeline_module():
    """Import once; the pipeline module pulls in a heavy dependency chain."""
    import docling.pipeline.video_pipeline as module

    return module


@pytest.fixture
def mock_asr_factory(video_pipeline_module):
    with patch.object(video_pipeline_module._AsrModelFactory, "create") as create:
        asr_model = Mock()
        create.return_value = asr_model
        yield asr_model


@pytest.fixture
def video_pipeline(video_pipeline_module, mock_asr_factory):
    options = VideoPipelineOptions(
        accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CPU),
        generate_frame_images=True,
        enable_diarization=False,
    )
    return video_pipeline_module.VideoPipeline(options)


def _make_video_conv_res(
    tmp_path: Path,
    *,
    filename: str = "clip.mp4",
    content: bytes = b"fake-video",
) -> ConversionResult:
    video_path = tmp_path / filename
    video_path.write_bytes(content)
    input_doc = InputDocument(
        path_or_stream=video_path,
        format=InputFormat.VIDEO,
        backend=NoOpBackend,
    )
    return ConversionResult(input=input_doc)


def _fake_extract_audio(_video_path: Path, wav_path: Path) -> bool:
    wav_path.write_bytes(b"fake-audio")
    return True


def _sample_frame(
    *, color: str, timestamp: float = 0.0, scene_id: int | None = 0
) -> VideoFrame:
    return VideoFrame(
        timestamp=timestamp,
        image=Image.new("RGB", (2, 2), color=color),
        scene_id=scene_id,
    )


def _stub_ffmpeg(video_pipeline_module, monkeypatch) -> None:
    monkeypatch.setattr(
        video_pipeline_module.shutil, "which", lambda _: "/usr/bin/ffmpeg"
    )


def _stub_sampler(
    video_pipeline_module, monkeypatch, sampler_cls, frames: list[VideoFrame]
) -> None:
    monkeypatch.setattr(sampler_cls, "sample", lambda self, _video_path: frames)


# --------------------------------------------------------------------------- #
# Helpers and backend contract
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("talk.mp4", "video/mp4"),
        ("clip.avi", "video/x-msvideo"),
        ("clip.mov", "video/quicktime"),
        ("clip.mkv", "video/x-matroska"),
        ("clip.webm", "video/webm"),
        ("unknown.bin", "video/mp4"),
    ],
)
def test_video_mimetype_mapping(
    video_pipeline_module, filename: str, expected: str
) -> None:
    assert video_pipeline_module._video_mimetype(filename) == expected


def test_is_backend_supported_only_accepts_noop_backend(
    video_pipeline_module, tmp_path: Path
) -> None:
    class _Dummy:
        pass

    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake-video")
    input_doc = InputDocument(
        path_or_stream=video_path,
        format=InputFormat.VIDEO,
        backend=NoOpBackend,
    )
    noop_backend = NoOpBackend(input_doc, video_path)

    assert video_pipeline_module.VideoPipeline.is_backend_supported(noop_backend)
    assert not video_pipeline_module.VideoPipeline.is_backend_supported(_Dummy())


def test_determine_status_empty_document_is_partial_success(
    video_pipeline, tmp_path: Path
) -> None:
    conv_res = _make_video_conv_res(tmp_path)
    conv_res.status = ConversionStatus.SUCCESS
    conv_res.errors = []
    conv_res.document = DoclingDocument(name="clip")

    assert (
        video_pipeline._determine_status(conv_res) == ConversionStatus.PARTIAL_SUCCESS
    )


def test_determine_status_with_text_is_success(video_pipeline, tmp_path: Path) -> None:
    conv_res = _make_video_conv_res(tmp_path)
    conv_res.document = DoclingDocument(name="clip")
    conv_res.document.add_text(label=DocItemLabel.TEXT, text="hello")
    conv_res.status = ConversionStatus.SUCCESS
    conv_res.errors = []

    assert video_pipeline._determine_status(conv_res) == ConversionStatus.SUCCESS


def test_determine_status_with_errors_is_failure(
    video_pipeline, tmp_path: Path
) -> None:
    conv_res = _make_video_conv_res(tmp_path)
    conv_res.document = DoclingDocument(name="clip")
    conv_res.document.add_text(label=DocItemLabel.TEXT, text="hello")
    conv_res.status = ConversionStatus.SUCCESS
    conv_res.errors = [
        ErrorItem(
            component_type=DoclingComponentType.PIPELINE,
            module_name="VideoPipeline",
            error_message="boom",
        )
    ]

    assert video_pipeline._determine_status(conv_res) == ConversionStatus.FAILURE


# --------------------------------------------------------------------------- #
# _extract_audio and _process_video error handling
# --------------------------------------------------------------------------- #


def test_process_video_missing_ffmpeg_records_failure(
    video_pipeline, video_pipeline_module, tmp_path: Path, monkeypatch
) -> None:
    conv_res = _make_video_conv_res(tmp_path)
    monkeypatch.setattr(video_pipeline_module.shutil, "which", lambda _: None)

    video_pipeline._process_video(conv_res)

    assert conv_res.status == ConversionStatus.FAILURE
    assert any(
        error.error_message == video_pipeline_module.MISSING_FFMPEG_MESSAGE
        for error in conv_res.errors
    )


def test_process_video_unsupported_input_type_records_failure(
    video_pipeline, video_pipeline_module, tmp_path: Path, monkeypatch
) -> None:
    conv_res = _make_video_conv_res(tmp_path)
    conv_res.input._backend.path_or_stream = 123  # type: ignore[assignment]
    _stub_ffmpeg(video_pipeline_module, monkeypatch)

    video_pipeline._process_video(conv_res)

    assert conv_res.status == ConversionStatus.FAILURE
    assert any(
        "Unsupported input type" in error.error_message for error in conv_res.errors
    )


def test_extract_audio_returns_false_when_ffmpeg_fails(
    video_pipeline_module, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        video_pipeline_module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stderr=b"boom"),
    )

    assert (
        video_pipeline_module._extract_audio(
            tmp_path / "clip.mp4", tmp_path / "out.wav"
        )
        is False
    )


def test_extract_audio_invokes_ffmpeg_for_16khz_mono_wav(
    video_pipeline_module, tmp_path: Path, monkeypatch
) -> None:
    captured: dict[str, list[str]] = {}

    def fake_run(args, **kwargs):
        captured["args"] = list(args)
        return SimpleNamespace(returncode=0, stderr=b"")

    monkeypatch.setattr(video_pipeline_module.subprocess, "run", fake_run)

    video_path = tmp_path / "clip.mp4"
    wav_path = tmp_path / "out.wav"
    assert video_pipeline_module._extract_audio(video_path, wav_path) is True
    assert captured["args"][:2] == ["ffmpeg", "-nostdin"]
    assert captured["args"][captured["args"].index("-ar") + 1] == "16000"
    assert captured["args"][captured["args"].index("-ac") + 1] == "1"
    assert str(video_path) in captured["args"]
    assert str(wav_path) in captured["args"]


# --------------------------------------------------------------------------- #
# _process_video happy paths
# --------------------------------------------------------------------------- #


def test_process_video_interleaves_frames_before_later_transcript(
    video_pipeline,
    video_pipeline_module,
    mock_asr_factory,
    tmp_path: Path,
    monkeypatch,
) -> None:
    mock_asr_factory.transcribe.return_value = [
        _ConversationItem(text="hello world.", start_time=1.0, end_time=2.0),
    ]
    frame = _sample_frame(color="red", timestamp=0.5, scene_id=3)

    _stub_ffmpeg(video_pipeline_module, monkeypatch)
    monkeypatch.setattr(video_pipeline_module, "_extract_audio", _fake_extract_audio)
    _stub_sampler(
        video_pipeline_module,
        monkeypatch,
        video_pipeline_module.FixedIntervalFrameSampler,
        [frame],
    )

    conv_res = _make_video_conv_res(tmp_path)
    video_pipeline._process_video(conv_res)

    assert conv_res.document is not None
    ordered = [item for item, _level in conv_res.document.iterate_items()]
    assert [type(item) for item in ordered] == [PictureItem, TextItem]
    assert ordered[0].source[0].start_time == 0.5
    assert ordered[0].source[0].identifier == "scene:3"
    assert ordered[1].text == "hello world."
    assert ordered[1].source[0].start_time == 1.0
    assert ordered[1].source[0].end_time == 2.0


def test_process_video_skips_whitespace_only_transcript(
    video_pipeline,
    video_pipeline_module,
    mock_asr_factory,
    tmp_path: Path,
    monkeypatch,
) -> None:
    mock_asr_factory.transcribe.return_value = [
        _ConversationItem(text="   ", start_time=0.0, end_time=0.4),
    ]
    video_pipeline.pipeline_options.generate_frame_images = False
    _stub_ffmpeg(video_pipeline_module, monkeypatch)
    monkeypatch.setattr(video_pipeline_module, "_extract_audio", _fake_extract_audio)

    conv_res = _make_video_conv_res(tmp_path)
    video_pipeline._process_video(conv_res)

    assert conv_res.document is not None
    assert conv_res.document.texts == []
    assert (
        video_pipeline._determine_status(conv_res) == ConversionStatus.PARTIAL_SUCCESS
    )


def test_process_video_audio_extraction_failure_yields_frames_only(
    video_pipeline,
    video_pipeline_module,
    mock_asr_factory,
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame = _sample_frame(color="blue")

    _stub_ffmpeg(video_pipeline_module, monkeypatch)
    monkeypatch.setattr(video_pipeline_module, "_extract_audio", lambda *_args: False)
    _stub_sampler(
        video_pipeline_module,
        monkeypatch,
        video_pipeline_module.FixedIntervalFrameSampler,
        [frame],
    )

    conv_res = _make_video_conv_res(tmp_path)
    video_pipeline._process_video(conv_res)

    mock_asr_factory.transcribe.assert_not_called()
    assert conv_res.document is not None
    assert conv_res.document.texts == []
    assert len(conv_res.document.pictures) == 1
    assert video_pipeline._determine_status(conv_res) == ConversionStatus.SUCCESS


def test_process_video_accepts_bytesio_input(
    video_pipeline,
    video_pipeline_module,
    mock_asr_factory,
    monkeypatch,
) -> None:
    mock_asr_factory.transcribe.return_value = [
        _ConversationItem(text="from stream", start_time=0.0, end_time=1.0),
    ]
    _stub_ffmpeg(video_pipeline_module, monkeypatch)
    monkeypatch.setattr(video_pipeline_module, "_extract_audio", _fake_extract_audio)
    video_pipeline.pipeline_options.generate_frame_images = False

    input_doc = InputDocument(
        path_or_stream=BytesIO(b"fake-video"),
        format=InputFormat.VIDEO,
        backend=NoOpBackend,
        filename="clip.mp4",
    )
    conv_res = ConversionResult(input=input_doc)
    video_pipeline._process_video(conv_res)

    assert conv_res.document is not None
    assert conv_res.document.texts[0].text == "from stream"


def test_process_video_uses_scene_change_sampler(
    video_pipeline_module,
    mock_asr_factory,
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame = _sample_frame(color="green", timestamp=2.0, scene_id=1)
    pipeline = video_pipeline_module.VideoPipeline(
        VideoPipelineOptions(
            accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CPU),
            generate_frame_images=True,
            frame_sampling_mode=VideoFrameSamplingMode.SCENE_CHANGE,
            enable_diarization=False,
        )
    )

    _stub_ffmpeg(video_pipeline_module, monkeypatch)
    monkeypatch.setattr(video_pipeline_module, "_extract_audio", lambda *_args: False)
    _stub_sampler(
        video_pipeline_module,
        monkeypatch,
        video_pipeline_module.SimpleSceneChangeFrameSampler,
        [frame],
    )

    conv_res = _make_video_conv_res(tmp_path)
    pipeline._process_video(conv_res)

    assert conv_res.document is not None
    assert len(conv_res.document.pictures) == 1
    assert conv_res.document.pictures[0].source[0].identifier == "scene:1"


def test_process_video_runs_diarization_when_enabled(
    video_pipeline_module,
    mock_asr_factory,
    tmp_path: Path,
    monkeypatch,
) -> None:
    mock_asr_factory.transcribe.return_value = [
        _ConversationItem(text="hello", start_time=0.0, end_time=1.0),
    ]
    diarization = DiarizationResult(
        segments=[SpeakerSegment(0.0, 1.0, "SPEAKER_00")],
        num_speakers=1,
        speaker_ids=["SPEAKER_00"],
    )
    pipeline = video_pipeline_module.VideoPipeline(
        VideoPipelineOptions(
            accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CPU),
            enable_diarization=True,
            generate_frame_images=False,
        )
    )

    _stub_ffmpeg(video_pipeline_module, monkeypatch)
    monkeypatch.setattr(video_pipeline_module, "_extract_audio", _fake_extract_audio)
    diarize_mock = Mock(return_value=diarization)
    monkeypatch.setattr(video_pipeline_module, "diarize", diarize_mock)

    conv_res = _make_video_conv_res(tmp_path)
    pipeline._process_video(conv_res)

    diarize_mock.assert_called_once()
    assert conv_res.document is not None
    assert conv_res.document.texts[0].text == "hello"
    assert conv_res.document.texts[0].source[0].voice == "SPEAKER_00"
