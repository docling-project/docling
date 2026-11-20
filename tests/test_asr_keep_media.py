# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from io import BytesIO
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from docling.backend.noop_backend import NoOpBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult, InputDocument
from docling.datamodel.pipeline_options import AsrPipelineOptions
from docling.pipeline.asr_pipeline import AsrPipeline
from docling.pipeline.asr_transcriber import _ConversationItem, _process_conversation

_CONVERSATION = [
    _ConversationItem(text="hello", start_time=0.0, end_time=1.0, speaker="A"),
    _ConversationItem(text="world", start_time=1.0, end_time=2.0, speaker="B"),
]


def _convert(
    path_or_stream: Path | BytesIO, name: str = "talk.mp3", keep_media: bool = False
) -> ConversionResult:
    model = Mock()
    model.run.side_effect = lambda conv_res: _process_conversation(
        _CONVERSATION, conv_res
    )
    if isinstance(path_or_stream, BytesIO):
        input_doc = InputDocument(
            path_or_stream=path_or_stream,
            format=InputFormat.AUDIO,
            backend=NoOpBackend,
            filename=name,
        )
    else:
        input_doc = InputDocument(
            path_or_stream=path_or_stream,
            format=InputFormat.AUDIO,
            backend=NoOpBackend,
        )
    with patch(
        "docling.pipeline.asr_pipeline._AsrModelFactory.create", return_value=model
    ):
        pipeline = AsrPipeline(AsrPipelineOptions(keep_media=keep_media))
    return pipeline._build_document(ConversionResult(input=input_doc))


@pytest.fixture
def audio_file(tmp_path: Path) -> Path:
    path = tmp_path / "talk.mp3"
    path.write_bytes(b"ID3 not decoded")
    return path


def test_transcript_is_one_track(audio_file):
    doc = _convert(audio_file).document

    assert len(doc.tracks) == 1
    track = doc.tracks[0]
    assert [ref.cref for ref in doc.body.children] == [track.self_ref]
    assert [ref.resolve(doc).text for ref in track.children] == ["hello", "world"]
    assert [ref.resolve(doc).source[0].voice for ref in track.children] == ["A", "B"]
    assert track.audio is None
    assert track.video is None


def test_keep_media_references_the_input(audio_file):
    track = _convert(audio_file, keep_media=True).document.tracks[0]

    assert track.audio is not None
    assert track.audio.mimetype == "audio/mpeg"
    assert track.audio.uri == audio_file.resolve()
    assert track.video is None


def test_keep_media_sets_the_video_of_a_video(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not decoded")
    track = _convert(video, keep_media=True).document.tracks[0]

    assert track.video is not None
    assert track.video.mimetype == "video/mp4"
    assert track.audio is None


def test_keep_media_skips_a_stream(caplog):
    track = _convert(BytesIO(b"ID3 not decoded"), keep_media=True).document.tracks[0]

    assert track.audio is None
    assert "only file inputs" in caplog.text


def test_keep_media_skips_an_unsupported_type(tmp_path, caplog):
    path = tmp_path / "talk.amr"
    path.write_bytes(b"not decoded")
    track = _convert(path, keep_media=True).document.tracks[0]

    assert track.audio is None
    assert "Media not kept" in caplog.text
