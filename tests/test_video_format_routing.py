"""Tests for InputFormat.VIDEO format registration.

Covers:
- .mp4/.avi/.mov/.mkv/.webm route to InputFormat.VIDEO
- .mp3/.wav/.flac stay in InputFormat.AUDIO
- video MIME types route to InputFormat.VIDEO
- audio MIME types stay in InputFormat.AUDIO
- VideoFormatOption uses VideoPipeline + NoOpBackend
- InputFormat.VIDEO is registered in DocumentConverter defaults
"""

import pytest

from docling.backend.noop_backend import NoOpBackend
from docling.datamodel.base_models import (
    FormatToExtensions,
    FormatToMimeType,
    InputFormat,
)
from docling.document_converter import DocumentConverter, VideoFormatOption
from docling.pipeline.video_pipeline import VideoPipeline

# --------------------------------------------------------------------------- #
# Extension routing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("ext", ["mp4", "avi", "mov", "mkv", "webm"])
def test_video_extensions_route_to_video(ext):
    assert InputFormat.VIDEO in FormatToExtensions
    assert ext in FormatToExtensions[InputFormat.VIDEO]


@pytest.mark.parametrize("ext", ["mp4", "avi", "mov"])
def test_video_extensions_not_in_audio(ext):
    assert ext not in FormatToExtensions[InputFormat.AUDIO]


@pytest.mark.parametrize("ext", ["wav", "mp3", "m4a", "aac", "ogg", "flac"])
def test_audio_extensions_stay_in_audio(ext):
    assert ext in FormatToExtensions[InputFormat.AUDIO]


# --------------------------------------------------------------------------- #
# MIME routing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "mime",
    [
        "video/mp4",
        "video/avi",
        "video/x-msvideo",
        "video/quicktime",
        "video/x-matroska",
        "video/webm",
    ],
)
def test_video_mimes_route_to_video(mime):
    assert InputFormat.VIDEO in FormatToMimeType
    assert mime in FormatToMimeType[InputFormat.VIDEO]


@pytest.mark.parametrize(
    "mime",
    [
        "video/mp4",
        "video/avi",
        "video/x-msvideo",
        "video/quicktime",
    ],
)
def test_video_mimes_not_in_audio(mime):
    assert mime not in FormatToMimeType[InputFormat.AUDIO]


@pytest.mark.parametrize(
    "mime",
    ["audio/x-wav", "audio/mpeg", "audio/flac", "audio/ogg"],
)
def test_audio_mimes_stay_in_audio(mime):
    assert mime in FormatToMimeType[InputFormat.AUDIO]


# --------------------------------------------------------------------------- #
# VideoFormatOption
# --------------------------------------------------------------------------- #


def test_video_format_option_uses_video_pipeline():
    opt = VideoFormatOption()
    assert opt.pipeline_cls is VideoPipeline


def test_video_format_option_uses_noop_backend():
    opt = VideoFormatOption()
    assert opt.backend is NoOpBackend


# --------------------------------------------------------------------------- #
# DocumentConverter default registration
# --------------------------------------------------------------------------- #


def test_video_registered_in_document_converter():
    dc = DocumentConverter()
    assert InputFormat.VIDEO in dc.format_to_options


def test_video_default_option_is_video_format_option():
    dc = DocumentConverter()
    opt = dc.format_to_options.get(InputFormat.VIDEO)
    assert opt is not None
    assert isinstance(opt, VideoFormatOption)
