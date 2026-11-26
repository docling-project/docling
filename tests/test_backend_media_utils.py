# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tests for slide media classification and reference helpers."""

from pathlib import Path

import pytest

from docling.backend.utils.media import MediaKind, media_kind, media_ref
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter


@pytest.mark.parametrize(
    ("location", "expected"),
    [
        ("https://cdn.example.com/talk.mp4?token=abc", "video"),
        ("https://cdn.example.com/talk.wmv?sig=xyz", "video"),
        ("https://cdn.example.com/tone.mp3#t=10", "audio"),
        ("https://cdn.example.com/note.wma?download=1#start", "audio"),
        (r"C:\media.v1\clip.mp4", "video"),
        (r"C:\dir.mp4\readme", None),
    ],
)
def test_media_kind_parses_url_paths_and_windows_paths(
    location: str, expected: MediaKind | None
) -> None:
    assert media_kind("application/vnd.sun.star.media", location) == expected
    assert media_kind(None, location) == expected


@pytest.mark.parametrize(
    ("mimetype", "expected"),
    [
        ("VIDEO/MP4", "video"),
        ("Audio/MPEG", "audio"),
        ("  video/webm  ", "video"),
        ("  AUDIO/ogg ", "audio"),
    ],
)
def test_media_kind_matches_mimetype_case_insensitively(
    mimetype: str, expected: MediaKind
) -> None:
    assert media_kind(mimetype) == expected


def test_media_kind_and_media_ref_agree_on_query_and_fragment_urls() -> None:
    video_url = "https://cdn.example.com/talk.mp4?token=abc"
    audio_url = "https://cdn.example.com/tone.mp3#t=10"

    video_kind = media_kind(None, video_url)
    audio_kind = media_kind(None, audio_url)

    assert video_kind == "video"
    assert audio_kind == "audio"
    video_ref = media_ref(video_kind, video_url)
    audio_ref = media_ref(audio_kind, audio_url)
    assert video_ref is not None and video_ref.mimetype == "video/mp4"
    assert audio_ref is not None and audio_ref.mimetype == "audio/mpeg"


def test_odp_linked_url_with_query_becomes_a_track(tmp_path: Path) -> None:
    odfdo = pytest.importorskip("odfdo")
    url = "https://cdn.example.com/talk.mp4?token=abc"

    odf = odfdo.Document("presentation")
    odf.body.clear()
    page = odfdo.DrawPage("page1", name="Slide One")
    frame = odfdo.Frame(size=("8cm", "4.5cm"), position=("1cm", "1cm"))
    frame.append(
        odfdo.Element.from_tag(
            f'<draw:plugin xlink:href="{url}" xlink:type="simple" xlink:show="embed"'
            ' xlink:actuate="onLoad" draw:mime-type="application/vnd.sun.star.media"/>'
        )
    )
    page.append(frame)
    odf.body.append(page)
    source = tmp_path / "linked_query.odp"
    odf.save(str(source))

    converter = DocumentConverter(allowed_formats=[InputFormat.ODP])
    doc = converter.convert(source).document

    assert len(doc.tracks) == 1
    track = doc.tracks[0]
    assert track.video is not None and track.audio is None
    assert (track.video.mimetype, str(track.video.uri)) == ("video/mp4", url)
