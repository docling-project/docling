# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tests for video and audio shapes on PowerPoint slides."""

import zipfile
from pathlib import Path

from docling_core.types.doc import DoclingDocument, TrackItem

from .test_backend_pptx import get_converter

_P14_MEDIA = "{http://schemas.microsoft.com/office/powerpoint/2010/main}media"


def _add_media(tmp_path: Path, file_name: str, mime_type: str):
    """Add a media shape the way PowerPoint stores it, and return the deck and shape.

    python-pptx writes the same XML as PowerPoint: an ``a:videoFile`` link plus a
    ``p14:media`` extension, both pointing at one part under ``ppt/media``.
    """
    from pptx import Presentation
    from pptx.util import Inches

    media_path = tmp_path / file_name
    media_path.write_bytes(b"not decoded by the backend")
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    title = slide.shapes.add_textbox(Inches(1), Inches(0.5), Inches(6), Inches(1))
    title.text_frame.text = "Slide with media"
    shape = slide.shapes.add_movie(
        str(media_path),
        Inches(1),
        Inches(2),
        Inches(4),
        Inches(2.25),
        mime_type=mime_type,
    )
    return prs, slide, shape


def _media(track: TrackItem) -> tuple[str, str, str]:
    """The kind, media type and location of the media of a track."""
    kind, ref = ("video", track.video) if track.video else ("audio", track.audio)
    assert ref is not None
    return kind, ref.mimetype, str(ref.uri)


def _convert(prs, path: Path) -> DoclingDocument:
    prs.save(path)
    return get_converter().convert(path).document


def test_pptx_embedded_video_is_a_track_with_its_media_path(tmp_path: Path):
    """An embedded video becomes a track on its slide, with its poster frame as cover.

    The track references the video by its path inside the package, so the video
    can be read from the deck, e.g. to keep or transcribe it.
    """
    prs, _, _ = _add_media(tmp_path, "clip.mp4", "video/mp4")
    deck = tmp_path / "video.pptx"
    doc = _convert(prs, deck)

    assert doc.pictures == []
    assert len(doc.tracks) == 1
    track = doc.tracks[0]
    assert track.image is not None
    assert track.prov[0].page_no == 1
    assert track.parent is not None and track.parent.cref == "#/groups/0"
    assert _media(track) == ("video", "video/mp4", "ppt/media/media1.mp4")
    # the reference only: the backend does not read the media
    assert track.video is not None and track.video._content is None
    with zipfile.ZipFile(deck) as package:
        assert package.read("ppt/media/media1.mp4") == b"not decoded by the backend"


def test_pptx_audio_is_a_track_with_its_media_path(tmp_path: Path):
    """An audio shape becomes a track with its speaker icon as cover.

    PowerPoint stores audio like video, with ``a:audioFile`` in place of
    ``a:videoFile``; python-pptx reads it as a plain picture.
    """
    from pptx.oxml.ns import qn

    prs, _, shape = _add_media(tmp_path, "tone.mp3", "audio/mpeg")
    shape._element.xpath("./p:nvPicPr/p:nvPr/a:videoFile")[0].tag = qn("a:audioFile")
    doc = _convert(prs, tmp_path / "audio.pptx")

    assert doc.pictures == []
    assert doc.tracks[0].image is not None
    assert _media(doc.tracks[0]) == ("audio", "audio/mpeg", "ppt/media/media1.mp3")


def test_pptx_linked_video_records_its_link(tmp_path: Path):
    """A video that is linked, not embedded, references the link target."""
    from pptx.opc.constants import RELATIONSHIP_TYPE as RT
    from pptx.oxml.ns import qn

    prs, slide, shape = _add_media(tmp_path, "clip.mp4", "video/mp4")
    link = "https://example.com/talk.mp4"
    video_rid = slide.part.relate_to(link, RT.VIDEO, is_external=True)
    media_rid = slide.part.relate_to(link, RT.MEDIA, is_external=True)
    shape._element.xpath("./p:nvPicPr/p:nvPr/a:videoFile")[0].set(
        qn("r:link"), video_rid
    )
    p14_media = next(shape._element.iter(_P14_MEDIA))
    del p14_media.attrib[qn("r:embed")]
    p14_media.set(qn("r:link"), media_rid)
    doc = _convert(prs, tmp_path / "linked.pptx")

    assert len(doc.tracks) == 1
    assert _media(doc.tracks[0]) == ("video", "video/mp4", link)


def test_pptx_video_in_a_legacy_container_is_a_track(tmp_path: Path):
    """The containers of older decks, such as .wmv, are tracks like the others."""
    prs, _, _ = _add_media(tmp_path, "clip.wmv", "video/x-ms-wmv")
    doc = _convert(prs, tmp_path / "legacy.pptx")

    assert doc.pictures == []
    assert _media(doc.tracks[0]) == ("video", "video/x-ms-wmv", "ppt/media/media1.wmv")


def test_pptx_video_in_an_unsupported_container_stays_a_picture(tmp_path: Path):
    """A track cannot reference a .3gp: the shape stays a picture of its poster frame."""
    prs, _, _ = _add_media(tmp_path, "clip.3gp", "video/3gpp")
    doc = _convert(prs, tmp_path / "unsupported.pptx")

    assert doc.tracks == []
    assert len(doc.pictures) == 1
    assert doc.pictures[0].image is not None


def test_pptx_picture_without_media_stays_a_picture(tmp_path: Path):
    """A plain picture is unchanged: it is no track."""
    from io import BytesIO

    from PIL import Image
    from pptx import Presentation
    from pptx.util import Inches

    image = BytesIO()
    Image.new("RGB", (40, 20), "red").save(image, format="PNG")
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.shapes.add_picture(image, Inches(1), Inches(1))
    doc = _convert(prs, tmp_path / "picture.pptx")

    assert len(doc.pictures) == 1
    assert doc.tracks == []
