# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Helpers to record the video and audio that a presentation plays.

A slide shows a video or audio object as a picture: a poster frame or a
speaker icon. The backends keep that picture and record on it where the media
file is, so a consumer can find the file and, for example, convert it with the
audio or video pipeline.
"""

from pathlib import PurePosixPath
from typing import Final, Literal, Optional

from docling_core.types.doc import PictureItem, PictureMeta

from docling.datamodel.base_models import FormatToExtensions, InputFormat

MediaKind = Literal["video", "audio"]

MEDIA_META_NAMESPACE: Final = "docling"
"""The media file is in the ``docling__video`` or ``docling__audio`` meta field."""


def media_kind(mimetype: Optional[str], location: str = "") -> Optional[MediaKind]:
    """Tell video from audio by the MIME type, or else by the file extension.

    The extensions are the ones the audio and video input formats accept.
    """
    major = (mimetype or "").partition("/")[0]
    if major == "video":
        return "video"
    if major == "audio":
        return "audio"
    suffix = PurePosixPath(location).suffix.lower().removeprefix(".")
    if suffix in FormatToExtensions[InputFormat.VIDEO]:
        return "video"
    if suffix in FormatToExtensions[InputFormat.AUDIO]:
        return "audio"
    return None


def set_media_meta(picture: PictureItem, kind: MediaKind, location: str) -> None:
    """Record on a picture the video or audio file that it stands for.

    The value is the path of the file inside the document package when the
    file is embedded, and the link target that the document gives when the file
    is linked. It is a plain string, so the Markdown export shows it under the
    picture.

    Args:
        picture: The picture that shows the media on the slide.
        kind: ``"video"`` or ``"audio"``; it names the meta field.
        location: The package path or the link target of the media file.
    """
    if picture.meta is None:
        picture.meta = PictureMeta()
    picture.meta.set_custom_field(
        namespace=MEDIA_META_NAMESPACE, name=kind, value=location
    )
