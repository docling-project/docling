# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Helpers to record and load the video and audio that a presentation plays.

A slide shows a video or audio object as a picture: a poster frame or a
speaker icon. The backends keep that picture and record on it where the media
file is. A pipeline can then load the file through the backend and convert it
with the audio or video pipeline.
"""

from pathlib import PurePosixPath
from typing import Final, Literal, Optional
from urllib.parse import unquote, urlparse

from docling_core.types.doc import PictureItem, PictureMeta

from docling.backend.utils.image_resource_loader import ImageResourceLoader
from docling.datamodel.backend_options import BaseBackendOptions
from docling.datamodel.base_models import FormatToExtensions, InputFormat
from docling.exceptions import OperationNotAllowed

MediaKind = Literal["video", "audio"]

MEDIA_META_NAMESPACE: Final = "docling"
"""The media file is in the ``docling__video`` or ``docling__audio`` meta field."""

MAX_LINKED_MEDIA_BYTES: Final = 512 * 1024 * 1024
"""Size limit for a linked media file that is downloaded from a remote URL."""

_MEDIA_KINDS: Final[tuple[MediaKind, ...]] = ("video", "audio")


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


def get_media_meta(picture: PictureItem) -> Optional[tuple[MediaKind, str]]:
    """Return the kind and the location of the media file that a picture stands for."""
    if picture.meta is None:
        return None
    fields = picture.meta.get_custom_part()
    for kind in _MEDIA_KINDS:
        location = fields.get(f"{MEDIA_META_NAMESPACE}__{kind}")
        if isinstance(location, str) and location:
            return kind, location
    return None


def media_file_name(location: str) -> str:
    """Return the file name in a package path, a URL or a Windows path."""
    return PurePosixPath(urlparse(location).path.replace("\\", "/")).name


def load_linked_media(
    location: str, options: BaseBackendOptions, base_path: Optional[str]
) -> Optional[bytes]:
    """Load a linked media file with the fetch rules of the backend options.

    A remote URL needs ``enable_remote_fetch`` and gets the same address checks
    as a remote image, with a limit of ``MAX_LINKED_MEDIA_BYTES``. A local file
    needs ``enable_local_fetch`` and a document that was read from a file
    (``base_path``). The media file must be in the folder of the document or
    below it, so an absolute link is refused.

    Raises:
        OperationNotAllowed: If the backend options do not allow the fetch.
        ValueError: If the file is outside the folder of the document, cannot
            be read, or is too large.
        requests.RequestException, urllib3.exceptions.HTTPError: If a download
            fails.
    """
    loader = ImageResourceLoader(
        enable_local_fetch=options.enable_local_fetch,
        enable_remote_fetch=options.enable_remote_fetch,
        max_remote_image_bytes=MAX_LINKED_MEDIA_BYTES,
    )
    if loader.is_remote_url(location):
        return loader.fetch_remote(location).content
    # Check the option first: the path checks below would otherwise report a
    # link that is not allowed as a broken one.
    if not options.enable_local_fetch:
        raise OperationNotAllowed(
            "Fetching local resources is only allowed when set explicitly. "
            "Set options.enable_local_fetch=True."
        )
    # A link target is a URI reference, so a space in a file name is "%20".
    path = loader.resolve_relative_path(unquote(location), base_path)
    return loader.load_image_data(path, base_path)
