# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Helpers to record and load the video and audio that a presentation plays.

A slide shows a video or audio object as a picture: a poster frame or a
speaker icon. The backends turn it into a track with that picture as its cover,
and a ``MediaRef`` that records where the media file is. A pipeline can then
load the file through the backend, keep it in the track, and convert it with
the audio or video pipeline.
"""

import logging
from pathlib import Path, PurePosixPath
from typing import Final, Literal, Optional
from urllib.parse import unquote, urlparse

from docling_core.types.doc import MediaRef, TrackItem
from docling_core.types.doc.common.reference import media_type_of
from pydantic import ValidationError

from docling.backend.utils.image_resource_loader import ImageResourceLoader
from docling.datamodel.backend_options import BaseBackendOptions
from docling.datamodel.base_models import FormatToExtensions, InputFormat
from docling.exceptions import OperationNotAllowed

_log = logging.getLogger(__name__)

MediaKind = Literal["video", "audio"]

MAX_LINKED_MEDIA_BYTES: Final = 512 * 1024 * 1024
"""Size limit for a linked media file that is downloaded from a remote URL."""

# Containers that older decks embed, besides those of the audio and video input formats.
_LEGACY_VIDEO_EXTENSIONS: Final = frozenset({"asf", "flv", "m4v", "mpeg", "mpg", "wmv"})
_LEGACY_AUDIO_EXTENSIONS: Final = frozenset({"aif", "aiff", "wma"})


def media_kind(mimetype: Optional[str], location: str = "") -> Optional[MediaKind]:
    """Tell video from audio by the MIME type, or else by the file extension.

    The extensions are the ones the audio and video input formats accept, and the
    containers that older decks embed (e.g. ``.wmv``), which are recorded even though
    the video pipeline cannot read them.
    """
    major = (mimetype or "").partition("/")[0]
    if major == "video":
        return "video"
    if major == "audio":
        return "audio"
    suffix = PurePosixPath(location).suffix.lower().removeprefix(".")
    if (
        suffix in FormatToExtensions[InputFormat.VIDEO]
        or suffix in _LEGACY_VIDEO_EXTENSIONS
    ):
        return "video"
    if (
        suffix in FormatToExtensions[InputFormat.AUDIO]
        or suffix in _LEGACY_AUDIO_EXTENSIONS
    ):
        return "audio"
    return None


def media_ref(kind: MediaKind, location: str) -> Optional[MediaRef]:
    """Return the reference of the video or audio file that a slide plays.

    The reference holds the location only: the path of the file inside the
    document package when the file is embedded, or the link target that the
    document gives when the file is linked. A pipeline loads the file through
    the backend (see ``load_media``).

    Returns:
        None when the file is in a container that a track cannot reference,
        such as a ``.3gp`` video.
    """
    mimetype = media_type_of(media_file_name(location))
    if kind == "audio" and mimetype == "video/mp4":
        # an MP4 container that a slide plays as audio
        mimetype = "audio/mp4"
    if mimetype is None or not mimetype.startswith(f"{kind}/"):
        _log.debug("Media file %s is not kept: unsupported container", location)
        return None
    try:
        return MediaRef(mimetype=mimetype, uri=location)
    except ValidationError:
        _log.debug("Media file %s is not kept: unsupported reference", location)
        return None


def track_media(track: TrackItem) -> Optional[tuple[MediaKind, MediaRef]]:
    """Return the kind and the reference of the media file of a track."""
    if track.video is not None:
        return "video", track.video
    if track.audio is not None:
        return "audio", track.audio
    return None


def media_location(ref: MediaRef) -> str:
    """Return the location that ``media_ref`` recorded: a package path or a link."""
    return ref.uri.as_posix() if isinstance(ref.uri, Path) else str(ref.uri)


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
