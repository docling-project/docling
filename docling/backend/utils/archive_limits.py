# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Limits on how much data a backend decompresses from a ZIP-based input.

Office Open XML and EPUB files are ZIP archives. A small file can hold members
that decompress to a very large size, and the libraries that parse them keep the
decompressed data, and the trees built from it, in memory. These helpers check
the sizes an archive declares before any member is read, and stop a read once
the decompressed bytes go above the limit.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final, Protocol
from zipfile import ZipFile, ZipInfo

from docling.exceptions import DocumentLoadError

_READ_CHUNK_BYTES: Final[int] = 1024 * 1024


class ArchiveLimitError(DocumentLoadError):
    """An archive declares, or decompresses to, more data than the limits allow."""


class _ArchiveLimitOptions(Protocol):
    """Backend options that carry the archive limit fields."""

    max_file_bytes: int
    max_total_bytes: int
    max_member_count: int


@dataclass(frozen=True)
class ArchiveLimits:
    """Limits on the decompressed content of a ZIP archive.

    Attributes:
        max_file_bytes: Largest decompressed size of one member.
        max_total_bytes: Largest decompressed size of all checked members.
        max_member_count: Largest number of checked members.
    """

    max_file_bytes: int
    max_total_bytes: int
    max_member_count: int

    @classmethod
    def from_options(cls, options: _ArchiveLimitOptions) -> "ArchiveLimits":
        """Build the limits from backend options with the same field names."""
        return cls(
            max_file_bytes=options.max_file_bytes,
            max_total_bytes=options.max_total_bytes,
            max_member_count=options.max_member_count,
        )


def check_archive_limits(
    archive: ZipFile,
    limits: ArchiveLimits,
    members: Iterable[ZipInfo] | None = None,
) -> None:
    """Check the declared sizes of archive members against the limits.

    No member is decompressed. ``ZipFile`` never returns more bytes for a member
    than its central directory entry declares, so this check also bounds what
    later reads of these members return.

    Args:
        archive: The open archive.
        limits: The limits to apply.
        members: The members to check. All members of the archive when None.

    Raises:
        ArchiveLimitError: If the members exceed one of the limits.
    """
    infos = archive.infolist() if members is None else list(members)
    if len(infos) > limits.max_member_count:
        raise ArchiveLimitError(
            f"Archive has {len(infos)} members, exceeding the max_member_count "
            f"limit of {limits.max_member_count}."
        )
    total_bytes = 0
    for info in infos:
        if info.file_size > limits.max_file_bytes:
            raise ArchiveLimitError(
                f"Archive member {info.filename} decompresses to {info.file_size} "
                f"bytes, exceeding the max_file_bytes limit of "
                f"{limits.max_file_bytes}."
            )
        total_bytes += info.file_size
    if total_bytes > limits.max_total_bytes:
        raise ArchiveLimitError(
            f"Archive decompresses to {total_bytes} bytes, exceeding the "
            f"max_total_bytes limit of {limits.max_total_bytes}."
        )


def read_member(archive: ZipFile, member: str | ZipInfo, max_bytes: int) -> bytes:
    """Read one archive member, decompressing at most ``max_bytes`` bytes.

    The declared size is checked first, so an oversized member is rejected
    without decompression. The read then counts the bytes it actually gets and
    stops once they go above the limit.

    Args:
        archive: The open archive.
        member: The member name or entry.
        max_bytes: The largest decompressed size to accept.

    Returns:
        The decompressed content of the member.

    Raises:
        KeyError: If the archive has no member with this name.
        ArchiveLimitError: If the member decompresses to more than ``max_bytes``.
    """
    info = member if isinstance(member, ZipInfo) else archive.getinfo(member)
    if info.file_size > max_bytes:
        raise ArchiveLimitError(
            f"Archive member {info.filename} decompresses to {info.file_size} "
            f"bytes, exceeding the limit of {max_bytes}."
        )
    chunks: list[bytes] = []
    size = 0
    with archive.open(info) as stream:
        while chunk := stream.read(min(_READ_CHUNK_BYTES, max_bytes - size + 1)):
            size += len(chunk)
            if size > max_bytes:
                raise ArchiveLimitError(
                    f"Archive member {info.filename} decompresses to more than "
                    f"{max_bytes} bytes."
                )
            chunks.append(chunk)
    return b"".join(chunks)
