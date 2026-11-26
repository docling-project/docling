# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Limits on the extended-header data that ``tarfile`` reads from a TAR input.

Before ``tarfile`` returns a member, it reads the payload of the extended headers
in front of that member (pax ``x``/``g`` headers and GNU long name / long link
headers) fully into memory. The size of this payload comes from the header. The
member limits of the backends (``max_file_bytes``, ``max_total_bytes``,
``max_member_count``) apply only to the members that ``tarfile`` returns, so they
do not see this payload. These helpers charge the declared payload size against
a per-archive budget before ``tarfile`` reads it.
"""

import tarfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Final

from docling.backend.utils.archive_limits import ArchiveLimitError

_EXTENDED_HEADER_TYPES: Final[frozenset[bytes]] = frozenset(
    {
        tarfile.XHDTYPE,
        tarfile.XGLTYPE,
        tarfile.SOLARIS_XHDTYPE,
        tarfile.GNUTYPE_LONGNAME,
        tarfile.GNUTYPE_LONGLINK,
    }
)


@dataclass(frozen=True)
class TarHeaderLimits:
    """Limits on the extended-header payload of one TAR archive.

    Valid pax and GNU long-name headers are a few hundred bytes, so the
    defaults leave a large margin.

    Attributes:
        max_header_bytes: Largest declared payload of one extended header.
        max_total_header_bytes: Largest declared payload of all extended
            headers in the archive.
    """

    max_header_bytes: int = 1024 * 1024
    max_total_header_bytes: int = 16 * 1024 * 1024


DEFAULT_TAR_HEADER_LIMITS: TarHeaderLimits = TarHeaderLimits()


class _TarHeaderBudget:
    """The extended-header payload that one archive has used so far."""

    def __init__(self, limits: TarHeaderLimits) -> None:
        self._limits = limits
        self._used_bytes = 0

    def charge(self, header: tarfile.TarInfo) -> None:
        """Add the declared payload size of ``header``, or raise if it is too large."""
        size = header.size
        if size < 0 or size > self._limits.max_header_bytes:
            raise ArchiveLimitError(
                f"TAR extended header at offset {header.offset} declares {size} "
                f"bytes, exceeding the limit of {self._limits.max_header_bytes} "
                "bytes for one header."
            )
        self._used_bytes += size
        if self._used_bytes > self._limits.max_total_header_bytes:
            raise ArchiveLimitError(
                f"TAR extended headers declare more than "
                f"{self._limits.max_total_header_bytes} bytes in total."
            )


def limited_tarinfo_class(limits: TarHeaderLimits) -> type[tarfile.TarInfo]:
    """Make a ``TarInfo`` class that applies ``limits`` to one archive.

    Each call makes a new class with its own budget. Use one class for each
    ``TarFile``, so that the total limit applies per archive.

    Args:
        limits: The limits to apply.

    Returns:
        A ``TarInfo`` subclass to give to ``tarfile.open(tarinfo=...)``.
    """
    budget = _TarHeaderBudget(limits)

    class _LimitedTarInfo(tarfile.TarInfo):
        # ``_proc_member`` is private, but ``tarfile`` documents it in its source
        # as the hook for subclasses. It is called for each header block before
        # the payload of an extended header is read, and for the chained headers
        # that follow it. Checked against the stdlib of Python 3.10 to 3.15.
        def _proc_member(self, tarfile_obj: tarfile.TarFile) -> tarfile.TarInfo:
            if self.type in _EXTENDED_HEADER_TYPES:
                budget.charge(self)
            elif self.type == tarfile.GNUTYPE_SPARSE:
                # Old GNU sparse members chain extension blocks with no declared
                # size, so no budget applies. METS-GBS packages do not use them.
                raise ArchiveLimitError(
                    f"TAR member at offset {self.offset} is a GNU sparse file, "
                    "which is not supported."
                )
            return super()._proc_member(tarfile_obj)  # type: ignore

        # ``_proc_pax`` calls this on the pax header for a GNU sparse 1.0 member.
        # It reads the sparse map from the member data, up to a count that the
        # data gives, so no declared size bounds the read. METS-GBS packages do
        # not use sparse members. Checked against the stdlib of Python 3.10 to
        # 3.15.
        def _proc_gnusparse_10(
            self,
            next: tarfile.TarInfo,
            pax_headers: dict[str, str],
            tarfile_obj: tarfile.TarFile,
        ) -> None:
            raise ArchiveLimitError(
                f"TAR member at offset {next.offset} is a GNU sparse 1.0 file, "
                "which is not supported."
            )

    return _LimitedTarInfo


def open_tar_gz(
    source: Path | BytesIO, limits: TarHeaderLimits | None = None
) -> tarfile.TarFile:
    """Open a gzip-compressed TAR archive for reading with header limits.

    Args:
        source: The archive path or stream.
        limits: The header limits. ``DEFAULT_TAR_HEADER_LIMITS`` when None.

    Returns:
        The open archive.

    Raises:
        ArchiveLimitError: If the headers read while opening exceed the limits.
            Later reads of members raise the same error.
        tarfile.TarError: If the input is not a valid gzip-compressed TAR archive.
    """
    tarinfo = limited_tarinfo_class(
        DEFAULT_TAR_HEADER_LIMITS if limits is None else limits
    )
    if isinstance(source, Path):
        return tarfile.open(name=source, mode="r:gz", tarinfo=tarinfo)
    return tarfile.open(fileobj=source, mode="r:gz", tarinfo=tarinfo)
