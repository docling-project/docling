"""Utilities for PDF embedded-file attachment extraction.

Thin wrapper over ``docling-parse`` attachment APIs. Isolates backend
differences and filename sanitization.
"""

from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO

_log = logging.getLogger(__name__)

MAX_ATTACHMENT_BYTES = 200 * 1024 * 1024

_WINDOWS_RESERVED = {
    "con",
    "prn",
    "aux",
    "nul",
    "com1",
    "com2",
    "com3",
    "com4",
    "com5",
    "com6",
    "com7",
    "com8",
    "com9",
    "lpt1",
    "lpt2",
    "lpt3",
    "lpt4",
    "lpt5",
    "lpt6",
    "lpt7",
    "lpt8",
    "lpt9",
}


@dataclass(frozen=True)
class RawPdfAttachment:
    """Attachment metadata without decoded payload."""

    index: int
    name: str
    mime_type: str | None
    size: int
    annotations: list[Any]  # List[FileAttachmentAnnotation]


def extract_pdf_attachments(
    source: Path | BytesIO,
    password: str | None = None,
) -> tuple[list[RawPdfAttachment], object | None]:
    """Extract attachment metadata from a PDF source.

    Loads the PDF via ``DoclingPdfParser`` and calls ``get_attachments()``.

    Returns:
        A tuple ``(attachments, doc)`` where ``doc`` is the underlying
        ``PdfDocument`` (caller must keep it alive while streaming). On
        non-docling-parse backends or extraction failure returns ``([], None)``.
    """
    try:
        from docling_parse.pdf_parser import DoclingPdfParser
    except ImportError:
        _log.warning("attachments skipped: docling-parse not installed")
        return [], None

    try:
        parser = DoclingPdfParser()
        if isinstance(source, Path):
            doc = parser.load(str(source), password=password)
        else:
            # BytesIO — read bytes and load from buffer
            pos = source.tell()
            try:
                data = source.getvalue()
            except Exception:
                source.seek(0)
                data = source.read()
                source.seek(pos)
            doc = parser.load_from_bytes(data, password=password)
        if doc is None or not doc.is_loaded():
            return [], None
        raw = doc.get_attachments()
    except Exception as exc:
        _log.warning("attachment extraction failed: %s", exc, exc_info=True)
        return [], None

    result: list[RawPdfAttachment] = []
    for idx, att in enumerate(raw):
        result.append(
            RawPdfAttachment(
                index=idx,
                name=att.name,
                mime_type=att.mime_type,
                size=att.size,
                annotations=list(att.annotations),
            )
        )
    return result, doc


def open_attachment_stream(doc: object, index: int) -> BinaryIO:
    """Open a readable stream for attachment ``index`` of ``doc``.

    Spec section 8 / docling-parse contract: ``max_size`` is required and
    enforces ``MAX_ATTACHMENT_BYTES`` (200 MB). Returns ``BinaryIO`` — either
    ``BytesIO`` (small payloads) or a spooled
    ``NamedTemporaryFile``/``_AttachmentDeletingFile`` for larger payloads
    (threshold enforced inside ``PdfDocument.get_attachment_stream``).
    Caller must close the stream (see ``document_converter`` ``finally``).
    """

    # doc is a PdfDocument
    return doc.get_attachment_stream(index, max_size=MAX_ATTACHMENT_BYTES)  # type: ignore[union-attr]


def sanitize_attachment_filename(raw_name: str, **kwargs: Any) -> str:
    """Sanitize attachment filename for filesystem use.

    Original ``AttachmentItem.name`` is preserved; this is only for the
    ``target`` filesystem path.

    Spec section 5/8: Windows-safe sanitization — Windows reserved names,
    trailing dots/spaces, and 200-char limit.
    """
    # Backward-compat for legacy ``name=`` keyword (renamed to ``raw_name``
    # to avoid shadowing ``Path.name`` and clarify intent).
    if kwargs:
        # Pop legacy alias if caller used sanitize_attachment_filename(name=...)
        raw_name = kwargs.pop("name", raw_name)  # type: ignore[assignment]
    # Basename only — strip directory components (use Path for platform-safe parsing)
    sanitized = Path(raw_name).name or "attachment"
    # Replace path separators and control chars
    sanitized = sanitized.replace("/", "_").replace("\\", "_")
    sanitized = re.sub(r"[\x00-\x1f\x7f]", "_", sanitized)
    # Replace characters illegal on Windows
    sanitized = re.sub(r'[<>:"|?*]', "_", sanitized)
    # Trim trailing dots/spaces (Windows) — only trailing, not leading
    # (e.g. ".hidden" must stay; "file. " -> "file")
    sanitized = sanitized.rstrip(" .")
    if not sanitized:
        sanitized = "attachment"
    # Avoid Windows reserved names (case-insensitive, stem only)
    stem = Path(sanitized).stem.lower()
    if stem in _WINDOWS_RESERVED:
        sanitized = f"_{sanitized}"
    # Limit length to 200 chars exactly as spec (preserve suffix)
    if len(sanitized) > 200:
        suffix = Path(sanitized).suffix
        if len(suffix) >= 200:
            sanitized = sanitized[:200]
        else:
            stem_part = Path(sanitized).stem[: 200 - len(suffix)]
            sanitized = stem_part + suffix
        # Re-strip trailing dots/spaces after truncation (spec 5)
        sanitized = sanitized.rstrip(" .")
        if not sanitized:
            sanitized = "attachment"
    return sanitized


def unique_target(
    base_dir: Path, candidate: str, seen: set[str], **kwargs: Any
) -> Path:
    """Return a unique path under ``base_dir`` for ``candidate``, using ``_1`` suffix on collision.

    Spec section 5: collision handling with ``_1``/``_2`` suffix. ``seen``
    tracks already-used filenames. Case-insensitive only on Windows
    (spec clarification); POSIX remains case-sensitive. Updated in place.
    """
    # Backward-compat for legacy ``dir=`` keyword (renamed to ``base_dir``
    # to avoid shadowing builtin ``dir`` and clarify Path semantics).
    if kwargs and "dir" in kwargs:
        base_dir = kwargs.pop("dir")  # type: ignore[assignment]

    # Spec 5: case-insensitive tracking only on Windows.
    # Prefer pathlib over os.path; OS detection uses sys.platform (pathlib
    # handles filesystem semantics, platform check handles case-sensitivity rule).
    is_windows = sys.platform == "win32"

    def _key(name: str) -> str:
        return name.lower() if is_windows else name

    key = _key(candidate)
    if key not in seen:
        seen.add(key)
        return base_dir / candidate
    stem = Path(candidate).stem
    suffix = Path(candidate).suffix
    counter = 1
    while True:
        alt = f"{stem}_{counter}{suffix}"
        key = _key(alt)
        if key not in seen:
            seen.add(key)
            return base_dir / alt
        counter += 1
