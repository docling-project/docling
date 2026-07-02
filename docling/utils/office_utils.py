import logging
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Union

_log = logging.getLogger(__name__)

_VBA_MARKER = "vbaProject.bin"


def warn_if_macros(path_or_stream: Union[BytesIO, Path]) -> None:
    """Warn if an OOXML file contains embedded VBA macros.

    Macros are not executed during conversion, but their presence indicates
    active content that callers may want to audit or reject upstream.
    """
    try:
        with zipfile.ZipFile(path_or_stream) as zf:
            names = zf.namelist()
            if any(n.startswith("/") or ".." in n for n in names):
                _log.warning(
                    "Skipping macro check: archive contains unsafe ZIP entry paths."
                )
                return
            has_macros = any(_VBA_MARKER in name for name in names)
        if has_macros:
            _log.warning(
                "Macro-enabled content detected (%s found in archive). "
                "Macros are not executed during conversion, but the source "
                "file may contain active content.",
                _VBA_MARKER,
            )
    except (zipfile.BadZipFile, OSError, ValueError):
        pass  # best-effort; the backend's own loader surfaces real errors
    finally:
        if isinstance(path_or_stream, BytesIO):
            path_or_stream.seek(0)
