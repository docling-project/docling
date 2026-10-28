# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import logging
from io import BytesIO
from pathlib import Path
from typing import Union

_log = logging.getLogger(__name__)

# Latin-1 decodes any byte sequence, so it has to be the last resort.
_ENCODINGS = ("utf-8", "cp1252", "latin-1")


def decode_latex_content(path_or_stream: Union[BytesIO, Path]) -> str:
    latex_text = ""
    if isinstance(path_or_stream, BytesIO):
        raw_bytes = path_or_stream.getvalue()

        for encoding in _ENCODINGS:
            try:
                latex_text = raw_bytes.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        if not latex_text:
            _log.warning("Failed to decode LaTeX content, using replacement mode")
            latex_text = raw_bytes.decode("utf-8", errors="replace")
    elif isinstance(path_or_stream, Path):
        for encoding in _ENCODINGS:
            try:
                with open(path_or_stream, encoding=encoding) as f:
                    latex_text = f.read()
                break
            except UnicodeDecodeError:
                continue
            except FileNotFoundError:
                _log.error(f"LaTeX file not found: {path_or_stream}")
                break
            except OSError as e:
                _log.error(f"Error reading LaTeX file: {e}")
                break
    return latex_text
