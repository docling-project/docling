from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import unicodedata
from pathlib import Path
from typing import ClassVar

from .availability import libreoffice_available
from .base import BackendName

_log = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SEC = 600


def _convert_arg_for(ext: str) -> str:
    if ext in ('.ppt', '.pptx'):
        return "pdf:impress_pdf_Export"
    if ext in ('.doc', '.docx'):
        return "pdf:writer_pdf_Export"
    if ext in ('.xls', '.xlsx', '.csv'):
        return "pdf:calc_pdf_Export"
    return "pdf"


class LibreOfficeConverter:
    name: ClassVar[BackendName] = "libreoffice"

    def is_available(self) -> bool:
        return libreoffice_available()

    def convert(self, file_path: str) -> str | None:
        try:
            in_path = Path(file_path).resolve()
            out_dir = in_path.parent
            pdf_path = in_path.with_suffix('.pdf')

            env = os.environ.copy()
            env.setdefault("LANG", "C.UTF-8")
            env.setdefault("LC_ALL", "C.UTF-8")

            convert_arg = _convert_arg_for(in_path.suffix.lower())

            try:
                in_path.name.encode('ascii')
                candidates = [in_path]
                tmp_dir = None
            except UnicodeEncodeError:
                tmp_dir = Path(tempfile.mkdtemp())
                ascii_name = unicodedata.normalize('NFKD', in_path.stem).encode('ascii', 'ignore').decode('ascii') or "file"
                ascii_copy = tmp_dir / f"{ascii_name}{in_path.suffix}"
                shutil.copy2(in_path, ascii_copy)
                candidates = [ascii_copy, in_path]

            timeout = int(os.environ.get("HWP_TO_PDF_TIMEOUT_SEC", DEFAULT_TIMEOUT_SEC))
            for cand in candidates:
                cmd = [
                    "soffice", "--headless",
                    "--convert-to", convert_arg,
                    "--outdir", str(out_dir),
                    str(cand),
                ]
                proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
                if proc.returncode == 0 and pdf_path.exists():
                    if tmp_dir:
                        shutil.rmtree(tmp_dir, ignore_errors=True)
                    return str(pdf_path)
                _log.warning(f"[hwp_to_pdf:libreoffice] stderr: {proc.stderr.strip()[:500]}")

            if tmp_dir:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            return None
        except subprocess.TimeoutExpired as e:
            _log.error(f"[hwp_to_pdf:libreoffice] timeout after {e.timeout}s for {file_path}")
            return None
        except Exception as e:
            _log.error(f"[hwp_to_pdf:libreoffice] error: {e}")
            return None
