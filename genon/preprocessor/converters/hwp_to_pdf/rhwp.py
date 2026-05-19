from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import ClassVar

from .availability import rhwp_available, rhwp_binary
from .base import BackendName

_log = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SEC = 600


class RhwpConverter:
    name: ClassVar[BackendName] = "rhwp"

    def is_available(self) -> bool:
        return rhwp_available()

    def convert(self, file_path: str) -> str | None:
        try:
            in_path = Path(file_path).resolve()
            out_path = in_path.with_suffix(".pdf")
            binary = str(rhwp_binary())

            env = os.environ.copy()
            env.setdefault("LANG", "C.UTF-8")
            env.setdefault("LC_ALL", "C.UTF-8")

            cmd = [binary, "export-pdf", str(in_path), "-o", str(out_path)]
            timeout = int(os.environ.get("HWP_TO_PDF_TIMEOUT_SEC", DEFAULT_TIMEOUT_SEC))

            _log.info(f"[hwp_to_pdf:rhwp] cmd: {cmd}")
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)

            if proc.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0:
                _log.info(f"[hwp_to_pdf:rhwp] success -> {out_path}")
                return str(out_path)

            _log.warning(
                f"[hwp_to_pdf:rhwp] FAILED rc={proc.returncode} "
                f"out_exists={out_path.exists()} stderr={proc.stderr[:500]!r}"
            )
            return None
        except subprocess.TimeoutExpired as e:
            _log.error(f"[hwp_to_pdf:rhwp] timeout after {e.timeout}s for {file_path}")
            return None
        except Exception as e:
            _log.error(f"[hwp_to_pdf:rhwp] error: {e}", exc_info=True)
            return None
