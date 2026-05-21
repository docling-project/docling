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
    """genos-rhwp 의 `rhwp` 바이너리를 컨테이너 안에서 subprocess 로 호출하는 client.

    이미지 빌드 시 Rust multi-stage 로 `genonai/genos-rhwp` 를 `cargo build --release
    --bin rhwp` 후 `/usr/local/bin/rhwp` 에 설치한다 (Dockerfile.opensource /
    Dockerfile.enterprise 의 `rhwp_builder` stage). 런타임에 외부 서비스/네트워크
    의존 없이 동작한다.

    CLI:
        rhwp export-pdf <input.hwp> -o <output.pdf>
    """

    name: ClassVar[BackendName] = "rhwp"

    def is_available(self) -> bool:
        return rhwp_available()

    def convert(self, file_path: str) -> str | None:
        try:
            in_path = Path(file_path).resolve()
            if not in_path.exists():
                _log.warning("[hwp_to_pdf:rhwp] input not found: %s", in_path)
                return None

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
