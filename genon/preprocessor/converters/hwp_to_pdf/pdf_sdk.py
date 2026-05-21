from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import ClassVar

from .availability import pdf_sdk_available, pdf_sdk_binary, pdf_sdk_home
from .base import BackendName

_log = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SEC = 600


def _patch_fontconfig(fonts_dir: str, font_cache: str, tmp_dir: str) -> str:
    src = os.path.join(fonts_dir, "fonts_gen.conf")
    dst = os.path.join(tmp_dir, "fonts.conf")
    with open(src, "r", encoding="utf-8") as f:
        conf = f.read()
    conf = re.sub(r"<dir>[^<]*</dir>", f"<dir>{fonts_dir}</dir>", conf, count=1)
    conf = re.sub(r"<cachedir>[^<]*</cachedir>", f"<cachedir>{font_cache}</cachedir>", conf, count=1)
    with open(dst, "w", encoding="utf-8") as f:
        f.write(conf)
    return dst


class PdfSdkConverter:
    name: ClassVar[BackendName] = "pdf_sdk"

    def is_available(self) -> bool:
        return pdf_sdk_available()

    def convert(self, file_path: str) -> str | None:
        sdk_out_dir: Path | None = None
        keep_out_dir = False
        try:
            sdk_home = str(pdf_sdk_home())
            binary = str(pdf_sdk_binary())
            fonts_dir = os.path.join(sdk_home, "fonts")
            moduledata = os.path.join(sdk_home, "moduledata")
            font_cache = os.path.join(sdk_home, "font_cache")
            os.makedirs(font_cache, exist_ok=True)

            in_path = Path(file_path).resolve()
            sdk_out_dir = Path(tempfile.mkdtemp(prefix="pdfsdk_out_"))

            _log.info(
                f"[hwp_to_pdf:pdf_sdk] preflight: "
                f"binary_exists={os.path.exists(binary)} "
                f"binary_x={os.access(binary, os.X_OK)} "
                f"fonts_dir_exists={os.path.exists(fonts_dir)} "
                f"moduledata_exists={os.path.exists(moduledata)} "
                f"input_exists={in_path.exists()} "
                f"input_size={in_path.stat().st_size if in_path.exists() else 'N/A'} "
                f"sdk_out_dir={sdk_out_dir}"
            )

            with tempfile.TemporaryDirectory() as tmp:
                patched_conf = _patch_fontconfig(fonts_dir, font_cache, tmp)

                env = os.environ.copy()
                env["LD_LIBRARY_PATH"] = f"{moduledata}:{env.get('LD_LIBRARY_PATH', '')}"
                env["FONTCONFIG_FILE"] = patched_conf
                env["FONTCONFIG_PATH"] = fonts_dir
                env.setdefault("LANG", "C.UTF-8")
                env.setdefault("LC_ALL", "C.UTF-8")

                cmd = [
                    binary,
                    "-i", str(in_path),
                    "-o", str(sdk_out_dir),
                    "-t", tmp,
                    "-f", fonts_dir,
                    "-e", "-1",
                    "-p", "1",
                ]
                timeout = int(os.environ.get("HWP_TO_PDF_TIMEOUT_SEC", DEFAULT_TIMEOUT_SEC))
                _log.info(f"[hwp_to_pdf:pdf_sdk] cmd: {cmd}")
                proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)

                produced = sorted(p for p in sdk_out_dir.glob("*") if p.is_file())
                _log.info(
                    f"[hwp_to_pdf:pdf_sdk] returncode={proc.returncode} "
                    f"produced_files={[p.name for p in produced]}"
                )
                _log.info(f"[hwp_to_pdf:pdf_sdk] stdout={proc.stdout!r}")
                _log.info(f"[hwp_to_pdf:pdf_sdk] stderr={proc.stderr!r}")

                pdf_files = [p for p in produced if p.suffix.lower() == ".pdf"]
                if proc.returncode == 0 and pdf_files:
                    produced_pdf = pdf_files[0]
                    target_pdf = in_path.with_suffix('.pdf')
                    try:
                        shutil.copy2(produced_pdf, target_pdf)
                        _log.info(f"[hwp_to_pdf:pdf_sdk] success -> {target_pdf}")
                        return str(target_pdf)
                    except OSError as e:
                        _log.warning(
                            f"[hwp_to_pdf:pdf_sdk] target copy failed ({e}); "
                            f"using temp path: {produced_pdf}"
                        )
                        keep_out_dir = True
                        return str(produced_pdf)

                _log.warning(
                    f"[hwp_to_pdf:pdf_sdk] FAILED - "
                    f"returncode={proc.returncode}, produced={[p.name for p in produced]}"
                )
                return None
        except subprocess.TimeoutExpired as e:
            _log.error(f"[hwp_to_pdf:pdf_sdk] timeout after {e.timeout}s for {file_path}")
            return None
        except Exception as e:
            _log.error(f"[hwp_to_pdf:pdf_sdk] error: {e}", exc_info=True)
            return None
        finally:
            if sdk_out_dir is not None and sdk_out_dir.exists() and not keep_out_dir:
                shutil.rmtree(sdk_out_dir, ignore_errors=True)
