import logging
import os
import platform
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import pypdfium2 as pdfium
from docling_core.types.doc.document import ImageRef

from docling.backend.latex.engines.base import RenderEngine

_log = logging.getLogger(__name__)


class TectonicEngine(RenderEngine):
    def __init__(self):
        self.cache_dir = Path.home() / ".cache" / "docling" / "tectonic"
        self.binary_path = None
        self._is_available = False

        self.install()

    def is_available(self) -> bool:
        return self._is_available

    def install(self):
        system_tectonic = shutil.which("tectonic")
        if system_tectonic:
            self.binary_path = Path(system_tectonic)
            self._is_available = True
            _log.info(f"Using system tectonic at {self.binary_path}")
            return

        self.binary_path = self.cache_dir / "tectonic"
        if self.binary_path.exists() and os.access(self.binary_path, os.X_OK):
            self._is_available = True
            return

        system = platform.system().lower()

        if system not in ["linux", "darwin"]:
            _log.warning(f"Tectonic engine is not supported on {system}")
            return

        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            _log.info("Downloading Tectonic binary using the official script...")

            subprocess.run(
                "curl --proto '=https' --tlsv1.2 -fsSL https://drop-sh.fullyjustified.net | sh",
                shell=True,
                cwd=self.cache_dir,
                check=True,
            )

            if self.binary_path.exists():
                self.binary_path.chmod(0o755)
                self._is_available = True
                _log.info(f"Tectonic successfully installed at {self.binary_path}")
            else:
                _log.warning("Tectonic binary not found after extraction.")

        except Exception as e:
            _log.warning(f"Failed to install Tectonic: {e}")

    def render(self, tikz_code: str, preamble: str = "") -> ImageRef | None:
        if not self.is_available():
            return None

        # Fallback preamble if none provided
        if not preamble.strip():
            preamble = "\\usepackage{tikz}\n\\usepackage{pgfplots}\n\\pgfplotsset{compat=newest}"

        # Minimal LaTeX document wrapping the Tikz code
        latex_doc = f"""\\documentclass[tikz, border=2pt]{{standalone}}
        {preamble}
        \\begin{{document}}
        {tikz_code}
        \\end{{document}}
"""

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            tex_file = temp_path / "diagram.tex"
            tex_file.write_text(latex_doc, encoding="utf-8")

            try:
                subprocess.run(
                    [str(self.binary_path), str(tex_file)],
                    cwd=temp_dir,
                    capture_output=True,
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                _log.warning(
                    f"Tectonic compilation failed: {e.stderr.decode('utf-8', errors='replace')}"
                )
                return None

            pdf_file = temp_path / "diagram.pdf"
            if not pdf_file.exists():
                _log.warning("Tectonic did not produce a PDF.")
                return None

            try:
                with pdfium.PdfDocument(pdf_file) as pdf:
                    page = pdf[0]
                    pil_image = page.render(scale=300 / 72).to_pil()
                    page.close()

                return ImageRef.from_pil(pil_image, dpi=300)
            except Exception as e:
                _log.warning(f"Failed to render PDF to image: {e}")
                return None
