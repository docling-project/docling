from __future__ import annotations

import os
import shutil
import subprocess
from io import BytesIO
from pathlib import Path
from tempfile import mkdtemp
from typing import TYPE_CHECKING, Callable, Optional

import pypdfium2
from PIL import Image, ImageChops

if TYPE_CHECKING:
    from docx.document import Document


def get_libreoffice_cmd(raise_if_unavailable: bool = False) -> Optional[str]:
    """Return the libreoffice cmd and optionally test it."""

    libreoffice_cmd = (
        shutil.which("libreoffice")
        or shutil.which("soffice")
        or (
            "/Applications/LibreOffice.app/Contents/MacOS/soffice"
            if os.path.isfile("/Applications/LibreOffice.app/Contents/MacOS/soffice")
            else None
        )
    )

    if raise_if_unavailable:
        if libreoffice_cmd is None:
            raise RuntimeError("Libreoffice not found")

        # The following test will raise if the libreoffice_cmd cannot be used
        subprocess.run(
            [
                libreoffice_cmd,
                "-h",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )

    return libreoffice_cmd


def convert_to_modern_format(
    source: BytesIO | Path,
    source_suffix: str,
    target_suffix: str,
    timeout_s: int = 120,
) -> BytesIO:
    """Convert a legacy binary Office file to modern Open XML format via LibreOffice.

    Both file paths and in-memory streams are accepted.  When a ``BytesIO`` is
    supplied the bytes are written to a temporary file (named with
    ``source_suffix`` so LibreOffice can detect the format) before invoking the
    CLI; the temporary file is removed together with the rest of the temp
    directory once the conversion finishes.

    Args:
        source: Path to the source file, or a ``BytesIO`` with its contents.
        source_suffix: File extension of the source format without leading dot
            (e.g. ``"doc"``, ``"xls"``, ``"ppt"``).  Required when *source* is
            a ``BytesIO`` so the temp file gets the right name; ignored for
            ``Path`` inputs (the path's own suffix is used instead).
        target_suffix: Target extension without leading dot (``"docx"``,
            ``"xlsx"``, or ``"pptx"``).
        timeout_s: Timeout in seconds for the LibreOffice subprocess.

    Returns:
        A ``BytesIO`` buffer with the converted file contents.

    Raises:
        RuntimeError: When LibreOffice is not installed, the subprocess fails,
            or the expected output file is not produced.
    """
    libreoffice_cmd = get_libreoffice_cmd()
    if libreoffice_cmd is None:
        raise RuntimeError(
            f"LibreOffice is required to convert a .{source_suffix} file to "
            f".{target_suffix}. Install LibreOffice and make sure it is on PATH."
        )

    tmp_dir = Path(mkdtemp())
    try:
        if isinstance(source, BytesIO):
            source.seek(0)
            input_path = tmp_dir / f"input.{source_suffix}"
            input_path.write_bytes(source.read())
        else:
            input_path = source

        subprocess.run(
            [
                libreoffice_cmd,
                "--headless",
                "--convert-to",
                target_suffix,
                "--outdir",
                str(tmp_dir),
                str(input_path),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=timeout_s,
        )

        converted_path = tmp_dir / (input_path.stem + "." + target_suffix)
        if not converted_path.exists():
            raise RuntimeError(
                f"LibreOffice did not produce the expected output: {converted_path}"
            )

        return BytesIO(converted_path.read_bytes())
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def get_docx_to_pdf_converter() -> Optional[Callable]:
    """
    Detects the best available DOCX to PDF tool and returns a conversion function.
    The returned function accepts (input_path, output_path).
    Returns None if no tool is available.
    """

    # Try LibreOffice
    libreoffice_cmd = get_libreoffice_cmd()

    if libreoffice_cmd:

        def convert_with_libreoffice(input_path, output_path):
            subprocess.run(
                [
                    libreoffice_cmd,
                    "--headless",
                    "--convert-to",
                    "pdf",
                    "--outdir",
                    os.path.dirname(output_path),
                    input_path,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )

            expected_output = os.path.join(
                os.path.dirname(output_path),
                os.path.splitext(os.path.basename(input_path))[0] + ".pdf",
            )
            if expected_output != output_path:
                os.rename(expected_output, output_path)

        return convert_with_libreoffice

    ## Space for other DOCX to PDF converters if available

    # No tools found
    return None


def crop_whitespace(image: Image.Image, bg_color=None, padding=0) -> Image.Image:
    if bg_color is None:
        bg_color = image.getpixel((0, 0))

    bg = Image.new(image.mode, image.size, bg_color)
    diff = ImageChops.difference(image, bg)
    bbox = diff.getbbox()

    if bbox:
        left, upper, right, lower = bbox
        left = max(0, left - padding)
        upper = max(0, upper - padding)
        right = min(image.width, right + padding)
        lower = min(image.height, lower + padding)
        return image.crop((left, upper, right, lower))
    else:
        return image


def get_pil_from_dml_docx(
    docx: Document, converter: Optional[Callable]
) -> Optional[Image.Image]:
    if converter is None:
        return None

    temp_dir = Path(mkdtemp())
    try:
        temp_docx = Path(temp_dir / "drawing_only.docx")
        temp_pdf = Path(temp_dir / "drawing_only.pdf")

        # 1) Save docx temporarily
        docx.save(str(temp_docx))

        # 2) Export to PDF
        converter(temp_docx, temp_pdf)

        # 3) Load PDF as PNG
        pdf = pypdfium2.PdfDocument(temp_pdf)
        page = pdf[0]
        image = crop_whitespace(page.render(scale=2).to_pil())
        page.close()
        pdf.close()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return image
