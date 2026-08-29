"""Convert PDFs to Markdown with OCR (Russian + Simplified Chinese).

Reads every PDF in /data (mounted read-only), runs the Docling PDF pipeline with
EasyOCR restricted to `ru` and `ch_sim`, and writes one Markdown file per input
to /out (mounted writable). Non-PDF files are skipped.

Usage (via docker-compose.yaml):
    docker compose run --rm docling
"""

import logging
import sys
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
_log = logging.getLogger("convert")

INPUT_DIR = Path("/data")
OUTPUT_DIR = Path("/out")
OCR_LANGUAGES = ["ru", "ch_sim"]


def main() -> int:
    if not INPUT_DIR.is_dir():
        _log.error("input directory %s is not mounted", INPUT_DIR)
        return 1

    pdf_paths = sorted(p for p in INPUT_DIR.rglob("*.pdf") if p.is_file())
    if not pdf_paths:
        _log.error("no PDF files found in %s", INPUT_DIR)
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.ocr_options = EasyOcrOptions(lang=OCR_LANGUAGES, use_gpu=False)

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )

    exit_code = 0
    for pdf_path in pdf_paths:
        _log.info("converting %s", pdf_path)
        try:
            result = converter.convert(pdf_path)
        except Exception as exc:  # noqa: BLE001 - keep processing the rest
            _log.error("failed: %s (%s)", pdf_path, exc)
            exit_code = 1
            continue

        md = result.document.export_to_markdown()
        out_file = OUTPUT_DIR / f"{pdf_path.stem}.md"
        out_file.write_text(md, encoding="utf-8")
        _log.info("wrote %s", out_file)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())