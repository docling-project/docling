# %% [markdown]
# Process a batch of broker research reports (Chinese PDFs) and export rich outputs.
#
# What this example does
# - Converts every PDF in an input directory with Docling's standard pipeline.
# - Exports one Markdown file per report with tables and referenced figures.
# - Prints a per-file summary (pages / tables / pictures) for quick QA.
#
# Why this matters
# - Broker reports mix dense tables, figures and footnotes; this recipe shows the
#   minimal production loop: convert -> export -> audit, which is the backbone of
#   research-report knowledge bases.
#
# How to run
# - From the repo root: `python docs/examples/process_broker_reports.py`.
# - Input: pass a directory via `--input` (default: the example data folder).
# - Outputs are written to `scratch/broker_reports/`.
#
# Key options
# - `ImageRefMode.REFERENCED` keeps the Markdown small and images on disk.
# - `PdfPipelineOptions` enable picture/table structures used in the summary.

# %%

import argparse
import logging
from pathlib import Path

from docling_core.types.doc import ImageRefMode

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

_log = logging.getLogger(__name__)


def build_converter() -> DocumentConverter:
    """Converter with picture and table structure enabled for report QA."""
    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_picture_images = True
    pipeline_options.generate_table_structure = True
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )


def export_report(converter: DocumentConverter, pdf_path: Path, out_dir: Path) -> dict:
    """Convert one report and export Markdown with referenced images."""
    result = converter.convert(pdf_path)
    doc = result.document
    report = {
        "file": pdf_path.name,
        "pages": len(doc.pages),
        "tables": len(doc.tables),
        "pictures": len(doc.pictures),
    }
    if doc.tables or doc.pictures:
        out_dir.mkdir(parents=True, exist_ok=True)
        md_path = out_dir / f"{pdf_path.stem}.md"
        try:
            doc.save_as_markdown(
                filename=md_path,
                artifacts_dir=out_dir / "images",
                image_mode=ImageRefMode.REFERENCED,
            )
        except TypeError:
            # Older docling-core: fall back to plain Markdown export.
            md_path.write_text(doc.export_to_markdown(), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-process broker research reports.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("docs/examples/data"),
        help="Directory containing report PDFs.",
    )
    args = parser.parse_args()

    out_dir = Path("scratch/broker_reports")
    converter = build_converter()
    reports = []
    for pdf_path in sorted(args.input.glob("*.pdf")):
        reports.append(export_report(converter, pdf_path, out_dir))
        _log.info("converted %s", pdf_path.name)

    print(f"\nProcessed {len(reports)} report(s) -> {out_dir}")
    for report in reports:
        print(
            f"  {report['file']}: {report['pages']} pages, "
            f"{report['tables']} tables, {report['pictures']} pictures"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
