# %% [markdown]
# Run conversion across multiple input formats and customize handling per type.
#
# What this example does
# - Demonstrates converting a mixed list of files (PDF, DOCX, PPTX, HTML, images, etc.).
# - Shows how to restrict `allowed_formats` and override `format_options` per format.
# - Writes results (Markdown, JSON, YAML) to `scratch/`.
#
# Prerequisites
# - Install Docling and any format-specific dependencies (e.g., for DOCX/PPTX parsing).
# - Ensure you can import `docling` from your Python environment.
# - YAML export requires `PyYAML` (`pip install pyyaml`).
#
# How to run
# - From the repository root, run: `python docs/examples/run_with_formats.py`.
# - Outputs are written under `scratch/` next to where you run the script.
# - If `scratch/` does not exist, create it before running.
#
# Customizing inputs
# - Update `input_paths` to include or remove files on your machine.
# - Non-whitelisted formats are ignored (see `allowed_formats`).
#
# Notes
# - `allowed_formats`: explicit whitelist of formats that will be processed.
# - `format_options`: per-format pipeline/backend overrides. Everything is optional; defaults exist.
# - Exports: per input, writes `<stem>.md`, `<stem>.json`, and `<stem>.yaml` in `scratch/`.

# %%

import json
import logging
import os
from pathlib import Path

import yaml

from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.settings import DEFAULT_PAGE_RANGE
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    WordFormatOption,
)
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline

_log = logging.getLogger(__name__)

# Check if running in CI
IS_CI = os.environ.get("CI", "").lower() in ("true", "1", "yes")


def main():
    input_paths = [
        Path("README.md"),
        Path("tests/data/html/sources/wiki_duck.html"),
        Path("tests/data/docx/sources/word_sample.docx"),
        Path("tests/data/docx/sources/lorem_ipsum.docx"),
        Path("tests/data/pptx/sources/powerpoint_sample.pptx"),
        Path(__file__).parent / "2305.03393v1-pg9-img.png",
        Path("tests/data/pdf/sources/2206.01062.pdf"),
        Path("tests/data/asciidoc/sources/asciidoc_01.asciidoc"),
    ]

    ## for defaults use:
    # doc_converter = DocumentConverter()

    ## to customize use:

    # Below we explicitly whitelist formats and override behavior for some of them.
    # You can omit this block and use the defaults (see above) for a quick start.
    doc_converter = DocumentConverter(  # all of the below is optional, has internal defaults.
        allowed_formats=[
            InputFormat.PDF,
            InputFormat.IMAGE,
            InputFormat.DOCX,
            InputFormat.HTML,
            InputFormat.PPTX,
            InputFormat.ASCIIDOC,
            InputFormat.CSV,
            InputFormat.MD,
        ],  # whitelist formats, non-matching files are ignored.
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=StandardPdfPipeline, backend=PyPdfiumDocumentBackend
            ),
            InputFormat.DOCX: WordFormatOption(
                pipeline_cls=SimplePipeline  # or set a backend, e.g., MsWordDocumentBackend
                # If you change the backend, remember to import it, e.g.:
                #   from docling.backend.msword_backend import MsWordDocumentBackend
            ),
        },
    )

    # Limit PDF pages under CI to keep runtime low. The range starts at page 1
    # so single-page inputs (e.g. images) stay within range and remain valid.
    page_range = (1, 2) if IS_CI else DEFAULT_PAGE_RANGE
    conv_results = doc_converter.convert_all(input_paths, page_range=page_range)

    for res in conv_results:
        out_path = Path("scratch")  # ensure this directory exists before running
        print(
            f"Document {res.input.file.name} converted."
            f"\nSaved markdown output to: {out_path!s}"
        )
        _log.debug(res.document._export_to_indented_text(max_text_len=16))
        # Export Docling document to Markdown:
        with (out_path / f"{res.input.file.stem}.md").open("w") as fp:
            fp.write(res.document.export_to_markdown())

        with (out_path / f"{res.input.file.stem}.json").open("w") as fp:
            fp.write(json.dumps(res.document.export_to_dict()))

        with (out_path / f"{res.input.file.stem}.yaml").open("w") as fp:
            fp.write(yaml.safe_dump(res.document.export_to_dict()))


if __name__ == "__main__":
    main()
