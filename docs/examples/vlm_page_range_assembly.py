# %% [markdown]
# Convert a PDF in page ranges with the VLM pipeline, then assemble one document.
#
# What this example does
# - Converts a PDF in separate page ranges with the VLM pipeline.
# - Checks that the ranges cover every page exactly once.
# - Concatenates the range documents into one complete `DoclingDocument`.
#
# Why
# - An external scheduler can send each range to a different worker, each with
#   its own converter. This example runs the ranges one after the other.
#
# Prerequisites
# - Install Docling with VLM extras and a backend (Transformers or MLX).
# - Ensure your environment can download model weights (e.g., from Hugging Face).
#
# How to run
# - From the repository root, run: `python docs/examples/vlm_page_range_assembly.py`.
# - The script prints the page numbers of the assembled document and its Markdown.
#
# Notes
# - `page_range=(first, last)` uses inclusive, one-based page numbers. The VLM
#   pipeline keeps the source page numbers in the returned document and its
#   provenance.
# - `DoclingDocument.concatenate` renumbers documents into a continuous sequence.
#   For an ordered partition without gaps that starts at page 1, each offset is
#   zero, so the source page numbers stay the same.
# - The VLM pipeline interprets each page independently. Range conversion keeps
#   the page-local reading order, lists, tables, headers, footers, and key/value
#   cell provenance of a whole conversion with the same page responses. It does
#   not join a table or list that continues onto another page. Model responses
#   can change with inference batching, so the predictions can differ.
# - Do not concatenate an incomplete set of ranges as a complete document. A
#   subset that starts after page 1, a missing page, or a failed range closes
#   gaps and shifts page numbers and provenance without an error.
# - Concatenation does not set `origin`, and it does not copy the conversion
#   status, errors, timings, or input metadata. Handle them separately.
# - Each range opens the source again. Measure the extra backend and rendering
#   work when you select a range size.

# %%

from collections.abc import Iterable
from pathlib import Path

from docling_core.types.doc import DoclingDocument

from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.document import ConversionResult
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline


def assemble_complete_document(
    results: Iterable[ConversionResult],
) -> DoclingDocument:
    """Concatenate range results that cover every source page exactly once."""
    results = list(results)
    if not results:
        raise ValueError("No ranges to assemble")
    for result in results:
        if result.status != ConversionStatus.SUCCESS:
            raise ValueError("A range did not complete successfully")
        if not result.document.pages:
            raise ValueError("A range has no pages")
    source_page_count = results[0].input.page_count
    results.sort(key=lambda result: min(result.document.pages))
    next_page = 1
    for result in results:
        pages = sorted(result.document.pages)
        if pages != list(range(next_page, next_page + len(pages))):
            raise ValueError("Missing, overlapping, or unordered pages")
        next_page += len(pages)
    if next_page != source_page_count + 1:
        raise ValueError("The final range is missing pages")
    return DoclingDocument.concatenate([result.document for result in results])


def main() -> None:
    data_folder = Path(__file__).parent / "../../tests/data"
    source = data_folder / "pdf/sources/2206.01062.pdf"

    # Each worker uses the same source and the same VLM configuration.
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_cls=VlmPipeline),
        }
    )

    # The page ranges are inclusive and one-based.
    ranges = [(1, 3), (4, 6), (7, 9)]
    results = [converter.convert(source, page_range=span) for span in ranges]

    document = assemble_complete_document(results)
    print(f"Assembled pages: {sorted(document.pages)}")
    print(document.export_to_markdown())


if __name__ == "__main__":
    main()
