# %% [markdown]
# Run the standard Docling pipeline on CPU-only machines while offloading
# OCR, layout, and table structure inference to remote GPU endpoints.
#
# Replace the endpoint URLs below with your own KServe/Triton deployments.
# The table-structure endpoint is expected to implement the JSON contract
# described in `KserveV2TableStructureModel`.
#
# For a local service that exposes all three model endpoints on one host, run
# `python docs/examples/remote_standard_pipeline_service.py`.
# Add `--demo` if you only want deterministic stub responses.

# %%

from pathlib import Path

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    KserveV2LayoutOptions,
    KserveV2OcrOptions,
    KserveV2TableStructureOptions,
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption


def main() -> None:
    input_doc_path = Path("tests/data/pdf/2206.01062.pdf")

    pipeline_options = PdfPipelineOptions(
        enable_remote_services=True,
        accelerator_options=AcceleratorOptions(
            device=AcceleratorDevice.CPU,
            num_threads=8,
        ),
        do_ocr=True,
        do_table_structure=True,
    )

    # Remote RapidOCR or equivalent OCR service.
    pipeline_options.ocr_options = KserveV2OcrOptions(
        url="http://ocr-gpu.example.com:8000",
        model_name="rapidocr",
        transport="http",
        scale=3.0,
        lang=["english"],
    )

    # Remote layout detection service.
    pipeline_options.layout_options = KserveV2LayoutOptions(
        url="http://layout-gpu.example.com:8000",
        model_name="layout-heron",
    )

    # Remote table-structure service. The endpoint receives a cropped table image
    # and a JSON payload with local text tokens and matching preferences.
    pipeline_options.table_structure_options = KserveV2TableStructureOptions(
        url="http://table-gpu.example.com:8000",
        model_name="table-structure",
        do_cell_matching=True,
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )

    result = converter.convert(input_doc_path)
    print(result.document.export_to_markdown())


if __name__ == "__main__":
    main()
