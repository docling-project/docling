from pathlib import Path

from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
    TesseractCliOcrOptions,
    TesseractOcrOptions,
)
from docling.datamodel.settings import settings
from docling.document_converter import DocumentConverter, PdfFormatOption


def main():
    input_doc = Path("./tests/data/pdf/2206.01062.pdf")

    # Explicitly set the accelerator
    # accelerator_options = AcceleratorOptions(
    #     num_threads=8, device=AcceleratorDevice.AUTO
    # )
    accelerator_options = AcceleratorOptions(
        num_threads=8, device=AcceleratorDevice.CPU
    )
    # accelerator_options = AcceleratorOptions(
    #     num_threads=8, device=AcceleratorDevice.MPS
    # )
    # accelerator_options = AcceleratorOptions(
    #     num_threads=8, device=AcceleratorDevice.CUDA
    # )

    # easyocr doesnt support cuda:N allocation, defaults to cuda:0
    # accelerator_options = AcceleratorOptions(num_threads=8, device="cuda:1")

    pipeline_options = PdfPipelineOptions()
    pipeline_options.accelerator_options = accelerator_options
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            )
        }
    )

    # Enable the profiling to measure the time spent
    settings.debug.profile_pipeline_timings = True

    # Convert the document
    conversion_result = converter.convert(input_doc)
    doc = conversion_result.document

    # List with total time per document
    doc_conversion_secs = conversion_result.timings["pipeline_total"].times

    md = doc.export_to_markdown()
    print(md)
    print(f"Conversion secs: {doc_conversion_secs}")


if __name__ == "__main__":
    main()
