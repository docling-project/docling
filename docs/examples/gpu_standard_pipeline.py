import logging
import time
from pathlib import Path

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    ThreadedPdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.threaded_standard_pdf_pipeline import ThreadedStandardPdfPipeline

_log = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO)

    data_folder = Path(__file__).parent / "../../tests/data"
    input_doc_paths = [
        data_folder / "pdf/2206.01062.pdf",
        # data_folder / "pdf/2203.01017v2.pdf",
        # data_folder / "pdf/2305.03393v1.pdf",
        # data_folder / "pdf/redp5110_sampled.pdf",
    ]

    pipeline_options = ThreadedPdfPipelineOptions(
        accelerator_options=AcceleratorOptions(
            device=AcceleratorDevice.CUDA,
        ),
        ocr_batch_size=4,
        layout_batch_size=32,
        table_batch_size=4,
    )
    pipeline_options.do_ocr = False

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=ThreadedStandardPdfPipeline,
                pipeline_options=pipeline_options,
            )
        }
    )

    start_time = time.time()
    conv_result = doc_converter.convert(input_doc_paths)  # noqa: F841
    end_time = time.time() - start_time

    # TODO: add timings which don't include models init.
    _log.info(f"Document converted in {end_time:.2f} seconds.")


if __name__ == "__main__":
    main()
