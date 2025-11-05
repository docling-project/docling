import datetime
import logging
import time
from pathlib import Path

import numpy as np
from pydantic import TypeAdapter

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import (
    ThreadedPdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.threaded_standard_pdf_pipeline import ThreadedStandardPdfPipeline
from docling.utils.profiling import ProfilingItem

_log = logging.getLogger(__name__)


def main():
    logging.getLogger("docling").setLevel(logging.WARNING)
    _log.setLevel(logging.INFO)

    data_folder = Path(__file__).parent / "../../tests/data"
    # input_doc_path = data_folder / "pdf" / "2305.03393v1.pdf"  # 14 pages
    input_doc_path = data_folder / "pdf" / "redp5110_sampled.pdf"  # 18 pages

    pipeline_options = ThreadedPdfPipelineOptions(
        accelerator_options=AcceleratorOptions(
            device=AcceleratorDevice.CUDA,
        ),
        ocr_batch_size=4,
        layout_batch_size=64,
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
    doc_converter.initialize_pipeline(InputFormat.PDF)
    init_runtime = time.time() - start_time
    _log.info(f"Pipeline initialized in {init_runtime:.2f} seconds.")

    start_time = time.time()
    conv_result = doc_converter.convert(input_doc_path)
    pipeline_runtime = time.time() - start_time
    assert conv_result.status == ConversionStatus.SUCCESS

    num_pages = len(conv_result.pages)
    _log.info(f"Document converted in {pipeline_runtime:.2f} seconds.")
    _log.info(f"  {num_pages / pipeline_runtime:.2f} pages/second.")


if __name__ == "__main__":
    main()
