import logging
import time
from pathlib import Path

from docling.datamodel import vlm_model_specs
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    VlmPipelineOptions,
)
from docling.datamodel.pipeline_options_vlm_model import ApiVlmOptions, ResponseFormat
from docling.datamodel.settings import settings
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline

_log = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO)

    BATCH_SIZE = 32

    settings.perf.page_batch_size = BATCH_SIZE
    settings.debug.profile_pipeline_timings = True

    data_folder = Path(__file__).parent / "../../tests/data"
    input_doc_paths = [
        data_folder / "pdf/2206.01062.pdf",
        # data_folder / "pdf/2203.01017v2.pdf",
        # data_folder / "pdf/2305.03393v1.pdf",
        # data_folder / "pdf/redp5110_sampled.pdf",
    ]

    vlm_options = ApiVlmOptions(
        url="http://localhost:1234/v1/chat/completions",  # LM studio defaults to port 1234, VLLM to 8000
        params=dict(
            model=vlm_model_specs.GRANITEDOCLING_TRANSFORMERS.repo_id,
            max_tokens=8192,
            skip_special_tokens=True,  # needed for VLLM
        ),
        prompt=vlm_model_specs.GRANITEDOCLING_TRANSFORMERS.prompt,
        timeout=90,
        scale=2.0,
        temperature=0.0,
        concurrency=BATCH_SIZE,
        response_format=ResponseFormat.DOCTAGS,
    )

    pipeline_options = VlmPipelineOptions(
        vlm_options=vlm_options,
        enable_remote_services=True,  # required when using a remote inference service.
    )

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=VlmPipeline,
                pipeline_options=pipeline_options,
            ),
        }
    )

    start_time = time.time()
    conv_result = doc_converter.convert(input_doc_paths)  # noqa: F841
    end_time = time.time() - start_time

    # TODO: add timings which don't include models init.
    _log.info(f"Document converted in {end_time:.2f} seconds.")


if __name__ == "__main__":
    main()
