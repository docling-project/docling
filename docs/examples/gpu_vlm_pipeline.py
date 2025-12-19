# %% [markdown]
#
# What this example does
# - Run a conversion using the best setup for GPU using VLM models
#
# Requirements
# - Python 3.10+
# - Install Docling: `pip install docling`
# - Install vLLM: `pip install vllm`
#
# How to run
# - `python docs/examples/gpu_vlm_pipeline.py`
#
# This example is part of a set of GPU optimization strategies. Read more about it in [GPU support](../../usage/gpu/)
#
# ### Start models with vllm
#
# ```console
# vllm serve ibm-granite/granite-docling-258M \
#   --host 127.0.0.1 --port 8000 \
#   --max-num-seqs 512 \
#   --max-num-batched-tokens 8192 \
#   --enable-chunked-prefill \
#   --gpu-memory-utilization 0.9
# ```
#
# ## Example code
# %%

import datetime
import logging
import time
from pathlib import Path

import numpy as np
from pydantic import TypeAdapter

from docling.datamodel import vlm_model_specs
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import (
    VlmPipelineOptions,
)
from docling.datamodel.pipeline_options_vlm_model import ApiVlmOptions, ResponseFormat
from docling.datamodel.settings import settings
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling.utils.profiling import ProfilingItem

_log = logging.getLogger(__name__)


def main():
    logging.getLogger("docling").setLevel(logging.WARNING)
    _log.setLevel(logging.INFO)

    BATCH_SIZE = 64

    settings.perf.page_batch_size = BATCH_SIZE
    settings.debug.profile_pipeline_timings = True

    data_folder = Path(__file__).parent / "../../tests/data"
    # input_doc_path = data_folder / "pdf" / "2305.03393v1.pdf"  # 14 pages
    input_doc_path = data_folder / "pdf" / "redp5110_sampled.pdf"  # 18 pages

    vlm_options = vlm_model_specs.GRANITEDOCLING_VLLM_API
    vlm_options.concurrency = BATCH_SIZE

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
    doc_converter.initialize_pipeline(InputFormat.PDF)
    end_time = time.time() - start_time
    _log.info(f"Pipeline initialized in {end_time:.2f} seconds.")

    now = datetime.datetime.now()
    conv_result = doc_converter.convert(input_doc_path)
    assert conv_result.status == ConversionStatus.SUCCESS

    num_pages = len(conv_result.pages)
    pipeline_runtime = conv_result.timings["pipeline_total"].times[0]
    _log.info(f"Document converted in {pipeline_runtime:.2f} seconds.")
    _log.info(f"  [efficiency]: {num_pages / pipeline_runtime:.2f} pages/second.")
    for stage in ("page_init", "vlm"):
        values = np.array(conv_result.timings[stage].times)
        _log.info(
            f"  [{stage}]: {np.min(values):.2f} / {np.median(values):.2f} / {np.max(values):.2f} seconds/page"
        )

    TimingsT = TypeAdapter(dict[str, ProfilingItem])
    timings_file = Path(f"result-timings-gpu-vlm-{now:%Y-%m-%d_%H-%M-%S}.json")
    with timings_file.open("wb") as fp:
        r = TimingsT.dump_json(conv_result.timings, indent=2)
        fp.write(r)
    _log.info(f"Profile details in {timings_file}.")


if __name__ == "__main__":
    main()
