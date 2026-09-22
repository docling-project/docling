# %% [markdown]
# Crop-based conversion with Granite for Docling.
#
# What this example does
# - Runs the experimental `LayoutCropVlmPipeline`: the Docling layout model finds the
#   regions of each page, Granite for Docling recognizes every region from its crop,
#   and the standard assembly and reading-order stages pack the results.
# - Talks to the model through an OpenAI-compatible server (e.g. vLLM).
# - Writes DocLang, Markdown and JSON for every input.
#
# Prerequisites
# - A server that serves a Granite for Docling checkpoint, for example:
#   `vllm serve <model-dir> --served-model-name granite-for-docling --port 8000`
#
# How to run
# - `python docs/examples/layout_crop_vlm_granite_for_docling.py INPUT [INPUT ...]`
# - `--url`, `--model` and `--out` select the server, the served model name and the
#   output directory.
#
# Notes
# - Each region is sent with the `<doclang> [<task>]` prompt matching its layout label
#   and the reply is forced to open with the matching element tag, so the detected
#   region type is never re-decided by the VLM.
# - Pictures are kept as pictures; their content is not sent to the VLM.

# %%

import argparse
import logging
import time
from pathlib import Path

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import VlmConvertOptions
from docling.datamodel.vlm_engine_options import ApiVlmEngineOptions
from docling.document_converter import (
    DocumentConverter,
    ImageFormatOption,
    PdfFormatOption,
)
from docling.experimental.datamodel.layout_crop_vlm_pipeline_options import (
    LayoutCropVlmOptions,
    LayoutCropVlmPipelineOptions,
)
from docling.experimental.pipeline.layout_crop_vlm_pipeline import LayoutCropVlmPipeline
from docling.models.inference_engines.vlm import VlmEngineType

_log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--url", default="http://localhost:8000/v1/chat/completions")
    parser.add_argument("--model", default="granite-for-docling")
    parser.add_argument("--out", type=Path, default=Path("scratch/layout_crop_vlm"))
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument(
        "--device",
        default="auto",
        help="Device of the layout model, e.g. `cpu` when the server owns the GPU.",
    )
    parser.add_argument(
        "--log-replies", action="store_true", help="Log the raw VLM reply per region."
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    if args.log_replies:
        logging.getLogger("docling.experimental.models.layout_crop_vlm_model").setLevel(
            logging.DEBUG
        )

    vlm_options = VlmConvertOptions.from_preset(
        "granite_for_docling_500m",
        engine_options=ApiVlmEngineOptions(
            engine_type=VlmEngineType.API,
            url=args.url,
            # DocLang tags are special tokens; the server must not strip them.
            params={"model": args.model, "skip_special_tokens": False},
            concurrency=args.concurrency,
            timeout=300,
        ),
    )
    pipeline_options = LayoutCropVlmPipelineOptions(
        enable_remote_services=True,
        accelerator_options=AcceleratorOptions(device=args.device),
        crop_vlm_options=LayoutCropVlmOptions(
            vlm_options=vlm_options,
            engine_batch_size=4 * args.concurrency,
        ),
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=LayoutCropVlmPipeline, pipeline_options=pipeline_options
            ),
            InputFormat.IMAGE: ImageFormatOption(
                pipeline_cls=LayoutCropVlmPipeline, pipeline_options=pipeline_options
            ),
        }
    )

    args.out.mkdir(parents=True, exist_ok=True)
    for source in args.inputs:
        start = time.time()
        result = converter.convert(source)
        doc = result.document
        _log.info(
            "%s: %s, %d pages, %d texts, %d tables in %.1fs",
            source.name,
            result.status.value,
            len(doc.pages),
            len(doc.texts),
            len(doc.tables),
            time.time() - start,
        )
        doc.save_as_doclang(args.out / f"{source.stem}.dclg.xml")
        doc.save_as_markdown(args.out / f"{source.stem}.md")
        doc.save_as_json(args.out / f"{source.stem}.json")


if __name__ == "__main__":
    main()
