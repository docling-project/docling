# %% [markdown]
# Compare different VLM models by running the VLM pipeline and timing outputs.
#
# What this example does
# - Iterates through a list of VLM model configurations and converts the same file.
# - Prints per-page generation times and saves JSON/MD/HTML to `scratch/`.
# - Summarizes total inference time and pages processed in a table.
#
# Requirements
# - Install `tabulate` for pretty printing (`pip install tabulate`).
#
# Prerequisites
# - Install Docling with VLM extras. Ensure models can be downloaded or are available.
#
# How to run
# - From the repo root: `python docs/examples/compare_vlm_models.py`.
# - Results are saved to `scratch/` with filenames including the model and framework.
#
# Notes
# - MLX models are skipped automatically on non-macOS platforms.
# - On CUDA systems, you can enable flash_attention_2 (see commented lines).
# - Running multiple VLMs can be GPU/CPU intensive and time‑consuming; ensure
#   enough VRAM/system RAM and close other memory‑heavy apps.

# %%

import json
import sys
import time
from pathlib import Path

from docling_core.types.doc import DocItemLabel, ImageRefMode
from docling_core.types.doc.document import DEFAULT_EXPORT_LABELS
from tabulate import tabulate

from docling.datamodel import vlm_model_specs
from docling.datamodel.accelerator_options import AcceleratorDevice
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    VlmPipelineOptions,
)
from docling.datamodel.pipeline_options_vlm_model import (
    InferenceFramework,
    InlineVlmOptions,
    ResponseFormat,
    TransformersModelType,
    TransformersPromptStyle,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline


def convert(sources: list[Path], converter: DocumentConverter):
    # Note: this helper assumes a single-item `sources` list. It returns after
    # processing the first source to keep runtime/output focused.
    model_id = pipeline_options.vlm_options.repo_id.replace("/", "_")
    framework = pipeline_options.vlm_options.inference_framework
    for source in sources:
        print("================================================")
        print("Processing...")
        print(f"Source: {source}")
        print("---")
        print(f"Model: {model_id}")
        print(f"Framework: {framework}")
        print("================================================")
        print("")

        res = converter.convert(source)

        print("")

        fname = f"{res.input.file.stem}-{model_id}-{framework}"

        inference_time = 0.0
        for i, page in enumerate(res.pages):
            inference_time += page.predictions.vlm_response.generation_time
            print("")
            print(
                f" ---------- Predicted page {i} in {pipeline_options.vlm_options.response_format} in {page.predictions.vlm_response.generation_time} [sec]:"
            )
            print(page.predictions.vlm_response.text)
            print(" ---------- ")

        print("===== Final output of the converted document =======")

        # Manual export for illustration. Below, `save_as_json()` writes the same
        # JSON again; kept intentionally to show both approaches.
        with (out_path / f"{fname}.json").open("w") as fp:
            fp.write(json.dumps(res.document.export_to_dict()))

        res.document.save_as_json(
            out_path / f"{fname}.json",
            image_mode=ImageRefMode.PLACEHOLDER,
        )
        print(f" => produced {out_path / fname}.json")

        res.document.save_as_markdown(
            out_path / f"{fname}.md",
            image_mode=ImageRefMode.PLACEHOLDER,
        )
        print(f" => produced {out_path / fname}.md")

        res.document.save_as_html(
            out_path / f"{fname}.html",
            image_mode=ImageRefMode.EMBEDDED,
            labels=[*DEFAULT_EXPORT_LABELS, DocItemLabel.FOOTNOTE],
            split_page_view=True,
        )
        print(f" => produced {out_path / fname}.html")

        pg_num = res.document.num_pages()
        print("")
        print(
            f"Total document prediction time: {inference_time:.2f} seconds, pages: {pg_num}"
        )
        print("====================================================")

        return [
            source,
            model_id,
            str(framework),
            pg_num,
            inference_time,
        ]


if __name__ == "__main__":
    sources = [
        "tests/data/pdf/2305.03393v1-pg9.pdf",
    ]

    out_path = Path("scratch")
    out_path.mkdir(parents=True, exist_ok=True)

    ## Definiton of more inline models
    llava_qwen = InlineVlmOptions(
        repo_id="llava-hf/llava-interleave-qwen-0.5b-hf",
        # prompt="Read text in the image.",
        prompt="Convert this page to markdown. Do not miss any text and only output the bare markdown!",
        # prompt="Parse the reading order of this document.",
        response_format=ResponseFormat.MARKDOWN,
        inference_framework=InferenceFramework.TRANSFORMERS,
        transformers_model_type=TransformersModelType.AUTOMODEL_IMAGETEXTTOTEXT,
        supported_devices=[AcceleratorDevice.CUDA, AcceleratorDevice.CPU],
        scale=2.0,
        temperature=0.0,
    )

    # Note that this is not the expected way of using the Dolphin model, but it shows the usage of a raw prompt.
    dolphin_oneshot = InlineVlmOptions(
        repo_id="ByteDance/Dolphin",
        prompt="<s>Read text in the image. <Answer/>",
        response_format=ResponseFormat.MARKDOWN,
        inference_framework=InferenceFramework.TRANSFORMERS,
        transformers_model_type=TransformersModelType.AUTOMODEL_IMAGETEXTTOTEXT,
        transformers_prompt_style=TransformersPromptStyle.RAW,
        supported_devices=[AcceleratorDevice.CUDA, AcceleratorDevice.CPU],
        scale=2.0,
        temperature=0.0,
    )

    ## Use VlmPipeline
    pipeline_options = VlmPipelineOptions()
    pipeline_options.generate_page_images = True

    ## On GPU systems, enable flash_attention_2 with CUDA:
    # pipeline_options.accelerator_options.device = AcceleratorDevice.CUDA
    # pipeline_options.accelerator_options.cuda_use_flash_attention2 = True

    vlm_models = [
        ## DocTags / SmolDocling models
        vlm_model_specs.SMOLDOCLING_MLX,
        vlm_model_specs.SMOLDOCLING_TRANSFORMERS,
        ## Markdown models (using MLX framework)
        vlm_model_specs.QWEN25_VL_3B_MLX,
        vlm_model_specs.PIXTRAL_12B_MLX,
        vlm_model_specs.GEMMA3_12B_MLX,
        ## Markdown models (using Transformers framework)
        vlm_model_specs.GRANITE_VISION_TRANSFORMERS,
        vlm_model_specs.PHI4_TRANSFORMERS,
        vlm_model_specs.PIXTRAL_12B_TRANSFORMERS,
        ## More inline models
        dolphin_oneshot,
        llava_qwen,
    ]

    # Remove MLX models if not on Mac
    if sys.platform != "darwin":
        vlm_models = [
            m for m in vlm_models if m.inference_framework != InferenceFramework.MLX
        ]

    rows = []
    for vlm_options in vlm_models:
        pipeline_options.vlm_options = vlm_options

        ## Set up pipeline for PDF or image inputs
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=VlmPipeline,
                    pipeline_options=pipeline_options,
                ),
                InputFormat.IMAGE: PdfFormatOption(
                    pipeline_cls=VlmPipeline,
                    pipeline_options=pipeline_options,
                ),
            },
        )

        row = convert(sources=sources, converter=converter)
        rows.append(row)

        print(
            tabulate(
                rows, headers=["source", "model_id", "framework", "num_pages", "time"]
            )
        )

        print("see if memory gets released ...")
        time.sleep(10)
