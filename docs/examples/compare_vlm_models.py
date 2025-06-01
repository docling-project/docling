# Compare VLM models
# ==================
#
# This example runs the VLM pipeline with different vision-language models.
# Their runtime as well output quality is compared.

import json
import time
from pathlib import Path

from docling_core.types.doc import DocItemLabel, ImageRefMode
from docling_core.types.doc.document import DEFAULT_EXPORT_LABELS
from tabulate import tabulate

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_model_specializations import (
    gemma_3_12b_mlx_conversion_options,
    granite_vision_vlm_conversion_options,
    granite_vision_vlm_ollama_conversion_options,
    phi_vlm_conversion_options,
    pixtral_12b_vlm_conversion_options,
    pixtral_12b_vlm_mlx_conversion_options,
    qwen25_vl_3b_vlm_mlx_conversion_options,
    smoldocling_vlm_conversion_options,
    smoldocling_vlm_mlx_conversion_options,
)
from docling.datamodel.pipeline_options import (
    VlmPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline


def convert(sources: list[Path], converter: DocumentConverter):
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

    ## Use VlmPipeline
    pipeline_options = VlmPipelineOptions()
    pipeline_options.generate_page_images = True

    ## On GPU systems, enable flash_attention_2 with CUDA:
    # pipeline_options.accelerator_options.device = AcceleratorDevice.CUDA
    # pipeline_options.accelerator_options.cuda_use_flash_attention2 = True

    rows = []
    for vlm_options in [
        ## DocTags / SmolDocling models
        smoldocling_vlm_conversion_options,
        # smoldocling_vlm_mlx_conversion_options,
        ## Markdown models (using MLX framework)
        # qwen25_vl_3b_vlm_mlx_conversion_options,
        # pixtral_12b_vlm_mlx_conversion_options,
        # gemma_3_12b_mlx_conversion_options,
        ## Markdown models (using Transformers framework)
        # granite_vision_vlm_conversion_options,
        phi_vlm_conversion_options,
        pixtral_12b_vlm_conversion_options,
    ]:
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
