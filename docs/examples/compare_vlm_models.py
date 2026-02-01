# %% [markdown]
# Compare different VLM models by running the VLM pipeline and timing outputs.
#
# What this example does
# - Iterates through a list of VLM presets and converts the same file.
# - Prints per-page generation times and saves JSON/MD/HTML to `scratch/`.
# - Summarizes total inference time and pages processed in a table.
# - Demonstrates the NEW preset-based approach with runtime overrides.
#
# Requirements
# - Install `tabulate` for pretty printing (`pip install tabulate`).
#
# Prerequisites
# - Install Docling with VLM extras. Ensure models can be downloaded or are available.
#
# How to run
# - From the repo root: `python docs/examples/compare_vlm_models.py`.
# - Results are saved to `scratch/` with filenames including the model and runtime.
#
# Notes
# - MLX models are skipped automatically on non-macOS platforms.
# - On CUDA systems, you can enable flash_attention_2 (see commented lines).
# - Running multiple VLMs can be GPU/CPU intensive and time-consuming; ensure
#   enough VRAM/system RAM and close other memory-heavy apps.

# %%

import json
import sys
import time
from pathlib import Path

from docling_core.types.doc import DocItemLabel, ImageRefMode
from docling_core.types.doc.document import DEFAULT_EXPORT_LABELS
from tabulate import tabulate

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    VlmConvertOptions,
    VlmPipelineOptions,
)
from docling.datamodel.vlm_runtime_options import (
    ApiVlmRuntimeOptions,
    MlxVlmRuntimeOptions,
    TransformersVlmRuntimeOptions,
    VlmRuntimeType,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline


def convert(
    sources: list[Path],
    converter: DocumentConverter,
    preset_name: str,
    runtime_type: VlmRuntimeType,
):
    # Note: this helper assumes a single-item `sources` list. It returns after
    # processing the first source to keep runtime/output focused.
    for source in sources:
        print("================================================")
        print("Processing...")
        print(f"Source: {source}")
        print("---")
        print(f"Preset: {preset_name}")
        print(f"Runtime: {runtime_type}")
        print("================================================")
        print("")

        # Measure actual conversion time
        start_time = time.time()
        res = converter.convert(source)
        end_time = time.time()
        wall_clock_time = end_time - start_time

        print("")

        fname = f"{res.input.file.stem}-{preset_name}-{runtime_type.value}"

        # Try to get timing from VLM response, but use wall clock as fallback
        inference_time = 0.0
        for i, page in enumerate(res.pages):
            if page.predictions.vlm_response is not None:
                gen_time = getattr(
                    page.predictions.vlm_response, "generation_time", 0.0
                )
                # Skip negative times (indicates timing not available)
                if gen_time >= 0:
                    inference_time += gen_time
                    print("")
                    print(f" ---------- Predicted page {i} in {gen_time:.2f} [sec]:")
                else:
                    print("")
                    print(f" ---------- Predicted page {i} (timing not available):")
                print(page.predictions.vlm_response.text)
                print(" ---------- ")
            else:
                print(f" ---------- Page {i}: No VLM response available ---------- ")

        # Use wall clock time if VLM timing not available
        if inference_time == 0.0:
            inference_time = wall_clock_time

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
            preset_name,
            str(runtime_type.value),
            pg_num,
            inference_time,
        ]


if __name__ == "__main__":
    sources = [
        "tests/data/pdf/2305.03393v1-pg9.pdf",
    ]

    out_path = Path("scratch")
    out_path.mkdir(parents=True, exist_ok=True)

    ## Use VlmPipeline with presets
    pipeline_options = VlmPipelineOptions()
    pipeline_options.generate_page_images = True

    ## On GPU systems, enable flash_attention_2 with CUDA:
    # pipeline_options.accelerator_options.device = AcceleratorDevice.CUDA
    # pipeline_options.accelerator_options.cuda_use_flash_attention2 = True

    # Define preset configurations to test
    # Each tuple is (preset_name, runtime_options)
    preset_configs = [
        # SmolDocling
        ("smoldocling", MlxVlmRuntimeOptions()),
        # GraniteDocling with different runtimes
        ("granite_docling", MlxVlmRuntimeOptions()),
        ("granite_docling", TransformersVlmRuntimeOptions()),
        # Granite models
        ("granite_vision", TransformersVlmRuntimeOptions()),
        # Other presets with MLX (macOS only)
        ("pixtral", MlxVlmRuntimeOptions()),
        ("qwen", MlxVlmRuntimeOptions()),
        ("gemma_12b", MlxVlmRuntimeOptions()),
        # Other presets with Ollama
        ("deepseek_ocr", ApiVlmRuntimeOptions(runtime_type=VlmRuntimeType.API_OLLAMA)),
        # Other presets with LM Studio
        (
            "deepseek_ocr",
            ApiVlmRuntimeOptions(runtime_type=VlmRuntimeType.API_LMSTUDIO),
        ),
    ]

    # Remove MLX configs if not on Mac
    if sys.platform != "darwin":
        preset_configs = [
            (preset, runtime)
            for preset, runtime in preset_configs
            if runtime.runtime_type != VlmRuntimeType.MLX
        ]

    rows = []
    for preset_name, runtime_options in preset_configs:
        # Create VLM options from preset with runtime override
        vlm_options = VlmConvertOptions.from_preset(
            preset_name,
            runtime_options=runtime_options,
        )

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

        row = convert(
            sources=sources,
            converter=converter,
            preset_name=preset_name,
            runtime_type=runtime_options.runtime_type,
        )
        rows.append(row)

        print(
            tabulate(rows, headers=["source", "preset", "runtime", "num_pages", "time"])
        )

        print("see if memory gets released ...")
        time.sleep(10)
