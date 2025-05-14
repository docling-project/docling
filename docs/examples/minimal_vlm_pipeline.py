import json
import time
from pathlib import Path

from docling_core.types.doc import DocItemLabel, ImageRefMode
from docling_core.types.doc.document import DEFAULT_EXPORT_LABELS

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    HuggingFaceVlmOptions,
    InferenceFramework,
    ResponseFormat,
    VlmPipelineOptions,
    smoldocling_vlm_mlx_conversion_options,
    smoldocling_vlm_conversion_options,
    granite_vision_vlm_conversion_options,
    granite_vision_vlm_ollama_conversion_options,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline

sources = [
    # "tests/data/2305.03393v1-pg9-img.png",
    "tests/data/pdf/2305.03393v1-pg9.pdf",
]

## Use experimental VlmPipeline
pipeline_options = VlmPipelineOptions()
# If force_backend_text = True, text from backend will be used instead of generated text
pipeline_options.force_backend_text = False

## On GPU systems, enable flash_attention_2 with CUDA:
# pipeline_options.accelerator_options.device = AcceleratorDevice.CUDA
# pipeline_options.accelerator_options.cuda_use_flash_attention2 = True

## Pick a VLM model. We choose SmolDocling-256M by default
# pipeline_options.vlm_options = smoldocling_vlm_conversion_options

## Pick a VLM model. Fast Apple Silicon friendly implementation for SmolDocling-256M via MLX
pipeline_options.vlm_options = smoldocling_vlm_mlx_conversion_options

## Alternative VLM models:
# pipeline_options.vlm_options = granite_vision_vlm_conversion_options

"""
pixtral_vlm_conversion_options = HuggingFaceVlmOptions(
     repo_id="mistralai/Pixtral-12B-Base-2409",
     prompt="OCR this image and export it in MarkDown.",
     response_format=ResponseFormat.MARKDOWN,
     inference_framework=InferenceFramework.TRANSFORMERS_LlavaForConditionalGeneration,
)
pipeline_options.vlm_options = pixtral_vlm_conversion_options
"""

"""
pixtral_vlm_conversion_options = HuggingFaceVlmOptions(
    repo_id="mistral-community/pixtral-12b",
    prompt="OCR this image and export it in MarkDown.",
    response_format=ResponseFormat.MARKDOWN,
    inference_framework=InferenceFramework.TRANSFORMERS_LlavaForConditionalGeneration,
)
pipeline_options.vlm_options = pixtral_vlm_conversion_options
"""

"""
phi_vlm_conversion_options = HuggingFaceVlmOptions(
    repo_id="microsoft/Phi-4-multimodal-instruct",
    # prompt="OCR the full page to markdown.",
    prompt="OCR this image and export it in MarkDown.",
    response_format=ResponseFormat.MARKDOWN,
    inference_framework=InferenceFramework.TRANSFORMERS_AutoModelForCausalLM,
)
pipeline_options.vlm_options = phi_vlm_conversion_options
"""

"""
pixtral_vlm_conversion_options = HuggingFaceVlmOptions(
    repo_id="mlx-community/pixtral-12b-bf16",
    prompt="Convert this page to markdown. Do not miss any text and only output the bare MarkDown!",
    response_format=ResponseFormat.MARKDOWN,
    inference_framework=InferenceFramework.MLX,
    scale=1.0,
)
pipeline_options.vlm_options = pixtral_vlm_conversion_options
"""

"""
qwen_vlm_conversion_options = HuggingFaceVlmOptions(
    repo_id="mlx-community/Qwen2.5-VL-3B-Instruct-bf16",
    prompt="Convert this full page to markdown. Do not miss any text and only output the bare MarkDown!",
    response_format=ResponseFormat.MARKDOWN,
    inference_framework=InferenceFramework.MLX,
)
pipeline_options.vlm_options = qwen_vlm_conversion_options
"""

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
    }
)

out_path = Path("scratch")
out_path.mkdir(parents=True, exist_ok=True)

for source in sources:
    start_time = time.time()
    print("================================================")
    print(f"Processing... {source}")
    print("================================================")
    print("")

    res = converter.convert(source)

    print("")
    #print(res.document.export_to_markdown())

    for i,page in enumerate(res.pages):
        print("")
        print(f" ---------- Predicted page {i} in {pipeline_options.vlm_options.response_format}:")
        print(page.predictions.vlm_response.text)
        print(f" ---------- ")

    print("===== Final output of the converted document =======")
    
    with (out_path / f"{res.input.file.stem}.json").open("w") as fp:
        fp.write(json.dumps(res.document.export_to_dict()))

    res.document.save_as_json(
        out_path / f"{res.input.file.stem}.json",
        image_mode=ImageRefMode.PLACEHOLDER,
    )
    print(f" => produced {out_path / res.input.file.stem}.json")
    
    res.document.save_as_markdown(
        out_path / f"{res.input.file.stem}.md",
        image_mode=ImageRefMode.PLACEHOLDER,
    )
    print(f" => produced {out_path / res.input.file.stem}.md")
    
    res.document.save_as_html(
        out_path / f"{res.input.file.stem}.html",
        image_mode=ImageRefMode.EMBEDDED,
        labels=[*DEFAULT_EXPORT_LABELS, DocItemLabel.FOOTNOTE],
        # split_page_view=True,
    )
    print(f" => produced {out_path / res.input.file.stem}.html")
    
    pg_num = res.document.num_pages()
    print("")
    inference_time = time.time() - start_time
    print(
        f"Total document prediction time: {inference_time:.2f} seconds, pages: {pg_num}"
    )
    print("====================================================")
    
