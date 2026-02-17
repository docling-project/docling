import logging
from enum import Enum

from pydantic import (
    AnyUrl,
)

from docling.datamodel.accelerator_options import AcceleratorDevice
from docling.datamodel.pipeline_options_vlm_model import (
    ApiVlmOptions,
    InferenceFramework,
    InlineVlmOptions,
    ResponseFormat,
    TransformersModelType,
    TransformersPromptStyle,
)

_log = logging.getLogger(__name__)


# Granite-Docling
GRANITEDOCLING_TRANSFORMERS = InlineVlmOptions(
    repo_id="ibm-granite/granite-docling-258M",
    prompt="Convert this page to docling.",
    response_format=ResponseFormat.DOCTAGS,
    inference_framework=InferenceFramework.TRANSFORMERS,
    transformers_model_type=TransformersModelType.AUTOMODEL_IMAGETEXTTOTEXT,
    supported_devices=[
        AcceleratorDevice.CPU,
        AcceleratorDevice.CUDA,
        AcceleratorDevice.XPU,
    ],
    extra_generation_config=dict(skip_special_tokens=False),
    scale=2.0,
    temperature=0.0,
    max_new_tokens=8192,
    stop_strings=["</doctag>", "<|end_of_text|>"],
)

GRANITEDOCLING_VLLM = GRANITEDOCLING_TRANSFORMERS.model_copy(deep=True)
GRANITEDOCLING_VLLM.inference_framework = InferenceFramework.VLLM

GRANITEDOCLING_MLX = InlineVlmOptions(
    repo_id="ibm-granite/granite-docling-258M-mlx",
    prompt="Convert this page to docling.",
    response_format=ResponseFormat.DOCTAGS,
    inference_framework=InferenceFramework.MLX,
    supported_devices=[AcceleratorDevice.MPS],
    scale=2.0,
    temperature=0.0,
    max_new_tokens=8192,
    stop_strings=["</doctag>", "<|end_of_text|>"],
)

GRANITEDOCLING_VLLM_API = ApiVlmOptions(
    url="http://localhost:8000/v1/chat/completions",  # LM studio defaults to port 1234, VLLM to 8000
    params=dict(
        model=GRANITEDOCLING_TRANSFORMERS.repo_id,
        max_tokens=4096,
        skip_special_tokens=True,
    ),
    prompt=GRANITEDOCLING_TRANSFORMERS.prompt,
    timeout=90,
    scale=2.0,
    temperature=0.0,
    concurrency=4,
    stop_strings=["</doctag>", "<|end_of_text|>"],
    response_format=ResponseFormat.DOCTAGS,
)

GRANITEDOCLING_OLLAMA = GRANITEDOCLING_VLLM_API.model_copy(deep=True)
GRANITEDOCLING_OLLAMA.url = AnyUrl("http://localhost:11434/v1/chat/completions")
GRANITEDOCLING_OLLAMA.params["model"] = "ibm/granite-docling:258m"

# SmolDocling
SMOLDOCLING_MLX = InlineVlmOptions(
    repo_id="docling-project/SmolDocling-256M-preview-mlx-bf16",
    prompt="Convert this page to docling.",
    response_format=ResponseFormat.DOCTAGS,
    inference_framework=InferenceFramework.MLX,
    supported_devices=[AcceleratorDevice.MPS],
    scale=2.0,
    temperature=0.0,
    stop_strings=["</doctag>", "<end_of_utterance>"],
)

SMOLDOCLING_TRANSFORMERS = InlineVlmOptions(
    repo_id="docling-project/SmolDocling-256M-preview",
    prompt="Convert this page to docling.",
    response_format=ResponseFormat.DOCTAGS,
    inference_framework=InferenceFramework.TRANSFORMERS,
    transformers_model_type=TransformersModelType.AUTOMODEL_IMAGETEXTTOTEXT,
    supported_devices=[
        AcceleratorDevice.CPU,
        AcceleratorDevice.CUDA,
        AcceleratorDevice.XPU,
    ],
    torch_dtype="bfloat16",
    scale=2.0,
    temperature=0.0,
    stop_strings=["</doctag>", "<end_of_utterance>"],
)

SMOLDOCLING_VLLM = InlineVlmOptions(
    repo_id="docling-project/SmolDocling-256M-preview",
    prompt="Convert this page to docling.",
    response_format=ResponseFormat.DOCTAGS,
    inference_framework=InferenceFramework.VLLM,
    supported_devices=[
        AcceleratorDevice.CUDA,
        AcceleratorDevice.XPU,
    ],
    scale=2.0,
    temperature=0.0,
    stop_strings=["</doctag>", "<end_of_utterance>"],
)

# SmolVLM-256M-Instruct
SMOLVLM256_TRANSFORMERS = InlineVlmOptions(
    repo_id="HuggingFaceTB/SmolVLM-256M-Instruct",
    prompt="Transcribe this image to plain text.",
    response_format=ResponseFormat.PLAINTEXT,
    inference_framework=InferenceFramework.TRANSFORMERS,
    transformers_model_type=TransformersModelType.AUTOMODEL_IMAGETEXTTOTEXT,
    supported_devices=[
        AcceleratorDevice.CPU,
        AcceleratorDevice.CUDA,
        # AcceleratorDevice.MPS,
        AcceleratorDevice.XPU,
    ],
    torch_dtype="bfloat16",
    scale=2.0,
    temperature=0.0,
)

# SmolVLM2-2.2b-Instruct
SMOLVLM256_MLX = InlineVlmOptions(
    repo_id="moot20/SmolVLM-256M-Instruct-MLX",
    prompt="Extract the text.",
    response_format=ResponseFormat.DOCTAGS,
    inference_framework=InferenceFramework.MLX,
    transformers_model_type=TransformersModelType.AUTOMODEL_IMAGETEXTTOTEXT,
    supported_devices=[
        AcceleratorDevice.MPS,
    ],
    scale=2.0,
    temperature=0.0,
)

SMOLVLM256_VLLM = InlineVlmOptions(
    repo_id="HuggingFaceTB/SmolVLM-256M-Instruct",
    prompt="Transcribe this image to plain text.",
    response_format=ResponseFormat.PLAINTEXT,
    inference_framework=InferenceFramework.VLLM,
    supported_devices=[
        AcceleratorDevice.CUDA,
        AcceleratorDevice.XPU,
    ],
    scale=2.0,
    temperature=0.0,
)


# GraniteVision
GRANITE_VISION_TRANSFORMERS = InlineVlmOptions(
    repo_id="ibm-granite/granite-vision-3.2-2b",
    prompt="Convert this page to markdown. Do not miss any text and only output the bare markdown!",
    response_format=ResponseFormat.MARKDOWN,
    inference_framework=InferenceFramework.TRANSFORMERS,
    transformers_model_type=TransformersModelType.AUTOMODEL_VISION2SEQ,
    supported_devices=[
        AcceleratorDevice.CPU,
        AcceleratorDevice.CUDA,
        AcceleratorDevice.MPS,
        AcceleratorDevice.XPU,
    ],
    scale=2.0,
    temperature=0.0,
)

GRANITE_VISION_VLLM = InlineVlmOptions(
    repo_id="ibm-granite/granite-vision-3.2-2b",
    prompt="Convert this page to markdown. Do not miss any text and only output the bare markdown!",
    response_format=ResponseFormat.MARKDOWN,
    inference_framework=InferenceFramework.VLLM,
    supported_devices=[
        AcceleratorDevice.CUDA,
        AcceleratorDevice.XPU,
    ],
    scale=2.0,
    temperature=0.0,
)

GRANITE_VISION_OLLAMA = ApiVlmOptions(
    url=AnyUrl("http://localhost:11434/v1/chat/completions"),
    params={"model": "granite3.2-vision:2b"},
    prompt="Convert this page to markdown. Do not miss any text and only output the bare markdown!",
    scale=1.0,
    timeout=120,
    response_format=ResponseFormat.MARKDOWN,
    temperature=0.0,
)

# Pixtral
PIXTRAL_12B_TRANSFORMERS = InlineVlmOptions(
    repo_id="mistral-community/pixtral-12b",
    prompt="Convert this page to markdown. Do not miss any text and only output the bare markdown!",
    response_format=ResponseFormat.MARKDOWN,
    inference_framework=InferenceFramework.TRANSFORMERS,
    transformers_model_type=TransformersModelType.AUTOMODEL_VISION2SEQ,
    supported_devices=[
        AcceleratorDevice.CPU,
        AcceleratorDevice.CUDA,
        AcceleratorDevice.XPU,
    ],
    scale=2.0,
    temperature=0.0,
)

PIXTRAL_12B_MLX = InlineVlmOptions(
    repo_id="mlx-community/pixtral-12b-bf16",
    prompt="Convert this page to markdown. Do not miss any text and only output the bare markdown!",
    response_format=ResponseFormat.MARKDOWN,
    inference_framework=InferenceFramework.MLX,
    supported_devices=[AcceleratorDevice.MPS],
    scale=2.0,
    temperature=0.0,
)

# Phi4
PHI4_TRANSFORMERS = InlineVlmOptions(
    repo_id="microsoft/Phi-4-multimodal-instruct",
    prompt="Convert this page to MarkDown. Do not miss any text and only output the bare markdown",
    trust_remote_code=True,
    response_format=ResponseFormat.MARKDOWN,
    inference_framework=InferenceFramework.TRANSFORMERS,
    transformers_model_type=TransformersModelType.AUTOMODEL_CAUSALLM,
    supported_devices=[
        AcceleratorDevice.CPU,
        AcceleratorDevice.CUDA,
        AcceleratorDevice.XPU,
    ],
    scale=2.0,
    temperature=0.0,
    extra_generation_config=dict(num_logits_to_keep=0),
)

# Qwen
QWEN25_VL_3B_MLX = InlineVlmOptions(
    repo_id="mlx-community/Qwen2.5-VL-3B-Instruct-bf16",
    prompt="Convert this page to markdown. Do not miss any text and only output the bare markdown!",
    response_format=ResponseFormat.MARKDOWN,
    inference_framework=InferenceFramework.MLX,
    supported_devices=[AcceleratorDevice.MPS],
    scale=2.0,
    temperature=0.0,
)

# GoT 2.0
GOT2_TRANSFORMERS = InlineVlmOptions(
    repo_id="stepfun-ai/GOT-OCR-2.0-hf",
    prompt="",
    response_format=ResponseFormat.MARKDOWN,
    inference_framework=InferenceFramework.TRANSFORMERS,
    transformers_prompt_style=TransformersPromptStyle.NONE,
    transformers_model_type=TransformersModelType.AUTOMODEL_IMAGETEXTTOTEXT,
    supported_devices=[
        AcceleratorDevice.CPU,
        AcceleratorDevice.CUDA,
        #    AcceleratorDevice.MPS,
        AcceleratorDevice.XPU,
    ],
    scale=2.0,
    temperature=0.0,
    stop_strings=["<|im_end|>"],
    extra_processor_kwargs={"format": True},
)


# Gemma-3
GEMMA3_12B_MLX = InlineVlmOptions(
    repo_id="mlx-community/gemma-3-12b-it-bf16",
    prompt="Convert this page to markdown. Do not miss any text and only output the bare markdown!",
    response_format=ResponseFormat.MARKDOWN,
    inference_framework=InferenceFramework.MLX,
    supported_devices=[AcceleratorDevice.MPS],
    scale=2.0,
    temperature=0.0,
)

GEMMA3_27B_MLX = InlineVlmOptions(
    repo_id="mlx-community/gemma-3-27b-it-bf16",
    prompt="Convert this page to markdown. Do not miss any text and only output the bare markdown!",
    response_format=ResponseFormat.MARKDOWN,
    inference_framework=InferenceFramework.MLX,
    supported_devices=[AcceleratorDevice.MPS],
    scale=2.0,
    temperature=0.0,
)

# Dolphin

DOLPHIN_TRANSFORMERS = InlineVlmOptions(
    repo_id="ByteDance/Dolphin",
    prompt="<s>Read text in the image. <Answer/>",
    response_format=ResponseFormat.MARKDOWN,
    inference_framework=InferenceFramework.TRANSFORMERS,
    transformers_model_type=TransformersModelType.AUTOMODEL_IMAGETEXTTOTEXT,
    transformers_prompt_style=TransformersPromptStyle.RAW,
    supported_devices=[
        AcceleratorDevice.CUDA,
        AcceleratorDevice.CPU,
        AcceleratorDevice.MPS,
        AcceleratorDevice.XPU,
    ],
    scale=2.0,
    temperature=0.0,
)

# DeepSeek-OCR
DEEPSEEKOCR_OLLAMA = ApiVlmOptions(
    url="http://localhost:11434/v1/chat/completions",
    params=dict(
        model="deepseek-ocr:3b",
        max_tokens=4096,
        skip_special_tokens=True,
    ),
    prompt="<|grounding|>Convert the document to markdown. ",
    timeout=90,
    scale=2.0,
    temperature=0.0,
    concurrency=4,
    response_format=ResponseFormat.DEEPSEEKOCR_MARKDOWN,
)

# NuExtract
NU_EXTRACT_2B_TRANSFORMERS = InlineVlmOptions(
    repo_id="numind/NuExtract-2.0-2B",
    revision="fe5b2f0b63b81150721435a3ca1129a75c59c74e",  # 489efed leads to MPS issues
    prompt="",  # This won't be used, template is passed separately
    torch_dtype="bfloat16",
    inference_framework=InferenceFramework.TRANSFORMERS,
    transformers_model_type=TransformersModelType.AUTOMODEL_IMAGETEXTTOTEXT,
    response_format=ResponseFormat.PLAINTEXT,
    supported_devices=[
        AcceleratorDevice.CPU,
        AcceleratorDevice.CUDA,
        AcceleratorDevice.MPS,
        AcceleratorDevice.XPU,
    ],
    scale=2.0,
    temperature=0.0,
)


class VlmModelType(str, Enum):
    SMOLDOCLING = "smoldocling"
    SMOLDOCLING_VLLM = "smoldocling_vllm"
    GRANITE_VISION = "granite_vision"
    GRANITE_VISION_VLLM = "granite_vision_vllm"
    GRANITE_VISION_OLLAMA = "granite_vision_ollama"
    GOT_OCR_2 = "got_ocr_2"
    GRANITEDOCLING = "granite_docling"
    GRANITEDOCLING_VLLM = "granite_docling_vllm"
    DEEPSEEKOCR_OLLAMA = "deepseekocr_ollama"
