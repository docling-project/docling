from enum import Enum
from typing import Any, Dict, List, Literal

from pydantic import AnyUrl, BaseModel
from typing_extensions import deprecated

from docling.datamodel.accelerator_options import AcceleratorDevice


class BaseVlmOptions(BaseModel):
    kind: str
    prompt: str


class ResponseFormat(str, Enum):
    DOCTAGS = "doctags"
    MARKDOWN = "markdown"
    HTML = "html"


class InferenceFramework(str, Enum):
    MLX = "mlx"
    TRANSFORMERS = "transformers"


class TransformersModelType(str, Enum):
    AUTOMODEL = "automodel"
    AUTOMODEL_VISION2SEQ = "automodel-vision2seq"
    AUTOMODEL_CAUSALLM = "automodel-causallm"


class InlineVlmOptions(BaseVlmOptions):
    kind: Literal["inline_model_options"] = "inline_model_options"

    repo_id: str
    trust_remote_code: bool = False
    load_in_8bit: bool = True
    llm_int8_threshold: float = 6.0
    quantized: bool = False

    inference_framework: InferenceFramework
    transformers_model_type: TransformersModelType = TransformersModelType.AUTOMODEL
    response_format: ResponseFormat

    supported_devices: List[AcceleratorDevice] = [
        AcceleratorDevice.CPU,
        AcceleratorDevice.CUDA,
        AcceleratorDevice.MPS,
    ]

    scale: float = 2.0

    temperature: float = 0.0
    stop_strings: List[str] = []
    extra_generation_config: Dict[str, Any] = {}

    use_kv_cache: bool = True
    max_new_tokens: int = 4096

    @property
    def repo_cache_folder(self) -> str:
        return self.repo_id.replace("/", "--")


@deprecated("Use InlineVlmOptions instead.")
class HuggingFaceVlmOptions(InlineVlmOptions):
    pass


class ApiVlmOptions(BaseVlmOptions):
    kind: Literal["api_model_options"] = "api_model_options"

    url: AnyUrl = AnyUrl(
        "http://localhost:11434/v1/chat/completions"
    )  # Default to ollama
    headers: Dict[str, str] = {}
    params: Dict[str, Any] = {}
    scale: float = 2.0
    timeout: float = 60
    concurrency: int = 1
    response_format: ResponseFormat
