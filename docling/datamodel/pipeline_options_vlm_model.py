from enum import Enum
from typing import Any, Dict, Literal

from pydantic import AnyUrl, BaseModel


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
    TRANSFORMERS_VISION2SEQ = "transformers-vision2seq"
    TRANSFORMERS_CAUSALLM = "transformers-causallm"


class HuggingFaceVlmOptions(BaseVlmOptions):
    kind: Literal["hf_model_options"] = "hf_model_options"

    repo_id: str
    load_in_8bit: bool = True
    llm_int8_threshold: float = 6.0
    quantized: bool = False

    inference_framework: InferenceFramework
    response_format: ResponseFormat

    scale: float = 2.0

    temperature: float = 0.0
    stop_strings: list[str] = []

    use_kv_cache: bool = True
    max_new_tokens: int = 4096

    @property
    def repo_cache_folder(self) -> str:
        return self.repo_id.replace("/", "--")


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
