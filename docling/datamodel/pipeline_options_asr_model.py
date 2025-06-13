from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import AnyUrl, BaseModel
from typing_extensions import deprecated

from docling.datamodel.accelerator_options import AcceleratorDevice
from docling.datamodel.pipeline_options_vlm_model import (
    InferenceFramework,
    TransformersModelType,
)


class BaseAsrOptions(BaseModel):
    kind: str
    # prompt: str


class AsrResponseFormat(str, Enum):
    WHISPER = "whisper"


class InlineAsrOptions(BaseAsrOptions):
    kind: Literal["inline_model_options"] = "inline_model_options"

    repo_id: str
    trust_remote_code: bool = False
    load_in_8bit: bool = True
    llm_int8_threshold: float = 6.0
    quantized: bool = False

    inference_framework: InferenceFramework
    transformers_model_type: TransformersModelType = TransformersModelType.AUTOMODEL
    response_format: AsrResponseFormat

    torch_dtype: Optional[str] = None
    supported_devices: List[AcceleratorDevice] = [
        AcceleratorDevice.CPU,
        AcceleratorDevice.CUDA,
        AcceleratorDevice.MPS,
    ]

    temperature: float = 0.0
    stop_strings: List[str] = []
    extra_generation_config: Dict[str, Any] = {}

    use_kv_cache: bool = True
    max_new_tokens: int = 4096

    @property
    def repo_cache_folder(self) -> str:
        return self.repo_id.replace("/", "--")
