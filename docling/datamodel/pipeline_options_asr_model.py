from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import AnyUrl, BaseModel
from typing_extensions import deprecated

from docling.datamodel.accelerator_options import AcceleratorDevice
from docling.datamodel.pipeline_options_vlm_model import (
    # InferenceFramework,
    TransformersModelType,
)


class BaseAsrOptions(BaseModel):
    kind: str
    # prompt: str


class InferenceAsrFramework(str, Enum):
    MLX = "mlx"
    # TRANSFORMERS = "transformers" # disabled for now
    WHISPER = "whisper"


class InlineAsrOptions(BaseAsrOptions):
    kind: Literal["inline_model_options"] = "inline_model_options"

    repo_id: str

    verbose: bool = False
    timestamps: bool = True

    temperature: float = 0.0
    max_new_tokens: int = 256
    max_time_chunk: float = 30.0

    torch_dtype: Optional[str] = None
    supported_devices: List[AcceleratorDevice] = [
        AcceleratorDevice.CPU,
        AcceleratorDevice.CUDA,
        AcceleratorDevice.MPS,
        AcceleratorDevice.XPU,
    ]

    @property
    def repo_cache_folder(self) -> str:
        return self.repo_id.replace("/", "--")


class InlineAsrNativeWhisperOptions(InlineAsrOptions):
    inference_framework: InferenceAsrFramework = InferenceAsrFramework.WHISPER

    language: str = "en"
    supported_devices: List[AcceleratorDevice] = [
        AcceleratorDevice.CPU,
        AcceleratorDevice.CUDA,
    ]
    word_timestamps: bool = True


class InlineAsrMlxWhisperOptions(InlineAsrOptions):
    """
    MLX Whisper options for Apple Silicon optimization.

    Uses mlx-whisper library for efficient inference on Apple Silicon devices.
    """

    inference_framework: InferenceAsrFramework = InferenceAsrFramework.MLX

    language: str = "en"
    task: str = "transcribe"  # "transcribe" or "translate"
    supported_devices: List[AcceleratorDevice] = [
        AcceleratorDevice.MPS,  # MLX is optimized for Apple Silicon
    ]
    word_timestamps: bool = True
    no_speech_threshold: float = 0.6  # Threshold for detecting speech
    logprob_threshold: float = -1.0  # Log probability threshold
    compression_ratio_threshold: float = 2.4  # Compression ratio threshold
