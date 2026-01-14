from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import AnyUrl, BaseModel, Field
from typing_extensions import deprecated

from docling.datamodel.accelerator_options import AcceleratorDevice
from docling.datamodel.pipeline_options_vlm_model import (
    # InferenceFramework,
    TransformersModelType,
)


class BaseAsrOptions(BaseModel):
    """Base configuration for automatic speech recognition models."""

    kind: Annotated[
        str,
        Field(
            description=(
                "Type identifier for the ASR options. Used for discriminating between different ASR "
                "configurations."
            ),
        ),
    ]


class InferenceAsrFramework(str, Enum):
    MLX = "mlx"
    # TRANSFORMERS = "transformers" # disabled for now
    WHISPER = "whisper"


class InlineAsrOptions(BaseAsrOptions):
    """Configuration for inline ASR models running locally."""

    kind: Literal["inline_model_options"] = "inline_model_options"
    repo_id: Annotated[
        str,
        Field(
            description=(
                "HuggingFace model repository ID for the ASR model. Must be a Whisper-compatible model for "
                "automatic speech recognition."
            ),
            examples=["openai/whisper-tiny", "openai/whisper-base"],
        ),
    ]
    verbose: bool = Field(
        default=False,
        description="Enable verbose logging output from the ASR model for debugging purposes.",
    )
    timestamps: bool = Field(
        default=True,
        description=(
            "Generate timestamps for transcribed segments. When enabled, each transcribed segment includes start "
            "and end times for temporal alignment with the audio."
        ),
    )
    temperature: float = Field(
        default=0.0,
        description=(
            "Sampling temperature for text generation. 0.0 uses greedy decoding (deterministic), higher values "
            "(e.g., 0.7-1.0) increase randomness. Recommended: 0.0 for consistent transcriptions."
        ),
    )
    max_new_tokens: int = Field(
        default=256,
        description=(
            "Maximum number of tokens to generate per transcription segment. Limits output length to prevent "
            "runaway generation. Adjust based on expected transcript length."
        ),
    )
    max_time_chunk: float = Field(
        default=30.0,
        description=(
            "Maximum duration in seconds for each audio chunk processed by the model. Audio longer than this is "
            "split into chunks. Whisper models are typically trained on 30-second segments."
        ),
    )
    torch_dtype: Optional[str] = Field(
        default=None,
        description=(
            "PyTorch data type for model weights. Options: `float32`, `float16`, `bfloat16`. Lower precision "
            "(float16/bfloat16) reduces memory usage and increases speed. If None, uses model default."
        ),
    )
    supported_devices: List[AcceleratorDevice] = Field(
        default=[
            AcceleratorDevice.CPU,
            AcceleratorDevice.CUDA,
            AcceleratorDevice.MPS,
            AcceleratorDevice.XPU,
        ],
        description="List of hardware accelerators supported by this ASR model configuration.",
    )

    @property
    def repo_cache_folder(self) -> str:
        return self.repo_id.replace("/", "--")


class InlineAsrNativeWhisperOptions(InlineAsrOptions):
    """Configuration for native Whisper ASR implementation."""

    inference_framework: InferenceAsrFramework = Field(
        default=InferenceAsrFramework.WHISPER,
        description="Inference framework for ASR. Uses native Whisper implementation for optimal performance.",
    )
    language: str = Field(
        default="en",
        description=(
            "Language code for transcription. Specifying the correct language improves accuracy. "
            "Use ISO 639-1 codes (e.g., `en`, `es`, `fr`)."
        ),
        examples=["en", "es", "fr", "de"],
    )
    supported_devices: List[AcceleratorDevice] = Field(
        default=[
            AcceleratorDevice.CPU,
            AcceleratorDevice.CUDA,
        ],
        description="Hardware accelerators supported by native Whisper. Supports CPU and CUDA only.",
    )
    word_timestamps: bool = Field(
        default=True,
        description=(
            "Generate word-level timestamps in addition to segment timestamps. Provides fine-grained temporal "
            "alignment for each word in the transcription."
        ),
    )


class InlineAsrMlxWhisperOptions(InlineAsrOptions):
    """MLX Whisper options for Apple Silicon optimization.

    Uses mlx-whisper library for efficient inference on Apple Silicon devices.
    """

    inference_framework: InferenceAsrFramework = Field(
        default=InferenceAsrFramework.MLX,
        description="Inference framework for ASR. Uses MLX for optimized performance on Apple Silicon (M1/M2/M3).",
    )
    language: str = Field(
        default="en",
        description=(
            "Language code for transcription. Specifying the correct language improves accuracy. "
            "Use ISO 639-1 codes (e.g., `en`, `es`, `fr`)."
        ),
        examples=["en", "es", "fr", "de"],
    )
    task: str = Field(
        default="transcribe",
        description=(
            "ASR task type. `transcribe` converts speech to text in the same language. `translate` converts speech "
            "to English text regardless of input language."
        ),
        examples=["transcribe", "translate"],
    )
    supported_devices: List[AcceleratorDevice] = Field(
        default=[AcceleratorDevice.MPS],
        description="Hardware accelerators supported by MLX Whisper. Optimized for Apple Silicon (MPS) only.",
    )
    word_timestamps: bool = Field(
        default=True,
        description=(
            "Generate word-level timestamps in addition to segment timestamps. Provides fine-grained temporal "
            "alignment for each word in the transcription."
        ),
    )
    no_speech_threshold: float = Field(
        default=0.6,
        description=(
            "Threshold for detecting speech vs. silence. Segments with no-speech probability above this threshold "
            "are considered silent. Range: 0.0-1.0. Higher values are more aggressive in filtering silence."
        ),
    )
    logprob_threshold: float = Field(
        default=-1.0,
        description=(
            "Log probability threshold for filtering low-confidence transcriptions. Segments with average log "
            "probability below this threshold are filtered out. More negative values are more permissive."
        ),
    )
    compression_ratio_threshold: float = Field(
        default=2.4,
        description=(
            "Compression ratio threshold for detecting repetitive or low-quality transcriptions. Segments with "
            "compression ratio above this threshold are filtered. Higher values are more permissive."
        ),
    )
