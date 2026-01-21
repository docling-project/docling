from enum import Enum
from typing import Annotated, Any, Literal, Optional, Union

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
                "Type identifier for the ASR options. Used for discriminating "
                "between different ASR configurations."
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
                "HuggingFace model repository ID for the ASR model. Must be a "
                "Whisper-compatible model for automatic speech recognition."
            ),
            examples=["openai/whisper-tiny", "openai/whisper-base"],
        ),
    ]
    verbose: Annotated[
        bool,
        Field(
            description=(
                "Enable verbose logging output from the ASR model for debugging "
                "purposes."
            )
        ),
    ] = False
    timestamps: Annotated[
        bool,
        Field(
            description=(
                "Generate timestamps for transcribed segments. When enabled, "
                "each transcribed segment includes start and end times for "
                "temporal alignment with the audio."
            )
        ),
    ] = True
    temperature: Annotated[
        float,
        Field(
            description=(
                "Sampling temperature for text generation. 0.0 uses greedy "
                "decoding (deterministic), higher values (e.g., 0.7-1.0) "
                "increase randomness. Recommended: 0.0 for consistent "
                "transcriptions."
            )
        ),
    ] = 0.0
    max_new_tokens: Annotated[
        int,
        Field(
            description=(
                "Maximum number of tokens to generate per transcription segment. "
                "Limits output length to prevent runaway generation. Adjust "
                "based on expected transcript length."
            )
        ),
    ] = 256
    max_time_chunk: Annotated[
        float,
        Field(
            description=(
                "Maximum duration in seconds for each audio chunk processed by "
                "the model. Audio longer than this is split into chunks. "
                "Whisper models are typically trained on 30-second segments."
            )
        ),
    ] = 30.0
    torch_dtype: Annotated[
        Optional[str],
        Field(
            description=(
                "PyTorch data type for model weights. Options: `float32`, "
                "`float16`, `bfloat16`. Lower precision (float16/bfloat16) "
                "reduces memory usage and increases speed. If None, uses model "
                "default."
            )
        ),
    ] = None
    supported_devices: Annotated[
        list[AcceleratorDevice],
        Field(
            description=(
                "List of hardware accelerators supported by this ASR model "
                "configuration."
            )
        ),
    ] = [
        AcceleratorDevice.CPU,
        AcceleratorDevice.CUDA,
        AcceleratorDevice.MPS,
        AcceleratorDevice.XPU,
    ]

    @property
    def repo_cache_folder(self) -> str:
        return self.repo_id.replace("/", "--")


class InlineAsrNativeWhisperOptions(InlineAsrOptions):
    """Configuration for native Whisper ASR implementation."""

    inference_framework: Annotated[
        InferenceAsrFramework,
        Field(
            description=(
                "Inference framework for ASR. Uses native Whisper "
                "implementation for optimal performance."
            )
        ),
    ] = InferenceAsrFramework.WHISPER
    language: Annotated[
        str,
        Field(
            description=(
                "Language code for transcription. Specifying the correct "
                "language improves accuracy. Use ISO 639-1 codes (e.g., `en`, "
                "`es`, `fr`)."
            ),
            examples=["en", "es", "fr", "de"],
        ),
    ] = "en"
    supported_devices: Annotated[
        list[AcceleratorDevice],
        Field(
            description=(
                "Hardware accelerators supported by native Whisper. Supports "
                "CPU and CUDA only."
            )
        ),
    ] = [
        AcceleratorDevice.CPU,
        AcceleratorDevice.CUDA,
    ]
    word_timestamps: Annotated[
        bool,
        Field(
            description=(
                "Generate word-level timestamps in addition to segment "
                "timestamps. Provides fine-grained temporal alignment for each "
                "word in the transcription."
            )
        ),
    ] = True


class InlineAsrMlxWhisperOptions(InlineAsrOptions):
    """MLX Whisper options for Apple Silicon optimization.

    Uses mlx-whisper library for efficient inference on Apple Silicon devices.
    """

    inference_framework: Annotated[
        InferenceAsrFramework,
        Field(
            description=(
                "Inference framework for ASR. Uses MLX for optimized "
                "performance on Apple Silicon (M1/M2/M3)."
            )
        ),
    ] = InferenceAsrFramework.MLX
    language: Annotated[
        str,
        Field(
            description=(
                "Language code for transcription. Specifying the correct "
                "language improves accuracy. Use ISO 639-1 codes (e.g., `en`, "
                "`es`, `fr`)."
            ),
            examples=["en", "es", "fr", "de"],
        ),
    ] = "en"
    task: Annotated[
        str,
        Field(
            description=(
                "ASR task type. `transcribe` converts speech to text in the "
                "same language. `translate` converts speech to English text "
                "regardless of input language."
            ),
            examples=["transcribe", "translate"],
        ),
    ] = "transcribe"
    supported_devices: Annotated[
        list[AcceleratorDevice],
        Field(
            description=(
                "Hardware accelerators supported by MLX Whisper. Optimized for "
                "Apple Silicon (MPS) only."
            )
        ),
    ] = [AcceleratorDevice.MPS]
    word_timestamps: Annotated[
        bool,
        Field(
            description=(
                "Generate word-level timestamps in addition to segment "
                "timestamps. Provides fine-grained temporal alignment for each "
                "word in the transcription."
            )
        ),
    ] = True
    no_speech_threshold: Annotated[
        float,
        Field(
            description=(
                "Threshold for detecting speech vs. silence. Segments with "
                "no-speech probability above this threshold are considered "
                "silent. Range: 0.0-1.0. Higher values are more aggressive in "
                "filtering silence."
            )
        ),
    ] = 0.6
    logprob_threshold: Annotated[
        float,
        Field(
            description=(
                "Log probability threshold for filtering low-confidence "
                "transcriptions. Segments with average log probability below "
                "this threshold are filtered out. More negative values are more "
                "permissive."
            )
        ),
    ] = -1.0
    compression_ratio_threshold: Annotated[
        float,
        Field(
            description=(
                "Compression ratio threshold for detecting repetitive or "
                "low-quality transcriptions. Segments with compression ratio "
                "above this threshold are filtered. Higher values are more "
                "permissive."
            )
        ),
    ] = 2.4
