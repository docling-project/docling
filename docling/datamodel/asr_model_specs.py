import logging
from enum import Enum

from pydantic import (
    AnyUrl,
)

from docling.datamodel.accelerator_options import AcceleratorDevice
from docling.datamodel.pipeline_options_asr_model import (
    # AsrResponseFormat,
    # ApiAsrOptions,
    InferenceAsrFramework,
    InlineAsrMlxWhisperOptions,
    InlineAsrNativeWhisperOptions,
    TransformersModelType,
)

_log = logging.getLogger(__name__)


def _get_whisper_tiny_model():
    """
    Get the best Whisper Tiny model for the current hardware.

    Automatically selects MLX Whisper Tiny for Apple Silicon (MPS) if available,
    otherwise falls back to native Whisper Tiny.
    """
    # Check if MPS is available (Apple Silicon)
    try:
        import torch

        has_mps = torch.backends.mps.is_built() and torch.backends.mps.is_available()
    except ImportError:
        has_mps = False

    # Check if mlx-whisper is available
    try:
        import mlx_whisper  # type: ignore

        has_mlx_whisper = True
    except ImportError:
        has_mlx_whisper = False

    # Use MLX Whisper if both MPS and mlx-whisper are available
    if has_mps and has_mlx_whisper:
        return InlineAsrMlxWhisperOptions(
            repo_id="mlx-community/whisper-tiny-mlx",
            inference_framework=InferenceAsrFramework.MLX,
            language="en",
            task="transcribe",
            word_timestamps=True,
            no_speech_threshold=0.6,
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.4,
        )
    else:
        return InlineAsrNativeWhisperOptions(
            repo_id="tiny",
            inference_framework=InferenceAsrFramework.WHISPER,
            verbose=True,
            timestamps=True,
            word_timestamps=True,
            temperature=0.0,
            max_new_tokens=256,
            max_time_chunk=30.0,
        )


# Create the model instance
WHISPER_TINY = _get_whisper_tiny_model()


def _get_whisper_small_model():
    """
    Get the best Whisper Small model for the current hardware.

    Automatically selects MLX Whisper Small for Apple Silicon (MPS) if available,
    otherwise falls back to native Whisper Small.
    """
    # Check if MPS is available (Apple Silicon)
    try:
        import torch

        has_mps = torch.backends.mps.is_built() and torch.backends.mps.is_available()
    except ImportError:
        has_mps = False

    # Check if mlx-whisper is available
    try:
        import mlx_whisper  # type: ignore

        has_mlx_whisper = True
    except ImportError:
        has_mlx_whisper = False

    # Use MLX Whisper if both MPS and mlx-whisper are available
    if has_mps and has_mlx_whisper:
        return InlineAsrMlxWhisperOptions(
            repo_id="mlx-community/whisper-small-mlx",
            inference_framework=InferenceAsrFramework.MLX,
            language="en",
            task="transcribe",
            word_timestamps=True,
            no_speech_threshold=0.6,
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.4,
        )
    else:
        return InlineAsrNativeWhisperOptions(
            repo_id="small",
            inference_framework=InferenceAsrFramework.WHISPER,
            verbose=True,
            timestamps=True,
            word_timestamps=True,
            temperature=0.0,
            max_new_tokens=256,
            max_time_chunk=30.0,
        )


# Create the model instance
WHISPER_SMALL = _get_whisper_small_model()


def _get_whisper_medium_model():
    """
    Get the best Whisper Medium model for the current hardware.

    Automatically selects MLX Whisper Medium for Apple Silicon (MPS) if available,
    otherwise falls back to native Whisper Medium.
    """
    # Check if MPS is available (Apple Silicon)
    try:
        import torch

        has_mps = torch.backends.mps.is_built() and torch.backends.mps.is_available()
    except ImportError:
        has_mps = False

    # Check if mlx-whisper is available
    try:
        import mlx_whisper  # type: ignore

        has_mlx_whisper = True
    except ImportError:
        has_mlx_whisper = False

    # Use MLX Whisper if both MPS and mlx-whisper are available
    if has_mps and has_mlx_whisper:
        return InlineAsrMlxWhisperOptions(
            repo_id="mlx-community/whisper-medium-mlx-8bit",
            inference_framework=InferenceAsrFramework.MLX,
            language="en",
            task="transcribe",
            word_timestamps=True,
            no_speech_threshold=0.6,
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.4,
        )
    else:
        return InlineAsrNativeWhisperOptions(
            repo_id="medium",
            inference_framework=InferenceAsrFramework.WHISPER,
            verbose=True,
            timestamps=True,
            word_timestamps=True,
            temperature=0.0,
            max_new_tokens=256,
            max_time_chunk=30.0,
        )


# Create the model instance
WHISPER_MEDIUM = _get_whisper_medium_model()


def _get_whisper_base_model():
    """
    Get the best Whisper Base model for the current hardware.

    Automatically selects MLX Whisper Base for Apple Silicon (MPS) if available,
    otherwise falls back to native Whisper Base.
    """
    # Check if MPS is available (Apple Silicon)
    try:
        import torch

        has_mps = torch.backends.mps.is_built() and torch.backends.mps.is_available()
    except ImportError:
        has_mps = False

    # Check if mlx-whisper is available
    try:
        import mlx_whisper  # type: ignore

        has_mlx_whisper = True
    except ImportError:
        has_mlx_whisper = False

    # Use MLX Whisper if both MPS and mlx-whisper are available
    if has_mps and has_mlx_whisper:
        return InlineAsrMlxWhisperOptions(
            repo_id="mlx-community/whisper-base-mlx",
            inference_framework=InferenceAsrFramework.MLX,
            language="en",
            task="transcribe",
            word_timestamps=True,
            no_speech_threshold=0.6,
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.4,
        )
    else:
        return InlineAsrNativeWhisperOptions(
            repo_id="base",
            inference_framework=InferenceAsrFramework.WHISPER,
            verbose=True,
            timestamps=True,
            word_timestamps=True,
            temperature=0.0,
            max_new_tokens=256,
            max_time_chunk=30.0,
        )


# Create the model instance
WHISPER_BASE = _get_whisper_base_model()


def _get_whisper_large_model():
    """
    Get the best Whisper Large model for the current hardware.

    Automatically selects MLX Whisper Large for Apple Silicon (MPS) if available,
    otherwise falls back to native Whisper Large.
    """
    # Check if MPS is available (Apple Silicon)
    try:
        import torch

        has_mps = torch.backends.mps.is_built() and torch.backends.mps.is_available()
    except ImportError:
        has_mps = False

    # Check if mlx-whisper is available
    try:
        import mlx_whisper  # type: ignore

        has_mlx_whisper = True
    except ImportError:
        has_mlx_whisper = False

    # Use MLX Whisper if both MPS and mlx-whisper are available
    if has_mps and has_mlx_whisper:
        return InlineAsrMlxWhisperOptions(
            repo_id="mlx-community/whisper-large-mlx-8bit",
            inference_framework=InferenceAsrFramework.MLX,
            language="en",
            task="transcribe",
            word_timestamps=True,
            no_speech_threshold=0.6,
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.4,
        )
    else:
        return InlineAsrNativeWhisperOptions(
            repo_id="large",
            inference_framework=InferenceAsrFramework.WHISPER,
            verbose=True,
            timestamps=True,
            word_timestamps=True,
            temperature=0.0,
            max_new_tokens=256,
            max_time_chunk=30.0,
        )


# Create the model instance
WHISPER_LARGE = _get_whisper_large_model()


def _get_whisper_turbo_model():
    """
    Get the best Whisper Turbo model for the current hardware.

    Automatically selects MLX Whisper Turbo for Apple Silicon (MPS) if available,
    otherwise falls back to native Whisper Turbo.
    """
    # Check if MPS is available (Apple Silicon)
    try:
        import torch

        has_mps = torch.backends.mps.is_built() and torch.backends.mps.is_available()
    except ImportError:
        has_mps = False

    # Check if mlx-whisper is available
    try:
        import mlx_whisper  # type: ignore

        has_mlx_whisper = True
    except ImportError:
        has_mlx_whisper = False

    # Use MLX Whisper if both MPS and mlx-whisper are available
    if has_mps and has_mlx_whisper:
        return InlineAsrMlxWhisperOptions(
            repo_id="mlx-community/whisper-turbo",
            inference_framework=InferenceAsrFramework.MLX,
            language="en",
            task="transcribe",
            word_timestamps=True,
            no_speech_threshold=0.6,
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.4,
        )
    else:
        return InlineAsrNativeWhisperOptions(
            repo_id="turbo",
            inference_framework=InferenceAsrFramework.WHISPER,
            verbose=True,
            timestamps=True,
            word_timestamps=True,
            temperature=0.0,
            max_new_tokens=256,
            max_time_chunk=30.0,
        )


# Create the model instance
WHISPER_TURBO = _get_whisper_turbo_model()

# Explicit MLX Whisper model options for users who want to force MLX usage
WHISPER_TINY_MLX = InlineAsrMlxWhisperOptions(
    repo_id="mlx-community/whisper-tiny-mlx",
    inference_framework=InferenceAsrFramework.MLX,
    language="en",
    task="transcribe",
    word_timestamps=True,
    no_speech_threshold=0.6,
    logprob_threshold=-1.0,
    compression_ratio_threshold=2.4,
)

WHISPER_SMALL_MLX = InlineAsrMlxWhisperOptions(
    repo_id="mlx-community/whisper-small-mlx",
    inference_framework=InferenceAsrFramework.MLX,
    language="en",
    task="transcribe",
    word_timestamps=True,
    no_speech_threshold=0.6,
    logprob_threshold=-1.0,
    compression_ratio_threshold=2.4,
)

WHISPER_MEDIUM_MLX = InlineAsrMlxWhisperOptions(
    repo_id="mlx-community/whisper-medium-mlx-8bit",
    inference_framework=InferenceAsrFramework.MLX,
    language="en",
    task="transcribe",
    word_timestamps=True,
    no_speech_threshold=0.6,
    logprob_threshold=-1.0,
    compression_ratio_threshold=2.4,
)

WHISPER_BASE_MLX = InlineAsrMlxWhisperOptions(
    repo_id="mlx-community/whisper-base-mlx",
    inference_framework=InferenceAsrFramework.MLX,
    language="en",
    task="transcribe",
    word_timestamps=True,
    no_speech_threshold=0.6,
    logprob_threshold=-1.0,
    compression_ratio_threshold=2.4,
)

WHISPER_LARGE_MLX = InlineAsrMlxWhisperOptions(
    repo_id="mlx-community/whisper-large-mlx-8bit",
    inference_framework=InferenceAsrFramework.MLX,
    language="en",
    task="transcribe",
    word_timestamps=True,
    no_speech_threshold=0.6,
    logprob_threshold=-1.0,
    compression_ratio_threshold=2.4,
)

WHISPER_TURBO_MLX = InlineAsrMlxWhisperOptions(
    repo_id="mlx-community/whisper-turbo",
    inference_framework=InferenceAsrFramework.MLX,
    language="en",
    task="transcribe",
    word_timestamps=True,
    no_speech_threshold=0.6,
    logprob_threshold=-1.0,
    compression_ratio_threshold=2.4,
)

# Explicit Native Whisper model options for users who want to force native usage
WHISPER_TINY_NATIVE = InlineAsrNativeWhisperOptions(
    repo_id="tiny",
    inference_framework=InferenceAsrFramework.WHISPER,
    verbose=True,
    timestamps=True,
    word_timestamps=True,
    temperature=0.0,
    max_new_tokens=256,
    max_time_chunk=30.0,
)

WHISPER_SMALL_NATIVE = InlineAsrNativeWhisperOptions(
    repo_id="small",
    inference_framework=InferenceAsrFramework.WHISPER,
    verbose=True,
    timestamps=True,
    word_timestamps=True,
    temperature=0.0,
    max_new_tokens=256,
    max_time_chunk=30.0,
)

WHISPER_MEDIUM_NATIVE = InlineAsrNativeWhisperOptions(
    repo_id="medium",
    inference_framework=InferenceAsrFramework.WHISPER,
    verbose=True,
    timestamps=True,
    word_timestamps=True,
    temperature=0.0,
    max_new_tokens=256,
    max_time_chunk=30.0,
)

WHISPER_BASE_NATIVE = InlineAsrNativeWhisperOptions(
    repo_id="base",
    inference_framework=InferenceAsrFramework.WHISPER,
    verbose=True,
    timestamps=True,
    word_timestamps=True,
    temperature=0.0,
    max_new_tokens=256,
    max_time_chunk=30.0,
)

WHISPER_LARGE_NATIVE = InlineAsrNativeWhisperOptions(
    repo_id="large",
    inference_framework=InferenceAsrFramework.WHISPER,
    verbose=True,
    timestamps=True,
    word_timestamps=True,
    temperature=0.0,
    max_new_tokens=256,
    max_time_chunk=30.0,
)

WHISPER_TURBO_NATIVE = InlineAsrNativeWhisperOptions(
    repo_id="turbo",
    inference_framework=InferenceAsrFramework.WHISPER,
    verbose=True,
    timestamps=True,
    word_timestamps=True,
    temperature=0.0,
    max_new_tokens=256,
    max_time_chunk=30.0,
)

# Note: The main WHISPER_* models (WHISPER_TURBO, WHISPER_BASE, etc.) automatically
# select the best implementation (MLX on Apple Silicon, Native elsewhere).
# Use the explicit _MLX or _NATIVE variants if you need to force a specific implementation.


class AsrModelType(str, Enum):
    # Auto-selecting models (choose best implementation for hardware)
    WHISPER_TINY = "whisper_tiny"
    WHISPER_SMALL = "whisper_small"
    WHISPER_MEDIUM = "whisper_medium"
    WHISPER_BASE = "whisper_base"
    WHISPER_LARGE = "whisper_large"
    WHISPER_TURBO = "whisper_turbo"

    # Explicit MLX models (force MLX implementation)
    WHISPER_TINY_MLX = "whisper_tiny_mlx"
    WHISPER_SMALL_MLX = "whisper_small_mlx"
    WHISPER_MEDIUM_MLX = "whisper_medium_mlx"
    WHISPER_BASE_MLX = "whisper_base_mlx"
    WHISPER_LARGE_MLX = "whisper_large_mlx"
    WHISPER_TURBO_MLX = "whisper_turbo_mlx"

    # Explicit Native models (force native implementation)
    WHISPER_TINY_NATIVE = "whisper_tiny_native"
    WHISPER_SMALL_NATIVE = "whisper_small_native"
    WHISPER_MEDIUM_NATIVE = "whisper_medium_native"
    WHISPER_BASE_NATIVE = "whisper_base_native"
    WHISPER_LARGE_NATIVE = "whisper_large_native"
    WHISPER_TURBO_NATIVE = "whisper_turbo_native"
