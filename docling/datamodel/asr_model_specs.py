# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import logging
from enum import Enum
from importlib.util import find_spec

from pydantic import (
    AnyUrl,
)

from docling.datamodel.accelerator_options import AcceleratorDevice
from docling.datamodel.asr_presets import (
    AsrEngineConfig,
    AsrModelPreset,
    AsrModelSpec,
)
from docling.datamodel.pipeline_options_asr_model import (
    # AsrResponseFormat,
    # ApiAsrOptions,
    InferenceAsrFramework,
    InlineAsrMlxWhisperOptions,
    InlineAsrNativeWhisperOptions,
    InlineAsrWhisperS2TOptions,
    TransformersModelType,
)

_log = logging.getLogger(__name__)


def _detect_hardware_and_libraries() -> tuple[bool, bool]:
    """
    Detect Apple Silicon MPS availability and whether mlx-whisper is installed.

    Returns:
        tuple: (has_mps, has_mlx_whisper)
            - has_mps: True if Apple Silicon MPS is available
            - has_mlx_whisper: True if mlx-whisper package is installed
    """
    # Check for Apple Silicon MPS
    try:
        import torch

        has_mps = torch.backends.mps.is_built() and torch.backends.mps.is_available()
    except ImportError:
        has_mps = False

    # Importing mlx_whisper initializes Metal; only the ASR model should do that.
    has_mlx_whisper = find_spec("mlx_whisper") is not None

    return has_mps, has_mlx_whisper


def _get_whisper_tiny_model():
    """
    Get the best Whisper Tiny model for the current hardware.

    Auto-selection:
    - MLX Whisper on Apple Silicon (if MPS available + mlx-whisper installed)
    - Native Whisper otherwise

    WhisperS2T is never auto-selected; use the explicit *_S2T model options to
    opt into the experimental WhisperS2T backend.
    """
    has_mps, has_mlx_whisper = _detect_hardware_and_libraries()

    # MLX on Apple Silicon
    if has_mps and has_mlx_whisper:
        return InlineAsrMlxWhisperOptions(
            repo_id="mlx-community/whisper-tiny-mlx",
            inference_framework=InferenceAsrFramework.MLX,
            task="transcribe",
            word_timestamps=True,
            no_speech_threshold=0.6,
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.4,
        )

    # Native Whisper (default)
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

    Auto-selection:
    - MLX Whisper on Apple Silicon (if MPS available + mlx-whisper installed)
    - Native Whisper otherwise

    WhisperS2T is never auto-selected; use the explicit *_S2T model options to
    opt into the experimental WhisperS2T backend.
    """
    has_mps, has_mlx_whisper = _detect_hardware_and_libraries()

    # MLX on Apple Silicon
    if has_mps and has_mlx_whisper:
        return InlineAsrMlxWhisperOptions(
            repo_id="mlx-community/whisper-small-mlx",
            inference_framework=InferenceAsrFramework.MLX,
            task="transcribe",
            word_timestamps=True,
            no_speech_threshold=0.6,
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.4,
        )

    # Native Whisper (default)
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

    Auto-selection:
    - MLX Whisper on Apple Silicon (if MPS available + mlx-whisper installed)
    - Native Whisper otherwise

    WhisperS2T is never auto-selected; use the explicit *_S2T model options to
    opt into the experimental WhisperS2T backend.
    """
    has_mps, has_mlx_whisper = _detect_hardware_and_libraries()

    # MLX on Apple Silicon
    if has_mps and has_mlx_whisper:
        return InlineAsrMlxWhisperOptions(
            repo_id="mlx-community/whisper-medium-mlx-8bit",
            inference_framework=InferenceAsrFramework.MLX,
            task="transcribe",
            word_timestamps=True,
            no_speech_threshold=0.6,
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.4,
        )

    # Native Whisper (default)
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

    Auto-selection:
    - MLX Whisper on Apple Silicon (if MPS available + mlx-whisper installed)
    - Native Whisper otherwise

    WhisperS2T is never auto-selected; use the explicit *_S2T model options to
    opt into the experimental WhisperS2T backend.
    """
    has_mps, has_mlx_whisper = _detect_hardware_and_libraries()

    # MLX on Apple Silicon
    if has_mps and has_mlx_whisper:
        return InlineAsrMlxWhisperOptions(
            repo_id="mlx-community/whisper-base-mlx",
            inference_framework=InferenceAsrFramework.MLX,
            task="transcribe",
            word_timestamps=True,
            no_speech_threshold=0.6,
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.4,
        )

    # Native Whisper (default)
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

    Auto-selection:
    - MLX Whisper on Apple Silicon (if MPS available + mlx-whisper installed)
    - Native Whisper otherwise

    WhisperS2T is never auto-selected; use the explicit *_S2T model options to
    opt into the experimental WhisperS2T backend.
    """
    has_mps, has_mlx_whisper = _detect_hardware_and_libraries()

    # MLX on Apple Silicon
    if has_mps and has_mlx_whisper:
        return InlineAsrMlxWhisperOptions(
            repo_id="mlx-community/whisper-large-mlx-8bit",
            inference_framework=InferenceAsrFramework.MLX,
            task="transcribe",
            word_timestamps=True,
            no_speech_threshold=0.6,
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.4,
        )

    # Native Whisper (default)
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

    Auto-selection:
    - MLX Whisper on Apple Silicon (if MPS available + mlx-whisper installed)
    - Native Whisper otherwise

    WhisperS2T is never auto-selected; use the explicit *_S2T model options to
    opt into the experimental WhisperS2T backend.
    """
    has_mps, has_mlx_whisper = _detect_hardware_and_libraries()

    # MLX on Apple Silicon
    if has_mps and has_mlx_whisper:
        return InlineAsrMlxWhisperOptions(
            repo_id="mlx-community/whisper-turbo",
            inference_framework=InferenceAsrFramework.MLX,
            task="transcribe",
            word_timestamps=True,
            no_speech_threshold=0.6,
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.4,
        )

    # Native Whisper (default)
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

# =============================================================================
# Explicit MLX Whisper model options for users who want to force MLX usage
# =============================================================================

WHISPER_TINY_MLX = InlineAsrMlxWhisperOptions(
    repo_id="mlx-community/whisper-tiny-mlx",
    inference_framework=InferenceAsrFramework.MLX,
    task="transcribe",
    word_timestamps=True,
    no_speech_threshold=0.6,
    logprob_threshold=-1.0,
    compression_ratio_threshold=2.4,
)

WHISPER_SMALL_MLX = InlineAsrMlxWhisperOptions(
    repo_id="mlx-community/whisper-small-mlx",
    inference_framework=InferenceAsrFramework.MLX,
    task="transcribe",
    word_timestamps=True,
    no_speech_threshold=0.6,
    logprob_threshold=-1.0,
    compression_ratio_threshold=2.4,
)

WHISPER_MEDIUM_MLX = InlineAsrMlxWhisperOptions(
    repo_id="mlx-community/whisper-medium-mlx-8bit",
    inference_framework=InferenceAsrFramework.MLX,
    task="transcribe",
    word_timestamps=True,
    no_speech_threshold=0.6,
    logprob_threshold=-1.0,
    compression_ratio_threshold=2.4,
)

WHISPER_BASE_MLX = InlineAsrMlxWhisperOptions(
    repo_id="mlx-community/whisper-base-mlx",
    inference_framework=InferenceAsrFramework.MLX,
    task="transcribe",
    word_timestamps=True,
    no_speech_threshold=0.6,
    logprob_threshold=-1.0,
    compression_ratio_threshold=2.4,
)

WHISPER_LARGE_MLX = InlineAsrMlxWhisperOptions(
    repo_id="mlx-community/whisper-large-mlx-8bit",
    inference_framework=InferenceAsrFramework.MLX,
    task="transcribe",
    word_timestamps=True,
    no_speech_threshold=0.6,
    logprob_threshold=-1.0,
    compression_ratio_threshold=2.4,
)

WHISPER_TURBO_MLX = InlineAsrMlxWhisperOptions(
    repo_id="mlx-community/whisper-turbo",
    inference_framework=InferenceAsrFramework.MLX,
    task="transcribe",
    word_timestamps=True,
    no_speech_threshold=0.6,
    logprob_threshold=-1.0,
    compression_ratio_threshold=2.4,
)

# =============================================================================
# Explicit Native Whisper model options for users who want to force native usage
# =============================================================================

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

# English-only OpenAI Whisper checkpoints (part of openai-whisper's registry)

WHISPER_TINY_EN_NATIVE = InlineAsrNativeWhisperOptions(
    repo_id="tiny.en",
    inference_framework=InferenceAsrFramework.WHISPER,
    verbose=True,
    timestamps=True,
    word_timestamps=True,
    temperature=0.0,
    max_new_tokens=256,
    max_time_chunk=30.0,
)

WHISPER_BASE_EN_NATIVE = InlineAsrNativeWhisperOptions(
    repo_id="base.en",
    inference_framework=InferenceAsrFramework.WHISPER,
    verbose=True,
    timestamps=True,
    word_timestamps=True,
    temperature=0.0,
    max_new_tokens=256,
    max_time_chunk=30.0,
)

WHISPER_SMALL_EN_NATIVE = InlineAsrNativeWhisperOptions(
    repo_id="small.en",
    inference_framework=InferenceAsrFramework.WHISPER,
    verbose=True,
    timestamps=True,
    word_timestamps=True,
    temperature=0.0,
    max_new_tokens=256,
    max_time_chunk=30.0,
)

WHISPER_MEDIUM_EN_NATIVE = InlineAsrNativeWhisperOptions(
    repo_id="medium.en",
    inference_framework=InferenceAsrFramework.WHISPER,
    verbose=True,
    timestamps=True,
    word_timestamps=True,
    temperature=0.0,
    max_new_tokens=256,
    max_time_chunk=30.0,
)

# Distil-Whisper models (English-only). These are not part of openai-whisper's
# registry; the ASR pipeline loads them from the OpenAI-format checkpoints
# published in the distil-whisper Hugging Face repos.

WHISPER_DISTIL_SMALL_EN_NATIVE = InlineAsrNativeWhisperOptions(
    repo_id="distil-small.en",
    inference_framework=InferenceAsrFramework.WHISPER,
    language="en",
    verbose=True,
    timestamps=True,
    word_timestamps=True,
    temperature=0.0,
    max_new_tokens=256,
    max_time_chunk=30.0,
)

WHISPER_DISTIL_MEDIUM_EN_NATIVE = InlineAsrNativeWhisperOptions(
    repo_id="distil-medium.en",
    inference_framework=InferenceAsrFramework.WHISPER,
    language="en",
    verbose=True,
    timestamps=True,
    word_timestamps=True,
    temperature=0.0,
    max_new_tokens=256,
    max_time_chunk=30.0,
)

WHISPER_DISTIL_LARGE_V3_NATIVE = InlineAsrNativeWhisperOptions(
    repo_id="distil-large-v3",
    inference_framework=InferenceAsrFramework.WHISPER,
    language="en",
    verbose=True,
    timestamps=True,
    word_timestamps=True,
    temperature=0.0,
    max_new_tokens=256,
    max_time_chunk=30.0,
)

WHISPER_DISTIL_LARGE_V3_5_NATIVE = InlineAsrNativeWhisperOptions(
    repo_id="distil-large-v3.5",
    inference_framework=InferenceAsrFramework.WHISPER,
    language="en",
    verbose=True,
    timestamps=True,
    word_timestamps=True,
    temperature=0.0,
    max_new_tokens=256,
    max_time_chunk=30.0,
)

# Distil-Whisper models are not part of openai-whisper's model registry, but
# their Hugging Face repos publish the checkpoint in the original OpenAI
# format, which whisper.load_model() accepts as a local file path.
_DISTIL_WHISPER_OPENAI_CHECKPOINTS: dict[str, tuple[str, str, str]] = {
    "distil-small.en": (
        "distil-whisper/distil-small.en",
        "original-model.bin",
        "9e4a67ca4569c30be43a3fe7fba1621e504f0093",
    ),
    "distil-medium.en": (
        "distil-whisper/distil-medium.en",
        "original-model.bin",
        "6e61418885eaf4d5cc9f64e508e80ac5b4c052b7",
    ),
    "distil-large-v3": (
        "distil-whisper/distil-large-v3-openai",
        "model.bin",
        "81941037893cb90dc82af45bc5dc5146ab0a818f",
    ),
    "distil-large-v3.5": (
        "distil-whisper/distil-large-v3.5-openai",
        "model.bin",
        "24c117e115e74a9979c49f65cff75c5552dc3652",
    ),
}


def _distil_whisper_prefetch_name(model_name: str) -> str:
    return "whisper_" + model_name.replace("-", "_").replace(".", "_")


# =============================================================================
# WhisperS2T Models (CTranslate2 backend - fastest option for CPU/CUDA)
# =============================================================================

# Tiny models
WHISPER_TINY_S2T = InlineAsrWhisperS2TOptions(
    repo_id="tiny",
    inference_framework=InferenceAsrFramework.WHISPER_S2T,
    language="en",
    task="transcribe",
    torch_dtype="float16",
    batch_size=16,
    beam_size=1,
)

WHISPER_TINY_EN_S2T = InlineAsrWhisperS2TOptions(
    repo_id="tiny.en",
    inference_framework=InferenceAsrFramework.WHISPER_S2T,
    language="en",
    task="transcribe",
    torch_dtype="float16",
    batch_size=16,
    beam_size=1,
)

# Base models
WHISPER_BASE_S2T = InlineAsrWhisperS2TOptions(
    repo_id="base",
    inference_framework=InferenceAsrFramework.WHISPER_S2T,
    language="en",
    task="transcribe",
    torch_dtype="float16",
    batch_size=12,
    beam_size=1,
)

WHISPER_BASE_EN_S2T = InlineAsrWhisperS2TOptions(
    repo_id="base.en",
    inference_framework=InferenceAsrFramework.WHISPER_S2T,
    language="en",
    task="transcribe",
    torch_dtype="float16",
    batch_size=12,
    beam_size=1,
)

# Small models
WHISPER_SMALL_S2T = InlineAsrWhisperS2TOptions(
    repo_id="small",
    inference_framework=InferenceAsrFramework.WHISPER_S2T,
    language="en",
    task="transcribe",
    torch_dtype="float16",
    batch_size=8,
    beam_size=1,
)

WHISPER_SMALL_EN_S2T = InlineAsrWhisperS2TOptions(
    repo_id="small.en",
    inference_framework=InferenceAsrFramework.WHISPER_S2T,
    language="en",
    task="transcribe",
    torch_dtype="float16",
    batch_size=8,
    beam_size=1,
)

WHISPER_DISTIL_SMALL_EN_S2T = InlineAsrWhisperS2TOptions(
    repo_id="distil-small.en",
    inference_framework=InferenceAsrFramework.WHISPER_S2T,
    language="en",
    task="transcribe",
    torch_dtype="float16",
    batch_size=10,
    beam_size=1,
)

# Medium models
WHISPER_MEDIUM_S2T = InlineAsrWhisperS2TOptions(
    repo_id="medium",
    inference_framework=InferenceAsrFramework.WHISPER_S2T,
    language="en",
    task="transcribe",
    torch_dtype="float16",
    batch_size=6,
    beam_size=1,
)

WHISPER_MEDIUM_EN_S2T = InlineAsrWhisperS2TOptions(
    repo_id="medium.en",
    inference_framework=InferenceAsrFramework.WHISPER_S2T,
    language="en",
    task="transcribe",
    torch_dtype="float16",
    batch_size=6,
    beam_size=1,
)

WHISPER_DISTIL_MEDIUM_EN_S2T = InlineAsrWhisperS2TOptions(
    repo_id="distil-medium.en",
    inference_framework=InferenceAsrFramework.WHISPER_S2T,
    language="en",
    task="transcribe",
    torch_dtype="float16",
    batch_size=8,
    beam_size=1,
)

# Large models
WHISPER_LARGE_V3_S2T = InlineAsrWhisperS2TOptions(
    repo_id="large-v3",
    inference_framework=InferenceAsrFramework.WHISPER_S2T,
    language="en",
    task="transcribe",
    torch_dtype="float16",
    batch_size=4,
    beam_size=1,
)

WHISPER_DISTIL_LARGE_V3_S2T = InlineAsrWhisperS2TOptions(
    repo_id="distil-large-v3",
    inference_framework=InferenceAsrFramework.WHISPER_S2T,
    language="en",
    task="transcribe",
    torch_dtype="float16",
    batch_size=6,
    beam_size=1,
)

WHISPER_DISTIL_LARGE_V3_5_S2T = InlineAsrWhisperS2TOptions(
    repo_id="distil-large-v3.5",
    inference_framework=InferenceAsrFramework.WHISPER_S2T,
    language="en",
    task="transcribe",
    torch_dtype="float16",
    batch_size=6,
    beam_size=1,
)

WHISPER_LARGE_V3_TURBO_S2T = InlineAsrWhisperS2TOptions(
    repo_id="large-v3-turbo",
    inference_framework=InferenceAsrFramework.WHISPER_S2T,
    language="en",
    task="transcribe",
    torch_dtype="float16",
    batch_size=6,
    beam_size=1,
)

_WHISPER_S2T_PRESETS: dict[str, InlineAsrWhisperS2TOptions] = {
    preset.repo_id: preset
    for preset in (
        WHISPER_TINY_S2T,
        WHISPER_TINY_EN_S2T,
        WHISPER_BASE_S2T,
        WHISPER_BASE_EN_S2T,
        WHISPER_SMALL_S2T,
        WHISPER_SMALL_EN_S2T,
        WHISPER_DISTIL_SMALL_EN_S2T,
        WHISPER_MEDIUM_S2T,
        WHISPER_MEDIUM_EN_S2T,
        WHISPER_DISTIL_MEDIUM_EN_S2T,
        WHISPER_LARGE_V3_S2T,
        WHISPER_DISTIL_LARGE_V3_S2T,
        WHISPER_DISTIL_LARGE_V3_5_S2T,
        WHISPER_LARGE_V3_TURBO_S2T,
    )
}

_WHISPER_S2T_COMPUTE_TYPES = ("float32", "float16", "bfloat16")


def _whisper_s2t_compute_type(torch_dtype: str | None, device: str) -> str:
    if (
        device == "cpu"
        or torch_dtype is None
        or torch_dtype not in _WHISPER_S2T_COMPUTE_TYPES
    ):
        return "float32"
    return torch_dtype


def _whisper_s2t_prefetch_name(model_name: str) -> str:
    return "whisper_" + model_name.replace("-", "_").replace(".", "_") + "_s2t"


# =============================================================================
# Note on auto-selecting models
# =============================================================================
# The main WHISPER_* models (WHISPER_TURBO, WHISPER_BASE, etc.) automatically
# select the best implementation based on available hardware and libraries:
#
# Priority order:
#   1. MLX Whisper - Used on Apple Silicon when mlx-whisper is installed
#   2. Native Whisper - Default on all other hardware
#
# WhisperS2T is an optional, experimental backend and is never auto-selected.
# Use the explicit _MLX, _NATIVE, or _S2T variants if you need to force a
# specific implementation regardless of hardware detection.
# =============================================================================


_WHISPER_EXPORTS = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx-8bit",
    "large": "mlx-community/whisper-large-mlx-8bit",
    "turbo": "mlx-community/whisper-turbo",
}

WHISPER_ASR_PRESETS: dict[str, AsrModelPreset] = {
    size: AsrModelPreset(
        preset_id=f"whisper_{size}",
        default_engine_type=InferenceAsrFramework.TRANSFORMERS,
        model_spec=AsrModelSpec(
            name=f"Whisper {size}",
            default_repo_id=(
                "openai/whisper-large-v3-turbo"
                if size == "turbo"
                else f"openai/whisper-{size}"
            ),
            supported_engines={
                InferenceAsrFramework.TRANSFORMERS,
                InferenceAsrFramework.MLX,
            },
            engine_overrides={
                InferenceAsrFramework.MLX: AsrEngineConfig(repo_id=mlx_repo_id)
            },
        ),
    )
    for size, mlx_repo_id in _WHISPER_EXPORTS.items()
}

WHISPER_TINY_TRANSFORMERS = WHISPER_ASR_PRESETS["tiny"].to_options()
WHISPER_BASE_TRANSFORMERS = WHISPER_ASR_PRESETS["base"].to_options()
WHISPER_SMALL_TRANSFORMERS = WHISPER_ASR_PRESETS["small"].to_options()
WHISPER_MEDIUM_TRANSFORMERS = WHISPER_ASR_PRESETS["medium"].to_options()
WHISPER_LARGE_TRANSFORMERS = WHISPER_ASR_PRESETS["large"].to_options()
WHISPER_TURBO_TRANSFORMERS = WHISPER_ASR_PRESETS["turbo"].to_options()

GRANITE_4_0_1B_SPEECH_PRESET = AsrModelPreset(
    preset_id="granite_4_0_1b_speech",
    default_engine_type=InferenceAsrFramework.TRANSFORMERS,
    model_spec=AsrModelSpec(
        name="Granite 4.0 1B Speech",
        default_repo_id="ibm-granite/granite-4.0-1b-speech",
        model_family="granite",
        supported_engines={InferenceAsrFramework.TRANSFORMERS},
    ),
)
NEMOTRON_3_5_ASR_STREAMING_PRESET = AsrModelPreset(
    preset_id="nemotron_3_5_asr_streaming_0_6b",
    default_engine_type=InferenceAsrFramework.TRANSFORMERS,
    model_spec=AsrModelSpec(
        name="Nemotron 3.5 ASR Streaming 0.6B",
        default_repo_id="nvidia/nemotron-3.5-asr-streaming-0.6b",
        model_family="nemotron",
        supported_engines={InferenceAsrFramework.TRANSFORMERS},
    ),
)

GRANITE_4_0_1B_SPEECH = GRANITE_4_0_1B_SPEECH_PRESET.to_options()
NEMOTRON_3_5_ASR_STREAMING = NEMOTRON_3_5_ASR_STREAMING_PRESET.to_options()


GRANITE_SPEECH_5_0_2B_PRESET = AsrModelPreset(
    preset_id="granite_speech_5_0_2b",
    default_engine_type=InferenceAsrFramework.TRANSFORMERS,
    model_spec=AsrModelSpec(
        name="Granite Speech 5.0 2B",
        default_repo_id="ibm-granite/granite-speech-5.0-2b",
        revision="f4e68169fa2a5751acc4b5da9957855e5e8fc3f1",
        model_family="granite",
        supported_engines={InferenceAsrFramework.TRANSFORMERS},
    ),
)
GRANITE_SPEECH_5_0_2B = GRANITE_SPEECH_5_0_2B_PRESET.to_options()

GRANITE_SPEECH_5_0_470M_TURBOCTC_PRESET = AsrModelPreset(
    preset_id="granite_speech_5_0_470m_turboctc",
    default_engine_type=InferenceAsrFramework.TRANSFORMERS,
    model_spec=AsrModelSpec(
        name="Granite Speech 5.0 470M TurboCTC",
        default_repo_id="ibm-granite/granite-speech-5.0-470m-turboctc",
        revision="1dd23f46f6d4e6eae2531982ba2e33220086dd97",
        model_family="granite_ctc",
        supported_engines={InferenceAsrFramework.TRANSFORMERS},
    ),
)
GRANITE_SPEECH_5_0_470M_TURBOCTC = GRANITE_SPEECH_5_0_470M_TURBOCTC_PRESET.to_options()

GRANITE_SPEECH_5_0_470M_TURBOCTC_NC_PRESET = AsrModelPreset(
    preset_id="granite_speech_5_0_470m_turboctc_nc",
    default_engine_type=InferenceAsrFramework.TRANSFORMERS,
    model_spec=AsrModelSpec(
        name="Granite Speech 5.0 470M TurboCTC NC",
        default_repo_id="ibm-granite/granite-speech-5.0-470m-turboctc-nc",
        revision="af0a8e8663defc902944ebdae0d29a5219be11a3",
        model_family="granite_ctc",
        supported_engines={InferenceAsrFramework.TRANSFORMERS},
    ),
)
GRANITE_SPEECH_5_0_470M_TURBOCTC_NC = (
    GRANITE_SPEECH_5_0_470M_TURBOCTC_NC_PRESET.to_options()
)

GRANITE_SPEECH_4_1_2B_PRESET = AsrModelPreset(
    preset_id="granite_speech_4_1_2b",
    default_engine_type=InferenceAsrFramework.TRANSFORMERS,
    model_spec=AsrModelSpec(
        name="Granite Speech 4.1 2B",
        default_repo_id="ibm-granite/granite-speech-4.1-2b",
        revision="de575db64086f84fdc79da4932d1076e965bc546",
        model_family="granite",
        supported_engines={InferenceAsrFramework.TRANSFORMERS},
    ),
)
GRANITE_SPEECH_4_1_2B = GRANITE_SPEECH_4_1_2B_PRESET.to_options()

GRANITE_SPEECH_4_1_2B_PLUS_PRESET = AsrModelPreset(
    preset_id="granite_speech_4_1_2b_plus",
    default_engine_type=InferenceAsrFramework.TRANSFORMERS,
    model_spec=AsrModelSpec(
        name="Granite Speech 4.1 2B Plus",
        default_repo_id="ibm-granite/granite-speech-4.1-2b-plus",
        revision="e69efe37f97b6283c61f5a8dadda2b3eb67765d6",
        model_family="granite",
        supported_engines={InferenceAsrFramework.TRANSFORMERS},
    ),
)
GRANITE_SPEECH_4_1_2B_PLUS = GRANITE_SPEECH_4_1_2B_PLUS_PRESET.to_options()

NEMOTRON_SPEECH_STREAMING_EN_0_6B_PRESET = AsrModelPreset(
    preset_id="nemotron_speech_streaming_en_0_6b",
    default_engine_type=InferenceAsrFramework.TRANSFORMERS,
    model_spec=AsrModelSpec(
        name="Nemotron Speech Streaming English 0.6B",
        default_repo_id="nvidia/nemotron-speech-streaming-en-0.6b",
        revision="ebe59e5a817142986528bbbee5dba8db7b38ed50",
        model_family="nemotron",
        supported_engines={InferenceAsrFramework.TRANSFORMERS},
    ),
)
NEMOTRON_SPEECH_STREAMING_EN_0_6B = (
    NEMOTRON_SPEECH_STREAMING_EN_0_6B_PRESET.to_options()
)

PARAKEET_CTC_1_1B_PRESET = AsrModelPreset(
    preset_id="parakeet_ctc_1_1b",
    default_engine_type=InferenceAsrFramework.TRANSFORMERS,
    model_spec=AsrModelSpec(
        name="Parakeet CTC 1.1B",
        default_repo_id="nvidia/parakeet-ctc-1.1b",
        revision="20e63a0fed6aedba145b74b826dbd41df0941730",
        model_family="parakeet",
        supported_engines={InferenceAsrFramework.TRANSFORMERS},
    ),
)
PARAKEET_CTC_1_1B = PARAKEET_CTC_1_1B_PRESET.to_options()


ADDITIONAL_ASR_PRESETS: dict[str, AsrModelPreset] = {
    preset.preset_id: preset
    for preset in (
        GRANITE_SPEECH_5_0_2B_PRESET,
        GRANITE_SPEECH_5_0_470M_TURBOCTC_PRESET,
        GRANITE_SPEECH_5_0_470M_TURBOCTC_NC_PRESET,
        GRANITE_SPEECH_4_1_2B_PRESET,
        GRANITE_SPEECH_4_1_2B_PLUS_PRESET,
        NEMOTRON_SPEECH_STREAMING_EN_0_6B_PRESET,
        PARAKEET_CTC_1_1B_PRESET,
    )
}


class AsrModelType(str, Enum):
    GRANITE_SPEECH_5_0_2B = "granite_speech_5_0_2b"
    GRANITE_SPEECH_5_0_470M_TURBOCTC = "granite_speech_5_0_470m_turboctc"
    GRANITE_SPEECH_5_0_470M_TURBOCTC_NC = "granite_speech_5_0_470m_turboctc_nc"
    GRANITE_SPEECH_4_1_2B = "granite_speech_4_1_2b"
    GRANITE_SPEECH_4_1_2B_PLUS = "granite_speech_4_1_2b_plus"
    NEMOTRON_SPEECH_STREAMING_EN_0_6B = "nemotron_speech_streaming_en_0_6b"
    PARAKEET_CTC_1_1B = "parakeet_ctc_1_1b"
    GRANITE_4_0_1B_SPEECH = "granite_4_0_1b_speech"
    NEMOTRON_3_5_ASR_STREAMING = "nemotron_3_5_asr_streaming"
    # Auto-selecting models (choose best implementation for hardware)
    WHISPER_TINY = "whisper_tiny"
    WHISPER_SMALL = "whisper_small"
    WHISPER_MEDIUM = "whisper_medium"
    WHISPER_BASE = "whisper_base"
    WHISPER_LARGE = "whisper_large"
    WHISPER_TURBO = "whisper_turbo"

    # Transformers-format Whisper checkpoints
    WHISPER_TINY_TRANSFORMERS = "whisper_tiny_transformers"
    WHISPER_BASE_TRANSFORMERS = "whisper_base_transformers"
    WHISPER_SMALL_TRANSFORMERS = "whisper_small_transformers"
    WHISPER_MEDIUM_TRANSFORMERS = "whisper_medium_transformers"
    WHISPER_LARGE_TRANSFORMERS = "whisper_large_transformers"
    WHISPER_TURBO_TRANSFORMERS = "whisper_turbo_transformers"

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
    WHISPER_TINY_EN_NATIVE = "whisper_tiny_en_native"
    WHISPER_BASE_EN_NATIVE = "whisper_base_en_native"
    WHISPER_SMALL_EN_NATIVE = "whisper_small_en_native"
    WHISPER_MEDIUM_EN_NATIVE = "whisper_medium_en_native"
    WHISPER_DISTIL_SMALL_EN_NATIVE = "whisper_distil_small_en_native"
    WHISPER_DISTIL_MEDIUM_EN_NATIVE = "whisper_distil_medium_en_native"
    WHISPER_DISTIL_LARGE_V3_NATIVE = "whisper_distil_large_v3_native"
    WHISPER_DISTIL_LARGE_V3_5_NATIVE = "whisper_distil_large_v3_5_native"

    # Explicit WhisperS2T models (CTranslate2 backend - fastest)
    WHISPER_TINY_S2T = "whisper_tiny_s2t"
    WHISPER_TINY_EN_S2T = "whisper_tiny_en_s2t"
    WHISPER_BASE_S2T = "whisper_base_s2t"
    WHISPER_BASE_EN_S2T = "whisper_base_en_s2t"
    WHISPER_SMALL_S2T = "whisper_small_s2t"
    WHISPER_SMALL_EN_S2T = "whisper_small_en_s2t"
    WHISPER_DISTIL_SMALL_EN_S2T = "whisper_distil_small_en_s2t"
    WHISPER_MEDIUM_S2T = "whisper_medium_s2t"
    WHISPER_MEDIUM_EN_S2T = "whisper_medium_en_s2t"
    WHISPER_DISTIL_MEDIUM_EN_S2T = "whisper_distil_medium_en_s2t"
    WHISPER_LARGE_V3_S2T = "whisper_large_v3_s2t"
    WHISPER_DISTIL_LARGE_V3_S2T = "whisper_distil_large_v3_s2t"
    WHISPER_DISTIL_LARGE_V3_5_S2T = "whisper_distil_large_v3_5_s2t"
    WHISPER_LARGE_V3_TURBO_S2T = "whisper_large_v3_turbo_s2t"
