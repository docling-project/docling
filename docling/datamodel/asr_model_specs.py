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
    InlineAsrWhisperS2TOptions,
    TransformersModelType,
)

_log = logging.getLogger(__name__)


def _detect_hardware_and_libraries():
    """
    Detect available hardware acceleration and installed ASR libraries.
    
    Returns:
        tuple: (has_mps, has_cuda, has_mlx_whisper, has_whisper_s2t)
            - has_mps: True if Apple Silicon MPS is available
            - has_cuda: True if NVIDIA CUDA is available
            - has_mlx_whisper: True if mlx-whisper package is installed
            - has_whisper_s2t: True if whisper-s2t-reborn package is installed
    """
    # Check for hardware acceleration
    try:
        import torch
        has_mps = torch.backends.mps.is_built() and torch.backends.mps.is_available()
        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_mps = False
        has_cuda = False

    # Check if mlx-whisper is available
    try:
        import mlx_whisper  # type: ignore
        has_mlx_whisper = True
    except ImportError:
        has_mlx_whisper = False

    # Check if whisper-s2t is available
    try:
        import whisper_s2t  # type: ignore
        has_whisper_s2t = True
    except ImportError:
        has_whisper_s2t = False

    return has_mps, has_cuda, has_mlx_whisper, has_whisper_s2t


def _get_whisper_tiny_model():
    """
    Get the best Whisper Tiny model for the current hardware.

    Auto-selection priority:
    1. MLX Whisper on Apple Silicon (if MPS available + mlx-whisper installed)
    2. WhisperS2T on CUDA (if CUDA available + whisper-s2t installed) - fastest for GPU
    3. WhisperS2T on CPU (if whisper-s2t installed) - faster than native on CPU
    4. Native Whisper (fallback)
    """
    has_mps, has_cuda, has_mlx_whisper, has_whisper_s2t = _detect_hardware_and_libraries()

    # Priority 1: MLX on Apple Silicon
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
    
    # Priority 2: WhisperS2T on CUDA (fastest for GPU inference)
    if has_cuda and has_whisper_s2t:
        return InlineAsrWhisperS2TOptions(
            repo_id="tiny",
            inference_framework=InferenceAsrFramework.WHISPER_S2T,
            language="en",
            task="transcribe",
            compute_type="float16",
            batch_size=16,
            beam_size=1,
        )
    
    # Priority 3: WhisperS2T on CPU (still faster than native Whisper)
    if has_whisper_s2t:
        return InlineAsrWhisperS2TOptions(
            repo_id="tiny",
            inference_framework=InferenceAsrFramework.WHISPER_S2T,
            language="en",
            task="transcribe",
            compute_type="float32",  # float32 is more reliable on CPU
            batch_size=16,
            beam_size=1,
        )
    
    # Priority 4: Native Whisper (fallback)
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

    Auto-selection priority:
    1. MLX Whisper on Apple Silicon (if MPS available + mlx-whisper installed)
    2. WhisperS2T on CUDA (if CUDA available + whisper-s2t installed) - fastest for GPU
    3. WhisperS2T on CPU (if whisper-s2t installed) - faster than native on CPU
    4. Native Whisper (fallback)
    """
    has_mps, has_cuda, has_mlx_whisper, has_whisper_s2t = _detect_hardware_and_libraries()

    # Priority 1: MLX on Apple Silicon
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
    
    # Priority 2: WhisperS2T on CUDA
    if has_cuda and has_whisper_s2t:
        return InlineAsrWhisperS2TOptions(
            repo_id="small",
            inference_framework=InferenceAsrFramework.WHISPER_S2T,
            language="en",
            task="transcribe",
            compute_type="float16",
            batch_size=8,
            beam_size=1,
        )
    
    # Priority 3: WhisperS2T on CPU
    if has_whisper_s2t:
        return InlineAsrWhisperS2TOptions(
            repo_id="small",
            inference_framework=InferenceAsrFramework.WHISPER_S2T,
            language="en",
            task="transcribe",
            compute_type="float32",
            batch_size=8,
            beam_size=1,
        )
    
    # Priority 4: Native Whisper (fallback)
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

    Auto-selection priority:
    1. MLX Whisper on Apple Silicon (if MPS available + mlx-whisper installed)
    2. WhisperS2T on CUDA (if CUDA available + whisper-s2t installed) - fastest for GPU
    3. WhisperS2T on CPU (if whisper-s2t installed) - faster than native on CPU
    4. Native Whisper (fallback)
    """
    has_mps, has_cuda, has_mlx_whisper, has_whisper_s2t = _detect_hardware_and_libraries()

    # Priority 1: MLX on Apple Silicon
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
    
    # Priority 2: WhisperS2T on CUDA
    if has_cuda and has_whisper_s2t:
        return InlineAsrWhisperS2TOptions(
            repo_id="medium",
            inference_framework=InferenceAsrFramework.WHISPER_S2T,
            language="en",
            task="transcribe",
            compute_type="float16",
            batch_size=6,
            beam_size=1,
        )
    
    # Priority 3: WhisperS2T on CPU
    if has_whisper_s2t:
        return InlineAsrWhisperS2TOptions(
            repo_id="medium",
            inference_framework=InferenceAsrFramework.WHISPER_S2T,
            language="en",
            task="transcribe",
            compute_type="float32",
            batch_size=6,
            beam_size=1,
        )
    
    # Priority 4: Native Whisper (fallback)
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

    Auto-selection priority:
    1. MLX Whisper on Apple Silicon (if MPS available + mlx-whisper installed)
    2. WhisperS2T on CUDA (if CUDA available + whisper-s2t installed) - fastest for GPU
    3. WhisperS2T on CPU (if whisper-s2t installed) - faster than native on CPU
    4. Native Whisper (fallback)
    """
    has_mps, has_cuda, has_mlx_whisper, has_whisper_s2t = _detect_hardware_and_libraries()

    # Priority 1: MLX on Apple Silicon
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
    
    # Priority 2: WhisperS2T on CUDA
    if has_cuda and has_whisper_s2t:
        return InlineAsrWhisperS2TOptions(
            repo_id="base",
            inference_framework=InferenceAsrFramework.WHISPER_S2T,
            language="en",
            task="transcribe",
            compute_type="float16",
            batch_size=12,
            beam_size=1,
        )
    
    # Priority 3: WhisperS2T on CPU
    if has_whisper_s2t:
        return InlineAsrWhisperS2TOptions(
            repo_id="base",
            inference_framework=InferenceAsrFramework.WHISPER_S2T,
            language="en",
            task="transcribe",
            compute_type="float32",
            batch_size=12,
            beam_size=1,
        )
    
    # Priority 4: Native Whisper (fallback)
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

    Auto-selection priority:
    1. MLX Whisper on Apple Silicon (if MPS available + mlx-whisper installed)
    2. WhisperS2T on CUDA (if CUDA available + whisper-s2t installed) - fastest for GPU
    3. WhisperS2T on CPU (if whisper-s2t installed) - faster than native on CPU
    4. Native Whisper (fallback)
    
    Note: For S2T, uses large-v3 which is the latest large model variant.
    """
    has_mps, has_cuda, has_mlx_whisper, has_whisper_s2t = _detect_hardware_and_libraries()

    # Priority 1: MLX on Apple Silicon
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
    
    # Priority 2: WhisperS2T on CUDA
    if has_cuda and has_whisper_s2t:
        return InlineAsrWhisperS2TOptions(
            repo_id="large-v3",
            inference_framework=InferenceAsrFramework.WHISPER_S2T,
            language="en",
            task="transcribe",
            compute_type="float16",
            batch_size=4,
            beam_size=1,
        )
    
    # Priority 3: WhisperS2T on CPU
    if has_whisper_s2t:
        return InlineAsrWhisperS2TOptions(
            repo_id="large-v3",
            inference_framework=InferenceAsrFramework.WHISPER_S2T,
            language="en",
            task="transcribe",
            compute_type="float32",
            batch_size=4,
            beam_size=1,
        )
    
    # Priority 4: Native Whisper (fallback)
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

    Auto-selection priority:
    1. MLX Whisper on Apple Silicon (if MPS available + mlx-whisper installed)
    2. WhisperS2T on CUDA (if CUDA available + whisper-s2t installed) - fastest for GPU
    3. WhisperS2T on CPU (if whisper-s2t installed) - faster than native on CPU
    4. Native Whisper (fallback)
    
    Note: Turbo is a distilled model optimized for speed. For S2T, we use
    distil-large-v3 as the closest equivalent to turbo's speed/quality tradeoff.
    """
    has_mps, has_cuda, has_mlx_whisper, has_whisper_s2t = _detect_hardware_and_libraries()

    # Priority 1: MLX on Apple Silicon
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
    
    # Priority 2: WhisperS2T on CUDA (using distil-large-v3 for turbo-like performance)
    if has_cuda and has_whisper_s2t:
        return InlineAsrWhisperS2TOptions(
            repo_id="distil-large-v3",
            inference_framework=InferenceAsrFramework.WHISPER_S2T,
            language="en",
            task="transcribe",
            compute_type="float16",
            batch_size=6,
            beam_size=1,
        )
    
    # Priority 3: WhisperS2T on CPU
    if has_whisper_s2t:
        return InlineAsrWhisperS2TOptions(
            repo_id="distil-large-v3",
            inference_framework=InferenceAsrFramework.WHISPER_S2T,
            language="en",
            task="transcribe",
            compute_type="float32",
            batch_size=6,
            beam_size=1,
        )
    
    # Priority 4: Native Whisper (fallback)
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

# =============================================================================
# WhisperS2T Models (CTranslate2 backend - fastest option for CPU/CUDA)
# =============================================================================

# Tiny models
WHISPER_TINY_S2T = InlineAsrWhisperS2TOptions(
    repo_id="tiny",
    inference_framework=InferenceAsrFramework.WHISPER_S2T,
    language="en",
    task="transcribe",
    compute_type="float16",
    batch_size=16,
    beam_size=1,
)

WHISPER_TINY_EN_S2T = InlineAsrWhisperS2TOptions(
    repo_id="tiny.en",
    inference_framework=InferenceAsrFramework.WHISPER_S2T,
    language="en",
    task="transcribe",
    compute_type="float16",
    batch_size=16,
    beam_size=1,
)

# Base models
WHISPER_BASE_S2T = InlineAsrWhisperS2TOptions(
    repo_id="base",
    inference_framework=InferenceAsrFramework.WHISPER_S2T,
    language="en",
    task="transcribe",
    compute_type="float16",
    batch_size=12,
    beam_size=1,
)

WHISPER_BASE_EN_S2T = InlineAsrWhisperS2TOptions(
    repo_id="base.en",
    inference_framework=InferenceAsrFramework.WHISPER_S2T,
    language="en",
    task="transcribe",
    compute_type="float16",
    batch_size=12,
    beam_size=1,
)

# Small models
WHISPER_SMALL_S2T = InlineAsrWhisperS2TOptions(
    repo_id="small",
    inference_framework=InferenceAsrFramework.WHISPER_S2T,
    language="en",
    task="transcribe",
    compute_type="float16",
    batch_size=8,
    beam_size=1,
)

WHISPER_SMALL_EN_S2T = InlineAsrWhisperS2TOptions(
    repo_id="small.en",
    inference_framework=InferenceAsrFramework.WHISPER_S2T,
    language="en",
    task="transcribe",
    compute_type="float16",
    batch_size=8,
    beam_size=1,
)

WHISPER_DISTIL_SMALL_EN_S2T = InlineAsrWhisperS2TOptions(
    repo_id="distil-small.en",
    inference_framework=InferenceAsrFramework.WHISPER_S2T,
    language="en",
    task="transcribe",
    compute_type="float16",
    batch_size=10,
    beam_size=1,
)

# Medium models
WHISPER_MEDIUM_S2T = InlineAsrWhisperS2TOptions(
    repo_id="medium",
    inference_framework=InferenceAsrFramework.WHISPER_S2T,
    language="en",
    task="transcribe",
    compute_type="float16",
    batch_size=6,
    beam_size=1,
)

WHISPER_MEDIUM_EN_S2T = InlineAsrWhisperS2TOptions(
    repo_id="medium.en",
    inference_framework=InferenceAsrFramework.WHISPER_S2T,
    language="en",
    task="transcribe",
    compute_type="float16",
    batch_size=6,
    beam_size=1,
)

WHISPER_DISTIL_MEDIUM_EN_S2T = InlineAsrWhisperS2TOptions(
    repo_id="distil-medium.en",
    inference_framework=InferenceAsrFramework.WHISPER_S2T,
    language="en",
    task="transcribe",
    compute_type="float16",
    batch_size=8,
    beam_size=1,
)

# Large models
WHISPER_LARGE_V3_S2T = InlineAsrWhisperS2TOptions(
    repo_id="large-v3",
    inference_framework=InferenceAsrFramework.WHISPER_S2T,
    language="en",
    task="transcribe",
    compute_type="float16",
    batch_size=4,
    beam_size=1,
)

WHISPER_DISTIL_LARGE_V3_S2T = InlineAsrWhisperS2TOptions(
    repo_id="distil-large-v3",
    inference_framework=InferenceAsrFramework.WHISPER_S2T,
    language="en",
    task="transcribe",
    compute_type="float16",
    batch_size=6,
    beam_size=1,
)

# =============================================================================
# Note on auto-selecting models
# =============================================================================
# The main WHISPER_* models (WHISPER_TURBO, WHISPER_BASE, etc.) automatically
# select the best implementation based on available hardware and libraries:
#
# Priority order:
#   1. MLX Whisper - Used on Apple Silicon when mlx-whisper is installed
#   2. WhisperS2T (CUDA) - Used when CUDA GPU and whisper-s2t are available
#   3. WhisperS2T (CPU) - Used when whisper-s2t is available but no GPU
#   4. Native Whisper - Fallback when no optimized backends are available
#
# Use the explicit _MLX, _NATIVE, or _S2T variants if you need to force a
# specific implementation regardless of hardware detection.
# =============================================================================


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
