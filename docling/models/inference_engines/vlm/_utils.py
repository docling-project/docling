"""Internal utilities for VLM runtimes.

This module contains shared utility functions used across different VLM runtime
implementations to avoid code duplication and ensure consistency.
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from PIL import Image

from docling.datamodel.pipeline_options_vlm_model import TransformersPromptStyle
from docling.models.utils.generation_utils import GenerationStopper

_log = logging.getLogger(__name__)


def normalize_image_to_pil(image: Union[Image.Image, np.ndarray]) -> Image.Image:
    """Convert any image format to RGB PIL Image.

    Args:
        image: Input image as PIL Image or numpy array

    Returns:
        RGB PIL Image

    Raises:
        ValueError: If numpy array has unsupported shape
    """
    # Handle numpy arrays
    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[2] in [3, 4]:
            # RGB or RGBA array
            image = Image.fromarray(image.astype(np.uint8))
        elif image.ndim == 2:
            # Grayscale array
            image = Image.fromarray(image.astype(np.uint8), mode="L")
        else:
            raise ValueError(f"Unsupported numpy array shape: {image.shape}")

    # Ensure RGB mode (handles RGBA, L, P, etc.)
    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def preprocess_image_batch(
    images: List[Union[Image.Image, np.ndarray]],
) -> List[Image.Image]:
    """Preprocess a batch of images to RGB PIL Images.

    Args:
        images: List of images as PIL Images or numpy arrays

    Returns:
        List of RGB PIL Images
    """
    return [normalize_image_to_pil(img) for img in images]


def extract_generation_stoppers(
    extra_config: Dict[str, Any],
) -> List[GenerationStopper]:
    """Extract and instantiate GenerationStopper instances from config.

    This handles both GenerationStopper instances and classes, instantiating
    classes as needed.

    Args:
        extra_config: Extra generation configuration dictionary

    Returns:
        List of GenerationStopper instances
    """
    stoppers: List[GenerationStopper] = []
    custom_criteria = extra_config.get("custom_stopping_criteria", [])

    for criteria in custom_criteria:
        if isinstance(criteria, GenerationStopper):
            # Already an instance
            stoppers.append(criteria)
        elif isinstance(criteria, type) and issubclass(criteria, GenerationStopper):
            # A class - instantiate it
            stoppers.append(criteria())
        # Ignore other types (e.g., HF StoppingCriteria for transformers)

    return stoppers


def resolve_model_artifacts_path(
    repo_id: str,
    revision: str,
    artifacts_path: Optional[Union[Path, str]],
    download_fn: Callable[[str, str], Path],
) -> Path:
    """Resolve the path to model artifacts, downloading if needed.

    This standardizes the logic for finding or downloading model artifacts
    across different runtimes.

    Args:
        repo_id: HuggingFace repository ID (e.g., "microsoft/Phi-3.5-vision-instruct")
        revision: Model revision (e.g., "main")
        artifacts_path: Optional path to cached artifacts directory
        download_fn: Function to download models, takes (repo_id, revision) and returns Path

    Returns:
        Path to the model artifacts directory

    Raises:
        FileNotFoundError: If artifacts_path is provided but model not found
    """
    repo_cache_folder = repo_id.replace("/", "--")

    artifacts_path = artifacts_path if artifacts_path is None else Path(artifacts_path)

    if artifacts_path is None:
        # No cache path provided - download
        return download_fn(repo_id, revision)
    elif (artifacts_path / repo_cache_folder).exists():
        # Cache path with repo-specific subfolder exists
        return artifacts_path / repo_cache_folder
    else:
        # Model not found in artifacts_path - raise clear error
        available_models = []
        if artifacts_path.exists():
            available_models = [p.name for p in artifacts_path.iterdir() if p.is_dir()]

        raise FileNotFoundError(
            f"Model '{repo_id}' not found in artifacts_path.\n"
            f"Expected location: {artifacts_path / repo_cache_folder}\n"
            f"Available models in {artifacts_path}: "
            f"{', '.join(available_models) if available_models else 'none'}\n\n"
            f"To fix this issue:\n"
            f"  1. Download the model: docling-tools models download-hf-repo {repo_id}\n"
            f"  2. Or remove --artifacts-path to enable auto-download\n"
            f"  3. Or use a different model that exists in your artifacts_path"
        )


def format_prompt_for_vlm(
    prompt: str,
    processor: Any,
    prompt_style: TransformersPromptStyle,
    repo_id: Optional[str] = None,
) -> Optional[str]:
    """Format a prompt according to the specified style.

    This centralizes prompt formatting logic that was previously duplicated
    across different model implementations.

    Args:
        prompt: User prompt text
        processor: Model processor with apply_chat_template method
        prompt_style: Style of prompt formatting to use
        repo_id: Optional model repository ID for model-specific formatting

    Returns:
        Formatted prompt string, or None if prompt_style is NONE
    """
    if prompt_style == TransformersPromptStyle.RAW:
        return prompt
    elif prompt_style == TransformersPromptStyle.NONE:
        return None
    elif repo_id == "microsoft/Phi-4-multimodal-instruct":
        # Special handling for Phi-4
        _log.debug("Using specialized prompt for Phi-4")
        user_prompt_prefix = "<|user|>"
        assistant_prompt = "<|assistant|>"
        prompt_suffix = "<|end|>"
        formatted = (
            f"{user_prompt_prefix}<|image_1|>{prompt}{prompt_suffix}{assistant_prompt}"
        )
        _log.debug(f"Formatted prompt for {repo_id}: {formatted}")
        return formatted
    elif prompt_style == TransformersPromptStyle.CHAT:
        # Standard chat template with image placeholder
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "This is a page from a document."},
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return processor.apply_chat_template(messages, add_generation_prompt=True)
    else:
        raise ValueError(
            f"Unknown prompt style: {prompt_style}. "
            f"Valid values are {', '.join(s.value for s in TransformersPromptStyle)}"
        )
