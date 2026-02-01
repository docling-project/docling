"""Model specifications and presets for VLM stages.

This module defines:
1. VlmModelSpec - Model configuration with runtime-specific overrides
2. StageModelPreset - Preset combining model, runtime, and stage config
3. StagePresetMixin - Mixin for stage options to manage presets
"""

import logging
from typing import Any, ClassVar, Dict, List, Optional, Set

from pydantic import BaseModel, Field

from docling.datamodel.pipeline_options_vlm_model import (
    ResponseFormat,
    TransformersModelType,
    TransformersPromptStyle,
)
from docling.datamodel.vlm_runtime_options import BaseVlmRuntimeOptions
from docling.models.runtimes.base import VlmRuntimeType

_log = logging.getLogger(__name__)


# =============================================================================
# RUNTIME-SPECIFIC MODEL CONFIGURATION
# =============================================================================


class RuntimeModelConfig(BaseModel):
    """Runtime-specific model configuration.

    Allows overriding model settings for specific runtimes.
    For example, MLX might use a different repo_id than Transformers.
    """

    repo_id: Optional[str] = Field(
        default=None, description="Override model repository ID for this runtime"
    )

    revision: Optional[str] = Field(
        default=None, description="Override model revision for this runtime"
    )

    torch_dtype: Optional[str] = Field(
        default=None,
        description="Override torch dtype for this runtime (e.g., 'bfloat16')",
    )

    extra_config: Dict[str, Any] = Field(
        default_factory=dict, description="Additional runtime-specific configuration"
    )

    def merge_with(
        self, base_repo_id: str, base_revision: str = "main"
    ) -> "RuntimeModelConfig":
        """Merge with base configuration.

        Args:
            base_repo_id: Base repository ID
            base_revision: Base revision

        Returns:
            Merged configuration with overrides applied
        """
        return RuntimeModelConfig(
            repo_id=self.repo_id or base_repo_id,
            revision=self.revision or base_revision,
            torch_dtype=self.torch_dtype,
            extra_config=self.extra_config,
        )


class ApiModelConfig(BaseModel):
    """API-specific model configuration.

    For API runtimes, configuration is simpler - just params to send.
    """

    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="API parameters (model name, max_tokens, etc.)",
    )

    def merge_with(self, base_params: Dict[str, Any]) -> "ApiModelConfig":
        """Merge with base parameters.

        Args:
            base_params: Base API parameters

        Returns:
            Merged configuration with overrides applied
        """
        merged_params = {**base_params, **self.params}
        return ApiModelConfig(params=merged_params)


# =============================================================================
# VLM MODEL SPECIFICATION
# =============================================================================


class VlmModelSpec(BaseModel):
    """Specification for a VLM model.

    This defines the model configuration that is independent of the runtime.
    It includes:
    - Default model repository ID
    - Prompt template
    - Response format
    - Runtime-specific overrides
    """

    name: str = Field(description="Human-readable model name")

    default_repo_id: str = Field(description="Default HuggingFace repository ID")

    revision: str = Field(default="main", description="Default model revision")

    prompt: str = Field(description="Prompt template for this model")

    response_format: ResponseFormat = Field(
        description="Expected response format from the model"
    )

    supported_runtimes: Optional[Set[VlmRuntimeType]] = Field(
        default=None, description="Set of supported runtimes (None = all supported)"
    )

    runtime_overrides: Dict[VlmRuntimeType, RuntimeModelConfig] = Field(
        default_factory=dict, description="Runtime-specific configuration overrides"
    )

    api_overrides: Dict[VlmRuntimeType, ApiModelConfig] = Field(
        default_factory=dict, description="API-specific configuration overrides"
    )

    trust_remote_code: bool = Field(
        default=False, description="Whether to trust remote code for this model"
    )

    stop_strings: List[str] = Field(
        default_factory=list, description="Stop strings for generation"
    )

    max_new_tokens: int = Field(
        default=4096, description="Maximum number of new tokens to generate"
    )

    def get_repo_id(self, runtime_type: VlmRuntimeType) -> str:
        """Get the repository ID for a specific runtime.

        Args:
            runtime_type: The runtime type

        Returns:
            Repository ID (with runtime override if applicable)
        """
        if runtime_type in self.runtime_overrides:
            override = self.runtime_overrides[runtime_type]
            return override.repo_id or self.default_repo_id
        return self.default_repo_id

    def get_revision(self, runtime_type: VlmRuntimeType) -> str:
        """Get the model revision for a specific runtime.

        Args:
            runtime_type: The runtime type

        Returns:
            Model revision (with runtime override if applicable)
        """
        if runtime_type in self.runtime_overrides:
            override = self.runtime_overrides[runtime_type]
            return override.revision or self.revision
        return self.revision

    def get_api_params(self, runtime_type: VlmRuntimeType) -> Dict[str, Any]:
        """Get API parameters for a specific runtime.

        Args:
            runtime_type: The runtime type

        Returns:
            API parameters (with runtime override if applicable)
        """
        base_params = {"model": self.default_repo_id}

        if runtime_type in self.api_overrides:
            override = self.api_overrides[runtime_type]
            return override.merge_with(base_params).params

        return base_params

    def is_runtime_supported(self, runtime_type: VlmRuntimeType) -> bool:
        """Check if a runtime is supported by this model.

        Args:
            runtime_type: The runtime type to check

        Returns:
            True if supported, False otherwise
        """
        if self.supported_runtimes is None:
            return True
        return runtime_type in self.supported_runtimes

    def get_runtime_config(self, runtime_type: VlmRuntimeType) -> RuntimeModelConfig:
        """Get RuntimeModelConfig for a specific runtime type.

        This is the single source of truth for generating runtime-specific
        configuration from the model spec.

        Args:
            runtime_type: The runtime type to get config for

        Returns:
            RuntimeModelConfig with repo_id, revision, and runtime-specific extra_config
        """
        # Get repo_id and revision (with runtime-specific overrides if present)
        repo_id = self.get_repo_id(runtime_type)
        revision = self.get_revision(runtime_type)

        # Get runtime-specific extra_config
        extra_config = {}
        if runtime_type in self.runtime_overrides:
            extra_config = self.runtime_overrides[runtime_type].extra_config.copy()

        return RuntimeModelConfig(
            repo_id=repo_id,
            revision=revision,
            extra_config=extra_config,
        )


# =============================================================================
# STAGE PRESET SYSTEM
# =============================================================================


class StageModelPreset(BaseModel):
    """A preset configuration combining stage, model, and prompt.

    Presets provide convenient named configurations that users can
    reference by ID instead of manually configuring everything.
    """

    preset_id: str = Field(
        description="Simple preset identifier (e.g., 'smolvlm', 'granite')"
    )

    name: str = Field(description="Human-readable preset name")

    description: str = Field(description="Description of what this preset does")

    model_spec: VlmModelSpec = Field(description="Model specification for this preset")

    scale: float = Field(default=2.0, description="Image scaling factor")

    max_size: Optional[int] = Field(default=None, description="Maximum image dimension")

    default_runtime_type: VlmRuntimeType = Field(
        default=VlmRuntimeType.AUTO_INLINE,
        description="Default runtime to use with this preset",
    )

    stage_options: Dict[str, Any] = Field(
        default_factory=dict, description="Additional stage-specific options"
    )

    @property
    def supported_runtimes(self) -> Set[VlmRuntimeType]:
        """Get supported runtimes from model spec."""
        if self.model_spec.supported_runtimes is None:
            return set(VlmRuntimeType)
        return self.model_spec.supported_runtimes


class StagePresetMixin:
    """Mixin for stage options classes that support presets.

    Each stage options class that uses this mixin manages its own presets.
    This is more decentralized than a global registry.

    Usage:
        class MyStageOptions(StagePresetMixin, BaseModel):
            ...

        # Register presets
        MyStageOptions.register_preset(preset1)
        MyStageOptions.register_preset(preset2)

        # Use presets
        options = MyStageOptions.from_preset("preset1")
    """

    # Class variable to store presets for this specific stage
    # Note: Each subclass gets its own _presets dict via __init_subclass__
    _presets: ClassVar[Dict[str, StageModelPreset]]

    def __init_subclass__(cls, **kwargs):
        """Initialize each subclass with its own preset registry.

        This ensures that each stage options class has an isolated preset
        registry, preventing namespace collisions across different stages.
        """
        super().__init_subclass__(**kwargs)
        # Each subclass gets its own _presets dictionary
        cls._presets = {}

    @classmethod
    def register_preset(cls, preset: StageModelPreset) -> None:
        """Register a preset for this stage options class.

        Args:
            preset: The preset to register

        Note:
            If preset ID already registered, it will be silently skipped.
            This allows for idempotent registration at module import time.
        """
        if preset.preset_id not in cls._presets:
            cls._presets[preset.preset_id] = preset
        else:
            _log.error(
                f"Preset '{preset.preset_id}' already registered for {cls.__name__}"
            )

    @classmethod
    def get_preset(cls, preset_id: str) -> StageModelPreset:
        """Get a specific preset.

        Args:
            preset_id: The preset identifier

        Returns:
            The requested preset

        Raises:
            KeyError: If preset not found
        """
        if preset_id not in cls._presets:
            raise KeyError(
                f"Preset '{preset_id}' not found for {cls.__name__}. "
                f"Available presets: {list(cls._presets.keys())}"
            )
        return cls._presets[preset_id]

    @classmethod
    def list_presets(cls) -> List[StageModelPreset]:
        """List all presets for this stage.

        Returns:
            List of presets
        """
        return list(cls._presets.values())

    @classmethod
    def list_preset_ids(cls) -> List[str]:
        """List all preset IDs for this stage.

        Returns:
            List of preset IDs
        """
        return list(cls._presets.keys())

    @classmethod
    def get_preset_info(cls) -> List[Dict[str, str]]:
        """Get summary info for all presets (useful for CLI).

        Returns:
            List of dicts with preset_id, name, description, model
        """
        return [
            {
                "preset_id": p.preset_id,
                "name": p.name,
                "description": p.description,
                "model": p.model_spec.name,
                "default_runtime": p.default_runtime_type.value,
            }
            for p in cls._presets.values()
        ]

    @classmethod
    def from_preset(
        cls,
        preset_id: str,
        runtime_options: Optional[BaseVlmRuntimeOptions] = None,
        **overrides,
    ):
        """Create options from a registered preset.

        Args:
            preset_id: The preset identifier
            runtime_options: Optional runtime override
            **overrides: Additional option overrides

        Returns:
            Instance of the stage options class
        """
        from docling.datamodel.vlm_runtime_options import (
            ApiVlmRuntimeOptions,
            AutoInlineVlmRuntimeOptions,
            MlxVlmRuntimeOptions,
            TransformersVlmRuntimeOptions,
            VllmVlmRuntimeOptions,
        )

        preset = cls.get_preset(preset_id)

        # Create runtime options if not provided
        if runtime_options is None:
            if preset.default_runtime_type == VlmRuntimeType.AUTO_INLINE:
                runtime_options = AutoInlineVlmRuntimeOptions()
            elif VlmRuntimeType.is_api_variant(preset.default_runtime_type):
                runtime_options = ApiVlmRuntimeOptions(
                    runtime_type=preset.default_runtime_type
                )
            elif preset.default_runtime_type == VlmRuntimeType.TRANSFORMERS:
                runtime_options = TransformersVlmRuntimeOptions()
            elif preset.default_runtime_type == VlmRuntimeType.MLX:
                runtime_options = MlxVlmRuntimeOptions()
            elif preset.default_runtime_type == VlmRuntimeType.VLLM:
                runtime_options = VllmVlmRuntimeOptions()
            else:
                runtime_options = AutoInlineVlmRuntimeOptions()

        # Create instance with preset values
        # Type ignore because cls is the concrete options class, not the mixin
        instance = cls(  # type: ignore[call-arg]
            model_spec=preset.model_spec,
            runtime_options=runtime_options,
            scale=preset.scale,
            max_size=preset.max_size,
            **preset.stage_options,
        )

        # Apply overrides
        for key, value in overrides.items():
            setattr(instance, key, value)

        return instance


# =============================================================================
# PRESET DEFINITIONS
# =============================================================================

# -----------------------------------------------------------------------------
# SHARED MODEL SPECS (for reuse across multiple stages)
# -----------------------------------------------------------------------------

# Shared Granite Docling model spec used across VLM_CONVERT and CODE_FORMULA stages
# Note: prompt and response_format are intentionally excluded here as they vary per stage
GRANITE_DOCLING_MODEL_SPEC_BASE = {
    "name": "Granite-Docling-258M",
    "default_repo_id": "ibm-granite/granite-docling-258M",
    "stop_strings": ["</doctag>", "<|end_of_text|>"],
    "max_new_tokens": 8192,
    "runtime_overrides": {
        VlmRuntimeType.MLX: RuntimeModelConfig(
            repo_id="ibm-granite/granite-docling-258M-mlx"
        ),
        VlmRuntimeType.TRANSFORMERS: RuntimeModelConfig(
            extra_config={
                "transformers_model_type": TransformersModelType.AUTOMODEL_IMAGETEXTTOTEXT,
                "extra_generation_config": {"skip_special_tokens": False},
            }
        ),
    },
    "api_overrides": {
        VlmRuntimeType.API_OLLAMA: ApiModelConfig(
            params={"model": "ibm/granite-docling:258m"}
        ),
    },
}

# -----------------------------------------------------------------------------
# VLM_CONVERT PRESETS (for full page conversion)
# -----------------------------------------------------------------------------

VLM_CONVERT_SMOLDOCLING = StageModelPreset(
    preset_id="smoldocling",
    name="SmolDocling",
    description="Lightweight DocTags model optimized for document conversion (256M parameters)",
    model_spec=VlmModelSpec(
        name="SmolDocling-256M",
        default_repo_id="docling-project/SmolDocling-256M-preview",
        prompt="Convert this page to docling.",
        response_format=ResponseFormat.DOCTAGS,
        stop_strings=["</doctag>", "<end_of_utterance>"],
        runtime_overrides={
            VlmRuntimeType.MLX: RuntimeModelConfig(
                repo_id="docling-project/SmolDocling-256M-preview-mlx-bf16"
            ),
            VlmRuntimeType.TRANSFORMERS: RuntimeModelConfig(
                torch_dtype="bfloat16",
                extra_config={
                    "transformers_model_type": TransformersModelType.AUTOMODEL_IMAGETEXTTOTEXT,
                },
            ),
        },
    ),
    scale=2.0,
    default_runtime_type=VlmRuntimeType.AUTO_INLINE,
)

VLM_CONVERT_GRANITE_DOCLING = StageModelPreset(
    preset_id="granite_docling",
    name="Granite-Docling",
    description="IBM Granite DocTags model for document conversion (258M parameters)",
    model_spec=VlmModelSpec(
        **GRANITE_DOCLING_MODEL_SPEC_BASE,
        prompt="Convert this page to docling.",
        response_format=ResponseFormat.DOCTAGS,
    ),
    scale=2.0,
    default_runtime_type=VlmRuntimeType.AUTO_INLINE,
)

VLM_CONVERT_DEEPSEEK_OCR = StageModelPreset(
    preset_id="deepseek_ocr",
    name="DeepSeek-OCR",
    description="DeepSeek OCR model via Ollama for document conversion (3B parameters)",
    model_spec=VlmModelSpec(
        name="DeepSeek-OCR-3B",
        default_repo_id="deepseek-ocr:3b",  # Ollama model name
        prompt="<|grounding|>Convert the document to markdown. ",
        response_format=ResponseFormat.DEEPSEEKOCR_MARKDOWN,
        supported_runtimes={VlmRuntimeType.API_OLLAMA},
        api_overrides={
            VlmRuntimeType.API_OLLAMA: ApiModelConfig(
                params={"model": "deepseek-ocr:3b", "max_tokens": 4096}
            ),
        },
    ),
    scale=2.0,
    default_runtime_type=VlmRuntimeType.API_OLLAMA,
)

VLM_CONVERT_GRANITE_VISION = StageModelPreset(
    preset_id="granite_vision",
    name="Granite-Vision",
    description="IBM Granite Vision model for markdown conversion (2B parameters)",
    model_spec=VlmModelSpec(
        name="Granite-Vision-3.3-2B",
        default_repo_id="ibm-granite/granite-vision-3.3-2b",
        prompt="Convert this page to markdown. Do not miss any text and only output the bare markdown!",
        response_format=ResponseFormat.MARKDOWN,
        supported_runtimes={
            VlmRuntimeType.TRANSFORMERS,
            VlmRuntimeType.API_OLLAMA,
            VlmRuntimeType.API_LMSTUDIO,
        },
        runtime_overrides={
            VlmRuntimeType.TRANSFORMERS: RuntimeModelConfig(
                extra_config={
                    "transformers_model_type": TransformersModelType.AUTOMODEL_IMAGETEXTTOTEXT,
                }
            ),
        },
        api_overrides={
            VlmRuntimeType.API_OLLAMA: ApiModelConfig(
                params={"model": "granite3.3-vision:2b"}
            ),
        },
    ),
    scale=2.0,
    default_runtime_type=VlmRuntimeType.AUTO_INLINE,
)

VLM_CONVERT_PIXTRAL = StageModelPreset(
    preset_id="pixtral",
    name="Pixtral-12B",
    description="Mistral Pixtral model for markdown conversion (12B parameters)",
    model_spec=VlmModelSpec(
        name="Pixtral-12B",
        default_repo_id="mistral-community/pixtral-12b",
        prompt="Convert this page to markdown. Do not miss any text and only output the bare markdown!",
        response_format=ResponseFormat.MARKDOWN,
        runtime_overrides={
            VlmRuntimeType.MLX: RuntimeModelConfig(
                repo_id="mlx-community/pixtral-12b-bf16"
            ),
            VlmRuntimeType.TRANSFORMERS: RuntimeModelConfig(
                extra_config={
                    "transformers_model_type": TransformersModelType.AUTOMODEL_VISION2SEQ,
                }
            ),
        },
    ),
    scale=2.0,
    default_runtime_type=VlmRuntimeType.AUTO_INLINE,
)

VLM_CONVERT_GOT_OCR = StageModelPreset(
    preset_id="got_ocr",
    name="GOT-OCR-2.0",
    description="GOT OCR 2.0 model for markdown conversion",
    model_spec=VlmModelSpec(
        name="GOT-OCR-2.0",
        default_repo_id="stepfun-ai/GOT-OCR-2.0-hf",
        prompt="",
        response_format=ResponseFormat.MARKDOWN,
        supported_runtimes={VlmRuntimeType.TRANSFORMERS},
        stop_strings=["<|im_end|>"],
        runtime_overrides={
            VlmRuntimeType.TRANSFORMERS: RuntimeModelConfig(
                extra_config={
                    "transformers_model_type": TransformersModelType.AUTOMODEL_IMAGETEXTTOTEXT,
                    "transformers_prompt_style": TransformersPromptStyle.NONE,
                    "extra_processor_kwargs": {"format": True},
                }
            ),
        },
    ),
    scale=2.0,
    default_runtime_type=VlmRuntimeType.TRANSFORMERS,
)

# -----------------------------------------------------------------------------
# PICTURE_DESCRIPTION PRESETS (for image captioning/description)
# -----------------------------------------------------------------------------

PICTURE_DESC_SMOLVLM = StageModelPreset(
    preset_id="smolvlm",
    name="SmolVLM-256M",
    description="Lightweight vision-language model for image descriptions (256M parameters)",
    model_spec=VlmModelSpec(
        name="SmolVLM-256M-Instruct",
        default_repo_id="HuggingFaceTB/SmolVLM-256M-Instruct",
        prompt="Describe this image in a few sentences.",
        response_format=ResponseFormat.PLAINTEXT,
        runtime_overrides={
            VlmRuntimeType.MLX: RuntimeModelConfig(
                repo_id="moot20/SmolVLM-256M-Instruct-MLX"
            ),
            VlmRuntimeType.TRANSFORMERS: RuntimeModelConfig(
                torch_dtype="bfloat16",
                extra_config={
                    "transformers_model_type": TransformersModelType.AUTOMODEL_IMAGETEXTTOTEXT,
                },
            ),
        },
        api_overrides={
            VlmRuntimeType.API_LMSTUDIO: ApiModelConfig(
                params={"model": "smolvlm-256m-instruct"}
            ),
        },
    ),
    scale=2.0,
    default_runtime_type=VlmRuntimeType.AUTO_INLINE,
    stage_options={
        "picture_area_threshold": 0.05,
    },
)

PICTURE_DESC_GRANITE_VISION = StageModelPreset(
    preset_id="granite_vision",
    name="Granite-Vision-3.3-2B",
    description="IBM Granite Vision model for detailed image descriptions (2B parameters)",
    model_spec=VlmModelSpec(
        name="Granite-Vision-3.3-2B",
        default_repo_id="ibm-granite/granite-vision-3.3-2b",
        prompt="What is shown in this image?",
        response_format=ResponseFormat.PLAINTEXT,
        supported_runtimes={
            VlmRuntimeType.TRANSFORMERS,
            VlmRuntimeType.API_OLLAMA,
            VlmRuntimeType.API_LMSTUDIO,
        },
        runtime_overrides={
            VlmRuntimeType.TRANSFORMERS: RuntimeModelConfig(
                extra_config={
                    "transformers_model_type": TransformersModelType.AUTOMODEL_IMAGETEXTTOTEXT,
                }
            ),
        },
        api_overrides={
            VlmRuntimeType.API_OLLAMA: ApiModelConfig(
                params={"model": "ibm/granite3.3-vision:2b"}
            ),
        },
    ),
    scale=2.0,
    default_runtime_type=VlmRuntimeType.AUTO_INLINE,
    stage_options={
        "picture_area_threshold": 0.05,
    },
)

PICTURE_DESC_PIXTRAL = StageModelPreset(
    preset_id="pixtral",
    name="Pixtral-12B",
    description="Mistral Pixtral model for detailed image descriptions (12B parameters)",
    model_spec=VlmModelSpec(
        name="Pixtral-12B",
        default_repo_id="mistral-community/pixtral-12b",
        prompt="Describe this image in detail.",
        response_format=ResponseFormat.PLAINTEXT,
        runtime_overrides={
            VlmRuntimeType.MLX: RuntimeModelConfig(
                repo_id="mlx-community/pixtral-12b-bf16"
            ),
            VlmRuntimeType.TRANSFORMERS: RuntimeModelConfig(
                extra_config={
                    "transformers_model_type": TransformersModelType.AUTOMODEL_VISION2SEQ,
                }
            ),
        },
    ),
    scale=2.0,
    default_runtime_type=VlmRuntimeType.AUTO_INLINE,
    stage_options={
        "picture_area_threshold": 0.05,
    },
)

PICTURE_DESC_QWEN = StageModelPreset(
    preset_id="qwen",
    name="Qwen2.5-VL-3B",
    description="Qwen vision-language model for image descriptions (3B parameters)",
    model_spec=VlmModelSpec(
        name="Qwen2.5-VL-3B-Instruct",
        default_repo_id="Qwen/Qwen2.5-VL-3B-Instruct",
        prompt="Describe this image.",
        response_format=ResponseFormat.PLAINTEXT,
        runtime_overrides={
            VlmRuntimeType.MLX: RuntimeModelConfig(
                repo_id="mlx-community/Qwen2.5-VL-3B-Instruct-bf16"
            ),
            VlmRuntimeType.TRANSFORMERS: RuntimeModelConfig(
                extra_config={
                    "transformers_model_type": TransformersModelType.AUTOMODEL_IMAGETEXTTOTEXT,
                }
            ),
        },
    ),
    scale=2.0,
    default_runtime_type=VlmRuntimeType.AUTO_INLINE,
    stage_options={
        "picture_area_threshold": 0.05,
    },
)

# -----------------------------------------------------------------------------
# CODE_FORMULA PRESETS (for code and formula extraction)
# -----------------------------------------------------------------------------

CODE_FORMULA_CODEFORMULAV2 = StageModelPreset(
    preset_id="codeformulav2",
    name="CodeFormulaV2",
    description="Specialized model for code and formula extraction",
    model_spec=VlmModelSpec(
        name="CodeFormulaV2",
        default_repo_id="docling-project/CodeFormulaV2",
        prompt="",
        response_format=ResponseFormat.PLAINTEXT,
    ),
    scale=2.0,
    default_runtime_type=VlmRuntimeType.AUTO_INLINE,
)

CODE_FORMULA_GRANITE_DOCLING = StageModelPreset(
    preset_id="granite_docling",
    name="Granite-Docling-CodeFormula",
    description="IBM Granite Docling model for code and formula extraction (258M parameters)",
    model_spec=VlmModelSpec(
        **GRANITE_DOCLING_MODEL_SPEC_BASE,
        prompt="",
        response_format=ResponseFormat.PLAINTEXT,
    ),
    scale=2.0,
    default_runtime_type=VlmRuntimeType.AUTO_INLINE,
)
