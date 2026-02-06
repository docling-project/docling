"""Model specifications and presets for VLM stages.

This module defines:
1. VlmModelSpec - Model configuration with engine-specific overrides
2. StageModelPreset - Preset combining model, engine, and stage config
3. StagePresetMixin - Mixin for stage options to manage presets
"""

import logging
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Set

from pydantic import BaseModel, Field

from docling.datamodel.pipeline_options_vlm_model import (
    ResponseFormat,
    TransformersModelType,
    TransformersPromptStyle,
)
from docling.datamodel.vlm_engine_options import BaseVlmEngineOptions
from docling.models.inference_engines.object_detection.base import (
    ObjectDetectionEngineType,
)
from docling.models.inference_engines.vlm.base import VlmEngineType

if TYPE_CHECKING:
    from docling.datamodel.object_detection_engine_options import (
        BaseObjectDetectionEngineOptions,
    )

_log = logging.getLogger(__name__)


# =============================================================================
# ENGINE-SPECIFIC MODEL CONFIGURATION
# =============================================================================


class EngineModelConfig(BaseModel):
    """Engine-specific model configuration.

    Allows overriding model settings for specific engines.
    For example, MLX might use a different repo_id than Transformers.
    """

    repo_id: Optional[str] = Field(
        default=None, description="Override model repository ID for this engine"
    )

    revision: Optional[str] = Field(
        default=None, description="Override model revision for this engine"
    )

    torch_dtype: Optional[str] = Field(
        default=None,
        description="Override torch dtype for this engine (e.g., 'bfloat16')",
    )

    extra_config: Dict[str, Any] = Field(
        default_factory=dict, description="Additional engine-specific configuration"
    )

    def merge_with(
        self, base_repo_id: str, base_revision: str = "main"
    ) -> "EngineModelConfig":
        """Merge with base configuration.

        Args:
            base_repo_id: Base repository ID
            base_revision: Base revision

        Returns:
            Merged configuration with overrides applied
        """
        return EngineModelConfig(
            repo_id=self.repo_id or base_repo_id,
            revision=self.revision or base_revision,
            torch_dtype=self.torch_dtype,
            extra_config=self.extra_config,
        )


class ApiModelConfig(BaseModel):
    """API-specific model configuration.

    For API engines, configuration is simpler - just params to send.
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

    This defines the model configuration that is independent of the engine.
    It includes:
    - Default model repository ID
    - Prompt template
    - Response format
    - Engine-specific overrides
    """

    name: str = Field(description="Human-readable model name")

    default_repo_id: str = Field(description="Default HuggingFace repository ID")

    revision: str = Field(default="main", description="Default model revision")

    prompt: str = Field(description="Prompt template for this model")

    response_format: ResponseFormat = Field(
        description="Expected response format from the model"
    )

    supported_engines: Optional[Set[VlmEngineType]] = Field(
        default=None, description="Set of supported engines (None = all supported)"
    )

    engine_overrides: Dict[VlmEngineType, EngineModelConfig] = Field(
        default_factory=dict, description="Engine-specific configuration overrides"
    )

    api_overrides: Dict[VlmEngineType, ApiModelConfig] = Field(
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

    def get_repo_id(self, engine_type: VlmEngineType) -> str:
        """Get the repository ID for a specific engine.

        Args:
            engine_type: The engine type

        Returns:
            Repository ID (with engine override if applicable)
        """
        if engine_type in self.engine_overrides:
            override = self.engine_overrides[engine_type]
            return override.repo_id or self.default_repo_id
        return self.default_repo_id

    def get_revision(self, engine_type: VlmEngineType) -> str:
        """Get the model revision for a specific engine.

        Args:
            engine_type: The engine type

        Returns:
            Model revision (with engine override if applicable)
        """
        if engine_type in self.engine_overrides:
            override = self.engine_overrides[engine_type]
            return override.revision or self.revision
        return self.revision

    def get_api_params(self, engine_type: VlmEngineType) -> Dict[str, Any]:
        """Get API parameters for a specific engine.

        Args:
            engine_type: The engine type

        Returns:
            API parameters (with engine override if applicable)
        """
        base_params = {"model": self.default_repo_id}

        if engine_type in self.api_overrides:
            override = self.api_overrides[engine_type]
            return override.merge_with(base_params).params

        return base_params

    def is_engine_supported(self, engine_type: VlmEngineType) -> bool:
        """Check if an engine is supported by this model.

        Args:
            engine_type: The engine type to check

        Returns:
            True if supported, False otherwise
        """
        if self.supported_engines is None:
            return True
        return engine_type in self.supported_engines

    def get_engine_config(self, engine_type: VlmEngineType) -> EngineModelConfig:
        """Get EngineModelConfig for a specific engine type.

        This is the single source of truth for generating engine-specific
        configuration from the model spec.

        Args:
            engine_type: The engine type to get config for

        Returns:
            EngineModelConfig with repo_id, revision, and engine-specific extra_config
        """
        # Get repo_id and revision (with engine-specific overrides if present)
        repo_id = self.get_repo_id(engine_type)
        revision = self.get_revision(engine_type)

        # Get engine-specific extra_config
        extra_config = {}
        if engine_type in self.engine_overrides:
            extra_config = self.engine_overrides[engine_type].extra_config.copy()

        return EngineModelConfig(
            repo_id=repo_id,
            revision=revision,
            extra_config=extra_config,
        )

    def has_explicit_engine_export(self, engine_type: VlmEngineType) -> bool:
        """Check if this model has an explicit export for the given engine.

        An explicit export means either:
        1. The engine has a different repo_id in engine_overrides, OR
        2. The engine is explicitly listed in supported_engines (not None)

        This is used by auto_inline to determine if it should attempt to use
        a specific engine. For example, MLX should only be used if there's
        an actual MLX export available (different repo_id) or if the model
        explicitly declares MLX support.

        Args:
            engine_type: The engine type to check

        Returns:
            True if there's an explicit export, False otherwise

        Examples:
            >>> # Model with MLX export (different repo_id)
            >>> spec = VlmModelSpec(
            ...     name="Test",
            ...     default_repo_id="org/model",
            ...     engine_overrides={
            ...         VlmEngineType.MLX: EngineModelConfig(repo_id="org/model-mlx")
            ...     }
            ... )
            >>> spec.has_explicit_engine_export(VlmEngineType.MLX)
            True

            >>> # Model without MLX export (same repo_id or no override)
            >>> spec = VlmModelSpec(name="Test", default_repo_id="org/model")
            >>> spec.has_explicit_engine_export(VlmEngineType.MLX)
            False

            >>> # Model with explicit supported_engines
            >>> spec = VlmModelSpec(
            ...     name="Test",
            ...     default_repo_id="org/model",
            ...     supported_engines={VlmEngineType.MLX}
            ... )
            >>> spec.has_explicit_engine_export(VlmEngineType.MLX)
            True
        """
        # If supported_engines is explicitly set and includes this engine
        if self.supported_engines is not None:
            return engine_type in self.supported_engines

        # Check if there's a different repo_id for this engine
        if engine_type in self.engine_overrides:
            override = self.engine_overrides[engine_type]
            if (
                override.repo_id is not None
                and override.repo_id != self.default_repo_id
            ):
                return True

        return False


# =============================================================================
# OBJECT DETECTION MODEL SPECIFICATION
# =============================================================================


class ObjectDetectionModelSpec(BaseModel):
    """Specification for an object detection model.

    Simpler than VlmModelSpec - no prompts, no preprocessing params.
    Preprocessing comes from HuggingFace preprocessor configs.
    Model files are assumed to be at the root of the HuggingFace repo.
    """

    name: str = Field(description="Human-readable model name")

    repo_id: str = Field(description="Default HuggingFace repository ID")

    revision: str = Field(default="main", description="Default model revision")

    engine_overrides: Dict["ObjectDetectionEngineType", EngineModelConfig] = Field(
        default_factory=dict,
        description="Engine-specific configuration overrides",
    )

    def get_engine_config(
        self, engine_type: "ObjectDetectionEngineType"
    ) -> EngineModelConfig:
        """Get EngineModelConfig for a specific object-detection engine.

        Args:
            engine_type: The engine type being requested

        Returns:
            EngineModelConfig populated with repo/revision and engine overrides
        """
        override = self.engine_overrides.get(engine_type)
        if override is not None:
            return override.merge_with(self.repo_id, self.revision)
        return EngineModelConfig(repo_id=self.repo_id, revision=self.revision)

    def get_repo_id(self, engine_type: "ObjectDetectionEngineType") -> str:
        """Get repository ID for specific engine.

        Args:
            engine_type: The engine type

        Returns:
            Repository ID (with engine override if applicable)
        """
        override = self.engine_overrides.get(engine_type)
        if override and override.repo_id:
            return override.repo_id
        return self.repo_id

    def get_revision(self, engine_type: "ObjectDetectionEngineType") -> str:
        """Get revision for specific engine.

        Args:
            engine_type: The engine type

        Returns:
            Model revision (with engine override if applicable)
        """
        override = self.engine_overrides.get(engine_type)
        if override and override.revision:
            return override.revision
        return self.revision


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

    default_engine_type: VlmEngineType = Field(
        default=VlmEngineType.AUTO_INLINE,
        description="Default engine to use with this preset",
    )

    stage_options: Dict[str, Any] = Field(
        default_factory=dict, description="Additional stage-specific options"
    )

    @property
    def supported_engines(self) -> Set[VlmEngineType]:
        """Get supported engines from model spec."""
        if self.model_spec.supported_engines is None:
            return set(VlmEngineType)
        return self.model_spec.supported_engines


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
                "default_engine": p.default_engine_type.value,
            }
            for p in cls._presets.values()
        ]

    @classmethod
    def from_preset(
        cls,
        preset_id: str,
        engine_options: Optional[BaseVlmEngineOptions] = None,
        **overrides,
    ):
        """Create options from a registered preset.

        Args:
            preset_id: The preset identifier
            engine_options: Optional engine override
            **overrides: Additional option overrides

        Returns:
            Instance of the stage options class
        """
        from docling.datamodel.vlm_engine_options import (
            ApiVlmEngineOptions,
            AutoInlineVlmEngineOptions,
            MlxVlmEngineOptions,
            TransformersVlmEngineOptions,
            VllmVlmEngineOptions,
        )

        preset = cls.get_preset(preset_id)

        # Create engine options if not provided
        if engine_options is None:
            if preset.default_engine_type == VlmEngineType.AUTO_INLINE:
                engine_options = AutoInlineVlmEngineOptions()
            elif VlmEngineType.is_api_variant(preset.default_engine_type):
                engine_options = ApiVlmEngineOptions(
                    engine_type=preset.default_engine_type
                )
            elif preset.default_engine_type == VlmEngineType.TRANSFORMERS:
                engine_options = TransformersVlmEngineOptions()
            elif preset.default_engine_type == VlmEngineType.MLX:
                engine_options = MlxVlmEngineOptions()
            elif preset.default_engine_type == VlmEngineType.VLLM:
                engine_options = VllmVlmEngineOptions()
            else:
                engine_options = AutoInlineVlmEngineOptions()

        # Create instance with preset values
        # Type ignore because cls is the concrete options class, not the mixin
        instance = cls(  # type: ignore[call-arg]
            model_spec=preset.model_spec,
            engine_options=engine_options,
            scale=preset.scale,
            max_size=preset.max_size,
            **preset.stage_options,
        )

        # Apply overrides
        for key, value in overrides.items():
            setattr(instance, key, value)

        return instance


class ObjectDetectionStagePreset(BaseModel):
    """Preset definition for object detection-powered stages."""

    preset_id: str = Field(description="Preset identifier")
    name: str = Field(description="Human-readable preset name")
    description: str = Field(description="Description of this preset")
    model_spec: ObjectDetectionModelSpec = Field(
        description="Object detection model specification"
    )
    default_engine_type: ObjectDetectionEngineType = Field(
        default=ObjectDetectionEngineType.ONNXRUNTIME,
        description="Default inference engine to use",
    )
    stage_options: Dict[str, Any] = Field(
        default_factory=dict, description="Additional stage-specific defaults"
    )


class ObjectDetectionStagePresetMixin:
    """Mixin to enable preset loading for object detection stages."""

    _presets: ClassVar[Dict[str, ObjectDetectionStagePreset]]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._presets = {}

    @classmethod
    def register_preset(cls, preset: ObjectDetectionStagePreset) -> None:
        if preset.preset_id not in cls._presets:
            cls._presets[preset.preset_id] = preset
        else:
            _log.error(
                f"Preset '{preset.preset_id}' already registered for {cls.__name__}"
            )

    @classmethod
    def get_preset(cls, preset_id: str) -> ObjectDetectionStagePreset:
        if preset_id not in cls._presets:
            raise KeyError(
                f"Preset '{preset_id}' not found for {cls.__name__}. "
                f"Available presets: {list(cls._presets.keys())}"
            )
        return cls._presets[preset_id]

    @classmethod
    def list_presets(cls) -> List[ObjectDetectionStagePreset]:
        return list(cls._presets.values())

    @classmethod
    def list_preset_ids(cls) -> List[str]:
        return list(cls._presets.keys())

    @classmethod
    def get_preset_info(cls) -> List[Dict[str, str]]:
        return [
            {
                "preset_id": p.preset_id,
                "name": p.name,
                "description": p.description,
                "model": p.model_spec.name,
                "default_engine": p.default_engine_type.value,
            }
            for p in cls._presets.values()
        ]

    @classmethod
    def from_preset(
        cls,
        preset_id: str,
        engine_options: Optional["BaseObjectDetectionEngineOptions"] = None,
        **overrides: Any,
    ):
        from docling.datamodel.object_detection_engine_options import (
            OnnxRuntimeObjectDetectionEngineOptions,
            TransformersObjectDetectionEngineOptions,
        )

        preset = cls.get_preset(preset_id)

        if engine_options is None:
            if preset.default_engine_type == ObjectDetectionEngineType.ONNXRUNTIME:
                engine_options = OnnxRuntimeObjectDetectionEngineOptions()
            elif preset.default_engine_type == ObjectDetectionEngineType.TRANSFORMERS:
                engine_options = TransformersObjectDetectionEngineOptions()
            else:
                raise ValueError(
                    f"Unsupported engine type {preset.default_engine_type} for presets"
                )

        instance = cls(  # type: ignore[call-arg]
            model_spec=preset.model_spec,
            engine_options=engine_options,
            **preset.stage_options,
        )

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
    "engine_overrides": {
        VlmEngineType.MLX: EngineModelConfig(
            repo_id="ibm-granite/granite-docling-258M-mlx"
        ),
        VlmEngineType.TRANSFORMERS: EngineModelConfig(
            extra_config={
                "transformers_model_type": TransformersModelType.AUTOMODEL_IMAGETEXTTOTEXT,
                "extra_generation_config": {"skip_special_tokens": False},
            }
        ),
    },
    "api_overrides": {
        VlmEngineType.API_OLLAMA: ApiModelConfig(
            params={"model": "ibm/granite-docling:258m"}
        ),
    },
}

# Shared Pixtral model spec used across VLM_CONVERT and PICTURE_DESCRIPTION stages
PIXTRAL_MODEL_SPEC_BASE = {
    "name": "Pixtral-12B",
    "default_repo_id": "mistral-community/pixtral-12b",
    "engine_overrides": {
        VlmEngineType.MLX: EngineModelConfig(repo_id="mlx-community/pixtral-12b-bf16"),
        VlmEngineType.TRANSFORMERS: EngineModelConfig(
            extra_config={
                "transformers_model_type": TransformersModelType.AUTOMODEL_VISION2SEQ,
            }
        ),
    },
}

# Shared Granite Vision model spec used across VLM_CONVERT and PICTURE_DESCRIPTION stages
GRANITE_VISION_MODEL_SPEC_BASE = {
    "name": "Granite-Vision-3.3-2B",
    "default_repo_id": "ibm-granite/granite-vision-3.3-2b",
    "supported_engines": {
        VlmEngineType.TRANSFORMERS,
        VlmEngineType.VLLM,
        VlmEngineType.API_OLLAMA,
        VlmEngineType.API_LMSTUDIO,
    },
    "engine_overrides": {
        VlmEngineType.TRANSFORMERS: EngineModelConfig(
            extra_config={
                "transformers_model_type": TransformersModelType.AUTOMODEL_IMAGETEXTTOTEXT,
            }
        ),
    },
    "api_overrides": {
        VlmEngineType.API_OLLAMA: ApiModelConfig(
            params={"model": "granite3.3-vision:2b"}
        ),
    },
}

# -----------------------------------------------------------------------------
# OBJECT DETECTION PRESETS
# -----------------------------------------------------------------------------

OBJECT_DETECTION_LAYOUT_HERON = ObjectDetectionStagePreset(
    preset_id="layout_heron_default",
    name="Layout Heron",
    description="RT-DETR layout-heron model (ResNet50)",
    model_spec=ObjectDetectionModelSpec(
        name="layout_heron",
        repo_id="docling-project/docling-layout-heron",
        revision="main",
    ),
    default_engine_type=ObjectDetectionEngineType.TRANSFORMERS,
)


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
        engine_overrides={
            VlmEngineType.MLX: EngineModelConfig(
                repo_id="docling-project/SmolDocling-256M-preview-mlx-bf16"
            ),
            VlmEngineType.TRANSFORMERS: EngineModelConfig(
                torch_dtype="bfloat16",
                extra_config={
                    "transformers_model_type": TransformersModelType.AUTOMODEL_IMAGETEXTTOTEXT,
                },
            ),
        },
    ),
    scale=2.0,
    default_engine_type=VlmEngineType.AUTO_INLINE,
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
    default_engine_type=VlmEngineType.AUTO_INLINE,
)

VLM_CONVERT_DEEPSEEK_OCR = StageModelPreset(
    preset_id="deepseek_ocr",
    name="DeepSeek-OCR",
    description="DeepSeek OCR model via Ollama/LM Studio for document conversion (3B parameters)",
    model_spec=VlmModelSpec(
        name="DeepSeek-OCR-3B",
        default_repo_id="deepseek-ocr:3b",  # Ollama model name
        prompt="<|grounding|>Convert the document to markdown. ",
        response_format=ResponseFormat.DEEPSEEKOCR_MARKDOWN,
        supported_engines={VlmEngineType.API_OLLAMA, VlmEngineType.API_LMSTUDIO},
        api_overrides={
            VlmEngineType.API_OLLAMA: ApiModelConfig(
                params={"model": "deepseek-ocr:3b", "max_tokens": 4096}
            ),
            VlmEngineType.API_LMSTUDIO: ApiModelConfig(
                params={"model": "deepseek-ocr", "max_tokens": 4096}
            ),
        },
    ),
    scale=2.0,
    default_engine_type=VlmEngineType.API_OLLAMA,
)

VLM_CONVERT_GRANITE_VISION = StageModelPreset(
    preset_id="granite_vision",
    name="Granite-Vision",
    description="IBM Granite Vision model for markdown conversion (2B parameters)",
    model_spec=VlmModelSpec(
        **GRANITE_VISION_MODEL_SPEC_BASE,
        prompt="Convert this page to markdown. Do not miss any text and only output the bare markdown!",
        response_format=ResponseFormat.MARKDOWN,
    ),
    scale=2.0,
    default_engine_type=VlmEngineType.AUTO_INLINE,
)

VLM_CONVERT_PIXTRAL = StageModelPreset(
    preset_id="pixtral",
    name="Pixtral-12B",
    description="Mistral Pixtral model for markdown conversion (12B parameters)",
    model_spec=VlmModelSpec(
        **PIXTRAL_MODEL_SPEC_BASE,
        prompt="Convert this page to markdown. Do not miss any text and only output the bare markdown!",
        response_format=ResponseFormat.MARKDOWN,
    ),
    scale=2.0,
    default_engine_type=VlmEngineType.AUTO_INLINE,
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
        supported_engines={VlmEngineType.TRANSFORMERS},
        stop_strings=["<|im_end|>"],
        engine_overrides={
            VlmEngineType.TRANSFORMERS: EngineModelConfig(
                extra_config={
                    "transformers_model_type": TransformersModelType.AUTOMODEL_IMAGETEXTTOTEXT,
                    "transformers_prompt_style": TransformersPromptStyle.NONE,
                    "extra_processor_kwargs": {"format": True},
                }
            ),
        },
    ),
    scale=2.0,
    default_engine_type=VlmEngineType.TRANSFORMERS,
)

VLM_CONVERT_PHI4 = StageModelPreset(
    preset_id="phi4",
    name="Phi-4",
    description="Microsoft Phi-4 multimodal model for markdown conversion",
    model_spec=VlmModelSpec(
        name="Phi-4-Multimodal-Instruct",
        default_repo_id="microsoft/Phi-4-multimodal-instruct",
        prompt="Convert this page to MarkDown. Do not miss any text and only output the bare markdown",
        response_format=ResponseFormat.MARKDOWN,
        trust_remote_code=True,
        supported_engines={
            VlmEngineType.TRANSFORMERS,
            VlmEngineType.VLLM,
        },
        engine_overrides={
            VlmEngineType.TRANSFORMERS: EngineModelConfig(
                extra_config={
                    "transformers_model_type": TransformersModelType.AUTOMODEL_CAUSALLM,
                    "extra_generation_config": {"num_logits_to_keep": 0},
                }
            ),
        },
    ),
    scale=2.0,
    default_engine_type=VlmEngineType.AUTO_INLINE,
)

VLM_CONVERT_QWEN = StageModelPreset(
    preset_id="qwen",
    name="Qwen2.5-VL-3B",
    description="Qwen vision-language model for markdown conversion (3B parameters)",
    model_spec=VlmModelSpec(
        name="Qwen2.5-VL-3B-Instruct",
        default_repo_id="Qwen/Qwen2.5-VL-3B-Instruct",
        prompt="Convert this page to markdown. Do not miss any text and only output the bare markdown!",
        response_format=ResponseFormat.MARKDOWN,
        engine_overrides={
            VlmEngineType.MLX: EngineModelConfig(
                repo_id="mlx-community/Qwen2.5-VL-3B-Instruct-bf16"
            ),
            VlmEngineType.TRANSFORMERS: EngineModelConfig(
                extra_config={
                    "transformers_model_type": TransformersModelType.AUTOMODEL_IMAGETEXTTOTEXT,
                }
            ),
        },
    ),
    scale=2.0,
    default_engine_type=VlmEngineType.AUTO_INLINE,
)

VLM_CONVERT_GEMMA_12B = StageModelPreset(
    preset_id="gemma_12b",
    name="Gemma-3-12B",
    description="Google Gemma-3 vision model for markdown conversion (12B parameters)",
    model_spec=VlmModelSpec(
        name="Gemma-3-12B-IT",
        default_repo_id="google/gemma-3-12b-it",
        prompt="Convert this page to markdown. Do not miss any text and only output the bare markdown!",
        response_format=ResponseFormat.MARKDOWN,
        supported_engines={VlmEngineType.MLX},
        engine_overrides={
            VlmEngineType.MLX: EngineModelConfig(
                repo_id="mlx-community/gemma-3-12b-it-bf16"
            ),
        },
    ),
    scale=2.0,
    default_engine_type=VlmEngineType.MLX,
)

VLM_CONVERT_GEMMA_27B = StageModelPreset(
    preset_id="gemma_27b",
    name="Gemma-3-27B",
    description="Google Gemma-3 vision model for markdown conversion (27B parameters)",
    model_spec=VlmModelSpec(
        name="Gemma-3-27B-IT",
        default_repo_id="google/gemma-3-27b-it",
        prompt="Convert this page to markdown. Do not miss any text and only output the bare markdown!",
        response_format=ResponseFormat.MARKDOWN,
        supported_engines={VlmEngineType.MLX},
        engine_overrides={
            VlmEngineType.MLX: EngineModelConfig(
                repo_id="mlx-community/gemma-3-27b-it-bf16"
            ),
        },
    ),
    scale=2.0,
    default_engine_type=VlmEngineType.MLX,
)

VLM_CONVERT_DOLPHIN = StageModelPreset(
    preset_id="dolphin",
    name="Dolphin",
    description="ByteDance Dolphin OCR model for markdown conversion",
    model_spec=VlmModelSpec(
        name="Dolphin",
        default_repo_id="ByteDance/Dolphin",
        prompt="<s>Read text in the image. <Answer/>",
        response_format=ResponseFormat.MARKDOWN,
        engine_overrides={
            VlmEngineType.TRANSFORMERS: EngineModelConfig(
                extra_config={
                    "transformers_model_type": TransformersModelType.AUTOMODEL_IMAGETEXTTOTEXT,
                    "transformers_prompt_style": TransformersPromptStyle.RAW,
                }
            ),
        },
    ),
    scale=2.0,
    default_engine_type=VlmEngineType.AUTO_INLINE,
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
        engine_overrides={
            VlmEngineType.MLX: EngineModelConfig(
                repo_id="moot20/SmolVLM-256M-Instruct-MLX"
            ),
            VlmEngineType.TRANSFORMERS: EngineModelConfig(
                torch_dtype="bfloat16",
                extra_config={
                    "transformers_model_type": TransformersModelType.AUTOMODEL_IMAGETEXTTOTEXT,
                },
            ),
        },
        api_overrides={
            VlmEngineType.API_LMSTUDIO: ApiModelConfig(
                params={"model": "smolvlm-256m-instruct"}
            ),
        },
    ),
    scale=2.0,
    default_engine_type=VlmEngineType.AUTO_INLINE,
    stage_options={
        "picture_area_threshold": 0.05,
    },
)

PICTURE_DESC_GRANITE_VISION = StageModelPreset(
    preset_id="granite_vision",
    name="Granite-Vision-3.3-2B",
    description="IBM Granite Vision model for detailed image descriptions (2B parameters)",
    model_spec=VlmModelSpec(
        **GRANITE_VISION_MODEL_SPEC_BASE,
        prompt="What is shown in this image?",
        response_format=ResponseFormat.PLAINTEXT,
    ),
    scale=2.0,
    default_engine_type=VlmEngineType.AUTO_INLINE,
    stage_options={
        "picture_area_threshold": 0.05,
    },
)

PICTURE_DESC_PIXTRAL = StageModelPreset(
    preset_id="pixtral",
    name="Pixtral-12B",
    description="Mistral Pixtral model for detailed image descriptions (12B parameters)",
    model_spec=VlmModelSpec(
        **PIXTRAL_MODEL_SPEC_BASE,
        prompt="Describe this image in detail.",
        response_format=ResponseFormat.PLAINTEXT,
    ),
    scale=2.0,
    default_engine_type=VlmEngineType.AUTO_INLINE,
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
        engine_overrides={
            VlmEngineType.MLX: EngineModelConfig(
                repo_id="mlx-community/Qwen2.5-VL-3B-Instruct-bf16"
            ),
            VlmEngineType.TRANSFORMERS: EngineModelConfig(
                extra_config={
                    "transformers_model_type": TransformersModelType.AUTOMODEL_IMAGETEXTTOTEXT,
                }
            ),
        },
    ),
    scale=2.0,
    default_engine_type=VlmEngineType.AUTO_INLINE,
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
        stop_strings=["</doctag>", "<end_of_utterance>"],
        engine_overrides={
            VlmEngineType.TRANSFORMERS: EngineModelConfig(
                extra_config={
                    "transformers_model_type": TransformersModelType.AUTOMODEL_IMAGETEXTTOTEXT,
                    "extra_generation_config": {"skip_special_tokens": False},
                }
            ),
        },
    ),
    scale=2.0,
    default_engine_type=VlmEngineType.AUTO_INLINE,
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
    default_engine_type=VlmEngineType.AUTO_INLINE,
)
