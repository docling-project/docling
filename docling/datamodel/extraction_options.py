# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import inspect
import json
from enum import Enum
from typing import TYPE_CHECKING, Optional

from pydantic import AnyUrl, BaseModel, ConfigDict, Field

from docling.datamodel.accelerator_options import AcceleratorDevice
from docling.datamodel.pipeline_options_vlm_model import (
    ApiVlmOptions,
    InferenceFramework,
    InlineVlmOptions,
    ResponseFormat,
    TransformersModelType,
)
from docling.datamodel.stage_model_specs import (
    StageModelPreset,
    StagePresetMixin,
    VlmModelSpec,
)
from docling.datamodel.vlm_engine_options import (
    ApiVlmEngineOptions,
    TransformersVlmEngineOptions,
)
from docling.models.inference_engines.vlm.base import (
    VlmEngineOptionsMixin,
    VlmEngineType,
)

if TYPE_CHECKING:
    from docling.datamodel.extraction import ExtractionTemplateType


class ExtractionPromptStyle(str, Enum):
    NUEXTRACT = "nuextract"
    GRANITE_VISION = "granite_vision"


class ChannelSelection(str, Enum):
    """Which payload channel(s) to send the model (dim 2).

    ``AUTO`` prefers the page image when the format has one, otherwise text.
    ``IMAGE_AND_TEXT`` is an explicit opt-in (never chosen by ``AUTO``).
    """

    AUTO = "auto"
    IMAGE = "image"
    TEXT = "text"
    IMAGE_AND_TEXT = "image_and_text"


def _build_extraction_prompt(template: str) -> str:
    """Wrap a serialized template in the Granite schema-instruction prompt.

    Kept as a module-level function (and re-exported from
    ``models.extraction.prompt_utils``) so it can be reused and tested on its own.
    """
    return (
        "Extract structured data from this document image.\n"
        "Return a JSON object matching this schema:\n\n"
        f"{template}\n\n"
        "Return null for fields you cannot find in the document.\n"
        "Return ONLY valid JSON, no other text."
    )


class ExtractionVlmModelSpec(VlmModelSpec):
    """Model specification for the extraction stage (modern spec/preset style).

    Extends the shared :class:`VlmModelSpec` with the per-*model* traits the
    extraction pipeline needs: the prompt style, which payload channels the
    model accepts, and the transformers load/generation settings the local
    engine reads. Both the template *serialization* and the prompt *embedding*
    are decided here (like ``build_prompt`` / ``decode_response`` on the convert
    side), so the pipeline never interprets the style and an illegal
    model/style/channel pairing cannot be constructed.
    """

    prompt_style: ExtractionPromptStyle = ExtractionPromptStyle.NUEXTRACT

    # Channel capability (dim 2 / R3). ``AUTO`` = (what the format offers) ∩
    # (what the model accepts). NuExtract accepts both; Granite is image-only.
    accepts_image: bool = True
    accepts_text: bool = False

    # Local transformers settings not present on the shared VlmModelSpec.
    torch_dtype: Optional[str] = None
    transformers_model_type: TransformersModelType = (
        TransformersModelType.AUTOMODEL_IMAGETEXTTOTEXT
    )
    supported_devices: list[AcceleratorDevice] = Field(
        default_factory=lambda: [
            AcceleratorDevice.CPU,
            AcceleratorDevice.CUDA,
            AcceleratorDevice.MPS,
            AcceleratorDevice.XPU,
        ]
    )
    extra_processor_kwargs: dict = Field(default_factory=dict)

    def serialize_template(self, template: "ExtractionTemplateType") -> str:
        """Serialize any of the four template forms to a schema string.

        Only a Pydantic *class* is style-dependent: NuExtract wants a sample
        instance (field name -> example value), GRANITE_VISION wants a real JSON
        Schema with field descriptions (the format from the Granite model card).
        """
        if isinstance(template, str):
            return template
        if isinstance(template, dict):
            return json.dumps(template, indent=2)
        if isinstance(template, BaseModel):
            return template.model_dump_json(indent=2)
        if inspect.isclass(template) and issubclass(template, BaseModel):
            if self.prompt_style is ExtractionPromptStyle.NUEXTRACT:
                from polyfactory.factories.pydantic_factory import ModelFactory

                class ExtractionTemplateFactory(ModelFactory[template]):  # type: ignore
                    __use_examples__ = True  # prefer Field(examples=...) when present
                    __use_defaults__ = True  # use field defaults over random values
                    __check_model__ = True  # avoid deprecation warnings

                return ExtractionTemplateFactory.build().model_dump_json(indent=2)  # type: ignore
            return json.dumps(template.model_json_schema(), indent=2)
        raise ValueError(f"Unsupported template type: {type(template)}")

    def build_extraction_prompt(self, template: "ExtractionTemplateType") -> str:
        """Turn a template into the final prompt text for this model's style.

        NuExtract feeds the serialized template through the model's own
        ``template=`` chat kwarg, so it is returned unwrapped. GRANITE_VISION
        wraps it in a plain-text schema instruction that any transformers or
        OpenAI-conformant API engine consumes.
        """
        text = self.serialize_template(template)
        if self.prompt_style is ExtractionPromptStyle.NUEXTRACT:
            return text
        return _build_extraction_prompt(text)


class ExtractionVlmOptions(StagePresetMixin, VlmEngineOptionsMixin, BaseModel):
    """Configuration for the VLM extraction stage (modern preset style).

    Symmetric with ``VlmConvertOptions``: pairs an :class:`ExtractionVlmModelSpec`
    with a runtime ``engine_options``. Use a preset for the common case::

        ExtractionVlmOptions.from_preset("nuextract_2b")

    The pipeline selects the execution model from ``engine_options.engine_type``
    (inline transformers vs. remote API); the prompt style and channel
    capability travel with ``model_spec``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_spec: ExtractionVlmModelSpec = Field(
        description="Model specification (repo, prompt style, capability, runtime)"
    )
    scale: float = Field(
        default=2.0, description="Image scaling factor for the image channel"
    )
    max_size: Optional[int] = Field(
        default=None, description="Maximum image dimension (width or height)"
    )

    @property
    def extraction_prompt_style(self) -> ExtractionPromptStyle:
        """The model's prompt style (lives on the spec)."""
        return self.model_spec.prompt_style

    def build_extraction_prompt(self, template: "ExtractionTemplateType") -> str:
        """Delegate to the spec (kept here so pipeline call sites stay stable)."""
        return self.model_spec.build_extraction_prompt(template)

    # -- lowering to the flat model-input types the execution models consume ---
    # The spec is the single source of truth; these derive the DTOs the two
    # extraction models already accept, so those models stay untouched.

    def to_inline_input(self) -> "InlineExtractionVlmOptions":
        spec = self.model_spec
        return InlineExtractionVlmOptions(
            extraction_prompt_style=spec.prompt_style,
            repo_id=spec.default_repo_id,
            revision=spec.revision,
            prompt=spec.prompt,
            torch_dtype=spec.torch_dtype,
            inference_framework=InferenceFramework.TRANSFORMERS,
            transformers_model_type=spec.transformers_model_type,
            response_format=spec.response_format,
            supported_devices=spec.supported_devices,
            trust_remote_code=spec.trust_remote_code,
            extra_processor_kwargs=spec.extra_processor_kwargs,
            extra_generation_config=spec.extra_generation_config,
            max_new_tokens=spec.max_new_tokens,
            scale=self.scale,
            max_size=self.max_size,
            temperature=spec.temperature,
        )

    @classmethod
    def from_legacy_inline_options(
        cls, inline: InlineVlmOptions, style: ExtractionPromptStyle
    ) -> "ExtractionVlmOptions":
        """Wrap a released-style flat ``InlineVlmOptions`` into the preset shape.

        Back-compat for the ``main`` surface, where ``vlm_options`` was a plain
        ``InlineVlmOptions`` and the prompt style lived on the pipeline options.
        """
        return cls(
            model_spec=ExtractionVlmModelSpec(
                name=inline.repo_id,
                prompt_style=style,
                accepts_image=True,
                accepts_text=style is ExtractionPromptStyle.NUEXTRACT,
                default_repo_id=inline.repo_id,
                revision=inline.revision,
                prompt=inline.prompt,
                torch_dtype=inline.torch_dtype,
                transformers_model_type=inline.transformers_model_type,
                response_format=inline.response_format,
                supported_devices=inline.supported_devices,
                trust_remote_code=inline.trust_remote_code,
                extra_processor_kwargs=inline.extra_processor_kwargs,
                extra_generation_config=inline.extra_generation_config,
                max_new_tokens=inline.max_new_tokens,
                temperature=inline.temperature,
            ),
            engine_options=TransformersVlmEngineOptions(
                trust_remote_code=inline.trust_remote_code
            ),
            scale=inline.scale,
            max_size=inline.max_size,
        )

    @classmethod
    def from_legacy_api_options(
        cls, api: ApiVlmOptions, style: ExtractionPromptStyle
    ) -> "ExtractionVlmOptions":
        """Wrap a flat ``ApiVlmOptions`` into the preset shape (branch back-compat)."""
        return cls(
            model_spec=ExtractionVlmModelSpec(
                name=str(api.params.get("model", "api-model")),
                prompt_style=style,
                accepts_image=True,
                accepts_text=style is ExtractionPromptStyle.NUEXTRACT,
                default_repo_id=str(api.params.get("model", "api-model")),
                prompt=api.prompt,
                response_format=api.response_format,
                temperature=api.temperature,
            ),
            engine_options=ApiVlmEngineOptions(
                engine_type=VlmEngineType.API,
                url=api.url,
                headers=api.headers,
                params=api.params,
                timeout=api.timeout,
                concurrency=api.concurrency,
            ),
            scale=api.scale,
            max_size=api.max_size,
        )

    def to_api_input(self) -> "ApiExtractionVlmOptions":
        spec = self.model_spec
        engine = self.engine_options
        assert isinstance(engine, ApiVlmEngineOptions), (
            "API extraction requires ApiVlmEngineOptions"
        )
        return ApiExtractionVlmOptions(
            extraction_prompt_style=spec.prompt_style,
            url=engine.url,
            headers=engine.headers,
            params=engine.params,
            timeout=engine.timeout,
            concurrency=engine.concurrency,
            prompt=spec.prompt,
            scale=self.scale,
            max_size=self.max_size,
            response_format=spec.response_format,
            temperature=spec.temperature,
        )


class ExtractionVlmOptionsMixin(BaseModel):
    """Carries the prompt style on the flat model-input DTOs (data only).

    The prompt behavior itself lives on :class:`ExtractionVlmModelSpec`; the
    execution models only need the style enum to pick their processor inputs.
    """

    extraction_prompt_style: ExtractionPromptStyle = ExtractionPromptStyle.NUEXTRACT


class InlineExtractionVlmOptions(ExtractionVlmOptionsMixin, InlineVlmOptions):
    """Internal local-transformers input for :class:`TransformersExtractionModel`.

    Derived from :class:`ExtractionVlmOptions` via ``to_inline_input``; not part
    of the user-facing surface.
    """


class ApiExtractionVlmOptions(ExtractionVlmOptionsMixin, ApiVlmOptions):
    """Internal remote-endpoint input for the extraction API models.

    Derived from :class:`ExtractionVlmOptions` via ``to_api_input``; not part of
    the user-facing surface.
    """


# =============================================================================
# PRESETS
# =============================================================================

NUEXTRACT_2B_SPEC = ExtractionVlmModelSpec(
    name="NuExtract 2.0 2B",
    prompt_style=ExtractionPromptStyle.NUEXTRACT,
    accepts_image=True,
    accepts_text=True,
    default_repo_id="numind/NuExtract-2.0-2B",
    revision="fe5b2f0b63b81150721435a3ca1129a75c59c74e",  # 489efed leads to MPS issues
    prompt="",  # unused: NuExtract carries the template out-of-band
    torch_dtype="bfloat16",
    transformers_model_type=TransformersModelType.AUTOMODEL_IMAGETEXTTOTEXT,
    response_format=ResponseFormat.PLAINTEXT,
    temperature=0.0,
)

GRANITE_VISION_4_1_SPEC = ExtractionVlmModelSpec(
    name="Granite Vision 4.1",
    prompt_style=ExtractionPromptStyle.GRANITE_VISION,
    accepts_image=True,
    accepts_text=False,  # Granite cannot take a text payload
    default_repo_id="ibm-granite/granite-vision-4.1-4b",
    revision="dd48e97503de471803850df70843cf9eb5da8712",
    prompt="",  # template is passed separately via extract()
    torch_dtype="bfloat16",
    transformers_model_type=TransformersModelType.AUTOMODEL_IMAGETEXTTOTEXT,
    response_format=ResponseFormat.PLAINTEXT,
    temperature=0.0,
    trust_remote_code=True,
)

ExtractionVlmOptions.register_preset(
    StageModelPreset(
        preset_id="nuextract_2b",
        name="NuExtract 2.0 2B",
        description="NuExtract structured extraction (local transformers).",
        model_spec=NUEXTRACT_2B_SPEC,
        default_engine_type=VlmEngineType.TRANSFORMERS,
        scale=2.0,
    )
)

ExtractionVlmOptions.register_preset(
    StageModelPreset(
        preset_id="granite_vision_4_1",
        name="Granite Vision 4.1",
        description="Granite Vision schema-instruction extraction (local transformers).",
        model_spec=GRANITE_VISION_4_1_SPEC,
        default_engine_type=VlmEngineType.TRANSFORMERS,
        scale=2.0,
    )
)


# =============================================================================
# NAMED SPECS (re-exported from vlm_model_specs for back-compat imports)
# =============================================================================

# NuExtract (local transformers) — modern preset style.
NU_EXTRACT_2B_TRANSFORMERS = ExtractionVlmOptions.from_preset("nuextract_2b")

# Granite Vision 4.1 (local transformers) — modern preset style.
GRANITE_VISION_4_1_TRANSFORMERS = ExtractionVlmOptions.from_preset("granite_vision_4_1")

# Granite Vision 4.1 served over an OpenAI-conformant endpoint (e.g. vLLM).
# Image-only; the spec carries GRANITE_VISION style, so it builds the
# schema-instruction prompt from the template itself and `prompt` is empty.
GRANITE_VISION_4_1_API = ExtractionVlmOptions(
    model_spec=ExtractionVlmModelSpec(
        name="Granite Vision 4.1 (API)",
        prompt_style=ExtractionPromptStyle.GRANITE_VISION,
        accepts_image=True,
        accepts_text=False,
        default_repo_id="ibm-granite/granite-vision-4.1-4b",
        prompt="",
        response_format=ResponseFormat.PLAINTEXT,
        temperature=0.0,
    ),
    engine_options=ApiVlmEngineOptions(
        engine_type=VlmEngineType.API,
        url=AnyUrl("http://localhost:8000/v1/chat/completions"),
        params={"model": "ibm-granite/granite-vision-4.1-4b"},
        timeout=120,
    ),
    scale=2.0,
)

# NuExtract served over an OpenAI-conformant endpoint (e.g. vLLM). NuExtract
# carries the template out-of-band, so this routes to `api_nuextract_request`
# (not the plain image-request path). Supports the text channel; `prompt` unused.
NU_EXTRACT_API = ExtractionVlmOptions(
    model_spec=ExtractionVlmModelSpec(
        name="NuExtract 2.0 8B (API)",
        prompt_style=ExtractionPromptStyle.NUEXTRACT,
        accepts_image=True,
        accepts_text=True,
        default_repo_id="numind/NuExtract-2.0-8B",
        prompt="",
        response_format=ResponseFormat.PLAINTEXT,
        temperature=0.0,
    ),
    engine_options=ApiVlmEngineOptions(
        engine_type=VlmEngineType.API,
        url=AnyUrl("http://localhost:8000/v1/chat/completions"),
        params={"model": "numind/NuExtract-2.0-8B"},
        timeout=120,
    ),
    scale=2.0,
)
