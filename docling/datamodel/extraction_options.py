# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import inspect
import json
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import AnyUrl, BaseModel, ConfigDict, Field, model_validator

from docling.datamodel.accelerator_options import AcceleratorDevice
from docling.datamodel.pipeline_options_vlm_model import (
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
    """Which payload channel(s) to send the model.

    ``AUTO`` prefers the page image when the format has one, otherwise text.
    ``IMAGE_AND_TEXT`` is an explicit opt-in (never chosen by ``AUTO``).
    """

    AUTO = "auto"
    IMAGE = "image"
    TEXT = "text"
    IMAGE_AND_TEXT = "image_and_text"


_SUPPORTED_EXTRACTION_ENGINES = {
    VlmEngineType.TRANSFORMERS,
    VlmEngineType.API,
    VlmEngineType.API_OLLAMA,
    VlmEngineType.API_LMSTUDIO,
    VlmEngineType.API_OPENAI,
}


def _build_extraction_prompt(template: str) -> str:
    """Wrap a serialized template in the Granite schema-instruction prompt."""
    return (
        "Extract structured data from this document image.\n"
        "Return a JSON object matching this schema:\n\n"
        f"{template}\n\n"
        "Return null for fields you cannot find in the document.\n"
        "Return ONLY valid JSON, no other text."
    )


class ExtractionVlmModelSpec(VlmModelSpec):
    """Model specification for structured extraction."""

    prompt_style: ExtractionPromptStyle = ExtractionPromptStyle.NUEXTRACT

    accepts_image: bool = True
    accepts_text: bool = False

    torch_dtype: str | None = None
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
                    __use_examples__ = True
                    __use_defaults__ = True
                    __check_model__ = True

                return ExtractionTemplateFactory.build().model_dump_json(indent=2)
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
    """Pair an extraction model specification with its inference engine."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_spec: ExtractionVlmModelSpec = Field(
        description="Model specification (repo, prompt style, capability, runtime)"
    )
    scale: float = Field(
        default=2.0, gt=0, description="Image scaling factor for the image channel"
    )
    max_size: int | None = Field(
        default=None,
        gt=0,
        description="Maximum image dimension (width or height)",
    )

    @model_validator(mode="after")
    def _validate_engine(self) -> "ExtractionVlmOptions":
        engine_type = self.engine_options.engine_type
        if not self.model_spec.is_engine_supported(engine_type):
            raise ValueError(
                f"Model {self.model_spec.name!r} does not support the "
                f"{engine_type.value} VLM engine"
            )
        if engine_type == VlmEngineType.TRANSFORMERS:
            if not isinstance(self.engine_options, TransformersVlmEngineOptions):
                raise ValueError(
                    "Transformers extraction requires TransformersVlmEngineOptions"
                )
        elif VlmEngineType.is_api_variant(engine_type):
            if not isinstance(self.engine_options, ApiVlmEngineOptions):
                raise ValueError("API extraction requires ApiVlmEngineOptions")
        else:
            raise ValueError(
                f"Extraction does not support the {engine_type.value} VLM engine"
            )
        return self

    def build_extraction_prompt(self, template: "ExtractionTemplateType") -> str:
        return self.model_spec.build_extraction_prompt(template)

    def get_api_params(self) -> dict[str, Any]:
        engine = self.engine_options
        assert isinstance(engine, ApiVlmEngineOptions)
        return {
            **self.model_spec.get_api_params(engine.engine_type),
            **engine.params,
        }

    @classmethod
    def from_legacy_inline_options(
        cls, inline: InlineVlmOptions, style: ExtractionPromptStyle
    ) -> "ExtractionVlmOptions":
        """Adapt the deprecated flat extraction options."""
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
                load_in_8bit=inline.load_in_8bit,
                llm_int8_threshold=inline.llm_int8_threshold,
                quantized=inline.quantized,
                torch_dtype=inline.torch_dtype,
                trust_remote_code=inline.trust_remote_code,
                use_kv_cache=inline.use_kv_cache,
            ),
            scale=inline.scale,
            max_size=inline.max_size,
        )


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
    supported_engines=_SUPPORTED_EXTRACTION_ENGINES,
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
    supported_engines=_SUPPORTED_EXTRACTION_ENGINES,
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


NU_EXTRACT_2B_TRANSFORMERS = ExtractionVlmOptions.from_preset("nuextract_2b")

GRANITE_VISION_4_1_TRANSFORMERS = ExtractionVlmOptions.from_preset("granite_vision_4_1")

GRANITE_VISION_4_1_API = ExtractionVlmOptions(
    model_spec=GRANITE_VISION_4_1_SPEC,
    engine_options=ApiVlmEngineOptions(
        engine_type=VlmEngineType.API,
        url=AnyUrl("http://localhost:8000/v1/chat/completions"),
        timeout=120,
    ),
    scale=2.0,
)

NU_EXTRACT_API = ExtractionVlmOptions(
    model_spec=ExtractionVlmModelSpec(
        name="NuExtract 2.0 8B (API)",
        prompt_style=ExtractionPromptStyle.NUEXTRACT,
        accepts_image=True,
        accepts_text=True,
        default_repo_id="numind/NuExtract-2.0-8B",
        prompt="",
        response_format=ResponseFormat.PLAINTEXT,
        supported_engines=_SUPPORTED_EXTRACTION_ENGINES,
        temperature=0.0,
    ),
    engine_options=ApiVlmEngineOptions(
        engine_type=VlmEngineType.API,
        url=AnyUrl("http://localhost:8000/v1/chat/completions"),
        timeout=120,
    ),
    scale=2.0,
)
