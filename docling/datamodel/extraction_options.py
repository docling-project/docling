# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import inspect
import json
from enum import Enum
from typing import TYPE_CHECKING

from pydantic import BaseModel

from docling.datamodel.pipeline_options_vlm_model import (
    ApiVlmOptions,
    InlineVlmOptions,
)

if TYPE_CHECKING:
    from docling.datamodel.extraction import ExtractionTemplateType


class ExtractionPromptStyle(str, Enum):
    NUEXTRACT = "nuextract"
    GRANITE_VISION = "granite_vision"


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


class ExtractionVlmOptionsMixin(BaseModel):
    """Adds extraction prompt behavior to a VLM options spec.

    The prompt style travels with the model spec (like ``build_prompt`` /
    ``decode_response`` on the convert side), so the pipeline never interprets
    it and an illegal model/style pairing cannot be constructed. Both the
    template *serialization* and the prompt *embedding* are decided here.
    """

    extraction_prompt_style: ExtractionPromptStyle = ExtractionPromptStyle.NUEXTRACT

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
            if self.extraction_prompt_style is ExtractionPromptStyle.NUEXTRACT:
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
        if self.extraction_prompt_style is ExtractionPromptStyle.NUEXTRACT:
            return text
        return _build_extraction_prompt(text)


class InlineExtractionVlmOptions(ExtractionVlmOptionsMixin, InlineVlmOptions):
    """Local HuggingFace transformers spec for the extraction pipeline."""


class ApiExtractionVlmOptions(ExtractionVlmOptionsMixin, ApiVlmOptions):
    """Remote OpenAI-conformant endpoint spec for the extraction pipeline."""
