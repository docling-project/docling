# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Data models for document extraction functionality."""

from typing import Annotated, Any, Literal

from PIL.Image import Image
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

from docling.datamodel.base_models import (
    ConversionStatus,
    ErrorItem,
    OpenAiResponseUsage,
    VlmStopReason,
)
from docling.datamodel.document import InputDocument


class TextContentItem(BaseModel):
    """A text payload item in a model request."""

    type: Literal["text"] = "text"
    text: str


class ImageContentItem(BaseModel):
    """An image payload item in a model request."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["image"] = "image"
    image: Image


ContentItem = TextContentItem | ImageContentItem


class ExtractionSpec(BaseModel):
    """What to extract: a validation contract and an optional model-input override.

    Each field has one role:

    - ``json_schema`` is the contract. It is a JSON Schema (Draft 2020-12) and
      it always defines the validation of the output. When no
      ``native_template`` is given, Docling also derives the model input from
      it with a best-effort conversion (a NuExtract native template, or the
      schema text in the prompt of a generic chat model). A schema that has no
      equivalent in the model input is rejected; add a ``native_template``.
    - ``native_template`` overrides the model input only. It is in the native
      template format of the model of the selected preset (today: NuExtract,
      for example ``{"total": "number", "id": "verbatim-string"}``). It goes to
      the model as is, is not checked, and never has an effect on validation.
      A preset whose model has no native template format rejects it.
    - ``instructions`` is free-text guidance added to the prompt in all cases.

    At least one of ``json_schema`` and ``native_template`` is required. The
    three combinations:

    | Request has                         | Model gets                  | Output validation |
    |-------------------------------------|-----------------------------|-------------------|
    | ``json_schema``                     | The converted schema        | ``json_schema``   |
    | ``json_schema`` + ``native_template`` | ``native_template``, as is | ``json_schema``   |
    | ``native_template``                 | ``native_template``, as is  | None (``not_requested``) |

    Docling does not check that the two fields agree. If they do not agree,
    the items get ``validation_status="failed"``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    json_schema: dict[str, JsonValue] | None = None
    native_template: dict[str, JsonValue] | None = None
    instructions: str | None = None

    @model_validator(mode="after")
    def _require_contract_or_template(self) -> "ExtractionSpec":
        if self.json_schema is None and self.native_template is None:
            raise ValueError(
                "An extraction spec requires json_schema or native_template"
            )
        return self

    @classmethod
    def from_pydantic(
        cls,
        model: type[BaseModel],
        *,
        native_template: dict[str, JsonValue] | None = None,
        instructions: str | None = None,
    ) -> "ExtractionSpec":
        """Use the model's JSON Schema; Python validators are not transferred."""
        return cls(
            json_schema=model.model_json_schema(),
            native_template=native_template,
            instructions=instructions,
        )


class DocumentScope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["document"] = "document"


class PageScope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["page"] = "page"
    page_no: int = Field(ge=1, strict=True)


ExtractionScope = Annotated[DocumentScope | PageScope, Field(discriminator="kind")]

# Outcome of validating one item against the spec's json_schema:
#   not_requested - no json_schema was supplied; validation was never in scope.
#   not_run       - a schema was supplied, but inference or JSON parsing failed
#                   first, so no parsed object ever existed to check.
#   passed/failed - schema supplied, parsing succeeded, object did/didn't validate.
ExtractionValidationStatus = Literal["not_requested", "not_run", "passed", "failed"]


class VlmInferenceMetadata(BaseModel):
    """Backend inference telemetry; absent for items with no model prediction.

    Bundles the generation-only fields so they don't read as universal
    extraction-result fields. Per-token output (``generated_tokens``) is
    deliberately not carried on the durable result — no consumer needs it.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    generation_time: float = -1
    num_tokens: int | None = None
    usage: dict[str, JsonValue] | None = None
    stop_reason: VlmStopReason = VlmStopReason.UNSPECIFIED

    @field_validator("usage", mode="before")
    @classmethod
    def _usage_values(cls, value: Any) -> Any:
        if isinstance(value, OpenAiResponseUsage):
            return value.model_dump(mode="json")
        return value


class ExtractionItem(BaseModel):
    """One durable outcome, without source content or image resources."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    scope: ExtractionScope
    extracted_data: dict[str, JsonValue] | None = None
    raw_text: str | None = None
    # Scoped inference/decode/validation failures for this item, as ErrorItems
    # (same type as the document-level DocumentExtractionResult.errors, which
    # instead cover failures where no item could be scoped). page_no carries the
    # scope, so the two lists are never merged or duplicated.
    errors: list[ErrorItem] = Field(default_factory=list)
    validation_status: ExtractionValidationStatus = "not_requested"
    inference_metadata: VlmInferenceMetadata | None = None


class DocumentExtractionResult(BaseModel):
    """Ordered extraction outcomes owned by one input document."""

    model_config = ConfigDict(extra="forbid")

    input: InputDocument
    status: ConversionStatus = ConversionStatus.PENDING
    errors: list[ErrorItem] = Field(default_factory=list)
    items: list[ExtractionItem] = Field(default_factory=list)


class ExtractedPageData(BaseModel):
    """Data model for extracted content from a single page."""

    page_no: int = Field(..., description="1-indexed page number")
    extracted_data: dict[str, Any] | None = Field(
        None, description="Extracted structured data from the page"
    )
    raw_text: str | None = Field(None, description="Raw extracted text")
    errors: list[str] = Field(
        default_factory=list,
        description="Any errors encountered during extraction for this page",
    )


class ExtractionResult(BaseModel):
    """Result of document extraction."""

    input: InputDocument
    status: ConversionStatus = ConversionStatus.PENDING
    errors: list[ErrorItem] = Field(default_factory=list)

    pages: list[ExtractedPageData] = Field(
        default_factory=list, description="Extracted data from each page"
    )


ExtractionTemplateType = str | dict[str, Any] | BaseModel | type[BaseModel]


def _legacy_result(result: DocumentExtractionResult) -> ExtractionResult:
    """Project page outcomes only at an outer legacy execution boundary."""
    return ExtractionResult(
        input=result.input,
        status=result.status,
        errors=result.errors,
        pages=[
            ExtractedPageData(
                page_no=item.scope.page_no,
                extracted_data=item.extracted_data,
                raw_text=item.raw_text,
                errors=[error.error_message for error in item.errors],
            )
            for item in result.items
            if isinstance(item.scope, PageScope)
        ],
    )
