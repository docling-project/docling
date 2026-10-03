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
    ValidationInfo,
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


class ExtractionTemplate(BaseModel):
    """Explicitly tagged model guidance, distinct from an output schema.

    Docling handles three non-interchangeable representations of what to
    extract. Only the last two are an ``ExtractionTemplate``; the first is
    ``ExtractionTarget.output_schema``:

    | Representation                | Example                  | Meaning |
    |-------------------------------|--------------------------|---------|
    | ``output_schema`` (JSON Schema) | ``{"total": {"type": "number"}}`` | A validation constraint, checked by a JSON Schema validator. |
    | ``format="nuextract"``        | ``{"total": "number"}``  | NuExtract's native typed template: leaf values name output *types* (``verbatim-string`` vs ``string``, ``date-time``, enum-as-array). Interpreted by NuExtract's chat template; shape-identical to the output, carries no schema/validation meaning. |
    | ``format="example_json"``     | ``{"total": 123.45}``    | A concrete illustrative output, injected into a generic model's prompt as an example. Explicitly *not* a constraint. |

    ``nuextract`` can express semantics ``output_schema`` cannot (e.g.
    ``verbatim-string`` — extract text exactly, scored differently from a
    generic string); ``example_json`` cannot express requiredness,
    nullability, enums, or item types and must never be read as a schema.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    # See the class docstring for the three representations and what each means.
    format: Literal["example_json", "nuextract"]
    value: dict[str, JsonValue]

    @field_validator("value", mode="before")
    @classmethod
    def _example_values(cls, value: Any, info: ValidationInfo) -> Any:
        if isinstance(value, BaseModel):
            if info.data.get("format") != "example_json":
                raise ValueError("Pydantic instances are example_json guidance only")
            return value.model_dump(mode="json")
        return value


class ExtractionTarget(BaseModel):
    """Portable output contract and explicitly tagged extraction guidance.

    ``output_schema`` and ``template`` carry non-overlapping meaning, so both
    are kept. ``output_schema`` is always the validation contract; ``template``
    is model guidance that can express what a schema cannot (a NuExtract native
    type such as ``verbatim-string``) or a plain example. At least one is
    required. The three supported combinations:

    - **schema only** — the preparation helper derives its own guidance (a
      NuExtract native template, or a prose "Output contract" block for generic
      chat). Most callers need only this.
    - **template only** — parsed as JSON, no validation; ``validation_status``
      stays ``not_requested``.
    - **both** — ``template`` guides inference, ``output_schema`` validates the
      answer. Required whenever a native semantic type (``verbatim-string``)
      matters, since the schema alone cannot express it.

    The caller keeps the two descriptions consistent; Docling does not build an
    equivalence checker. For a generic model an example alongside a schema is
    unproven guidance — A/B it before relying on it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    output_schema: dict[str, JsonValue] | None = None
    template: ExtractionTemplate | None = None
    instructions: str | None = None

    @model_validator(mode="after")
    def _require_contract_or_guidance(self) -> "ExtractionTarget":
        if self.output_schema is None and self.template is None:
            raise ValueError("An extraction target requires output_schema or template")
        return self

    @classmethod
    def from_pydantic(
        cls,
        model: type[BaseModel],
        *,
        template: ExtractionTemplate | None = None,
        instructions: str | None = None,
    ) -> "ExtractionTarget":
        """Use the model's JSON Schema; Python validators are not transferred."""
        return cls(
            output_schema=model.model_json_schema(),
            template=template,
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

# Outcome of validating one item against the target's output_schema:
#   not_requested - no output_schema was supplied; validation was never in scope.
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
