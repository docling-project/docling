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
    VlmPredictionToken,
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
    """Explicitly tagged model guidance, distinct from an output schema."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

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
    """Portable output contract and explicitly tagged extraction guidance."""

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
ExtractionValidationStatus = Literal["not_requested", "not_run", "passed", "failed"]


class ExtractionItem(BaseModel):
    """One durable outcome, without source content or image resources."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    scope: ExtractionScope
    extracted_data: dict[str, JsonValue] | None = None
    raw_text: str | None = None
    errors: list[str] = Field(default_factory=list)
    validation_status: ExtractionValidationStatus = "not_requested"
    generated_tokens: list[VlmPredictionToken] = Field(default_factory=list)
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
                errors=item.errors,
            )
            for item in result.items
            if isinstance(item.scope, PageScope)
        ],
    )
