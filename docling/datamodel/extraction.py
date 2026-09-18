# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Data models for document extraction functionality."""

from typing import Any, Literal

from PIL.Image import Image
from pydantic import BaseModel, ConfigDict, Field

from docling.datamodel.base_models import ConversionStatus, ErrorItem, VlmStopReason
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
