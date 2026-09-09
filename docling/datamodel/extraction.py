# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Data models for document extraction functionality."""

from typing import Any, Dict, List, Literal, Optional, Type, Union

from PIL.Image import Image
from pydantic import BaseModel, ConfigDict, Field

from docling.datamodel.base_models import ConversionStatus, ErrorItem, VlmStopReason
from docling.datamodel.document import InputDocument


class TextContentItem(BaseModel):
    """A text payload item in a model request (dim 2)."""

    type: Literal["text"] = "text"
    text: str


class ImageContentItem(BaseModel):
    """An image payload item in a model request (dim 2)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["image"] = "image"
    image: Image


# One entry of an ordered model-request payload. A request is a
# ``list[ContentItem]`` (image and/or text); the schema/template rides a
# separate channel, not the content (see the extraction plan, dim 2).
ContentItem = Union[TextContentItem, ImageContentItem]


class ExtractedPageData(BaseModel):
    """Data model for extracted content from a single page."""

    page_no: int = Field(..., description="1-indexed page number")
    extracted_data: Optional[Dict[str, Any]] = Field(
        None, description="Extracted structured data from the page"
    )
    raw_text: Optional[str] = Field(None, description="Raw extracted text")
    errors: List[str] = Field(
        default_factory=list,
        description="Any errors encountered during extraction for this page",
    )


class ExtractionResult(BaseModel):
    """Result of document extraction."""

    input: InputDocument
    status: ConversionStatus = ConversionStatus.PENDING
    errors: List[ErrorItem] = []

    # Pages field - always a list for consistency
    pages: List[ExtractedPageData] = Field(
        default_factory=list, description="Extracted data from each page"
    )


# Type alias for template parameters that can be string, dict, or BaseModel
ExtractionTemplateType = Union[str, Dict[str, Any], BaseModel, Type[BaseModel]]
