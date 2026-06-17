"""Data models for document extraction functionality."""

from typing import Any, Optional, Union

from pydantic import BaseModel, Field

from docling.datamodel.base_models import ConversionStatus, ErrorItem
from docling.datamodel.document import InputDocument


class ExtractedPageData(BaseModel):
    """Data model for extracted content from a single page."""

    page_no: int = Field(..., description="1-indexed page number")
    extracted_data: Optional[dict[str, Any]] = Field(
        None, description="Extracted structured data from the page"
    )
    raw_text: Optional[str] = Field(None, description="Raw extracted text")
    errors: list[str] = Field(
        default_factory=list,
        description="Any errors encountered during extraction for this page",
    )


class ExtractionResult(BaseModel):
    """Result of document extraction."""

    input: InputDocument
    status: ConversionStatus = ConversionStatus.PENDING
    errors: list[ErrorItem] = []

    # Pages field - always a list for consistency
    pages: list[ExtractedPageData] = Field(
        default_factory=list, description="Extracted data from each page"
    )


# Type alias for template parameters that can be string, dict, or BaseModel
ExtractionTemplateType = Union[str, dict[str, Any], BaseModel, type[BaseModel]]
