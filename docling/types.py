"""Shared type definitions for the docling package."""

from typing import Any, Dict, Type, Union

from pydantic import BaseModel

# Type alias for template parameters that can be string, dict, or BaseModel
ExtractionTemplateType = Union[str, Dict[str, Any], BaseModel, Type[BaseModel]]
