"""Internal options for the experimental TableCrops layout model."""

from typing import ClassVar

from docling.datamodel.pipeline_options import BaseLayoutOptions

__all__ = ["TableCropsLayoutOptions"]


class TableCropsLayoutOptions(BaseLayoutOptions):
    """Options for TableCropsLayoutModel (internal-only)."""

    kind: ClassVar[str] = "docling_experimental_table_crops_layout"
