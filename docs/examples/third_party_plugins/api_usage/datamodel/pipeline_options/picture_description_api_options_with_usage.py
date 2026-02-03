from typing import Any, ClassVar, Dict, List, Literal, Optional, Union

from pydantic import (
    AnyUrl,
    Field,
)

from docling.datamodel.pipeline_options import PictureDescriptionApiOptions


class PictureDescriptionApiOptionsWithUsage(PictureDescriptionApiOptions):
    """DescriptionAnnotation."""

    kind: ClassVar[Literal["api_usage"]] = "api_usage"
