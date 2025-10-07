from typing import Any, ClassVar, Dict, List, Literal, Optional, Union

from pydantic import (
    AnyUrl,
    BaseModel,
    ConfigDict,
    Field,
)

from docling.datamodel.pipeline_options import PictureDescriptionBaseOptions


class PictureDescriptionApiOptionsWithUsage(PictureDescriptionBaseOptions):
    """DescriptionAnnotation."""

    kind: ClassVar[Literal["api_usage"]] = "api_usage"

    url: AnyUrl = AnyUrl("http://localhost:8000/v1/chat/completions")
    headers: Dict[str, str] = {}
    params: Dict[str, Any] = {}
    timeout: float = 20
    concurrency: int = 1

    prompt: str = "Describe this image in a few sentences."
    provenance: str = ""
    # Key inside the response 'usage' (or similar) which will be used to extract
    # the token/response text. Example: 'content' or 'text'. If None, no
    # token extraction will be performed by default.
    token_extract_key: Optional[str] = Field(
        None,
        description=(
            "Key in the response usage dict whose value contains the token/"
            "response to extract. For example 'content' or 'text'."
        ),
    )
