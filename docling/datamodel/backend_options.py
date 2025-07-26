from enum import Enum

from pydantic import BaseModel


class ImageOptions(str, Enum):
    """Image options for HTML backend."""

    NONE = "none"
    EMBEDDED = "embedded"
    REFERENCED = "referenced"


class BackendOptions(BaseModel):
    """
    Options for the document backend.
    This class is used to pass options to the backend when processing documents.
    """


class HTMLBackendOptions(BackendOptions):
    """
    Options specific to the HTML backend.
    This class can be extended to include options specific to HTML processing.
    """

    image_options: ImageOptions = ImageOptions.NONE
