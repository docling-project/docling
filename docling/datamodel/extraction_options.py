from enum import Enum


class ExtractionPromptStyle(str, Enum):
    NUEXTRACT = "nuextract"
    GRANITE_VISION = "granite_vision"
