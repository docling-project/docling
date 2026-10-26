# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from enum import Enum


class ExtractionPromptStyle(str, Enum):
    NUEXTRACT = "nuextract"
    GRANITE_VISION = "granite_vision"
