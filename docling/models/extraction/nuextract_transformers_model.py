# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Main's NuExtract entry point forwards to the shared extraction implementation."""

from pathlib import Path

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.pipeline_options_vlm_model import InlineVlmOptions
from docling.models.extraction.transformers_extraction_model import (
    TransformersExtractionModel,
)


class NuExtractTransformersModel(TransformersExtractionModel):
    def __init__(
        self,
        enabled: bool,
        artifacts_path: Path | None,
        accelerator_options: AcceleratorOptions,
        vlm_options: InlineVlmOptions,
    ):
        super().__init__(enabled, artifacts_path, accelerator_options, vlm_options)
