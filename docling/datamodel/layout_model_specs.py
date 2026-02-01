import logging
from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

from pydantic import BaseModel, Field

from docling.datamodel.accelerator_options import AcceleratorDevice

_log = logging.getLogger(__name__)


class LayoutModelConfig(BaseModel):
    """Configuration for document layout analysis models from HuggingFace."""

    name: Annotated[
        str,
        Field(
            description=(
                "Human-readable name identifier for the layout model. Used for "
                "logging, debugging, and model selection."
            ),
            examples=["docling_layout_heron", "docling_layout_egret_large"],
        ),
    ]
    repo_id: Annotated[
        str,
        Field(
            description=(
                "HuggingFace repository ID where the model is hosted. Used to "
                "download model weights and configuration files from "
                "HuggingFace Hub."
            ),
            examples=[
                "docling-project/docling-layout-heron",
                "docling-project/docling-layout-egret-large",
            ],
        ),
    ]
    revision: Annotated[
        str,
        Field(
            description=(
                "Git revision (branch, tag, or commit hash) of the model "
                "repository to use. Allows pinning to specific model versions "
                "for reproducibility."
            ),
            examples=["main", "v1.0.0"],
        ),
    ]
    model_path: Annotated[
        str,
        Field(
            description=(
                "Relative path within the repository to model artifacts. Empty "
                "string indicates artifacts are in the repository root. Used "
                "for repositories with multiple models or nested structures."
            ),
        ),
    ]
    supported_devices: Annotated[
        list[AcceleratorDevice],
        Field(
            description=(
                "List of hardware accelerators supported by this model. The "
                "model can only run on devices in this list."
            )
        ),
    ] = [
        AcceleratorDevice.CPU,
        AcceleratorDevice.CUDA,
        AcceleratorDevice.MPS,
        AcceleratorDevice.XPU,
    ]

    @property
    def model_repo_folder(self) -> str:
        return self.repo_id.replace("/", "--")


# HuggingFace Layout Models

# Default Docling Layout Model
DOCLING_LAYOUT_V2 = LayoutModelConfig(
    name="docling_layout_v2",
    repo_id="docling-project/docling-layout-old",
    revision="main",
    model_path="",
)

DOCLING_LAYOUT_HERON = LayoutModelConfig(
    name="docling_layout_heron",
    repo_id="docling-project/docling-layout-heron",
    revision="main",
    model_path="",
)

DOCLING_LAYOUT_HERON_101 = LayoutModelConfig(
    name="docling_layout_heron_101",
    repo_id="docling-project/docling-layout-heron-101",
    revision="main",
    model_path="",
)

DOCLING_LAYOUT_EGRET_MEDIUM = LayoutModelConfig(
    name="docling_layout_egret_medium",
    repo_id="docling-project/docling-layout-egret-medium",
    revision="main",
    model_path="",
)

DOCLING_LAYOUT_EGRET_LARGE = LayoutModelConfig(
    name="docling_layout_egret_large",
    repo_id="docling-project/docling-layout-egret-large",
    revision="main",
    model_path="",
)

DOCLING_LAYOUT_EGRET_XLARGE = LayoutModelConfig(
    name="docling_layout_egret_xlarge",
    repo_id="docling-project/docling-layout-egret-xlarge",
    revision="main",
    model_path="",
)

# Example for a hypothetical alternative model
# ALTERNATIVE_LAYOUT = LayoutModelConfig(
#     name="alternative_layout",
#     repo_id="someorg/alternative-layout",
#     revision="main",
#     model_path="model_artifacts/layout_alt",
# )


class LayoutModelType(str, Enum):
    DOCLING_LAYOUT_V2 = "docling_layout_v2"
    DOCLING_LAYOUT_HERON = "docling_layout_heron"
    DOCLING_LAYOUT_HERON_101 = "docling_layout_heron_101"
    DOCLING_LAYOUT_EGRET_MEDIUM = "docling_layout_egret_medium"
    DOCLING_LAYOUT_EGRET_LARGE = "docling_layout_egret_large"
    DOCLING_LAYOUT_EGRET_XLARGE = "docling_layout_egret_xlarge"
    # ALTERNATIVE_LAYOUT = "alternative_layout"
