import logging
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from docling.datamodel.accelerator_options import AcceleratorDevice

_log = logging.getLogger(__name__)


class LayoutModelConfig(BaseModel):
    name: str
    repo_id: str
    revision: str
    model_path: str
    supported_devices: list[AcceleratorDevice] = [
        AcceleratorDevice.CPU,
        AcceleratorDevice.CUDA,
        AcceleratorDevice.MPS,
    ]

    @property
    def model_repo_folder(self) -> str:
        return self.repo_id.replace("/", "--")


# HuggingFace Layout Models

# Default Docling Layout Model
DOCLING_LAYOUT_V2 = LayoutModelConfig(
    name="docling_layout_v2",
    repo_id="ds4sd/docling-layout-old",
    revision="main",
    model_path="",
)

DOCLING_LAYOUT_HERON = LayoutModelConfig(
    name="docling_layout_heron",
    repo_id="ds4sd/docling-layout-heron",
    revision="main",
    model_path="",
)

DOCLING_LAYOUT_HERON_101 = LayoutModelConfig(
    name="docling_layout_heron_101",
    repo_id="ds4sd/docling-layout-heron-101",
    revision="main",
    model_path="",
)

DOCLING_LAYOUT_EGRET_MEDIUM = LayoutModelConfig(
    name="docling_layout_egret_medium",
    repo_id="ds4sd/docling-layout-egret-medium",
    revision="main",
    model_path="",
)

DOCLING_LAYOUT_EGRET_LARGE = LayoutModelConfig(
    name="docling_layout_egret_large",
    repo_id="ds4sd/docling-layout-egret-large",
    revision="main",
    model_path="",
)

DOCLING_LAYOUT_EGRET_XLARGE = LayoutModelConfig(
    name="docling_layout_egret_xlarge",
    repo_id="ds4sd/docling-layout-egret-xlarge",
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
