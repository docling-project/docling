# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Deprecated layout model specs, kept alive only by `LayoutOptions`.

This whole module is removable once the deprecated `LayoutOptions` is dropped.
`LayoutModelConfig` is structurally `ObjectDetectionModelSpec` minus
`engine_overrides`, and every constant below has an equivalent preset
registered on `LayoutObjectDetectionOptions` (see `stage_model_specs.py`):

    DOCLING_LAYOUT_HERON        -> "layout_heron_default"
    DOCLING_LAYOUT_HERON_101    -> "layout_heron_101"
    DOCLING_LAYOUT_EGRET_MEDIUM -> "layout_egret_medium"
    DOCLING_LAYOUT_EGRET_LARGE  -> "layout_egret_large"
    DOCLING_LAYOUT_EGRET_XLARGE -> "layout_egret_xlarge"
    DOCLING_LAYOUT_V2           -> none; unsupported, falls back to Heron

They are retained purely so `LayoutOptions(model_spec=DOCLING_LAYOUT_*)` keeps
working. Deleting them is a public interface break, so it waits for the same
release that removes `LayoutOptions`; `_translate` in
`models/stages/layout/layout_model.py` goes at the same time.
"""

from typing import Annotated

from pydantic import BaseModel, Field


class LayoutModelConfig(BaseModel):
    """Configuration for document layout analysis models from HuggingFace.

    Deprecated together with `LayoutOptions`; use `ObjectDetectionModelSpec`.
    """

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


# HuggingFace Layout Models. Removable with `LayoutOptions` — see module
# docstring for the preset each one maps to.

# Unsupported: `_translate` warns and substitutes Heron.
DOCLING_LAYOUT_V2 = LayoutModelConfig(
    name="docling_layout_v2",
    repo_id="docling-project/docling-layout-old",
    revision="main",
)

DOCLING_LAYOUT_HERON = LayoutModelConfig(
    name="docling_layout_heron",
    repo_id="docling-project/docling-layout-heron",
    revision="main",
)

DOCLING_LAYOUT_HERON_101 = LayoutModelConfig(
    name="docling_layout_heron_101",
    repo_id="docling-project/docling-layout-heron-101",
    revision="main",
)

DOCLING_LAYOUT_EGRET_MEDIUM = LayoutModelConfig(
    name="docling_layout_egret_medium",
    repo_id="docling-project/docling-layout-egret-medium",
    revision="main",
)

DOCLING_LAYOUT_EGRET_LARGE = LayoutModelConfig(
    name="docling_layout_egret_large",
    repo_id="docling-project/docling-layout-egret-large",
    revision="main",
)

DOCLING_LAYOUT_EGRET_XLARGE = LayoutModelConfig(
    name="docling_layout_egret_xlarge",
    repo_id="docling-project/docling-layout-egret-xlarge",
    revision="main",
)

# Custom models should be defined as an ObjectDetectionModelSpec and passed to
# LayoutObjectDetectionOptions, not added here.
