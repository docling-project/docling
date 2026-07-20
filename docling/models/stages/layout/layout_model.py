"""Deprecated layout stage kept as a thin shim over the object-detection model.

This whole module is removable once `LayoutOptions` is dropped: it exists only
so `PdfPipelineOptions(layout_options=LayoutOptions(...))` still resolves
through `LayoutFactory`. Nothing here runs inference — `LayoutModel` inherits
all of that from `LayoutObjectDetectionModel`.
"""

import warnings
from pathlib import Path
from typing import Optional

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.layout_model_specs import (
    DOCLING_LAYOUT_HERON,
    DOCLING_LAYOUT_V2,
)
from docling.datamodel.pipeline_options import (
    LayoutObjectDetectionOptions,
    LayoutOptions,
)
from docling.datamodel.stage_model_specs import ObjectDetectionModelSpec
from docling.models.stages.layout.layout_object_detection_model import (
    LayoutObjectDetectionModel,
)


def _translate(options: LayoutOptions) -> LayoutObjectDetectionOptions:
    """Map deprecated `LayoutOptions` onto `LayoutObjectDetectionOptions`.

    Removable with `LayoutOptions`: `LayoutModelConfig` is structurally
    `ObjectDetectionModelSpec` minus `engine_overrides`, so this is a
    field copy plus the retired-model fallback.
    """
    cfg = options.model_spec
    if cfg.repo_id == DOCLING_LAYOUT_V2.repo_id:
        warnings.warn(
            "DOCLING_LAYOUT_V2 is no longer supported; "
            "using DOCLING_LAYOUT_HERON instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        cfg = DOCLING_LAYOUT_HERON
    return LayoutObjectDetectionOptions(
        model_spec=ObjectDetectionModelSpec(
            name=cfg.name, repo_id=cfg.repo_id, revision=cfg.revision
        ),
        # engine_options omitted on purpose: the object-detection default
        # applies, so there is no second place to keep in sync.
        keep_empty_clusters=options.keep_empty_clusters,
        skip_cell_assignment=options.skip_cell_assignment,
        create_orphan_clusters=options.create_orphan_clusters,
    )


class LayoutModel(LayoutObjectDetectionModel):
    """Deprecated. Use `LayoutObjectDetectionModel` / `LayoutObjectDetectionOptions`."""

    def __init__(
        self,
        artifacts_path: Optional[Path],
        accelerator_options: AcceleratorOptions,
        options: LayoutOptions,
        enable_remote_services: bool = False,
    ) -> None:
        warnings.warn(
            "LayoutModel is deprecated and will be removed in a future release. "
            "Use LayoutObjectDetectionModel with LayoutObjectDetectionOptions.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(
            artifacts_path=artifacts_path,
            accelerator_options=accelerator_options,
            options=_translate(options),
            enable_remote_services=enable_remote_services,
        )

    @classmethod
    def get_options_type(cls) -> type[LayoutOptions]:  # ty: ignore[invalid-method-override]
        # LayoutFactory dispatches on the exact options type, so the shim must
        # keep claiming LayoutOptions rather than inheriting the parent's.
        return LayoutOptions

