# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Options for the crop-based layout+VLM pipeline."""

from docling_core.types.doc import DocItemLabel
from pydantic import BaseModel, Field, model_validator

from docling.datamodel.pipeline_options import (
    BaseLayoutOptions,
    LayoutObjectDetectionOptions,
    ThreadedPdfPipelineOptions,
    VlmConvertOptions,
)
from docling.datamodel.pipeline_options_vlm_model import ResponseFormat
from docling.datamodel.vlm_prompts import DocLangTask, doclang_prompt


class CropTask(BaseModel):
    """How one kind of layout region is sent to the VLM."""

    prompt: str = Field(description="User prompt sent along with the crop.")
    response_prefix: str | None = Field(
        default=None,
        description=(
            "Opening of the reply the model is forced to continue from. Pins the "
            "element type to the one the layout model detected."
        ),
    )
    max_tiles: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Upper bound on the image tiles the processor cuts the crop into "
            "(Granite Docling v2 `max_patches`). None keeps the processor's "
            "own limit."
        ),
    )


# Granite Docling v2 is trained on `<doclang> [<task>]` prompts for single
# elements; without `<layout>` in the task list the reply carries no location
# tokens, which is what we want because the layout model owns the boxes.
DOCLANG_OCR_TASK = CropTask(
    prompt=doclang_prompt([DocLangTask.OCR]), response_prefix="<doclang>"
)
DOCLANG_TABLE_TASK = CropTask(
    prompt=doclang_prompt([DocLangTask.TABLE]), response_prefix="<doclang><table>"
)
DOCLANG_FORMULA_TASK = CropTask(
    prompt=doclang_prompt([DocLangTask.FORMULA]),
    response_prefix="<doclang><formula>",
)
# A code element opens with a `<label>` naming its language, which only the model
# can tell; prefilling past `<code>` makes it skip the label and derail.
DOCLANG_CODE_TASK = CropTask(
    prompt=doclang_prompt([DocLangTask.CODE]), response_prefix="<doclang>"
)


def _default_crop_tasks() -> dict[DocItemLabel, CropTask]:
    return {
        DocItemLabel.TABLE: DOCLANG_TABLE_TASK,
        DocItemLabel.DOCUMENT_INDEX: DOCLANG_TABLE_TASK,
        DocItemLabel.FORMULA: DOCLANG_FORMULA_TASK,
        DocItemLabel.CODE: DOCLANG_CODE_TASK,
    }


# The processor stretches every image to the tile grid that best matches its
# aspect ratio, with up to 32 tiles, so a small crop is blown up to many tiles.
# On OmniDocBench crops, Granite Docling v2 500M reads text, formulas and code
# as well or better with at most 8 tiles and an 8-pixel minimum margin, at less
# than half the image tokens. The 2B model reads small text worse with that cap,
# so it keeps the processor's limits. Tables always keep the full grid.
# default_repo_id -> (crop_min_padding in points, max_tiles for non-table tasks)
_MODEL_CROP_DEFAULTS: dict[str, tuple[float, int]] = {
    "ibm-granite/granite-docling-v2-500m": (4.0, 8),
}
_FULL_GRID_LABELS = {DocItemLabel.TABLE, DocItemLabel.DOCUMENT_INDEX}


def _with_max_tiles(task: CropTask, max_tiles: int) -> CropTask:
    if task.max_tiles is not None:
        return task
    return task.model_copy(update={"max_tiles": max_tiles})


class LayoutCropVlmOptions(BaseModel):
    """Options of the stage that recognizes layout regions with a VLM."""

    vlm_options: VlmConvertOptions = Field(
        default_factory=lambda: VlmConvertOptions.from_preset(
            "granite_for_docling_500m"
        ),
        description="Model and engine that recognize the crops.",
    )
    default_task: CropTask = Field(
        default=DOCLANG_OCR_TASK,
        description="Task for every text-like region without an entry in `tasks`.",
    )
    tasks: dict[DocItemLabel, CropTask] = Field(
        default_factory=_default_crop_tasks,
        description="Task per layout label, overriding `default_task`.",
    )
    use_response_prefix: bool = Field(
        default=True,
        description=(
            "Force the reply to open with the task's `response_prefix`. Disable for "
            "API servers that cannot continue a partial assistant message."
        ),
    )
    crop_scale: float = Field(
        default=2.0,
        description="Render scale of the crops, relative to 72 DPI.",
    )
    crop_padding: float = Field(
        default=0.02,
        description="Margin added around each region, as a fraction of its size.",
    )
    crop_min_padding: float = Field(
        default=0.0,
        description=(
            "Smallest margin around each region, in points. A fractional margin "
            "alone leaves almost no room around narrow regions. Defaults to 4 "
            "points (8 pixels at the default `crop_scale`) for Granite for "
            "Docling 500M, 0 otherwise."
        ),
    )
    native_canvas_max_aspect: float = Field(
        default=0.5,
        description=(
            "Crops with width / height below this ratio (vertical text) are not "
            "stretched to the tile grid: they keep their aspect ratio, upscaled to "
            "a short side of 40 pixels at most 3x, centred on a canvas of whole "
            "512-pixel tiles. 0 disables."
        ),
    )
    engine_batch_size: int = Field(
        default=16,
        ge=1,
        description="Crops handed to the VLM engine in one call.",
    )
    max_new_tokens: int | None = Field(
        default=None,
        description="Token budget per crop. Defaults to the model spec's budget.",
    )

    @model_validator(mode="after")
    def apply_model_crop_defaults(self) -> "LayoutCropVlmOptions":
        """Apply the crop sizing measured for the selected model.

        Only fields the caller left unset are filled in.
        """
        repo_id = self.vlm_options.model_spec.default_repo_id
        defaults = _MODEL_CROP_DEFAULTS.get(repo_id)
        if defaults is None:
            return self
        min_padding, max_tiles = defaults
        if "crop_min_padding" not in self.model_fields_set:
            self.crop_min_padding = min_padding
        if "default_task" not in self.model_fields_set:
            self.default_task = _with_max_tiles(self.default_task, max_tiles)
        if "tasks" not in self.model_fields_set:
            self.tasks = {
                label: task
                if label in _FULL_GRID_LABELS
                else _with_max_tiles(task, max_tiles)
                for label, task in self.tasks.items()
            }
        return self

    @model_validator(mode="after")
    def validate_response_format(self) -> "LayoutCropVlmOptions":
        response_format = self.vlm_options.model_spec.response_format
        if response_format != ResponseFormat.DOCLANG:
            raise ValueError(
                "The crop-based layout+VLM pipeline parses DocLang replies, "
                f"but the model spec declares {response_format}."
            )
        return self


class LayoutCropVlmPipelineOptions(ThreadedPdfPipelineOptions):
    """Pipeline options for the crop-based layout+VLM pipeline.

    The layout model finds the regions of a page and a VLM recognizes each region
    from its crop; OCR and the table-structure model are not used.
    """

    do_ocr: bool = False
    do_table_structure: bool = False

    # Regions carry no text before the VLM ran, so they must survive
    # post-processing empty.
    layout_options: BaseLayoutOptions = Field(
        default_factory=lambda: LayoutObjectDetectionOptions(
            skip_cell_assignment=True, keep_empty_clusters=True
        ),
    )

    crop_vlm_options: LayoutCropVlmOptions = Field(
        default_factory=LayoutCropVlmOptions,
    )
    crop_vlm_batch_size: int = Field(
        default=4,
        description="Pages whose crops are sent to the VLM engine together.",
    )
