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


# Granite for Docling is trained on `<doclang> [<task>]` prompts for single
# elements; without `<layout>` in the task list the reply carries no location
# tokens, which is what we want because the layout model owns the boxes.
DOCLANG_OCR_TASK = CropTask(prompt="<doclang> [<ocr>]", response_prefix="<doclang>")
DOCLANG_TABLE_TASK = CropTask(
    prompt="<doclang> [<table>]", response_prefix="<doclang><table>"
)
DOCLANG_FORMULA_TASK = CropTask(
    prompt="<doclang> [<formula>]", response_prefix="<doclang><formula>"
)
# A code element opens with a `<label>` naming its language, which only the model
# can tell; prefilling past `<code>` makes it skip the label and derail.
DOCLANG_CODE_TASK = CropTask(prompt="<doclang> [<code>]", response_prefix="<doclang>")


def _default_crop_tasks() -> dict[DocItemLabel, CropTask]:
    return {
        DocItemLabel.TABLE: DOCLANG_TABLE_TASK,
        DocItemLabel.DOCUMENT_INDEX: DOCLANG_TABLE_TASK,
        DocItemLabel.FORMULA: DOCLANG_FORMULA_TASK,
        DocItemLabel.CODE: DOCLANG_CODE_TASK,
    }


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
    engine_batch_size: int = Field(
        default=16,
        description="Crops handed to the VLM engine in one call.",
    )
    max_new_tokens: int | None = Field(
        default=None,
        description="Token budget per crop. Defaults to the model spec's budget.",
    )

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
