"""Options for the threaded layout+VLM pipeline."""

from typing import Union

from docling.datamodel.layout_model_specs import DOCLING_LAYOUT_HERON
from docling.datamodel.pipeline_options import LayoutOptions, PaginatedPipelineOptions
from docling.datamodel.pipeline_options_vlm_model import (
    ApiVlmOptions,
    InlineVlmOptions,
)
from docling.datamodel.vlm_model_specs import GRANITEDOCLING_TRANSFORMERS


class ThreadedLayoutVlmPipelineOptions(PaginatedPipelineOptions):
    """Pipeline options for the threaded layout+VLM pipeline."""

    images_scale: float = 2.0

    # VLM configuration (will be enhanced with layout awareness by the pipeline)
    vlm_options: Union[InlineVlmOptions, ApiVlmOptions] = GRANITEDOCLING_TRANSFORMERS

    # Layout model configuration
    layout_options: LayoutOptions = LayoutOptions(
        model_spec=DOCLING_LAYOUT_HERON, skip_cell_assignment=True
    )

    # Threading and batching controls
    layout_batch_size: int = 4
    vlm_batch_size: int = 4
    batch_timeout_seconds: float = 2.0
    queue_max_size: int = 50
