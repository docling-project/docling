# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from unittest.mock import Mock, patch

from docling.datamodel.accelerator_options import AcceleratorDevice
from docling.datamodel.pipeline_options import VlmExtractionPipelineOptions
from docling.datamodel.vlm_engine_options import TransformersVlmEngineOptions
from docling.datamodel.vlm_model_specs import GRANITE_VISION_4_1_TRANSFORMERS


@patch(
    "docling.models.extraction.transformers_extraction_model.TransformersExtractionModel",
)
def test_pipeline_preserves_transformers_engine_options(mock_model_cls: Mock) -> None:
    from docling.pipeline.extraction_vlm_pipeline import ExtractionVlmPipeline

    vlm_options = GRANITE_VISION_4_1_TRANSFORMERS.model_copy(
        update={
            "engine_options": TransformersVlmEngineOptions(
                device=AcceleratorDevice.CPU,
                torch_dtype="float16",
                trust_remote_code=False,
                compile_model=False,
            )
        }
    )
    options = VlmExtractionPipelineOptions(vlm_options=vlm_options)
    _ = ExtractionVlmPipeline(pipeline_options=options)
    mock_model_cls.assert_called_once()
    call_kwargs = mock_model_cls.call_args.kwargs
    assert call_kwargs["vlm_options"] is vlm_options
