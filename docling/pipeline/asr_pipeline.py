import logging
import re
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Union, cast

from docling.backend.abstract_backend import AbstractDocumentBackend

from docling.datamodel.document import ConversionResult, InputDocument
from docling.datamodel.pipeline_options import (
    AsrPipelineOptions,
)
from docling.datamodel.pipeline_options_vlm_model import (
    InferenceFramework,
)
from docling.datamodel.pipeline_options_asr_model import (
    InlineAsrOptions,
    AsrResponseFormat,
)
from docling.datamodel.settings import settings
from docling.pipeline.base_pipeline import BasePipeline
from docling.utils.profiling import ProfilingScope, TimeRecorder
from docling.datamodel.document import ConversionResult, InputDocument

_log = logging.getLogger(__name__)


class AsrPipeline(BasePipeline):
    def __init__(self, pipeline_options: AsrPipelineOptions):
        super().__init__(pipeline_options)
        self.keep_backend = True

        self.pipeline_options: AsrPipelineOptions

        artifacts_path: Optional[Path] = None
        if pipeline_options.artifacts_path is not None:
            artifacts_path = Path(pipeline_options.artifacts_path).expanduser()
        elif settings.artifacts_path is not None:
            artifacts_path = Path(settings.artifacts_path).expanduser()

        if artifacts_path is not None and not artifacts_path.is_dir():
            raise RuntimeError(
                f"The value of {artifacts_path=} is not valid. "
                "When defined, it must point to a folder containing all models required by the pipeline."
            )

    def _build_document(self, conv_res: ConversionResult) -> ConversionResult:
        total_elapsed_time = 0.0
        with TimeRecorder(conv_res, "doc_build", scope=ProfilingScope.DOCUMENT):
            print("do something")

        return conv_res

    """
    def _determine_status(self, conv_res: ConversionResult) -> ConversionStatus:
        status = ConversionStatus()        
        return status
    """
    
    @classmethod    
    def is_backend_supported(cls, backend: AbstractDocumentBackend):
        return True
