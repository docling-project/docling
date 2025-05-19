import logging
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Union, cast

from docling.backend.abstract_backend import (
    AbstractDocumentBackend,
    DeclarativeDocumentBackend,
)
from docling.datamodel.base_models import ConversionStatus
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    AsrPipelineOptions,
    HuggingFaceAsrOptions,
    InferenceFramework,
    PipelineOptions,
    ResponseFormat,
)
from docling.models.hf_asr_models.asr_nemo import AsrNemoModel
from docling.pipeline.base_pipeline import BasePipeline
from docling.utils.profiling import ProfilingScope, TimeRecorder

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

        if isinstance(self.pipeline_options.asr_options, HuggingFaceAsrOptions):
            asr_options = cast(HuggingFaceAsrOptions, self.pipeline_options.asr_options)
            if asr_options.inference_framework == InferenceFramework.ASR_NENO:
                self.build_pipe = [
                    AsrNemoModel(
                        enabled=True,  # must be always enabled for this pipeline to make sense.
                        artifacts_path=artifacts_path,
                        accelerator_options=pipeline_options.accelerator_options,
                        asr_options=asr_options,
                    ),
                ]
            else:
                _log.error(f"{asr_options.inference_framework} is not supported")

        else:
            _log.error("ASR is not supported")

    def _build_document(self, conv_res: ConversionResult) -> ConversionResult:
        pass

    def _assemble_document(self, conv_res: ConversionResult) -> ConversionResult:
        return conv_res

    def _determine_status(self, conv_res: ConversionResult) -> ConversionStatus:
        pass

    def _unload(self, conv_res: ConversionResult):
        pass

    @classmethod
    def get_default_options(cls) -> PipelineOptions:
        pass

    @classmethod
    def is_backend_supported(cls, backend: AbstractDocumentBackend):
        pass
