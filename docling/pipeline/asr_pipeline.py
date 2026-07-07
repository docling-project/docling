import logging

from docling_core.types.doc import DoclingDocument

from docling.backend.abstract_backend import AbstractDocumentBackend
from docling.backend.noop_backend import NoOpBackend
from docling.datamodel.base_models import ConversionStatus
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import AsrPipelineOptions

# Re-export moved symbols so existing imports
# (`from docling.pipeline.asr_pipeline import _NativeWhisperModel`, etc.)
# keep resolving. New code should import from asr_transcriber directly.
from docling.pipeline.asr_transcriber import (
    _AsrModelFactory,
    _AsrTranscriber,
    _ConversationItem,
    _ConversationWord,
    _MlxWhisperModel,
    _NativeWhisperModel,
    _process_conversation,
    _WhisperS2TModel,
)
from docling.pipeline.base_pipeline import BasePipeline
from docling.utils.profiling import ProfilingScope, TimeRecorder

_log = logging.getLogger(__name__)


class AsrPipeline(BasePipeline):
    def __init__(self, pipeline_options: AsrPipelineOptions):
        super().__init__(pipeline_options)
        self.keep_backend = True

        self.pipeline_options: AsrPipelineOptions = pipeline_options
        self._model: _AsrTranscriber = _AsrModelFactory.create(
            asr_options=self.pipeline_options.asr_options,
            artifacts_path=self.artifacts_path,
            accelerator_options=pipeline_options.accelerator_options,
        )

    def _has_text(self, document: "DoclingDocument") -> bool:
        if not document or not document.texts:
            return False
        for item in document.texts:
            if item.text and item.text.strip():
                return True
        return False

    def _determine_status(self, conv_res: ConversionResult) -> ConversionStatus:
        if conv_res.status == ConversionStatus.FAILURE or conv_res.errors:
            return ConversionStatus.FAILURE
        if not self._has_text(conv_res.document):
            _log.warning(
                "ASR conversion resulted in an empty document."
                f"File: {conv_res.input.file.name}"
            )
            return ConversionStatus.PARTIAL_SUCCESS
        return ConversionStatus.SUCCESS

    @classmethod
    def get_default_options(cls) -> AsrPipelineOptions:
        return AsrPipelineOptions()

    def _build_document(self, conv_res: ConversionResult) -> ConversionResult:
        _log.info("Transcribing audio document %s.", conv_res.input.file.name)
        with TimeRecorder(conv_res, "doc_build", scope=ProfilingScope.DOCUMENT):
            self._model.run(conv_res=conv_res)
        return conv_res

    @classmethod
    def is_backend_supported(cls, backend: AbstractDocumentBackend):
        return isinstance(backend, NoOpBackend)
