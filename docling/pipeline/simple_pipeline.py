import logging

from docling.backend.abstract_backend import (
    AbstractDocumentBackend,
    DeclarativeDocumentBackend,
)
from docling.datamodel.base_models import ConversionStatus
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import PipelineOptions
from docling.pipeline.base_pipeline import BasePipeline
from docling.utils.profiling import ProfilingScope, TimeRecorder

_log = logging.getLogger(__name__)


class SimplePipeline(BasePipeline):
    """SimpleModelPipeline.

    This class is used at the moment for formats / backends
    which produce straight DoclingDocument output.
    """

    def __init__(self, pipeline_options: PipelineOptions):
        super().__init__(pipeline_options)

    def _build_document(self, conv_res: ConversionResult) -> ConversionResult:
        if not isinstance(conv_res.input._backend, DeclarativeDocumentBackend):
            raise RuntimeError(
                f"The selected backend {type(conv_res.input._backend).__name__} for {conv_res.input.file} is not a declarative backend. "
                f"Can not convert this with simple pipeline. "
                f"Please check your format configuration on DocumentConverter."
            )
            # conv_res.status = ConversionStatus.FAILURE
            # return conv_res

        # Instead of running a page-level pipeline to build up the document structure,
        # the backend is expected to be of type DeclarativeDocumentBackend, which can output
        # a DoclingDocument straight.
        with TimeRecorder(conv_res, "doc_build", scope=ProfilingScope.DOCUMENT):
            conv_res.document = conv_res.input._backend.convert()
        return conv_res

    def _determine_status(self, conv_res: ConversionResult) -> ConversionStatus:
        # This is called only if the previous steps didn't raise.
        # Since we don't have anything else to evaluate, we can
        # safely return SUCCESS.
        return ConversionStatus.SUCCESS

    @classmethod
    def get_default_options(cls) -> PipelineOptions:
        return PipelineOptions()

    @classmethod
    def is_backend_supported(cls, backend: AbstractDocumentBackend):
        return isinstance(backend, DeclarativeDocumentBackend)
