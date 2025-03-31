import functools
import logging
import time
import traceback
from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, List

from docling_core.types.doc import DoclingDocument, NodeItem

from docling.backend.abstract_backend import AbstractDocumentBackend
from docling.backend.pdf_backend import PdfDocumentBackend
from docling.datamodel.base_models import (
    ConversionStatus,
    DoclingComponentType,
    ErrorItem,
    Page,
)
from docling.datamodel.document import ConversionResult, InputDocument
from docling.datamodel.pipeline_options import PipelineOptions
from docling.datamodel.settings import settings
from docling.models.base_model import GenericEnrichmentModel
from docling.utils.profiling import ProfilingScope, TimeRecorder
from docling.utils.utils import chunkify

_log = logging.getLogger(__name__)


class BasePipeline(ABC):
    def __init__(self, pipeline_options: PipelineOptions):
        self.pipeline_options = pipeline_options
        self.keep_images = False
        self.build_pipe: List[Callable] = []
        self.enrichment_pipe: List[GenericEnrichmentModel[Any]] = []

    def execute(self, in_doc: InputDocument, raises_on_error: bool) -> ConversionResult:
        conv_res = ConversionResult(input=in_doc)

        _log.info(f"Processing document {in_doc.file.name}")
        try:
            with TimeRecorder(
                conv_res, "pipeline_total", scope=ProfilingScope.DOCUMENT
            ):
                # These steps are building and assembling the structure of the
                # output DoclingDocument.
                conv_res = self._build_document(conv_res)
                conv_res = self._assemble_document(conv_res)
                # From this stage, all operations should rely only on conv_res.output
                conv_res = self._enrich_document(conv_res)
                conv_res.status = self._determine_status(conv_res)
        except Exception as e:
            conv_res.status = ConversionStatus.FAILURE
            if raises_on_error:
                raise e
        finally:
            self._unload(conv_res)

        return conv_res

    @abstractmethod
    def _build_document(self, conv_res: ConversionResult) -> ConversionResult:
        pass

    def _assemble_document(self, conv_res: ConversionResult) -> ConversionResult:
        return conv_res

    def _enrich_document(self, conv_res: ConversionResult) -> ConversionResult:

        def _prepare_elements(
            conv_res: ConversionResult, model: GenericEnrichmentModel[Any]
        ) -> Iterable[NodeItem]:
            for doc_element, _level in conv_res.document.iterate_items():
                prepared_element = model.prepare_element(
                    conv_res=conv_res, element=doc_element
                )
                if prepared_element is not None:
                    yield prepared_element

        with TimeRecorder(conv_res, "doc_enrich", scope=ProfilingScope.DOCUMENT):
            for model in self.enrichment_pipe:
                for element_batch in chunkify(
                    _prepare_elements(conv_res, model),
                    model.elements_batch_size,
                ):
                    for element in model(
                        doc=conv_res.document, element_batch=element_batch
                    ):  # Must exhaust!
                        pass

        return conv_res

    @abstractmethod
    def _determine_status(self, conv_res: ConversionResult) -> ConversionStatus:
        pass

    def _unload(self, conv_res: ConversionResult):
        pass

    @classmethod
    @abstractmethod
    def get_default_options(cls) -> PipelineOptions:
        pass

    @classmethod
    @abstractmethod
    def is_backend_supported(cls, backend: AbstractDocumentBackend):
        pass

    # def _apply_on_elements(self, element_batch: Iterable[NodeItem]) -> Iterable[Any]:
    #    for model in self.build_pipe:
    #        element_batch = model(element_batch)
    #
    #    yield from element_batch


class PaginatedPipeline(BasePipeline):  # TODO this is a bad name.

    def __init__(self, pipeline_options: PipelineOptions):
        super().__init__(pipeline_options)
        self.keep_backend = False

    def _apply_on_pages(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        for model in self.build_pipe:
            page_batch = model(conv_res, page_batch)

        yield from page_batch

    def _build_document(self, conv_res: ConversionResult) -> ConversionResult:

        if not isinstance(conv_res.input._backend, PdfDocumentBackend):
            raise RuntimeError(
                f"The selected backend {type(conv_res.input._backend).__name__} for {conv_res.input.file} is not a PDF backend. "
                f"Can not convert this with a PDF pipeline. "
                f"Please check your format configuration on DocumentConverter."
            )
            # conv_res.status = ConversionStatus.FAILURE
            # return conv_res

        total_elapsed_time = 0.0
        with TimeRecorder(conv_res, "doc_build", scope=ProfilingScope.DOCUMENT):

            for i in range(0, conv_res.input.page_count):
                start_page, end_page = conv_res.input.limits.page_range
                if (start_page - 1) <= i <= (end_page - 1):
                    conv_res.pages.append(Page(page_no=i))

            try:
                # Iterate batches of pages (page_batch_size) in the doc
                for page_batch in chunkify(
                    conv_res.pages, settings.perf.page_batch_size
                ):
                    start_batch_time = time.monotonic()

                    # 1. Initialise the page resources
                    init_pages = map(
                        functools.partial(self.initialize_page, conv_res), page_batch
                    )

                    # 2. Run pipeline stages
                    pipeline_pages = self._apply_on_pages(conv_res, init_pages)

                    for p in pipeline_pages:  # Must exhaust!

                        # Cleanup cached images
                        if not self.keep_images:
                            p._image_cache = {}

                        # Cleanup page backends
                        if not self.keep_backend and p._backend is not None:
                            p._backend.unload()

                    end_batch_time = time.monotonic()
                    total_elapsed_time += end_batch_time - start_batch_time
                    if (
                        self.pipeline_options.document_timeout is not None
                        and total_elapsed_time > self.pipeline_options.document_timeout
                    ):
                        _log.warning(
                            f"Document processing time ({total_elapsed_time:.3f} seconds) exceeded the specified timeout of {self.pipeline_options.document_timeout:.3f} seconds"
                        )
                        conv_res.status = ConversionStatus.PARTIAL_SUCCESS
                        break

                    _log.debug(
                        f"Finished converting page batch time={end_batch_time:.3f}"
                    )

            except Exception as e:
                conv_res.status = ConversionStatus.FAILURE
                trace = "\n".join(
                    traceback.format_exception(type(e), e, e.__traceback__)
                )
                _log.warning(
                    f"Encountered an error during conversion of document {conv_res.input.document_hash}:\n"
                    f"{trace}"
                )
                raise e

        return conv_res

    def _unload(self, conv_res: ConversionResult) -> ConversionResult:
        for page in conv_res.pages:
            if page._backend is not None:
                page._backend.unload()

        if conv_res.input._backend:
            conv_res.input._backend.unload()

        return conv_res

    def _determine_status(self, conv_res: ConversionResult) -> ConversionStatus:
        status = ConversionStatus.SUCCESS
        for page in conv_res.pages:
            if page._backend is None or not page._backend.is_valid():
                conv_res.errors.append(
                    ErrorItem(
                        component_type=DoclingComponentType.DOCUMENT_BACKEND,
                        module_name=type(page._backend).__name__,
                        error_message=f"Page {page.page_no} failed to parse.",
                    )
                )
                status = ConversionStatus.PARTIAL_SUCCESS

        return status

    # Initialise and load resources for a page
    @abstractmethod
    def initialize_page(self, conv_res: ConversionResult, page: Page) -> Page:
        pass
