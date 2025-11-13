import hashlib
import logging
import sys
import threading
import time
import warnings
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Optional, Type, Union

from pydantic import ConfigDict, model_validator, validate_call
from typing_extensions import Self

from docling.backend.abstract_backend import AbstractDocumentBackend
from docling.backend.image_backend import ImageDocumentBackend
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import (
    BaseFormatOption,
    ConversionStatus,
    DoclingComponentType,
    DocumentStream,
    ErrorItem,
    InputFormat,
)
from docling.datamodel.document import (
    InputDocument,
    _DocumentConversionInput,  # intentionally reused builder
)
from docling.datamodel.extraction import ExtractionResult, ExtractionTemplateType
from docling.datamodel.pipeline_options import PipelineOptions
from docling.datamodel.settings import (
    DEFAULT_PAGE_RANGE,
    DocumentLimits,
    PageRange,
    settings,
)
from docling.exceptions import ConversionError
from docling.pipeline.base_extraction_pipeline import BaseExtractionPipeline
from docling.pipeline.extraction_vlm_pipeline import ExtractionVlmPipeline
from docling.utils.utils import chunkify

_log = logging.getLogger(__name__)
_PIPELINE_CACHE_LOCK = threading.Lock()


class ExtractionFormatOption(BaseFormatOption):
    """Per-format configuration for extraction.

    Notes:
        - `pipeline_cls` must subclass `BaseExtractionPipeline`.
        - `pipeline_options` is typed as `PipelineOptions` which MUST inherit from
          `BaseOptions` (as used by `BaseExtractionPipeline`).
        - `backend` is the document-opening backend used by `_DocumentConversionInput`.
    """

    pipeline_cls: Type[BaseExtractionPipeline]

    @model_validator(mode="after")
    def set_optional_field_default(self) -> Self:
        if self.pipeline_options is None:
            # `get_default_options` comes from BaseExtractionPipeline
            self.pipeline_options = self.pipeline_cls.get_default_options()  # type: ignore[assignment]
        return self


def _get_default_extraction_option(fmt: InputFormat) -> ExtractionFormatOption:
    """Return the default extraction option for a given input format.

    Defaults mirror the converter's *backend* choices, while the pipeline is
    the VLM extractor. This duplication will be removed when we deduplicate
    the format registry between convert/extract.
    """
    format_to_default_backend: dict[InputFormat, Type[AbstractDocumentBackend]] = {
        InputFormat.IMAGE: ImageDocumentBackend,
        InputFormat.PDF: PyPdfiumDocumentBackend,
    }

    backend = format_to_default_backend.get(fmt)
    if backend is None:
        raise RuntimeError(f"No default extraction backend configured for {fmt}")

    return ExtractionFormatOption(
        pipeline_cls=ExtractionVlmPipeline,
        backend=backend,
    )


class DocumentExtractor:
    """Standalone extractor class.

    Public API:
        - `extract(...) -> ExtractionResult`
        - `extract_all(...) -> Iterator[ExtractionResult]`

    Implementation intentionally reuses `_DocumentConversionInput` to build
    `InputDocument` with the correct backend per format.
    """

    def __init__(
        self,
        allowed_formats: Optional[list[InputFormat]] = None,
        extraction_format_options: Optional[
            dict[InputFormat, ExtractionFormatOption]
        ] = None,
    ) -> None:
        self.allowed_formats: list[InputFormat] = (
            allowed_formats if allowed_formats is not None else list(InputFormat)
        )
        # Build per-format options with defaults, then apply any user overrides
        overrides = extraction_format_options or {}
        self.extraction_format_to_options: dict[InputFormat, ExtractionFormatOption] = {
            fmt: overrides.get(fmt, _get_default_extraction_option(fmt))
            for fmt in self.allowed_formats
        }

        # Cache pipelines by (class, options-hash)
        self._initialized_pipelines: dict[
            tuple[Type[BaseExtractionPipeline], str], BaseExtractionPipeline
        ] = {}

    # ---------------------------- Public API ---------------------------------

    @validate_call(config=ConfigDict(strict=True))
    def extract(
        self,
        source: Union[Path, str, DocumentStream],
        template: ExtractionTemplateType,
        headers: Optional[dict[str, str]] = None,
        raises_on_error: bool = True,
        max_num_pages: int = sys.maxsize,
        max_file_size: int = sys.maxsize,
        page_range: PageRange = DEFAULT_PAGE_RANGE,
    ) -> ExtractionResult:
        all_res = self.extract_all(
            source=[source],
            headers=headers,
            raises_on_error=raises_on_error,
            max_num_pages=max_num_pages,
            max_file_size=max_file_size,
            page_range=page_range,
            template=template,
        )
        return next(all_res)

    @validate_call(config=ConfigDict(strict=True))
    def extract_all(
        self,
        source: Iterable[Union[Path, str, DocumentStream]],
        template: ExtractionTemplateType,
        headers: Optional[dict[str, str]] = None,
        raises_on_error: bool = True,
        max_num_pages: int = sys.maxsize,
        max_file_size: int = sys.maxsize,
        page_range: PageRange = DEFAULT_PAGE_RANGE,
    ) -> Iterator[ExtractionResult]:
        warnings.warn(
            "The extract API is currently experimental and may change without prior notice.\n"
            "Only PDF and image formats are supported.",
            UserWarning,
            stacklevel=2,
        )

        limits = DocumentLimits(
            max_num_pages=max_num_pages,
            max_file_size=max_file_size,
            page_range=page_range,
        )
        conv_input = _DocumentConversionInput(
            path_or_stream_iterator=source, limits=limits, headers=headers
        )

        ext_res_iter = self._extract(
            conv_input, raises_on_error=raises_on_error, template=template
        )

        had_result = False
        for ext_res in ext_res_iter:
            had_result = True
            if raises_on_error and ext_res.status not in {
                ConversionStatus.SUCCESS,
                ConversionStatus.PARTIAL_SUCCESS,
            }:
                raise ConversionError(
                    f"Extraction failed for: {ext_res.input.file} with status: {ext_res.status}"
                )
            else:
                yield ext_res

        if not had_result and raises_on_error:
            raise ConversionError(
                "Extraction failed because the provided file has no recognizable format or it wasn't in the list of allowed formats."
            )

    # --------------------------- Internal engine ------------------------------

    def _extract(
        self,
        conv_input: _DocumentConversionInput,
        raises_on_error: bool,
        template: ExtractionTemplateType,
    ) -> Iterator[ExtractionResult]:
        start_time = time.monotonic()

        for input_batch in chunkify(
            conv_input.docs(self.extraction_format_to_options),
            settings.perf.doc_batch_size,
        ):
            _log.info("Going to extract document batch...")
            process_func = partial(
                self._process_document_extraction,
                raises_on_error=raises_on_error,
                template=template,
            )

            if (
                settings.perf.doc_batch_concurrency > 1
                and settings.perf.doc_batch_size > 1
            ):
                with ThreadPoolExecutor(
                    max_workers=settings.perf.doc_batch_concurrency
                ) as pool:
                    for item in pool.map(
                        process_func,
                        input_batch,
                    ):
                        yield item
            else:
                for item in map(
                    process_func,
                    input_batch,
                ):
                    elapsed = time.monotonic() - start_time
                    start_time = time.monotonic()
                    _log.info(
                        f"Finished extracting document {item.input.file.name} in {elapsed:.2f} sec."
                    )
                    yield item

    def _process_document_extraction(
        self,
        in_doc: InputDocument,
        raises_on_error: bool,
        template: ExtractionTemplateType,
    ) -> ExtractionResult:
        valid = (
            self.allowed_formats is not None and in_doc.format in self.allowed_formats
        )
        if valid:
            return self._execute_extraction_pipeline(
                in_doc, raises_on_error=raises_on_error, template=template
            )
        else:
            error_message = f"File format not allowed: {in_doc.file}"
            if raises_on_error:
                raise ConversionError(error_message)
            else:
                error_item = ErrorItem(
                    component_type=DoclingComponentType.USER_INPUT,
                    module_name="",
                    error_message=error_message,
                )
                return ExtractionResult(
                    input=in_doc, status=ConversionStatus.SKIPPED, errors=[error_item]
                )

    def _execute_extraction_pipeline(
        self,
        in_doc: InputDocument,
        raises_on_error: bool,
        template: ExtractionTemplateType,
    ) -> ExtractionResult:
        if not in_doc.valid:
            if raises_on_error:
                raise ConversionError(f"Input document {in_doc.file} is not valid.")
            else:
                return ExtractionResult(input=in_doc, status=ConversionStatus.FAILURE)

        pipeline = self._get_pipeline(in_doc.format)
        if pipeline is None:
            if raises_on_error:
                raise ConversionError(
                    f"No extraction pipeline could be initialized for {in_doc.file}."
                )
            else:
                return ExtractionResult(input=in_doc, status=ConversionStatus.FAILURE)

        return pipeline.execute(
            in_doc, raises_on_error=raises_on_error, template=template
        )

    def _get_pipeline(
        self, doc_format: InputFormat
    ) -> Optional[BaseExtractionPipeline]:
        """Retrieve or initialize a pipeline, reusing instances based on class and options."""
        fopt = self.extraction_format_to_options.get(doc_format)
        if fopt is None or fopt.pipeline_options is None:
            return None

        pipeline_class = fopt.pipeline_cls
        pipeline_options = fopt.pipeline_options
        options_hash = self._get_pipeline_options_hash(pipeline_options)

        cache_key = (pipeline_class, options_hash)
        with _PIPELINE_CACHE_LOCK:
            if cache_key not in self._initialized_pipelines:
                _log.info(
                    f"Initializing extraction pipeline for {pipeline_class.__name__} with options hash {options_hash}"
                )
                self._initialized_pipelines[cache_key] = pipeline_class(
                    pipeline_options=pipeline_options  # type: ignore[arg-type]
                )
            else:
                _log.debug(
                    f"Reusing cached extraction pipeline for {pipeline_class.__name__} with options hash {options_hash}"
                )

            return self._initialized_pipelines[cache_key]

    @staticmethod
    def _get_pipeline_options_hash(pipeline_options: PipelineOptions) -> str:
        """Generate a stable hash of pipeline options to use as part of the cache key."""
        options_str = str(pipeline_options.model_dump())
        return hashlib.md5(
            options_str.encode("utf-8"), usedforsecurity=False
        ).hexdigest()
