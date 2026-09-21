# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import logging
import sys
import threading
import time
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Optional, Type, Union, overload

from pydantic import ConfigDict, model_validator, validate_call
from typing_extensions import Self

from docling.backend.abstract_backend import AbstractDocumentBackend
from docling.backend.image_backend import ImageDocumentBackend
from docling.backend.xml.doclang_archive_backend import DocLangArchiveBackend
from docling.datamodel.base_models import (
    BaseFormatOption,
    ConversionStatus,
    DoclingComponentType,
    DocumentStream,
    ErrorItem,
    FailureCategory,
    InputFormat,
)
from docling.datamodel.document import (
    InputDocument,
    _DocumentConversionInput,  # intentionally reused builder
    build_invalid_input_errors,
)
from docling.datamodel.extraction import (
    DocumentExtractionResult,
    ExtractionResult,
    ExtractionTarget,
    ExtractionTemplateType,
    PageScope,
    _legacy_result,
)
from docling.datamodel.pipeline_options import PipelineOptions
from docling.datamodel.settings import (
    DEFAULT_PAGE_RANGE,
    DocumentLimits,
    PageRange,
    settings,
)
from docling.exceptions import ConversionError
from docling.models.extraction.prompt_utils import normalize_extraction_call
from docling.pipeline.base_extraction_pipeline import BaseExtractionPipeline
from docling.pipeline.extraction_vlm_pipeline import ExtractionVlmPipeline
from docling.utils.pipeline_cache import create_pipeline_options_hash
from docling.utils.utils import chunkify

_log = logging.getLogger(__name__)
_PIPELINE_CACHE_LOCK = threading.Lock()
DEFAULT_EXTRACTION_FORMATS = [
    InputFormat.IMAGE,
    InputFormat.PDF,
    InputFormat.DOCX,
    InputFormat.HTML,
    InputFormat.MD,
    InputFormat.DCLX,
]


class ExtractionFormatOption(BaseFormatOption):
    """Per-format extraction configuration."""

    pipeline_cls: Type[BaseExtractionPipeline]
    backend: Optional[Type[AbstractDocumentBackend]] = None

    @model_validator(mode="after")
    def set_optional_field_default(self) -> Self:
        if self.pipeline_options is None:
            # `get_default_options` comes from BaseExtractionPipeline
            self.pipeline_options = self.pipeline_cls.get_default_options()  # type: ignore[assignment]
        return self


def _get_default_extraction_option(fmt: InputFormat) -> ExtractionFormatOption:
    """Return the default extraction option for a supported format."""
    if fmt == InputFormat.PDF:
        from docling.backend.docling_parse_backend import (
            ThreadedDoclingParseDocumentBackend,
        )

        backend: Type[AbstractDocumentBackend] | None = (
            ThreadedDoclingParseDocumentBackend
        )
    elif fmt == InputFormat.DOCX:
        from docling.backend.msword_backend import MsWordDocumentBackend

        backend = MsWordDocumentBackend
    elif fmt == InputFormat.HTML:
        from docling.backend.html_backend import HTMLDocumentBackend

        backend = HTMLDocumentBackend
    elif fmt == InputFormat.MD:
        from docling.backend.md_backend import MarkdownDocumentBackend

        backend = MarkdownDocumentBackend
    else:
        backend = {
            InputFormat.IMAGE: ImageDocumentBackend,
            InputFormat.DCLX: DocLangArchiveBackend,
        }.get(fmt)
    if backend is None:
        raise RuntimeError(f"No default extraction backend configured for {fmt}")

    return ExtractionFormatOption(
        pipeline_cls=ExtractionVlmPipeline,
        backend=backend,
    )


class DocumentExtractor:
    """Extract structured data from supported document formats."""

    def __init__(
        self,
        allowed_formats: Optional[list[InputFormat]] = None,
        extraction_format_options: Optional[
            dict[InputFormat, ExtractionFormatOption]
        ] = None,
    ) -> None:
        self.allowed_formats: list[InputFormat] = (
            allowed_formats
            if allowed_formats is not None
            else list(DEFAULT_EXTRACTION_FORMATS)
        )
        overrides = extraction_format_options or {}
        self.extraction_format_to_options: dict[
            InputFormat, ExtractionFormatOption
        ] = {}
        for fmt in self.allowed_formats:
            fopt = overrides.get(fmt)
            if fopt is None:
                fopt = _get_default_extraction_option(fmt)
            elif fopt.backend is None:
                fopt = fopt.model_copy(
                    update={"backend": _get_default_extraction_option(fmt).backend}
                )
            self.extraction_format_to_options[fmt] = fopt

        self._initialized_pipelines: dict[
            tuple[Type[BaseExtractionPipeline], str], BaseExtractionPipeline
        ] = {}

    @overload
    def extract(
        self,
        source: Union[Path, str, DocumentStream],
        template: ExtractionTemplateType,
        headers: Optional[dict[str, str]] = None,
        raises_on_error: bool = True,
        max_num_pages: int = sys.maxsize,
        max_file_size: int = sys.maxsize,
        page_range: PageRange = DEFAULT_PAGE_RANGE,
        *,
        target: None = None,
    ) -> ExtractionResult: ...

    @overload
    def extract(
        self,
        source: Union[Path, str, DocumentStream],
        template: ExtractionTemplateType | None = None,
        headers: Optional[dict[str, str]] = None,
        raises_on_error: bool = True,
        max_num_pages: int = sys.maxsize,
        max_file_size: int = sys.maxsize,
        page_range: PageRange = DEFAULT_PAGE_RANGE,
        *,
        target: ExtractionTarget,
    ) -> DocumentExtractionResult: ...

    @validate_call(config=ConfigDict(strict=True))
    def extract(
        self,
        source: Union[Path, str, DocumentStream],
        template: ExtractionTemplateType | None = None,
        headers: Optional[dict[str, str]] = None,
        raises_on_error: bool = True,
        max_num_pages: int = sys.maxsize,
        max_file_size: int = sys.maxsize,
        page_range: PageRange = DEFAULT_PAGE_RANGE,
        *,
        target: ExtractionTarget | None = None,
    ) -> ExtractionResult | DocumentExtractionResult:
        owned = normalize_extraction_call(template, target)
        results = self._extract_all(
            source=[source],
            headers=headers,
            raises_on_error=raises_on_error,
            max_num_pages=max_num_pages,
            max_file_size=max_file_size,
            page_range=page_range,
            target=owned,
        )
        return next(results)

    @overload
    def extract_all(
        self,
        source: Iterable[Union[Path, str, DocumentStream]],
        template: ExtractionTemplateType,
        headers: Optional[dict[str, str]] = None,
        raises_on_error: bool = True,
        max_num_pages: int = sys.maxsize,
        max_file_size: int = sys.maxsize,
        page_range: PageRange = DEFAULT_PAGE_RANGE,
        *,
        target: None = None,
    ) -> Iterator[ExtractionResult]: ...

    @overload
    def extract_all(
        self,
        source: Iterable[Union[Path, str, DocumentStream]],
        template: ExtractionTemplateType | None = None,
        headers: Optional[dict[str, str]] = None,
        raises_on_error: bool = True,
        max_num_pages: int = sys.maxsize,
        max_file_size: int = sys.maxsize,
        page_range: PageRange = DEFAULT_PAGE_RANGE,
        *,
        target: ExtractionTarget,
    ) -> Iterator[DocumentExtractionResult]: ...

    @validate_call(config=ConfigDict(strict=True))
    def extract_all(
        self,
        source: Iterable[Union[Path, str, DocumentStream]],
        template: ExtractionTemplateType | None = None,
        headers: Optional[dict[str, str]] = None,
        raises_on_error: bool = True,
        max_num_pages: int = sys.maxsize,
        max_file_size: int = sys.maxsize,
        page_range: PageRange = DEFAULT_PAGE_RANGE,
        *,
        target: ExtractionTarget | None = None,
    ) -> Iterator[ExtractionResult | DocumentExtractionResult]:
        owned = normalize_extraction_call(template, target)
        results = self._extract_all(
            source=source,
            headers=headers,
            raises_on_error=raises_on_error,
            max_num_pages=max_num_pages,
            max_file_size=max_file_size,
            page_range=page_range,
            target=owned,
        )
        return results

    def _extract_all(
        self,
        source: Iterable[Union[Path, str, DocumentStream]],
        headers: Optional[dict[str, str]],
        raises_on_error: bool,
        max_num_pages: int,
        max_file_size: int,
        page_range: PageRange,
        *,
        target: ExtractionTarget | str,
    ) -> Iterator[ExtractionResult | DocumentExtractionResult]:
        limits = DocumentLimits(
            max_num_pages=max_num_pages,
            max_file_size=max_file_size,
            page_range=page_range,
        )
        conv_input = _DocumentConversionInput(
            path_or_stream_iterator=source, limits=limits, headers=headers
        )

        ext_res_iter = self._extract(
            conv_input, raises_on_error=raises_on_error, target=target
        )

        had_result = False
        for ext_res in ext_res_iter:
            had_result = True
            if raises_on_error and ext_res.status not in {
                ConversionStatus.SUCCESS,
                ConversionStatus.PARTIAL_SUCCESS,
            }:
                error_messages = [err.error_message for err in ext_res.errors]
                for item in ext_res.items:
                    scope = (
                        f"Page {item.scope.page_no}"
                        if isinstance(item.scope, PageScope)
                        else "Document"
                    )
                    error_messages.extend(
                        f"{scope}: {error.error_message}" for error in item.errors
                    )
                error_details = (
                    f" Errors: {'; '.join(error_messages)}" if error_messages else ""
                )
                raise ConversionError(
                    f"Extraction failed for: {ext_res.input.file} with status: {ext_res.status.value}.{error_details}"
                )
            else:
                yield _legacy_result(ext_res) if isinstance(target, str) else ext_res

        if not had_result and raises_on_error:
            raise ConversionError(
                "Extraction failed because the provided file has no recognizable format or it wasn't in the list of allowed formats."
            )

    def _extract(
        self,
        conv_input: _DocumentConversionInput,
        raises_on_error: bool,
        target: ExtractionTarget | str,
    ) -> Iterator[DocumentExtractionResult]:
        start_time = time.monotonic()

        for input_batch in chunkify(
            conv_input.docs(self.extraction_format_to_options),
            settings.perf.doc_batch_size,
        ):
            _log.info("Going to extract document batch...")
            process_func = partial(
                self._process_document_extraction,
                raises_on_error=raises_on_error,
                target=target,
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
        target: ExtractionTarget | str,
    ) -> DocumentExtractionResult:
        valid = (
            self.allowed_formats is not None and in_doc.format in self.allowed_formats
        )
        if valid:
            return self._execute_extraction_pipeline(
                in_doc, raises_on_error=raises_on_error, target=target
            )
        else:
            error_message = f"File format not allowed: {in_doc.file}"
            error_item = ErrorItem(
                component_type=DoclingComponentType.USER_INPUT,
                module_name="",
                error_message=error_message,
                category=FailureCategory.POLICY,
            )
            self._unload_unexecuted_input(in_doc)
            return DocumentExtractionResult(
                input=in_doc, status=ConversionStatus.SKIPPED, errors=[error_item]
            )

    def _execute_extraction_pipeline(
        self,
        in_doc: InputDocument,
        raises_on_error: bool,
        target: ExtractionTarget | str,
    ) -> DocumentExtractionResult:
        if not in_doc.valid:
            self._unload_unexecuted_input(in_doc)
            return DocumentExtractionResult(
                input=in_doc,
                status=ConversionStatus.FAILURE,
                errors=build_invalid_input_errors(in_doc),
            )

        try:
            pipeline = self._get_pipeline(in_doc.format)
        except Exception:
            self._unload_unexecuted_input(in_doc)
            raise
        if pipeline is None:
            self._unload_unexecuted_input(in_doc)
            if raises_on_error:
                raise ConversionError(
                    f"No extraction pipeline could be initialized for {in_doc.file}."
                )
            else:
                return DocumentExtractionResult(
                    input=in_doc, status=ConversionStatus.FAILURE
                )

        return pipeline._execute(in_doc, raises_on_error=raises_on_error, target=target)

    @staticmethod
    def _unload_unexecuted_input(in_doc: InputDocument) -> None:
        # Input rejection before backend construction leaves _backend unset.
        try:
            backend = in_doc._backend
        except AttributeError:
            return
        backend.unload()

    def _get_pipeline(
        self, doc_format: InputFormat
    ) -> Optional[BaseExtractionPipeline]:
        """Retrieve or initialize a pipeline, reusing instances based on class and options."""
        fopt = self.extraction_format_to_options.get(doc_format)
        if fopt is None or fopt.pipeline_options is None:
            return None

        pipeline_class = fopt.pipeline_cls
        pipeline_options = fopt.pipeline_options
        options_hash = create_pipeline_options_hash(pipeline_options)

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
