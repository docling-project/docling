import hashlib
import logging
import sys
import threading
import time
import warnings
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import Optional, Type, Union

from pydantic import ConfigDict, model_validator, validate_call
from typing_extensions import Self

from docling.backend.abstract_backend import (
    AbstractDocumentBackend,
)
from docling.backend.asciidoc_backend import AsciiDocBackend
from docling.backend.csv_backend import CsvDocumentBackend
from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
from docling.backend.html_backend import HTMLDocumentBackend
from docling.backend.image_backend import ImageDocumentBackend
from docling.backend.json.docling_json_backend import DoclingJSONBackend
from docling.backend.md_backend import MarkdownDocumentBackend
from docling.backend.mets_gbs_backend import MetsGbsDocumentBackend
from docling.backend.msexcel_backend import MsExcelDocumentBackend
from docling.backend.mspowerpoint_backend import MsPowerpointDocumentBackend
from docling.backend.msword_backend import MsWordDocumentBackend
from docling.backend.noop_backend import NoOpBackend
from docling.backend.webvtt_backend import WebVTTDocumentBackend
from docling.backend.xml.jats_backend import JatsDocumentBackend
from docling.backend.xml.uspto_backend import PatentUsptoDocumentBackend
from docling.datamodel.backend_options import (
    BackendOptions,
    HTMLBackendOptions,
    MarkdownBackendOptions,
    PdfBackendOptions,
)
from docling.datamodel.base_models import (
    BaseFormatOption,
    ConversionStatus,
    DoclingComponentType,
    DocumentStream,
    ErrorItem,
    InputFormat,
)
from docling.datamodel.document import (
    ConversionResult,
    InputDocument,
    _DocumentConversionInput,
)
from docling.datamodel.pipeline_options import PipelineOptions
from docling.datamodel.settings import (
    DEFAULT_PAGE_RANGE,
    DocumentLimits,
    PageRange,
    settings,
)
from docling.exceptions import ConversionError
from docling.pipeline.asr_pipeline import AsrPipeline
from docling.pipeline.base_pipeline import BasePipeline
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.utils.utils import chunkify

_log = logging.getLogger(__name__)
_PIPELINE_CACHE_LOCK = threading.Lock()


class FormatOption(BaseFormatOption):
    pipeline_cls: Type[BasePipeline]
    backend_options: Optional[BackendOptions] = None

    @model_validator(mode="after")
    def set_optional_field_default(self) -> Self:
        if self.pipeline_options is None:
            self.pipeline_options = self.pipeline_cls.get_default_options()

        return self


class CsvFormatOption(FormatOption):
    pipeline_cls: Type = SimplePipeline
    backend: Type[AbstractDocumentBackend] = CsvDocumentBackend


class ExcelFormatOption(FormatOption):
    pipeline_cls: Type = SimplePipeline
    backend: Type[AbstractDocumentBackend] = MsExcelDocumentBackend


class WordFormatOption(FormatOption):
    pipeline_cls: Type = SimplePipeline
    backend: Type[AbstractDocumentBackend] = MsWordDocumentBackend


class PowerpointFormatOption(FormatOption):
    pipeline_cls: Type = SimplePipeline
    backend: Type[AbstractDocumentBackend] = MsPowerpointDocumentBackend


class MarkdownFormatOption(FormatOption):
    pipeline_cls: Type = SimplePipeline
    backend: Type[AbstractDocumentBackend] = MarkdownDocumentBackend
    backend_options: Optional[MarkdownBackendOptions] = None


class AsciiDocFormatOption(FormatOption):
    pipeline_cls: Type = SimplePipeline
    backend: Type[AbstractDocumentBackend] = AsciiDocBackend


class HTMLFormatOption(FormatOption):
    pipeline_cls: Type = SimplePipeline
    backend: Type[AbstractDocumentBackend] = HTMLDocumentBackend
    backend_options: Optional[HTMLBackendOptions] = None


class PatentUsptoFormatOption(FormatOption):
    pipeline_cls: Type = SimplePipeline
    backend: Type[PatentUsptoDocumentBackend] = PatentUsptoDocumentBackend


class XMLJatsFormatOption(FormatOption):
    pipeline_cls: Type = SimplePipeline
    backend: Type[AbstractDocumentBackend] = JatsDocumentBackend


class ImageFormatOption(FormatOption):
    pipeline_cls: Type = StandardPdfPipeline
    backend: Type[AbstractDocumentBackend] = ImageDocumentBackend


class PdfFormatOption(FormatOption):
    pipeline_cls: Type = StandardPdfPipeline
    backend: Type[AbstractDocumentBackend] = DoclingParseV4DocumentBackend
    backend_options: Optional[PdfBackendOptions] = None


class AudioFormatOption(FormatOption):
    pipeline_cls: Type = AsrPipeline
    backend: Type[AbstractDocumentBackend] = NoOpBackend


def _get_default_option(format: InputFormat) -> FormatOption:
    format_to_default_options = {
        InputFormat.CSV: CsvFormatOption(),
        InputFormat.XLSX: ExcelFormatOption(),
        InputFormat.DOCX: WordFormatOption(),
        InputFormat.PPTX: PowerpointFormatOption(),
        InputFormat.MD: MarkdownFormatOption(),
        InputFormat.ASCIIDOC: AsciiDocFormatOption(),
        InputFormat.HTML: HTMLFormatOption(),
        InputFormat.XML_USPTO: PatentUsptoFormatOption(),
        InputFormat.XML_JATS: XMLJatsFormatOption(),
        InputFormat.METS_GBS: FormatOption(
            pipeline_cls=StandardPdfPipeline, backend=MetsGbsDocumentBackend
        ),
        InputFormat.IMAGE: ImageFormatOption(),
        InputFormat.PDF: PdfFormatOption(),
        InputFormat.JSON_DOCLING: FormatOption(
            pipeline_cls=SimplePipeline, backend=DoclingJSONBackend
        ),
        InputFormat.AUDIO: AudioFormatOption(),
        InputFormat.VTT: FormatOption(
            pipeline_cls=SimplePipeline, backend=WebVTTDocumentBackend
        ),
    }
    if (options := format_to_default_options.get(format)) is not None:
        return options
    else:
        raise RuntimeError(f"No default options configured for {format}")


class DocumentConverter:
    """Convert documents of various input formats to Docling documents.

    `DocumentConverter` is the main entry point for converting documents in Docling.
    It handles various input formats (PDF, DOCX, PPTX, images, HTML, Markdown, etc.)
    and provides both single-document and batch conversion capabilities.

    The conversion methods return a `ConversionResult` instance for each document,
    which wraps a `DoclingDocument` object if the conversion was successful, along
    with metadata about the conversion process.

    Attributes:
        allowed_formats: Allowed input formats.
        format_to_options: Mapping of formats to their options.
        initialized_pipelines: Cache of initialized pipelines keyed by
            (pipeline class, options hash).
    """

    _default_download_filename = "file"

    def __init__(
        self,
        allowed_formats: Optional[list[InputFormat]] = None,
        format_options: Optional[dict[InputFormat, FormatOption]] = None,
    ) -> None:
        """Initialize the converter based on format preferences.

        Args:
            allowed_formats: List of allowed input formats. By default, any
                format supported by Docling is allowed.
            format_options: Dictionary of format-specific options.
        """
        self.allowed_formats: list[InputFormat] = (
            allowed_formats if allowed_formats is not None else list(InputFormat)
        )

        # Normalize format options: ensure IMAGE format uses ImageDocumentBackend
        # for backward compatibility (old code might use PdfFormatOption or other backends for images)
        normalized_format_options: dict[InputFormat, FormatOption] = {}
        if format_options:
            for format, option in format_options.items():
                if (
                    format == InputFormat.IMAGE
                    and option.backend is not ImageDocumentBackend
                ):
                    warnings.warn(
                        f"Using {option.backend.__name__} for InputFormat.IMAGE is deprecated. "
                        "Images should use ImageDocumentBackend via ImageFormatOption. "
                        "Automatically correcting the backend, please update your code to avoid this warning.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                    # Convert to ImageFormatOption while preserving pipeline and backend options
                    normalized_format_options[format] = ImageFormatOption(
                        pipeline_cls=option.pipeline_cls,
                        pipeline_options=option.pipeline_options,
                        backend_options=option.backend_options,
                    )
                else:
                    normalized_format_options[format] = option

        self.format_to_options: dict[InputFormat, FormatOption] = {
            format: (
                _get_default_option(format=format)
                if (custom_option := normalized_format_options.get(format)) is None
                else custom_option
            )
            for format in self.allowed_formats
        }
        self.initialized_pipelines: dict[
            tuple[Type[BasePipeline], str], BasePipeline
        ] = {}

    def _get_initialized_pipelines(
        self,
    ) -> dict[tuple[Type[BasePipeline], str], BasePipeline]:
        return self.initialized_pipelines

    def _get_pipeline_options_hash(self, pipeline_options: PipelineOptions) -> str:
        """Generate a hash of pipeline options to use as part of the cache key."""
        options_str = str(pipeline_options.model_dump())
        return hashlib.md5(
            options_str.encode("utf-8"), usedforsecurity=False
        ).hexdigest()

    def initialize_pipeline(self, format: InputFormat):
        """Initialize the conversion pipeline for the selected format.

        Args:
            format: The input format for which to initialize the pipeline.

        Raises:
            ConversionError: If no pipeline could be initialized for the
                given format.
            RuntimeError: If `artifacts_path` is set in
                `docling.datamodel.settings.settings` when required by
                the pipeline, but points to a non-directory file.
            FileNotFoundError: If local model files are not found.
        """
        pipeline = self._get_pipeline(doc_format=format)
        if pipeline is None:
            raise ConversionError(
                f"No pipeline could be initialized for format {format}"
            )

    @validate_call(config=ConfigDict(strict=True))
    def convert(
        self,
        source: Union[Path, str, DocumentStream],  # TODO review naming
        headers: Optional[dict[str, str]] = None,
        raises_on_error: bool = True,
        max_num_pages: int = sys.maxsize,
        max_file_size: int = sys.maxsize,
        page_range: PageRange = DEFAULT_PAGE_RANGE,
    ) -> ConversionResult:
        """Convert one document fetched from a file path, URL, or DocumentStream.

        Note: If the document content is given as a string (Markdown or HTML
        content), use the `convert_string` method.

        Args:
            source: Source of input document given as file path, URL, or
                DocumentStream.
            headers: Optional headers given as a dictionary of string key-value pairs,
                in case of URL input source.
            raises_on_error: Whether to raise an error on the first conversion failure.
                If False, errors are captured in the ConversionResult objects.
            max_num_pages: Maximum number of pages accepted per document.
                Documents exceeding this number will not be converted.
            max_file_size: Maximum file size to convert.
            page_range: Range of pages to convert.

        Returns:
            The conversion result, which contains a `DoclingDocument` in the `document`
                attribute, and metadata about the conversion process.

        Raises:
            ConversionError: An error occurred during conversion.
        """
        all_res = self.convert_all(
            source=[source],
            raises_on_error=raises_on_error,
            max_num_pages=max_num_pages,
            max_file_size=max_file_size,
            headers=headers,
            page_range=page_range,
        )
        return next(all_res)

    @validate_call(config=ConfigDict(strict=True))
    def convert_all(
        self,
        source: Iterable[Union[Path, str, DocumentStream]],  # TODO review naming
        headers: Optional[dict[str, str]] = None,
        raises_on_error: bool = True,
        max_num_pages: int = sys.maxsize,
        max_file_size: int = sys.maxsize,
        page_range: PageRange = DEFAULT_PAGE_RANGE,
    ) -> Iterator[ConversionResult]:
        """Convert multiple documents from file paths, URLs, or DocumentStreams.

        Args:
            source: Source of input documents given as an iterable of file paths, URLs,
                or DocumentStreams.
            headers: Optional headers given as a (single) dictionary of string
                key-value pairs, in case of URL input source.
            raises_on_error: Whether to raise an error on the first conversion failure.
            max_num_pages: Maximum number of pages to convert.
            max_file_size: Maximum number of pages accepted per document. Documents
                exceeding this number will be skipped.
            page_range: Range of pages to convert in each document.

        Yields:
            The conversion results, each containing a `DoclingDocument` in the
                `document` attribute and metadata about the conversion process.

        Raises:
            ConversionError: An error occurred during conversion.
        """
        limits = DocumentLimits(
            max_num_pages=max_num_pages,
            max_file_size=max_file_size,
            page_range=page_range,
        )
        conv_input = _DocumentConversionInput(
            path_or_stream_iterator=source, limits=limits, headers=headers
        )
        conv_res_iter = self._convert(conv_input, raises_on_error=raises_on_error)

        had_result = False
        for conv_res in conv_res_iter:
            had_result = True
            if raises_on_error and conv_res.status not in {
                ConversionStatus.SUCCESS,
                ConversionStatus.PARTIAL_SUCCESS,
            }:
                error_details = ""
                if conv_res.errors:
                    error_messages = [err.error_message for err in conv_res.errors]
                    error_details = f" Errors: {'; '.join(error_messages)}"
                raise ConversionError(
                    f"Conversion failed for: {conv_res.input.file} with status: "
                    f"{conv_res.status}.{error_details}"
                )
            else:
                yield conv_res

        if not had_result and raises_on_error:
            raise ConversionError(
                "Conversion failed because the provided file has no recognizable "
                "format or it wasn't in the list of allowed formats."
            )

    @validate_call(config=ConfigDict(strict=True))
    def convert_string(
        self,
        content: str,
        format: InputFormat,
        name: Optional[str] = None,
    ) -> ConversionResult:
        """Convert a document given as a string using the specified format.

        Only Markdown (`InputFormat.MD`) and HTML (`InputFormat.HTML`) formats
        are supported. The content is wrapped in a `DocumentStream` and passed
        to the main conversion pipeline.

        Args:
            content: The document content as a string.
            format: The format of the input content.
            name: The filename to associate with the document. If not provided, a
                timestamp-based name is generated. The appropriate file extension (`md`
                or `html`) is appended if missing.

        Returns:
            The conversion result, which contains a `DoclingDocument` in the `document`
                attribute, and metadata about the conversion process.

        Raises:
            ValueError: If format is neither `InputFormat.MD` nor `InputFormat.HTML`.
            ConversionError: An error occurred during conversion.
        """
        name = name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        if format == InputFormat.MD:
            if not name.endswith(".md"):
                name += ".md"

            buff = BytesIO(content.encode("utf-8"))
            doc_stream = DocumentStream(name=name, stream=buff)

            return self.convert(doc_stream)
        elif format == InputFormat.HTML:
            if not name.endswith(".html"):
                name += ".html"

            buff = BytesIO(content.encode("utf-8"))
            doc_stream = DocumentStream(name=name, stream=buff)

            return self.convert(doc_stream)
        else:
            raise ValueError(f"format {format} is not supported in `convert_string`")

    def _convert(
        self, conv_input: _DocumentConversionInput, raises_on_error: bool
    ) -> Iterator[ConversionResult]:
        start_time = time.monotonic()

        for input_batch in chunkify(
            conv_input.docs(self.format_to_options),
            settings.perf.doc_batch_size,  # pass format_options
        ):
            _log.info("Going to convert document batch...")
            process_func = partial(
                self._process_document, raises_on_error=raises_on_error
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
                        f"Finished converting document {item.input.file.name} in {elapsed:.2f} sec."
                    )
                    yield item

    def _get_pipeline(self, doc_format: InputFormat) -> Optional[BasePipeline]:
        """Retrieve or initialize a pipeline, reusing instances based on class and options."""
        fopt = self.format_to_options.get(doc_format)

        if fopt is None or fopt.pipeline_options is None:
            return None

        pipeline_class = fopt.pipeline_cls
        pipeline_options = fopt.pipeline_options
        options_hash = self._get_pipeline_options_hash(pipeline_options)

        # Use a composite key to cache pipelines
        cache_key = (pipeline_class, options_hash)

        with _PIPELINE_CACHE_LOCK:
            if cache_key not in self.initialized_pipelines:
                _log.info(
                    f"Initializing pipeline for {pipeline_class.__name__} with options hash {options_hash}"
                )
                self.initialized_pipelines[cache_key] = pipeline_class(
                    pipeline_options=pipeline_options
                )
            else:
                _log.debug(
                    f"Reusing cached pipeline for {pipeline_class.__name__} with options hash {options_hash}"
                )

            return self.initialized_pipelines[cache_key]

    def _process_document(
        self, in_doc: InputDocument, raises_on_error: bool
    ) -> ConversionResult:
        valid = (
            self.allowed_formats is not None and in_doc.format in self.allowed_formats
        )
        if valid:
            conv_res = self._execute_pipeline(in_doc, raises_on_error=raises_on_error)
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
                conv_res = ConversionResult(
                    input=in_doc, status=ConversionStatus.SKIPPED, errors=[error_item]
                )

        return conv_res

    def _execute_pipeline(
        self, in_doc: InputDocument, raises_on_error: bool
    ) -> ConversionResult:
        if in_doc.valid:
            pipeline = self._get_pipeline(in_doc.format)
            if pipeline is not None:
                conv_res = pipeline.execute(in_doc, raises_on_error=raises_on_error)
            else:
                if raises_on_error:
                    raise ConversionError(
                        f"No pipeline could be initialized for {in_doc.file}."
                    )
                else:
                    conv_res = ConversionResult(
                        input=in_doc,
                        status=ConversionStatus.FAILURE,
                    )
        else:
            if raises_on_error:
                raise ConversionError(f"Input document {in_doc.file} is not valid.")
            else:
                # invalid doc or not of desired format
                conv_res = ConversionResult(
                    input=in_doc,
                    status=ConversionStatus.FAILURE,
                )
                # TODO add error log why it failed.

        return conv_res
