import hashlib
import logging
import os
import shutil
import sys
import tempfile
import threading
import time
import warnings
import gc
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import Optional, Type, Union

from pydantic import ConfigDict, Field, model_validator, validate_call
from typing_extensions import Self

from docling.backend.abstract_backend import (
    AbstractDocumentBackend,
)
from docling.backend.asciidoc_backend import AsciiDocBackend
from docling.backend.csv_backend import CsvDocumentBackend
from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.backend.html_backend import HTMLDocumentBackend
from docling.backend.image_backend import ImageDocumentBackend
from docling.backend.json.docling_json_backend import DoclingJSONBackend
from docling.backend.latex_backend import LatexDocumentBackend
from docling.backend.md_backend import MarkdownDocumentBackend
from docling.backend.mets_gbs_backend import MetsGbsDocumentBackend
from docling.backend.msexcel_backend import MsExcelDocumentBackend
from docling.backend.mspowerpoint_backend import MsPowerpointDocumentBackend
from docling.backend.msword_backend import MsWordDocumentBackend
from docling.backend.noop_backend import NoOpBackend
from docling.backend.webvtt_backend import WebVTTDocumentBackend
from docling.backend.xml.jats_backend import JatsDocumentBackend
from docling.backend.xml.uspto_backend import PatentUsptoDocumentBackend
from docling.backend.xml.xbrl_backend import XBRLDocumentBackend
from docling.datamodel.backend_options import (
    BackendOptions,
    HTMLBackendOptions,
    LatexBackendOptions,
    MarkdownBackendOptions,
    PdfBackendOptions,
    XBRLBackendOptions,
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
from docling.datamodel.pipeline_options import ConvertPipelineOptions, PipelineOptions
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


class ReferenceCountedBackend:
    """Wrapper for managing shared backend lifecycle across chunks.

    This class ensures that a shared backend (used by memory stream chunks)
    is not unloaded until all chunks have completed processing. It addresses
    the C++ garbage collection warning that occurs when pypdfium2's Document
    is destroyed before its child Pages.

    The mechanism:
    1. Each chunk calls backend.unload() when done processing
    2. Instead of immediately unloading, we decrement a counter
    3. Only actually unload when counter reaches zero (last chunk)
    This prevents premature underload while pages from other chunks are still in use.
    """

    def __init__(self, backend: AbstractDocumentBackend, num_chunks: int):
        self._backend = backend
        self._orig_unload = backend.unload
        self._ref_count = num_chunks  # Starts at number of chunks
        self._lock = threading.Lock()

    def unload_deferred(self) -> None:
        """Called each time a chunk finishes and tries to unload the backend.

        Decrements the reference count and only calls the original unload()
        when all chunks are done (count reaches 0).
        """
        with self._lock:
            self._ref_count -= 1
            if self._ref_count <= 0:
                try:
                    self._orig_unload()
                except Exception as e:
                    _log.warning(f"Error during backend unload: {e}")

    def __getattr__(self, name: str):
        """Delegate all other attributes to the wrapped backend."""
        return getattr(self._backend, name)


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


class XBRLFormatOption(FormatOption):
    pipeline_cls: Type = SimplePipeline
    backend: Type[AbstractDocumentBackend] = XBRLDocumentBackend
    backend_options: XBRLBackendOptions | None = None


class ImageFormatOption(FormatOption):
    pipeline_cls: Type = StandardPdfPipeline
    backend: Type[AbstractDocumentBackend] = ImageDocumentBackend


class PdfFormatOption(FormatOption):
    pipeline_cls: Type = StandardPdfPipeline
    backend: Type[AbstractDocumentBackend] = DoclingParseDocumentBackend
    backend_options: Optional[PdfBackendOptions] = None


class AudioFormatOption(FormatOption):
    pipeline_cls: Type = AsrPipeline
    backend: Type[AbstractDocumentBackend] = NoOpBackend


class LatexFormatOption(FormatOption):
    """Format options for LaTeX documents."""

    pipeline_cls: Type = SimplePipeline
    backend: Type[AbstractDocumentBackend] = LatexDocumentBackend
    backend_options: Optional[LatexBackendOptions] = None


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
        InputFormat.XML_XBRL: XBRLFormatOption(),
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
        InputFormat.LATEX: LatexFormatOption(),
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

    For processing exceptionally large documents without exceeding memory limits,
    configure `page_chunk_size` in the pipeline options to stream partial
    `ConversionResult` chunks iteratively via `convert_all()`.

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

        Examples:
            Create a converter with default settings (all formats allowed):

            >>> converter = DocumentConverter()

            Allow only PDF and DOCX formats:

            >>> from docling.datamodel.base_models import InputFormat
            >>> converter = DocumentConverter(
            ...     allowed_formats=[InputFormat.PDF, InputFormat.DOCX]
            ... )

            Customize pipeline options for PDF:

            >>> from docling.datamodel.pipeline_options import PdfPipelineOptions
            >>> converter = DocumentConverter(
            ...     format_options={
            ...         InputFormat.PDF: PdfFormatOption(
            ...             pipeline_options=PdfPipelineOptions()
            ...         ),
            ...     }
            ... )
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

        Note: If `page_chunk_size` is enabled in the pipeline options, this method will
        only return the first chunk. To stream all chunks of a large document,
        use the `convert_all` method instead.

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

        Examples:
            Convert a local PDF file:

            >>> from pathlib import Path
            >>> converter = DocumentConverter()
            >>> result = converter.convert("path/to/document.pdf")
            >>> print(result.document.export_to_markdown())

            Convert a document from a URL:

            >>> result = converter.convert("https://example.com/paper.pdf")

            Convert from an in-memory stream:

            >>> from io import BytesIO
            >>> from docling.datamodel.base_models import DocumentStream
            >>> buf = BytesIO(b"<html><body>Hello</body></html>")
            >>> stream = DocumentStream(name="page.html", stream=buf)
            >>> result = converter.convert(stream)
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
            max_num_pages: Maximum number of pages accepted per document.
                Documents exceeding this number will not be converted.
            max_file_size: Maximum file size in bytes. Documents exceeding this
                limit will be skipped.
            page_range: Range of pages to convert in each document.

        Yields:
            The conversion results. If `page_chunk_size` is configured in the pipeline options,
            this will yield multiple `ConversionResult` objects per document, representing
            sequential page chunks. Because these chunks are treated as independent documents
            internally, they are processed in parallel. You should increase `doc_batch_concurrency`
            to run multiple chunks at once (e.g., 500 total pages / 50 page chunks = 10 concurrency (10 chunks))
            Otherwise, it yields one `ConversionResult` per document, containing the full `DoclingDocument`.

        Raises:
            ConversionError: An error occurred during conversion.

        Examples:
            Convert a batch of local files:

            >>> from pathlib import Path
            >>> converter = DocumentConverter()
            >>> paths = list(Path("docs/").glob("*.pdf"))
            >>> for result in converter.convert_all(paths):
            ...     print(result.document.export_to_markdown()[:100])

            Convert with a file size limit of 20 MB:

            >>> results = converter.convert_all(
            ...     paths, max_file_size=20 * 1024 * 1024
            ... )
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

        Note: Page chunking (`page_chunk_size`) is not applicable to this method
        as it operates on plain strings (Markdown/HTML), which do not have pages.

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

        Examples:
            Convert a Markdown string:

            >>> from docling.datamodel.base_models import InputFormat
            >>> converter = DocumentConverter()
            >>> result = converter.convert_string(
            ...     "# Title\nSome text.", format=InputFormat.MD
            ... )
            >>> print(result.document.export_to_markdown())

            Convert an HTML string:

            >>> result = converter.convert_string(
            ...     "<h1>Title</h1><p>Some text.</p>",
            ...     format=InputFormat.HTML,
            ...     name="my_page",
            ... )
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
        # Track reference-counted backends for memory streams (shared backends)
        ref_counted_backends: dict[int, ReferenceCountedBackend] = {}
        # Track backends that need explicit cleanup (independent backends for local file chunks)
        independent_backends: list[AbstractDocumentBackend] = []
        # Track temporary files created for memory stream chunking (cleaned up in finally)
        temp_files_to_clean: list[str] = []

        def _expand_into_chunks(
            docs_iter: Iterable[InputDocument],
        ) -> Iterator[InputDocument]:
            for in_doc in docs_iter:
                if not in_doc.valid:
                    yield in_doc
                    continue

                fopt = self.format_to_options.get(in_doc.format)
                chunk_size = (
                    getattr(fopt.pipeline_options, "page_chunk_size", None)
                    if fopt and getattr(fopt, "pipeline_options", None)
                    else None
                ) or settings.perf.page_chunk_size

                start_page = in_doc.limits.page_range[0]
                end_page = (
                    min(in_doc.page_count, in_doc.limits.page_range[1])
                    if in_doc.page_count > 0
                    else in_doc.limits.page_range[1]
                )

                if (
                    chunk_size
                    and in_doc.page_count > 0
                    and (end_page - start_page + 1) > chunk_size
                ):
                    page_groups = list(
                        chunkify(iter(range(start_page, end_page + 1)), chunk_size)
                    )

                    is_local_file = False
                    file_path = None
                    if in_doc.file is not None:
                        try:
                            file_path = Path(in_doc.file)
                            if file_path.exists() and file_path.is_file():
                                is_local_file = True
                        except Exception:
                            pass

                    # If we have a memory stream, prefer materializing (converting) it to a temp file so each
                    # chunk can create an independent backend (thread-safe and avoids GC warnings).
                    # If materialization fails, fall back to the reference-counted shared backend.
                    if not is_local_file:
                        original_stream = getattr(in_doc, "_path_or_stream", None)
                        if original_stream is None and hasattr(
                            in_doc._backend, "path_or_stream"
                        ):
                            original_stream = in_doc._backend.path_or_stream

                        if original_stream is not None:
                            try:
                                # Ensure we read from the start
                                if hasattr(original_stream, "seek"):
                                    original_stream.seek(0)

                                fd, temp_path = tempfile.mkstemp(suffix=".pdf")
                                with os.fdopen(fd, "wb") as f:
                                    if hasattr(original_stream, "read"):
                                        shutil.copyfileobj(original_stream, f)
                                    else:
                                        f.write(bytes(original_stream))

                                file_path = Path(temp_path)
                                temp_files_to_clean.append(str(file_path))
                                is_local_file = True
                            except Exception as e:
                                _log.debug(
                                    "Unable to materialize stream to temp file for chunking; "
                                    "falling back to shared backend. Error: %s",
                                    e,
                                )

                    if not is_local_file:
                        # For memory streams that could not be materialized into a temp file,
                        # share the backend but ensure it is unloaded only once all chunks finish.
                        backend_id = id(in_doc._backend)
                        if backend_id not in ref_counted_backends:
                            ref_counter = ReferenceCountedBackend(
                                in_doc._backend,
                                num_chunks=len(page_groups),
                            )
                            ref_counted_backends[backend_id] = ref_counter
                            # Replace the backend's unload with our wrapper
                            in_doc._backend.unload = ref_counter.unload_deferred

                    for page_group in page_groups:
                        chunk_limits = DocumentLimits(
                            max_num_pages=in_doc.limits.max_num_pages,
                            max_file_size=in_doc.limits.max_file_size,
                            page_range=(page_group[0], page_group[-1]),
                        )

                        if is_local_file:
                            # Create a fresh, thread-safe InputDocument for local (or temp) files
                            chunk_doc = InputDocument(
                                path_or_stream=file_path,
                                format=in_doc.format,
                                backend=fopt.backend,
                                backend_options=in_doc.backend_options,
                                limits=chunk_limits,
                            )
                            # Track this backend for explicit cleanup
                            independent_backends.append(chunk_doc._backend)
                        else:  # memory stream (shares same parser)
                            chunk_doc = in_doc.model_copy()
                            chunk_doc._backend = in_doc._backend
                            chunk_doc.limits = chunk_limits
                        yield chunk_doc
                else:
                    # No chunking active (or document fits in one chunk)
                    # → keep original document (default behavior)
                    in_doc.limits = DocumentLimits(
                        max_num_pages=in_doc.limits.max_num_pages,
                        max_file_size=in_doc.limits.max_file_size,
                        page_range=(start_page, end_page),
                    )
                    yield in_doc

        try:
            for input_batch in chunkify(
                _expand_into_chunks(conv_input.docs(self.format_to_options)),
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
                            f"Finished converting document chunk {item.input.file.name} in {elapsed:.2f} sec."
                        )
                        yield item
        finally:
            # Explicit cleanup to prevent C++ GC warnings at library shutdown

            # 1. Deferred unloads for reference-counted backends are automatically
            #    triggered by each chunk's _unload call in the pipeline
            #    (no need to manually trigger here, they're already counted down)

            # 2. Explicitly unload independent backends (per-chunk backends for local files)
            #    to ensure C++ objects are destroyed before library teardown
            for backend in independent_backends:
                try:
                    if hasattr(backend, "unload") and callable(backend.unload):
                        backend.unload()
                except Exception as e:
                    _log.debug(f"Error during independent backend cleanup: {e}")

            # 3. Remove any temporary files created for memory streams
            for temp_path in temp_files_to_clean:
                try:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                except Exception as e:
                    _log.debug(f"Error cleaning up temp file {temp_path}: {e}")

            # 4. Force garbage collection to ensure all C++ objects are properly
            #    destroyed while the pypdfium2 library is still active.
            #    This helps avoid "library is destroyed" warnings at Python shutdown.
            gc.collect()

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
                    _log.warning(
                        "No pipeline could be initialized for %s.", in_doc.file
                    )
                    conv_res = ConversionResult(
                        input=in_doc,
                        status=ConversionStatus.FAILURE,
                    )
        else:
            if raises_on_error:
                raise ConversionError(f"Input document {in_doc.file} is not valid.")
            else:
                _log.warning("Input document %s is not valid.", in_doc.file)
                conv_res = ConversionResult(
                    input=in_doc,
                    status=ConversionStatus.FAILURE,
                )

        return conv_res
