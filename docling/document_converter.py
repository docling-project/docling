import hashlib
import logging
import sys
import threading
import time
import warnings
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import lru_cache, partial
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Type, Union

from pydantic import ConfigDict, Field, model_validator, validate_call
from typing_extensions import Self

from docling.backend.abstract_backend import (
    AbstractDocumentBackend,
)
from docling.backend.asciidoc_backend import AsciiDocBackend
from docling.backend.boxnote_backend import BoxNoteDocumentBackend
from docling.backend.csv_backend import CsvDocumentBackend
from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.backend.ebcdic_backend import EbcdicDocumentBackend
from docling.backend.email_backend import EmailDocumentBackend
from docling.backend.epub_backend import EpubDocumentBackend
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
from docling.backend.opendocument_backend import (
    OdpDocumentBackend,
    OdsDocumentBackend,
    OdtDocumentBackend,
)
from docling.backend.webvtt_backend import WebVTTDocumentBackend
from docling.backend.xml.doclang_archive_backend import DocLangArchiveBackend
from docling.backend.xml.doclang_backend import DocLangDocumentBackend
from docling.backend.xml.jats_backend import JatsDocumentBackend
from docling.backend.xml.uspto_backend import PatentUsptoDocumentBackend
from docling.backend.xml.xbrl_backend import XBRLDocumentBackend
from docling.datamodel.backend_options import (
    BackendOptions,
    EbcdicBackendOptions,
    EpubBackendOptions,
    HTMLBackendOptions,
    LatexBackendOptions,
    MarkdownBackendOptions,
    MetsGbsBackendOptions,
    MsWordBackendOptions,
    PdfBackendOptions,
    XBRLBackendOptions,
)
from docling.datamodel.base_models import (
    BaseFormatOption,
    ConversionStatus,
    DoclingComponentType,
    DocumentStream,
    ErrorItem,
    FailureCategory,
    HttpSource,
    InputFormat,
)
from docling.datamodel.document import (
    ConversionResult,
    InputDocument,
    _DocumentConversionInput,
    build_invalid_input_errors,
    get_input_rejection_cause,
)
from docling.datamodel.pipeline_options import (
    ConvertPipelineOptions,
    PdfPipelineOptions,
    PipelineOptions,
)
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
from docling.pipeline.video_pipeline import VideoPipeline
from docling.utils.utils import chunkify

_log = logging.getLogger(__name__)
_PIPELINE_CACHE_LOCK = threading.Lock()


def _resolve_pdf_password(in_doc: InputDocument) -> str | None:
    """Resolve PDF password from ``InputDocument`` backend options.

    Checks ``in_doc.backend_options`` first, then ``in_doc._backend.options``.
    Handles ``SecretStr``-like objects via ``get_secret_value()``.
    """

    def _extract_password_value(password_obj: object) -> str | None:
        if password_obj is None:
            return None
        # Narrowly scoped probe for SecretStr vs plain str:
        # PdfBackendOptions uses SecretStr which exposes get_secret_value();
        # other backends may store plain str. Use try/except AttributeError
        # instead of broad hasattr().
        try:
            return password_obj.get_secret_value()  # type: ignore[union-attr]
        except AttributeError:
            return str(password_obj)
        except Exception:
            return None

    # Probe InputDocument.backend_options.password. Backend options vary by
    # format (PdfBackendOptions vs other backends), so attribute existence
    # is probed narrowly with try/except AttributeError.
    try:
        backend_options = in_doc.backend_options
        if backend_options is not None:
            try:
                pwd = backend_options.password  # type: ignore[attr-defined]
            except AttributeError:
                # BackendOptions subtype without password (e.g., non-PDF)
                pwd = None
            if pwd is not None:
                value = _extract_password_value(pwd)
                if value is not None:
                    return value
    except AttributeError:
        pass
    except Exception:
        return None

    # Fallback: probe instantiated backend's options. Different backends
    # (DoclingParseDocumentBackend vs PyPdfium vs ThreadedDoclingParse)
    # expose options differently, so probing with narrow try/except is required.
    try:
        try:
            backend = in_doc._backend  # type: ignore[attr-defined]
        except AttributeError:
            backend = None
        if backend is not None:
            try:
                backend_options_from_backend = backend.options  # type: ignore[union-attr]
            except AttributeError:
                # Backend compat: not all backends expose .options
                backend_options_from_backend = None
            if backend_options_from_backend is not None:
                try:
                    pwd2 = backend_options_from_backend.password  # type: ignore[attr-defined]
                except AttributeError:
                    pwd2 = None
                if pwd2 is not None:
                    return _extract_password_value(pwd2)
    except Exception:
        return None
    return None


@lru_cache(maxsize=1)
def _get_supported_extensions() -> frozenset[str]:
    """Return the set of lower-cased extensions known to ``FormatToExtensions``."""
    from docling.datamodel.base_models import FormatToExtensions

    return frozenset(e.lower() for exts in FormatToExtensions.values() for e in exts)


@lru_cache(maxsize=1)
def _get_extension_to_format_map() -> dict[str, InputFormat]:
    """Return mapping from lower-cased extension to ``InputFormat``."""
    from docling.datamodel.base_models import FormatToExtensions

    return {e.lower(): fmt for fmt, exts in FormatToExtensions.items() for e in exts}


class FormatOption(BaseFormatOption):
    pipeline_cls: Type[BasePipeline]
    backend_options: Optional[BackendOptions] = None

    def backend_options_for_input(
        self, source: Path | str | DocumentStream
    ) -> BackendOptions | None:
        return self.backend_options

    @model_validator(mode="after")
    def set_optional_field_default(self) -> Self:
        if self.pipeline_options is None:
            self.pipeline_options = self.pipeline_cls.get_default_options()

        return self


class BoxNoteFormatOption(FormatOption):
    pipeline_cls: Type = SimplePipeline
    backend: Type[AbstractDocumentBackend] = BoxNoteDocumentBackend


class CsvFormatOption(FormatOption):
    pipeline_cls: Type = SimplePipeline
    backend: Type[AbstractDocumentBackend] = CsvDocumentBackend


class ExcelFormatOption(FormatOption):
    pipeline_cls: Type = SimplePipeline
    backend: Type[AbstractDocumentBackend] = MsExcelDocumentBackend


class WordFormatOption(FormatOption):
    pipeline_cls: Type = SimplePipeline
    backend: Type[AbstractDocumentBackend] = MsWordDocumentBackend
    backend_options: Optional[MsWordBackendOptions] = None


class PowerpointFormatOption(FormatOption):
    pipeline_cls: Type = SimplePipeline
    backend: Type[AbstractDocumentBackend] = MsPowerpointDocumentBackend


class OdtFormatOption(FormatOption):
    pipeline_cls: Type = SimplePipeline
    backend: Type[AbstractDocumentBackend] = OdtDocumentBackend


class OdsFormatOption(FormatOption):
    pipeline_cls: Type = SimplePipeline
    backend: Type[AbstractDocumentBackend] = OdsDocumentBackend


class OdpFormatOption(FormatOption):
    pipeline_cls: Type = SimplePipeline
    backend: Type[AbstractDocumentBackend] = OdpDocumentBackend


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

    def backend_options_for_input(
        self, source: Path | str | DocumentStream
    ) -> HTMLBackendOptions | None:
        options = self.backend_options
        if (
            options is None
            or options.source_uri is not None
            or isinstance(source, DocumentStream)
        ):
            return options

        return HTMLBackendOptions.model_validate(
            {**options.model_dump(), "source_uri": source}
        )


class PatentUsptoFormatOption(FormatOption):
    pipeline_cls: Type = SimplePipeline
    backend: Type[PatentUsptoDocumentBackend] = PatentUsptoDocumentBackend


class XMLJatsFormatOption(FormatOption):
    pipeline_cls: Type = SimplePipeline
    backend: Type[AbstractDocumentBackend] = JatsDocumentBackend


class XMLDocLangFormatOption(FormatOption):
    pipeline_cls: Type = SimplePipeline
    backend: Type[AbstractDocumentBackend] = DocLangDocumentBackend


class DclxFormatOption(FormatOption):
    pipeline_cls: Type = SimplePipeline
    backend: Type[AbstractDocumentBackend] = DocLangArchiveBackend


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


class MetsGbsFormatOption(FormatOption):
    pipeline_cls: Type = StandardPdfPipeline
    backend: Type[AbstractDocumentBackend] = MetsGbsDocumentBackend
    backend_options: MetsGbsBackendOptions | None = None


class AudioFormatOption(FormatOption):
    pipeline_cls: Type = AsrPipeline
    backend: Type[AbstractDocumentBackend] = NoOpBackend


class VideoFormatOption(FormatOption):
    """Format option for video input, processed via VideoPipeline."""

    pipeline_cls: Type = VideoPipeline
    backend: Type[AbstractDocumentBackend] = NoOpBackend


class LatexFormatOption(FormatOption):
    """Format options for LaTeX documents."""

    pipeline_cls: Type = SimplePipeline
    backend: Type[AbstractDocumentBackend] = LatexDocumentBackend
    backend_options: Optional[LatexBackendOptions] = None


class EmailFormatOption(FormatOption):
    pipeline_cls: Type = SimplePipeline
    backend: Type[AbstractDocumentBackend] = EmailDocumentBackend


class EpubFormatOption(FormatOption):
    pipeline_cls: Type = SimplePipeline
    backend: Type[AbstractDocumentBackend] = EpubDocumentBackend
    backend_options: EpubBackendOptions | None = None


class EbcdicFormatOption(FormatOption):
    pipeline_cls: Type = SimplePipeline
    backend: Type[AbstractDocumentBackend] = EbcdicDocumentBackend
    backend_options: EbcdicBackendOptions | None = None


def _get_default_option(format: InputFormat) -> FormatOption:
    format_to_default_options = {
        InputFormat.CSV: CsvFormatOption(),
        InputFormat.BOXNOTE: BoxNoteFormatOption(),
        InputFormat.XLSX: ExcelFormatOption(),
        InputFormat.XLS: ExcelFormatOption(),
        InputFormat.DOCX: WordFormatOption(),
        InputFormat.DOC: WordFormatOption(),
        InputFormat.PPTX: PowerpointFormatOption(),
        InputFormat.PPT: PowerpointFormatOption(),
        InputFormat.ODT: OdtFormatOption(),
        InputFormat.ODS: OdsFormatOption(),
        InputFormat.ODP: OdpFormatOption(),
        InputFormat.MD: MarkdownFormatOption(),
        InputFormat.ASCIIDOC: AsciiDocFormatOption(),
        InputFormat.HTML: HTMLFormatOption(),
        InputFormat.XML_USPTO: PatentUsptoFormatOption(),
        InputFormat.XML_JATS: XMLJatsFormatOption(),
        InputFormat.XML_DOCLANG: XMLDocLangFormatOption(),
        InputFormat.DCLX: DclxFormatOption(),
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
        InputFormat.VIDEO: VideoFormatOption(),
        InputFormat.VTT: FormatOption(
            pipeline_cls=SimplePipeline, backend=WebVTTDocumentBackend
        ),
        InputFormat.LATEX: LatexFormatOption(),
        InputFormat.EMAIL: EmailFormatOption(),
        InputFormat.EPUB: EpubFormatOption(),
        InputFormat.EBCDIC: EbcdicFormatOption(),
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
        source: Union[Path, str, DocumentStream, HttpSource],  # TODO review naming
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
            source: Source of input document given as file path, URL,
                DocumentStream, or HttpSource (a URL bundled with its own headers).
            headers: Optional headers given as a dictionary of string key-value pairs,
                in case of URL input source. Ignored for HttpSource inputs, which
                carry their own headers (these override the batch headers per key).
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
        source: Iterable[
            Union[Path, str, DocumentStream, HttpSource]
        ],  # TODO review naming
        headers: Optional[dict[str, str]] = None,
        raises_on_error: bool = True,
        max_num_pages: int = sys.maxsize,
        max_file_size: int = sys.maxsize,
        page_range: PageRange = DEFAULT_PAGE_RANGE,
    ) -> Iterator[ConversionResult]:
        """Convert multiple documents from file paths, URLs, or DocumentStreams.

        Args:
            source: Source of input documents given as an iterable of file paths, URLs,
                DocumentStreams, or HttpSources (a URL bundled with its own headers).
            headers: Optional headers given as a (single) dictionary of string
                key-value pairs, in case of URL input source. Per-source HttpSource
                headers override these (merged per key) for that source only.
            raises_on_error: Whether to raise an error on the first conversion failure.
            max_num_pages: Maximum number of pages accepted per document.
                Documents exceeding this number will not be converted.
            max_file_size: Maximum file size in bytes. Documents exceeding this
                limit will be skipped.
            page_range: Range of pages to convert in each document.

        Yields:
            The conversion results, each containing a `DoclingDocument` in the
                `document` attribute and metadata about the conversion process.

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
                # Chain the underlying exception (when one was captured during
                # input construction) so callers can classify failures via
                # ``__cause__`` — e.g. an encrypted PDF surfaces the original
                # ``PdfiumError``. See issue #1920.
                raise ConversionError(
                    f"Conversion failed for: {conv_res.input.file} with status: "
                    f"{conv_res.status.value}.{error_details}"
                ) from get_input_rejection_cause(conv_res.input)
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

        Only Markdown (`InputFormat.MD`), HTML (`InputFormat.HTML`), and DocLang
        (`InputFormat.XML_DOCLANG`) formats are supported. The content is wrapped
        in a `DocumentStream` and passed to the main conversion pipeline.

        Args:
            content: The document content as a string.
            format: The format of the input content.
            name: The filename to associate with the document. If not provided, a
                timestamp-based name is generated. The appropriate file extension is
                appended if missing.

        Returns:
            The conversion result, which contains a `DoclingDocument` in the `document`
                attribute, and metadata about the conversion process.

        Raises:
            ValueError: If format is not supported by `convert_string`.
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
        elif format == InputFormat.XML_DOCLANG:
            if not name.endswith((".dclg", ".dclg.xml")):
                name += ".dclg.xml"

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
            error_item = ErrorItem(
                component_type=DoclingComponentType.USER_INPUT,
                module_name="",
                error_message=error_message,
                category=FailureCategory.POLICY,
            )
            conv_res = ConversionResult(
                input=in_doc, status=ConversionStatus.SKIPPED, errors=[error_item]
            )

        if conv_res.status != ConversionStatus.SUCCESS:
            return conv_res
        if in_doc.format != InputFormat.PDF:
            return conv_res
        opts = self.format_to_options.get(InputFormat.PDF)
        # Direct attribute access: FormatOption contract guarantees
        # pipeline_options exists, so broad getattr with default is unnecessary.
        pdf_opts = opts.pipeline_options if opts is not None else None
        # Narrowly scoped check for PdfPipelineOptions-specific fields.
        # Only PdfPipelineOptions defines process_attachments /
        # attachments_max_depth; other PipelineOptions subtypes do not, so
        # we guard with isinstance instead of broad getattr(obj, "field", default).
        if not isinstance(pdf_opts, PdfPipelineOptions):
            return conv_res
        if not pdf_opts.process_attachments:
            return conv_res
        depth = int(pdf_opts.attachments_max_depth)
        try:
            self._process_pdf_attachments(conv_res, in_doc, pdf_opts, depth)
        except Exception as exc:
            _log.warning(
                "attachment processing failed for %s: %s",
                in_doc.file,
                exc,
                exc_info=True,
            )
        return conv_res

    def _process_pdf_attachments(  # noqa: C901
        self,
        parent_res: ConversionResult,
        in_doc: InputDocument,
        pdf_opts: PdfPipelineOptions | None,
        depth: int,
    ) -> None:
        from pathlib import Path as _Path

        from docling_core.types.doc import AttachmentStatus
        from docling_core.types.doc.common.reference import ProvenanceItem

        from docling.datamodel.base_models import FormatToExtensions

        # Build extension maps — reuse cached helpers to avoid rebuilding per attachment
        # (see _get_supported_extensions / _get_extension_to_format_map)
        supported_exts: frozenset[str] = _get_supported_extensions()
        ext_to_format: dict[str, InputFormat] = _get_extension_to_format_map()

        # Resolve password if available — delegate to narrowed helper that
        # probes InputDocument.backend_options vs backend.options with
        # third-party backend compat handling.
        password = _resolve_pdf_password(in_doc)

        # Collect raw attachments: prefer backend's PdfDocument to avoid re-parse
        raw_attachments = None
        pdf_document: Any | None = (
            None  # PdfDocument when available; Any avoids hard dep on docling-parse
        )
        # Narrowly scoped probe for backend's private _backend attr.
        # InputDocument always creates _backend for valid docs, but invalid
        # docs may lack it, so AttributeError is possible — probe narrowly.
        try:
            backend = in_doc._backend  # type: ignore[attr-defined]
        except AttributeError:
            backend = None
        try:
            # Try direct backend PdfDocument
            if backend is not None:
                # DoclingParseDocumentBackend exposes .dp_doc (PdfDocument);
                # PyPdfium backend does not. Probe narrowly with try/except
                # — third-party backend compat.
                try:
                    pdf_document_candidate = backend.dp_doc  # type: ignore[union-attr]
                except AttributeError:
                    pdf_document_candidate = None
                # PdfDocument from docling-parse exposes get_attachments();
                # other document types do not. Narrow probe required for
                # backend compat — use direct attribute access, not broad getattr.
                has_get_attachments = False
                if pdf_document_candidate is not None:
                    try:
                        _ = pdf_document_candidate.get_attachments  # type: ignore[union-attr]
                        has_get_attachments = True
                    except AttributeError:
                        has_get_attachments = False
                # Optimization: reuse already-loaded PdfDocument to avoid re-parse.
                # This fast-path avoids a second DoclingPdfParser load for the same file.
                if pdf_document_candidate is not None and has_get_attachments:
                    try:
                        raw_list = pdf_document_candidate.get_attachments()
                        # Convert to RawPdfAttachment-like objects
                        from docling.utils.pdf_attachments import RawPdfAttachment

                        raw_attachments = []
                        for idx, attachment_meta in enumerate(raw_list):
                            raw_attachments.append(
                                RawPdfAttachment(
                                    index=idx,
                                    name=attachment_meta.name,
                                    mime_type=attachment_meta.mime_type,
                                    size=attachment_meta.size,
                                    annotations=list(attachment_meta.annotations),
                                )
                            )
                        pdf_document = pdf_document_candidate
                    except Exception as exc:
                        _log.warning(
                            "failed to get attachments via backend PdfDocument: %s",
                            exc,
                            exc_info=True,
                        )
                        raw_attachments = None
        except Exception as exc:
            # Narrowly scoped probe for InputDocument.file — in_doc always has file
            # in normal flow, but fallback is needed for defensive logging.
            try:
                _file = in_doc.file  # type: ignore[attr-defined]
            except AttributeError:
                _file = "unknown"
            _log.warning(
                "attachment fast-path failed for %s: %s",
                _file,
                exc,
                exc_info=True,
            )
            raw_attachments = None

        if raw_attachments is None:
            # Fallback: re-parse from source via extract_pdf_attachments (spec 3).
            # This path handles PyPdfium, threaded, and BytesIO parents.
            # For backends that do not expose dp_doc, log the spec warning but
            # still attempt re-parse — the file may still be extractable.
            if backend is not None:
                try:
                    _dp = backend.dp_doc  # type: ignore[union-attr]
                except AttributeError:
                    _dp = None
                if _dp is None:
                    # Non-docling-parse backend — extraction via re-parse may still succeed
                    _log.warning(
                        "backend %s does not support extraction", type(backend).__name__
                    )
            from docling.utils.pdf_attachments import extract_pdf_attachments

            source = None
            # Try backend's path_or_stream first. Different backends store
            # source differently (Path vs BytesIO vs not at all). Narrow probe
            # with direct attribute access for backend compat
            # (DoclingParse vs PyPdfium vs threaded).
            if backend is not None:
                try:
                    _ = backend.path_or_stream  # type: ignore[union-attr]
                    has_path_or_stream = True
                except AttributeError:
                    # Backend compat: PyPdfium/threaded may not expose path_or_stream
                    has_path_or_stream = False
                if has_path_or_stream:
                    try:
                        path_or_stream = backend.path_or_stream  # type: ignore[union-attr]
                    except AttributeError:
                        path_or_stream = None
                    if isinstance(path_or_stream, _Path) and path_or_stream.exists():
                        source = path_or_stream
                    elif isinstance(path_or_stream, BytesIO):
                        try:
                            path_or_stream.seek(0)
                        except Exception:
                            pass
                        source = path_or_stream
                    elif path_or_stream is not None:
                        # Handle BytesIO-like objects (e.g., SpooledTemporaryFile, _AttachmentDeletingFile)
                        # Duck-typing: try seek/read
                        try:
                            path_or_stream.seek(0)  # type: ignore[union-attr]
                            source = path_or_stream  # type: ignore[assignment]
                        except Exception:
                            pass
            if source is None:
                # Try file path on disk — may be PurePath for BytesIO inputs, so guard exists()
                try:
                    p = _Path(str(in_doc.file))
                    if p.exists() and p.is_file():
                        source = p
                except Exception as exc:
                    _log.warning(
                        "failed to resolve file path for %s: %s",
                        in_doc.file,
                        exc,
                    )
                    source = None
            if source is None:
                _log.warning(
                    "attachments skipped: cannot resolve source for %s", in_doc.file
                )
                return
            extracted, pdf_document = extract_pdf_attachments(source, password=password)
            raw_attachments = extracted

        if not raw_attachments:
            return

        for raw_attachment in raw_attachments:
            ext = _Path(raw_attachment.name).suffix.lower().lstrip(".")
            is_supported = ext in supported_exts

            status: AttachmentStatus | None = None
            target: str | None = None
            if depth <= 0:
                status = "depth_limited"
                # Still need to create AttachmentItem(s)
            elif not is_supported:
                status = "unsupported"
                _log.warning("attachment %s unsupported .%s", raw_attachment.name, ext)
            else:
                # Attempt conversion
                fmt = ext_to_format.get(ext)
                if fmt is None or fmt not in self.format_to_options:
                    status = "unsupported"
                    _log.warning(
                        "attachment %s unsupported .%s (no handler)",
                        raw_attachment.name,
                        ext,
                    )
                else:
                    # Resolve backend before streaming; if no backend, mark unsupported per spec (remove NoOpBackend fallback)
                    fmt_option = self.format_to_options.get(fmt)
                    if fmt_option is not None:
                        child_backend = fmt_option.backend
                        child_backend_options = fmt_option.backend_options
                    else:
                        child_backend = None
                        child_backend_options = None
                    if child_backend is None:
                        status = "unsupported"
                        _log.warning(
                            "attachment %s unsupported .%s (no backend)",
                            raw_attachment.name,
                            ext,
                        )
                    else:
                        stream_raw = None
                        child_res = None
                        try:
                            if pdf_document is None:
                                raise RuntimeError(
                                    "no PdfDocument available for streaming"
                                )
                            from docling.utils.pdf_attachments import (
                                open_attachment_stream,
                            )

                            stream_raw = open_attachment_stream(
                                pdf_document, raw_attachment.index
                            )
                            # Preserve docling-parse's spill optimization:
                            # small payloads (≤8 MB) arrive as BytesIO and can be
                            # passed directly to DocumentStream; larger payloads
                            # arrive as a spooled temp file (BinaryIO) which
                            # DocumentStream/InputDocument currently typing as
                            # BytesIO-only — fall back to materializing into
                            # BytesIO for that case to satisfy the type contract.
                            try:
                                stream_raw.seek(0)
                            except Exception:
                                pass
                            if isinstance(stream_raw, BytesIO):
                                doc_stream = DocumentStream(
                                    name=raw_attachment.name, stream=stream_raw
                                )
                            else:
                                # Spooled file (NamedTemporaryFile / _AttachmentDeletingFile):
                                # copy into BytesIO to satisfy DocumentStream's
                                # BytesIO-only type until core widens to BinaryIO.
                                import shutil

                                child_bytes = BytesIO()
                                shutil.copyfileobj(stream_raw, child_bytes)
                                child_bytes.seek(0)
                                doc_stream = DocumentStream(
                                    name=raw_attachment.name, stream=child_bytes
                                )
                            original_pdf_option = None
                            if fmt == InputFormat.PDF:
                                try:
                                    child_pdf_opts = pdf_opts.model_copy(  # type: ignore[union-attr]
                                        update={"attachments_max_depth": depth - 1}
                                    )
                                    original_pdf_option = self.format_to_options.get(
                                        InputFormat.PDF
                                    )
                                    self.format_to_options[InputFormat.PDF] = (
                                        PdfFormatOption(
                                            pipeline_options=child_pdf_opts,
                                            backend=child_backend,
                                            backend_options=child_backend_options,
                                        )
                                    )
                                except Exception as exc:
                                    _log.warning(
                                        "failed to copy pdf options for child %s: %s",
                                        raw_attachment.name,
                                        exc,
                                    )
                                    original_pdf_option = None
                            try:
                                child_input = InputDocument(
                                    path_or_stream=doc_stream.stream,
                                    format=fmt,
                                    backend=child_backend,
                                    backend_options=child_backend_options,
                                    filename=raw_attachment.name,
                                    limits=in_doc.limits,
                                )
                                child_res = self._process_document(
                                    child_input, raises_on_error=False
                                )
                                if child_res.status == ConversionStatus.SUCCESS:
                                    status = "converted"
                                    parent_res._attachment_results.append(child_res)
                                else:
                                    status = "failed"
                                    _log.warning(
                                        "attachment %s conversion failed: %s",
                                        raw_attachment.name,
                                        child_res.errors,
                                    )
                            finally:
                                if original_pdf_option is not None:
                                    self.format_to_options[InputFormat.PDF] = (
                                        original_pdf_option
                                    )
                        except Exception as e:
                            status = "failed"
                            _log.warning(
                                "attachment %s failed: %s",
                                raw_attachment.name,
                                e,
                                exc_info=True,
                            )
                        finally:
                            if stream_raw is not None:
                                try:
                                    stream_raw.close()
                                except Exception:
                                    pass
                        if status is None:
                            status = "failed"

            # Attach to parent document — prov drives inline vs section
            # status is guaranteed AttachmentStatus after the guard above
            assert status is not None
            try:
                if raw_attachment.annotations:
                    for annotation in raw_attachment.annotations:
                        # Narrowly scoped probe for bbox compat: docling-parse
                        # FileAttachmentAnnotation bbox may expose
                        # to_bounding_box() (converted bbox) or already be a
                        # BoundingBox. Try/except AttributeError avoids broad
                        # hasattr on third-party annotation types.
                        try:
                            bbox = annotation.bbox.to_bounding_box()  # type: ignore[union-attr]
                        except AttributeError:
                            bbox = annotation.bbox
                        except Exception:
                            bbox = annotation.bbox
                        prov = ProvenanceItem(
                            page_no=int(annotation.page_no) + 1,
                            bbox=bbox,
                            charspan=(0, 0),
                        )
                        parent_res.document.add_attachment(
                            name=raw_attachment.name,
                            mime_type=raw_attachment.mime_type,
                            size=int(raw_attachment.size)
                            if raw_attachment.size
                            else None,
                            target=target,
                            status=status,
                            prov=prov,
                        )
                else:
                    parent_res.document.add_attachment(
                        name=raw_attachment.name,
                        mime_type=raw_attachment.mime_type,
                        size=int(raw_attachment.size) if raw_attachment.size else None,
                        target=target,
                        status=status,
                    )
            except Exception as e:
                _log.warning(
                    "failed to create AttachmentItem for %s: %s",
                    raw_attachment.name,
                    e,
                    exc_info=True,
                )

        # Cleanup fallback PdfDocument once, after all attachments are streamed,
        # to avoid Windows file lock. Keep backend's own PdfDocument alive.
        if pdf_document is not None:
            try:
                backend_dp = None
                if backend is not None:
                    try:
                        backend_dp = backend.dp_doc  # type: ignore[union-attr]
                    except AttributeError:
                        backend_dp = None
                if pdf_document is not backend_dp:
                    try:
                        pdf_document.unload()  # type: ignore[union-attr]
                    except Exception:
                        pass
            except Exception:
                pass

    def _unload_input_document(self, in_doc: InputDocument) -> None:
        # Narrowly scoped probe for private _backend attr; InputDocument
        # creates it during _init_doc but invalid docs may lack it.
        try:
            backend = in_doc._backend  # type: ignore[attr-defined]
        except AttributeError:
            backend = None
        if backend is not None:
            backend.unload()

    def _execute_pipeline(
        self, in_doc: InputDocument, raises_on_error: bool
    ) -> ConversionResult:
        if in_doc.valid:
            pipeline_started = False
            try:
                pipeline = self._get_pipeline(in_doc.format)
                if pipeline is not None:
                    pipeline_started = True
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
            finally:
                if not pipeline_started:
                    self._unload_input_document(in_doc)
        else:
            try:
                _log.warning("Input document %s is not valid.", in_doc.file)
                conv_res = ConversionResult(
                    input=in_doc,
                    status=ConversionStatus.FAILURE,
                    errors=build_invalid_input_errors(in_doc),
                )
            finally:
                self._unload_input_document(in_doc)

        return conv_res
