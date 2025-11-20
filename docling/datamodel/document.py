import csv
import importlib
import json
import logging
import platform
import re
import sys
import tarfile
import zipfile
from collections.abc import Iterable, Mapping
from datetime import datetime
from enum import Enum
from io import BytesIO
from pathlib import Path, PurePath
from typing import (
    TYPE_CHECKING,
    Annotated,
    Literal,
    Optional,
    Type,
    Union,
    cast,
)

import filetype

# DO NOT REMOVE; explicitly exposed from this location
from docling_core.types.doc import (
    DocItem,
    DocItemLabel,
    DoclingDocument,
    PictureItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
)
from docling_core.types.doc.document import ListItem
from docling_core.types.legacy_doc.base import (
    BaseText,
    Figure,
    GlmTableCell,
    PageDimensions,
    PageReference,
    Prov,
    Ref,
    Table as DsSchemaTable,
    TableCell,
)
from docling_core.types.legacy_doc.document import (
    CCSDocumentDescription as DsDocumentDescription,
    CCSFileInfoObject as DsFileInfoObject,
    ExportedCCSDocument as DsDocument,
)
from docling_core.utils.file import resolve_source_to_stream
from docling_core.utils.legacy import docling_document_to_legacy
from pydantic import BaseModel, Field
from typing_extensions import deprecated

from docling.backend.abstract_backend import (
    AbstractDocumentBackend,
    DeclarativeDocumentBackend,
    PaginatedDocumentBackend,
)
from docling.datamodel.backend_options import BackendOptions
from docling.datamodel.base_models import (
    AssembledUnit,
    ConfidenceReport,
    ConversionStatus,
    DocumentStream,
    ErrorItem,
    FormatToExtensions,
    FormatToMimeType,
    InputFormat,
    MimeTypeToFormat,
    Page,
)
from docling.datamodel.settings import DocumentLimits
from docling.utils.profiling import ProfilingItem
from docling.utils.utils import create_file_hash

if TYPE_CHECKING:
    from docling.datamodel.base_models import BaseFormatOption
    from docling.document_converter import FormatOption

_log = logging.getLogger(__name__)

layout_label_to_ds_type = {
    DocItemLabel.TITLE: "title",
    DocItemLabel.DOCUMENT_INDEX: "table",
    DocItemLabel.SECTION_HEADER: "subtitle-level-1",
    DocItemLabel.CHECKBOX_SELECTED: "checkbox-selected",
    DocItemLabel.CHECKBOX_UNSELECTED: "checkbox-unselected",
    DocItemLabel.CAPTION: "caption",
    DocItemLabel.PAGE_HEADER: "page-header",
    DocItemLabel.PAGE_FOOTER: "page-footer",
    DocItemLabel.FOOTNOTE: "footnote",
    DocItemLabel.TABLE: "table",
    DocItemLabel.FORMULA: "equation",
    DocItemLabel.LIST_ITEM: "paragraph",
    DocItemLabel.CODE: "paragraph",
    DocItemLabel.PICTURE: "figure",
    DocItemLabel.TEXT: "paragraph",
    DocItemLabel.PARAGRAPH: "paragraph",
    DocItemLabel.FORM: DocItemLabel.FORM.value,
    DocItemLabel.KEY_VALUE_REGION: DocItemLabel.KEY_VALUE_REGION.value,
}

_EMPTY_DOCLING_DOC = DoclingDocument(name="dummy")


class InputDocument(BaseModel):
    """A document as an input of a Docling conversion."""

    file: Annotated[
        PurePath, Field(description="A path representation the input document.")
    ]
    document_hash: Annotated[
        str,
        Field(description="A stable hash of the path or stream of the input document."),
    ]
    valid: bool = Field(True, description="Whether this is is a valid input document.")
    backend_options: Optional[BackendOptions] = Field(
        None, description="Custom options for backends."
    )
    limits: DocumentLimits = Field(
        DocumentLimits(), description="Limits in the input document for the conversion."
    )
    format: Annotated[InputFormat, Field(description="The document format.")]

    filesize: Optional[int] = Field(
        None, description="Size of the input file, in bytes."
    )
    page_count: int = Field(0, description="Number of pages in the input document.")

    _backend: AbstractDocumentBackend

    def __init__(
        self,
        path_or_stream: Union[BytesIO, Path],
        format: InputFormat,
        backend: Type[AbstractDocumentBackend],
        backend_options: Optional[BackendOptions] = None,
        filename: Optional[str] = None,
        limits: Optional[DocumentLimits] = None,
    ) -> None:
        super().__init__(
            file="",
            document_hash="",
            format=InputFormat.PDF,
            backend_options=backend_options,
        )  # initialize with dummy values
        self.limits = limits or DocumentLimits()
        self.format = format

        try:
            if isinstance(path_or_stream, Path):
                self.file = path_or_stream
                self.filesize = path_or_stream.stat().st_size
                if self.filesize > self.limits.max_file_size:
                    self.valid = False
                else:
                    self.document_hash = create_file_hash(path_or_stream)
                    self._init_doc(backend, path_or_stream)

            elif isinstance(path_or_stream, BytesIO):
                assert filename is not None, (
                    "Can't construct InputDocument from stream without providing "
                    "filename arg."
                )
                self.file = PurePath(filename)
                self.filesize = path_or_stream.getbuffer().nbytes

                if self.filesize > self.limits.max_file_size:
                    self.valid = False
                else:
                    self.document_hash = create_file_hash(path_or_stream)
                    self._init_doc(backend, path_or_stream)
            else:
                raise RuntimeError(
                    f"Unexpected type path_or_stream: {type(path_or_stream)}"
                )

            # For paginated backends, check if the maximum page count is exceeded.
            if self.valid and self._backend.is_valid():
                if self._backend.supports_pagination() and isinstance(
                    self._backend, PaginatedDocumentBackend
                ):
                    self.page_count = self._backend.page_count()
                    if not self.page_count <= self.limits.max_num_pages:
                        self.valid = False
                    elif self.page_count < self.limits.page_range[0]:
                        self.valid = False

        except (FileNotFoundError, OSError) as e:
            self.valid = False
            _log.exception(
                f"File {self.file.name} not found or cannot be opened.", exc_info=e
            )
            # raise
        except RuntimeError as e:
            self.valid = False
            _log.exception(
                "An unexpected error occurred while opening the document "
                f"{self.file.name}",
                exc_info=e,
            )
            # raise

    def _init_doc(
        self,
        backend: Type[AbstractDocumentBackend],
        path_or_stream: Union[BytesIO, Path],
    ) -> None:
        if self.backend_options:
            self._backend = backend(
                self,
                path_or_stream=path_or_stream,
                options=self.backend_options,
            )
        else:
            self._backend = backend(self, path_or_stream=path_or_stream)

        if not self._backend.is_valid():
            self.valid = False


class DocumentFormat(str, Enum):
    V2 = "v2"
    V1 = "v1"


class DoclingVersion(BaseModel):
    docling_version: str = importlib.metadata.version("docling")
    docling_core_version: str = importlib.metadata.version("docling-core")
    docling_ibm_models_version: str = importlib.metadata.version("docling-ibm-models")
    docling_parse_version: str = importlib.metadata.version("docling-parse")
    platform_str: str = platform.platform()
    py_impl_version: str = sys.implementation.cache_tag
    py_lang_version: str = platform.python_version()


class ConversionAssets(BaseModel):
    version: DoclingVersion = DoclingVersion()
    # When the assets were saved (ISO string from datetime.now())
    timestamp: Optional[str] = None

    status: ConversionStatus = ConversionStatus.PENDING  # failure, success
    errors: list[ErrorItem] = []  # structure to keep errors

    pages: list[Page] = []
    timings: dict[str, ProfilingItem] = {}
    confidence: ConfidenceReport = Field(default_factory=ConfidenceReport)

    document: DoclingDocument = _EMPTY_DOCLING_DOC

    @property
    @deprecated("Use document instead.")
    def legacy_document(self):
        return docling_document_to_legacy(self.document)

    def save(
        self,
        *,
        filename: Union[str, Path],
        indent: Optional[int] = 2,
    ):
        """Serialize the full ConversionAssets to JSON."""
        if isinstance(filename, str):
            filename = Path(filename)
        # Build an in-memory ZIP archive containing JSON for each asset
        buf = BytesIO()

        def to_jsonable(obj):
            try:
                # pydantic v2 models
                if hasattr(obj, "model_dump"):
                    return obj.model_dump(mode="json")  # type: ignore[attr-defined]
            except TypeError:
                # some models may not accept mode argument
                return obj.model_dump()  # type: ignore[attr-defined]

            # enums
            try:
                from enum import Enum

                if isinstance(obj, Enum):
                    return obj.value
            except Exception:
                pass

            # containers
            if isinstance(obj, list):
                return [to_jsonable(x) for x in obj]
            if isinstance(obj, dict):
                return {k: to_jsonable(v) for k, v in obj.items()}

            # passthrough primitives
            return obj

        with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:

            def write_json(name: str, payload) -> None:
                data = json.dumps(
                    to_jsonable(payload), ensure_ascii=False, indent=indent
                )
                zf.writestr(name, data.encode("utf-8"))

            # Update and persist a save timestamp
            self.timestamp = datetime.now().isoformat()
            write_json("timestamp.json", self.timestamp)

            # Store each component in its own JSON file
            write_json("version.json", self.version)
            write_json("status.json", self.status)
            write_json("errors.json", self.errors)
            write_json("pages.json", self.pages)
            write_json("timings.json", self.timings)
            write_json("confidence.json", self.confidence)
            # For the document, ensure stable schema via export_to_dict
            doc_dict = self.document.export_to_dict()
            zf.writestr(
                "document.json",
                json.dumps(doc_dict, ensure_ascii=False, indent=indent).encode("utf-8"),
            )

        # Persist the ZIP to disk
        buf.seek(0)
        if filename.parent and not filename.parent.exists():
            filename.parent.mkdir(parents=True, exist_ok=True)
        with filename.open("wb") as f:
            f.write(buf.getvalue())

    @classmethod
    def load(cls, filename: Union[str, Path]) -> "ConversionAssets":
        """Load a ConversionAssets."""
        if isinstance(filename, str):
            filename = Path(filename)

        # Read the ZIP and deserialize all items
        version_info: DoclingVersion = DoclingVersion()
        timestamp: Optional[str] = None
        status = ConversionStatus.PENDING
        errors: list[ErrorItem] = []
        pages: list[Page] = []
        timings: dict[str, ProfilingItem] = {}
        confidence = ConfidenceReport()
        document: DoclingDocument = _EMPTY_DOCLING_DOC

        with zipfile.ZipFile(filename, mode="r") as zf:

            def read_json(name: str):
                try:
                    with zf.open(name, "r") as fp:
                        return json.loads(fp.read().decode("utf-8"))
                except KeyError:
                    return None

            # version
            if (data := read_json("version.json")) is not None:
                try:
                    version_info = DoclingVersion.model_validate(data)
                except Exception as exc:
                    _log.error(f"Could not read version: {exc}")

            # timestamp
            if (data := read_json("timestamp.json")) is not None:
                if isinstance(data, str):
                    timestamp = data

            # status
            if (data := read_json("status.json")) is not None:
                try:
                    status = ConversionStatus(data)
                except Exception:
                    status = ConversionStatus.PENDING

            # errors
            if (data := read_json("errors.json")) is not None and isinstance(
                data, list
            ):
                errors = [ErrorItem.model_validate(item) for item in data]

            # pages
            if (data := read_json("pages.json")) is not None and isinstance(data, list):
                pages = [Page.model_validate(item) for item in data]

            # timings
            if (data := read_json("timings.json")) is not None and isinstance(
                data, dict
            ):
                timings = {k: ProfilingItem.model_validate(v) for k, v in data.items()}

            # confidence
            if (data := read_json("confidence.json")) is not None and isinstance(
                data, dict
            ):
                confidence = ConfidenceReport.model_validate(data)

            # document
            if (data := read_json("document.json")) is not None and isinstance(
                data, dict
            ):
                document = DoclingDocument.model_validate(data)

        return cls(
            version=version_info,
            timestamp=timestamp,
            status=status,
            errors=errors,
            pages=pages,
            timings=timings,
            confidence=confidence,
            document=document,
        )


class ConversionResult(ConversionAssets):
    input: InputDocument
    assembled: AssembledUnit = AssembledUnit()


class _DummyBackend(AbstractDocumentBackend):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def is_valid(self) -> bool:
        return False

    @classmethod
    def supported_formats(cls) -> set[InputFormat]:
        return set()

    @classmethod
    def supports_pagination(cls) -> bool:
        return False

    def unload(self):
        return super().unload()


class _DocumentConversionInput(BaseModel):
    path_or_stream_iterator: Iterable[Union[Path, str, DocumentStream]]
    headers: Optional[dict[str, str]] = None
    limits: Optional[DocumentLimits] = DocumentLimits()

    def docs(
        self,
        format_options: Mapping[InputFormat, "BaseFormatOption"],
    ) -> Iterable[InputDocument]:
        for item in self.path_or_stream_iterator:
            obj = (
                resolve_source_to_stream(item, self.headers)
                if isinstance(item, str)
                else item
            )
            format = self._guess_format(obj)
            backend: Type[AbstractDocumentBackend]
            backend_options: Optional[BackendOptions] = None
            if not format or format not in format_options:
                _log.error(
                    f"Input document {obj.name} with format {format} does not match "
                    f"any allowed format: ({format_options.keys()})"
                )
                backend = _DummyBackend
            else:
                options = format_options[format]
                backend = options.backend
                if "backend_options" in options.model_fields_set:
                    backend_options = cast("FormatOption", options).backend_options

            path_or_stream: Union[BytesIO, Path]
            if isinstance(obj, Path):
                path_or_stream = obj
            elif isinstance(obj, DocumentStream):
                path_or_stream = obj.stream
            else:
                raise RuntimeError(f"Unexpected obj type in iterator: {type(obj)}")

            yield InputDocument(
                path_or_stream=path_or_stream,
                format=format,  # type: ignore[arg-type]
                filename=obj.name,
                limits=self.limits,
                backend=backend,
                backend_options=backend_options,
            )

    def _guess_format(self, obj: Union[Path, DocumentStream]) -> Optional[InputFormat]:
        content = b""  # empty binary blob
        formats: list[InputFormat] = []

        if isinstance(obj, Path):
            mime = filetype.guess_mime(str(obj))
            if mime is None:
                ext = obj.suffix[1:]
                mime = _DocumentConversionInput._mime_from_extension(ext)
            if mime is None:  # must guess from
                with obj.open("rb") as f:
                    content = f.read(1024)  # Read first 1KB
            if mime is not None and mime.lower() == "application/zip":
                mime_root = "application/vnd.openxmlformats-officedocument"
                if obj.suffixes[-1].lower() == ".xlsx":
                    mime = mime_root + ".spreadsheetml.sheet"
                elif obj.suffixes[-1].lower() == ".docx":
                    mime = mime_root + ".wordprocessingml.document"
                elif obj.suffixes[-1].lower() == ".pptx":
                    mime = mime_root + ".presentationml.presentation"

        elif isinstance(obj, DocumentStream):
            content = obj.stream.read(8192)
            obj.stream.seek(0)
            mime = filetype.guess_mime(content)
            if mime is None:
                ext = (
                    obj.name.rsplit(".", 1)[-1]
                    if ("." in obj.name and not obj.name.startswith("."))
                    else ""
                )
                mime = _DocumentConversionInput._mime_from_extension(ext.lower())
            if mime is not None and mime.lower() == "application/zip":
                objname = obj.name.lower()
                mime_root = "application/vnd.openxmlformats-officedocument"
                if objname.endswith(".xlsx"):
                    mime = mime_root + ".spreadsheetml.sheet"
                elif objname.endswith(".docx"):
                    mime = mime_root + ".wordprocessingml.document"
                elif objname.endswith(".pptx"):
                    mime = mime_root + ".presentationml.presentation"

        if mime is not None and mime.lower() == "application/gzip":
            if detected_mime := _DocumentConversionInput._detect_mets_gbs(obj):
                mime = detected_mime

        mime = mime or _DocumentConversionInput._detect_html_xhtml(content)
        mime = mime or _DocumentConversionInput._detect_csv(content)
        mime = mime or "text/plain"
        formats = MimeTypeToFormat.get(mime, [])
        _log.info(f"detected formats: {formats}")

        if formats:
            if len(formats) == 1 and mime not in ("text/plain"):
                return formats[0]
            else:  # ambiguity in formats
                return _DocumentConversionInput._guess_from_content(
                    content, mime, formats
                )
        else:
            return None

    @staticmethod
    def _guess_from_content(
        content: bytes, mime: str, formats: list[InputFormat]
    ) -> Optional[InputFormat]:
        """Guess the input format of a document by checking part of its content."""
        input_format: Optional[InputFormat] = None

        if mime == "application/xml":
            content_str = content.decode("utf-8")
            match_doctype = re.search(r"<!DOCTYPE [^>]+>", content_str)
            if match_doctype:
                xml_doctype = match_doctype.group()
                if InputFormat.XML_USPTO in formats and any(
                    item in xml_doctype
                    for item in (
                        "us-patent-application-v4",
                        "us-patent-grant-v4",
                        "us-grant-025",
                        "patent-application-publication",
                    )
                ):
                    input_format = InputFormat.XML_USPTO

                if InputFormat.XML_JATS in formats and (
                    "JATS-journalpublishing" in xml_doctype
                    or "JATS-archive" in xml_doctype
                ):
                    input_format = InputFormat.XML_JATS

        elif mime == "text/plain":
            content_str = content.decode("utf-8")
            if InputFormat.XML_USPTO in formats and content_str.startswith("PATN\r\n"):
                input_format = InputFormat.XML_USPTO

        return input_format

    @staticmethod
    def _mime_from_extension(ext):
        mime = None
        if ext in FormatToExtensions[InputFormat.ASCIIDOC]:
            mime = FormatToMimeType[InputFormat.ASCIIDOC][0]
        elif ext in FormatToExtensions[InputFormat.HTML]:
            mime = FormatToMimeType[InputFormat.HTML][0]
        elif ext in FormatToExtensions[InputFormat.MD]:
            mime = FormatToMimeType[InputFormat.MD][0]
        elif ext in FormatToExtensions[InputFormat.CSV]:
            mime = FormatToMimeType[InputFormat.CSV][0]
        elif ext in FormatToExtensions[InputFormat.JSON_DOCLING]:
            mime = FormatToMimeType[InputFormat.JSON_DOCLING][0]
        elif ext in FormatToExtensions[InputFormat.PDF]:
            mime = FormatToMimeType[InputFormat.PDF][0]
        elif ext in FormatToExtensions[InputFormat.DOCX]:
            mime = FormatToMimeType[InputFormat.DOCX][0]
        elif ext in FormatToExtensions[InputFormat.PPTX]:
            mime = FormatToMimeType[InputFormat.PPTX][0]
        elif ext in FormatToExtensions[InputFormat.XLSX]:
            mime = FormatToMimeType[InputFormat.XLSX][0]
        elif ext in FormatToExtensions[InputFormat.VTT]:
            mime = FormatToMimeType[InputFormat.VTT][0]

        return mime

    @staticmethod
    def _detect_html_xhtml(
        content: bytes,
    ) -> Optional[Literal["application/xhtml+xml", "application/xml", "text/html"]]:
        """Guess the mime type of an XHTML, HTML, or XML file from its content.

        Args:
            content: A short piece of a document from its beginning.

        Returns:
            The mime type of an XHTML, HTML, or XML file, or None if the content does
              not match any of these formats.
        """
        content_str = content.decode("ascii", errors="ignore").lower()
        # Remove XML comments
        content_str = re.sub(r"<!--(.*?)-->", "", content_str, flags=re.DOTALL)
        content_str = content_str.lstrip()

        if re.match(r"<\?xml", content_str):
            if "xhtml" in content_str[:1000]:
                return "application/xhtml+xml"
            else:
                return "application/xml"

        if re.match(
            r"(<script.*?>.*?</script>\s*)?(<!doctype\s+html|<html|<head|<body)",
            content_str,
            re.DOTALL,
        ):
            return "text/html"

        p = re.compile(
            r"<!doctype\s+(?P<root>[a-zA-Z_:][a-zA-Z0-9_:.-]*)\s+.*>\s*<(?P=root)\b"
        )
        if p.search(content_str):
            return "application/xml"

        return None

    @staticmethod
    def _detect_csv(
        content: bytes,
    ) -> Optional[Literal["text/csv"]]:
        """Guess the mime type of a CSV file from its content.

        Args:
            content: A short piece of a document from its beginning.

        Returns:
            The mime type of a CSV file, or None if the content does
              not match any of the format.
        """
        content_str = content.decode("ascii", errors="ignore").strip()

        # Ensure there's at least one newline (CSV is usually multi-line)
        if "\n" not in content_str:
            return None

        # Use csv.Sniffer to detect CSV characteristics
        try:
            dialect = csv.Sniffer().sniff(content_str)
            if dialect.delimiter in {",", ";", "\t", "|"}:  # Common delimiters
                return "text/csv"
        except csv.Error:
            return None

        return None

    @staticmethod
    def _detect_mets_gbs(
        obj: Union[Path, DocumentStream],
    ) -> Optional[Literal["application/mets+xml"]]:
        content = obj if isinstance(obj, Path) else obj.stream
        tar: tarfile.TarFile
        member: tarfile.TarInfo
        with tarfile.open(
            name=content if isinstance(content, Path) else None,
            fileobj=content if isinstance(content, BytesIO) else None,
            mode="r:gz",
        ) as tar:
            for member in tar.getmembers():
                if member.name.endswith(".xml"):
                    file = tar.extractfile(member)
                    if file is not None:
                        content_str = file.read().decode(errors="ignore")
                        if "http://www.loc.gov/METS/" in content_str:
                            return "application/mets+xml"
        return None
