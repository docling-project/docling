from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Type, Union

import numpy as np
from docling_core.types.doc import (
    BoundingBox,
    DocItemLabel,
    NodeItem,
    PictureDataType,
    Size,
    TableCell,
)
from docling_core.types.doc.base import PydanticSerCtxKey, round_pydantic_float
from docling_core.types.doc.document import Orientation
from docling_core.types.doc.page import SegmentedPdfPage, TextCell
from docling_core.types.io import (
    DocumentStream as DocumentStream,
)

# DO NOT REMOVE; explicitly exposed from this location
from PIL.Image import Image
from pydantic import (
    AnyUrl,
    BaseModel,
    ConfigDict,
    Field,
    FieldSerializationInfo,
    PrivateAttr,
    computed_field,
    field_serializer,
    field_validator,
    model_validator,
)

if TYPE_CHECKING:
    from docling.backend.pdf_backend import PdfPageBackend
    from docling.datamodel.backend_options import BackendOptions

from docling.backend.abstract_backend import AbstractDocumentBackend
from docling.datamodel.pipeline_options import PipelineOptions


class BaseFormatOption(BaseModel):
    """Base class for format options used by _DocumentConversionInput."""

    pipeline_options: PipelineOptions | None = None
    backend: Type[AbstractDocumentBackend]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def backend_options_for_input(
        self, source: Path | str | DocumentStream
    ) -> "BackendOptions | None":
        return None


class ConversionStatus(str, Enum):
    PENDING = "pending"
    STARTED = "started"
    FAILURE = "failure"
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    SKIPPED = "skipped"


class InputFormat(str, Enum):
    """A document format supported by document backend parsers."""

    DOCX = "docx"
    PPTX = "pptx"
    HTML = "html"
    IMAGE = "image"
    PDF = "pdf"
    ASCIIDOC = "asciidoc"
    MD = "md"
    CSV = "csv"
    XLSX = "xlsx"
    XML_USPTO = "xml_uspto"
    XML_JATS = "xml_jats"
    XML_XBRL = "xml_xbrl"
    XML_DOCLANG = "xml_doclang"
    METS_GBS = "mets_gbs"
    JSON_DOCLING = "json_docling"
    AUDIO = "audio"
    VTT = "vtt"
    LATEX = "latex"
    EMAIL = "email"
    EPUB = "epub"


class OutputFormat(str, Enum):
    MARKDOWN = "md"
    JSON = "json"
    YAML = "yaml"
    HTML = "html"
    HTML_SPLIT_PAGE = "html_split_page"
    TEXT = "text"
    DOCTAGS = "doctags"
    VTT = "vtt"
    DOCLANG = "doclang"


FormatToExtensions: dict[InputFormat, list[str]] = {
    InputFormat.DOCX: ["docx", "dotx", "docm", "dotm"],
    InputFormat.PPTX: ["pptx", "potx", "ppsx", "pptm", "potm", "ppsm"],
    InputFormat.PDF: ["pdf"],
    InputFormat.MD: ["md", "txt", "text", "qmd", "rmd", "Rmd"],
    InputFormat.HTML: ["html", "htm", "xhtml"],
    InputFormat.XML_JATS: ["xml", "nxml"],
    InputFormat.XML_XBRL: ["xml", "xbrl"],
    InputFormat.XML_DOCLANG: ["dclg", "dclg.xml"],
    InputFormat.IMAGE: ["jpg", "jpeg", "png", "tif", "tiff", "bmp", "webp"],
    InputFormat.ASCIIDOC: ["adoc", "asciidoc", "asc"],
    InputFormat.CSV: ["csv"],
    InputFormat.XLSX: ["xlsx", "xlsm"],
    InputFormat.XML_USPTO: ["xml", "txt"],
    InputFormat.METS_GBS: ["tar.gz"],
    InputFormat.JSON_DOCLING: ["json"],
    InputFormat.AUDIO: ["wav", "mp3", "m4a", "aac", "ogg", "flac", "mp4", "avi", "mov"],
    InputFormat.VTT: ["vtt"],
    InputFormat.LATEX: ["tex", "latex"],
    InputFormat.EMAIL: ["eml"],
    InputFormat.EPUB: ["epub"],
}

FormatToMimeType: dict[InputFormat, list[str]] = {
    InputFormat.DOCX: [
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.template",
    ],
    InputFormat.PPTX: [
        "application/vnd.openxmlformats-officedocument.presentationml.template",
        "application/vnd.openxmlformats-officedocument.presentationml.slideshow",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ],
    InputFormat.HTML: ["text/html", "application/xhtml+xml"],
    InputFormat.XML_JATS: ["application/xml"],
    InputFormat.XML_XBRL: ["application/xml", "application/xhtml+xml"],
    InputFormat.XML_DOCLANG: ["application/xml"],
    InputFormat.IMAGE: [
        "image/png",
        "image/jpeg",
        "image/tiff",
        "image/gif",
        "image/bmp",
        "image/webp",
    ],
    InputFormat.PDF: ["application/pdf"],
    InputFormat.ASCIIDOC: ["text/asciidoc"],
    InputFormat.MD: ["text/markdown", "text/x-markdown", "text/plain"],
    InputFormat.CSV: ["text/csv"],
    InputFormat.XLSX: [
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    ],
    InputFormat.XML_USPTO: ["application/xml", "text/plain"],
    InputFormat.METS_GBS: ["application/mets+xml"],
    InputFormat.JSON_DOCLING: ["application/json"],
    InputFormat.AUDIO: [
        "audio/x-wav",
        "audio/mpeg",
        "audio/wav",
        "audio/mp3",
        "audio/mp4",
        "audio/m4a",
        "audio/aac",
        "audio/ogg",
        "audio/flac",
        "audio/x-flac",
        "video/mp4",
        "video/avi",
        "video/x-msvideo",
        "video/quicktime",
    ],
    InputFormat.VTT: ["text/vtt"],
    InputFormat.LATEX: ["text/x-tex", "application/x-tex", "text/x-latex"],
    InputFormat.EMAIL: ["message/rfc822"],
    InputFormat.EPUB: ["application/epub+zip"],
}

MimeTypeToFormat: dict[str, list[InputFormat]] = {
    mime: [fmt for fmt in FormatToMimeType if mime in FormatToMimeType[fmt]]
    for value in FormatToMimeType.values()
    for mime in value
}


class DocInputType(str, Enum):
    PATH = "path"
    STREAM = "stream"


class DoclingComponentType(str, Enum):
    DOCUMENT_BACKEND = "document_backend"
    MODEL = "model"
    DOC_ASSEMBLER = "doc_assembler"
    USER_INPUT = "user_input"
    PIPELINE = "pipeline"


class VlmStopReason(str, Enum):
    LENGTH = "length"  # max tokens reached
    STOP_SEQUENCE = "stop_sequence"  # Custom stopping criteria met
    END_OF_SEQUENCE = "end_of_sequence"  # Model generated end-of-text token
    CONTENT_FILTERED = "content_filter"  # Content filtered by API provider
    UNSPECIFIED = "unspecified"  # Defaul none value


class FailureCategory(str, Enum):
    """Shared category enum for both task-level and document/page-level errors.

    A single enum is used for ``PublicFailureInfo.category`` (task scope) and
    ``ErrorItem.category`` (document/page scope) so the jobkit bridge can assign
    one to the other without translation. Some members are only meaningful at one
    scope; that is intentional.

    Task-scope members (produced for ``PublicFailureInfo``, not by pipeline code):
        CAPACITY, TARGET_UNAVAILABLE, INTERNAL.

    Document/page-scope members (produced for ``ErrorItem`` by pipelines/input):
        BACKEND_FAILURE, GENERATION_FAILURE.

    Shared members (used at both scopes):
        POLICY, SOURCE_UNAVAILABLE, TIMEOUT.

    UNKNOWN is the backward-compatible default for uncategorized errors; it is
    distinct from INTERNAL (a known internal service defect).
    """

    POLICY = "policy"
    CAPACITY = "capacity"
    SOURCE_UNAVAILABLE = "source_unavailable"
    TARGET_UNAVAILABLE = "target_unavailable"
    TIMEOUT = "timeout"
    INTERNAL = "internal"
    BACKEND_FAILURE = "backend_failure"
    GENERATION_FAILURE = "generation_failure"
    UNKNOWN = "unknown"


class ErrorItem(BaseModel):
    """Structured error information from document conversion.

    Attributes:
        component_type: The component that generated the error.
        module_name: The module where the error occurred.
        error_message: Human-readable error description.
        category: Semantic category of the error for filtering.
        page_no: 1-indexed page the error is attributable to, or None for
            document-scoped errors.
    """

    component_type: DoclingComponentType
    module_name: str
    error_message: str
    category: FailureCategory = FailureCategory.UNKNOWN
    page_no: int | None = None


class Cluster(BaseModel):
    id: int
    label: DocItemLabel
    bbox: BoundingBox
    confidence: float = 1.0
    cells: list[TextCell] = []
    children: list["Cluster"] = []  # Add child cluster support

    @field_serializer("confidence")
    def _serialize(self, value: float, info: FieldSerializationInfo) -> float:
        return round_pydantic_float(value, info.context, PydanticSerCtxKey.CONFID_PREC)


class BasePageElement(BaseModel):
    label: DocItemLabel
    id: int
    page_no: int
    cluster: Cluster
    text: str | None = None


class LayoutPrediction(BaseModel):
    clusters: list[Cluster] = []


class VlmPredictionToken(BaseModel):
    text: str = ""
    token: int = -1
    logprob: float = -1


class VlmPrediction(BaseModel):
    text: str = ""
    generated_tokens: list[VlmPredictionToken] = []
    generation_time: float = -1
    num_tokens: int | None = None
    usage: Any | None = None
    stop_reason: VlmStopReason = VlmStopReason.UNSPECIFIED
    input_prompt: str | None = None


@dataclass(frozen=True)
class ApiImageRequestResult:
    """Image API response with optional provider usage metadata."""

    text: str
    num_tokens: int | None
    stop_reason: VlmStopReason
    usage: Any | None = None


@dataclass(frozen=True)
class ApiImageStreamingRequestResult:
    """Streaming image API response with optional provider usage metadata."""

    text: str
    num_tokens: int | None
    usage: Any | None = None


class ContainerElement(
    BasePageElement
):  # Used for Form and Key-Value-Regions, only for typing.
    pass


class Table(BasePageElement):
    otsl_seq: list[str]
    num_rows: int = 0
    num_cols: int = 0
    orientation: Orientation = Orientation.ROT_0
    table_cells: list[TableCell]


class TableStructurePrediction(BaseModel):
    table_map: dict[int, Table] = {}


class TextElement(BasePageElement):
    text: str
    hyperlink: Optional[Union[AnyUrl, Path]] = None


class FigureElement(BasePageElement):
    annotations: list[PictureDataType] = []
    provenance: str | None = None
    predicted_class: str | None = None
    confidence: float | None = None

    @field_serializer("confidence")
    def _serialize(
        self, value: float | None, info: FieldSerializationInfo
    ) -> float | None:
        return (
            round_pydantic_float(value, info.context, PydanticSerCtxKey.CONFID_PREC)
            if value is not None
            else None
        )


class FigureClassificationPrediction(BaseModel):
    figure_count: int = 0
    figure_map: dict[int, FigureElement] = {}


class EquationPrediction(BaseModel):
    equation_count: int = 0
    equation_map: dict[int, TextElement] = {}


class PagePredictions(BaseModel):
    layout: LayoutPrediction | None = None
    tablestructure: TableStructurePrediction | None = None
    figures_classification: FigureClassificationPrediction | None = None
    equations_prediction: EquationPrediction | None = None
    vlm_response: VlmPrediction | None = None


PageElement = Union[TextElement, Table, FigureElement, ContainerElement]


class AssembledUnit(BaseModel):
    elements: list[PageElement] = []
    body: list[PageElement] = []
    headers: list[PageElement] = []


class ItemAndImageEnrichmentElement(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    item: NodeItem
    image: Image


class Page(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    page_no: int
    # page_hash: Optional[str] = None
    size: Size | None = None
    parsed_page: SegmentedPdfPage | None = None
    predictions: PagePredictions = PagePredictions()
    assembled: AssembledUnit | None = None

    _backend: Optional["PdfPageBackend"] = (
        None  # Internal PDF backend. By default it is cleared during assembling.
    )
    _default_image_scale: float = 1.0  # Default image scale for external usage.
    _image_cache: dict[
        float, Image
    ] = {}  # Cache of images in different scales. By default it is cleared during assembling.

    @property
    def cells(self) -> list[TextCell]:
        """Return text cells as a read-only view of parsed_page.textline_cells."""
        if self.parsed_page is not None:
            return self.parsed_page.textline_cells
        else:
            return []

    def get_image(
        self,
        scale: float = 1.0,
        max_size: int | None = None,
        cropbox: BoundingBox | None = None,
    ) -> Image | None:
        if self._backend is None:
            return self._image_cache.get(scale, None)

        if max_size:
            assert self.size is not None
            scale = min(scale, max_size / max(self.size.as_tuple()))

        if scale not in self._image_cache:
            if cropbox is None:
                self._image_cache[scale] = self._backend.get_page_image(scale=scale)
            else:
                return self._backend.get_page_image(scale=scale, cropbox=cropbox)

        if cropbox is None:
            return self._image_cache[scale]
        else:
            page_im = self._image_cache[scale]
            assert self.size is not None
            return page_im.crop(
                cropbox.to_top_left_origin(page_height=self.size.height)
                .scaled(scale=scale)
                .as_tuple()
            )

    @property
    def image(self) -> Image | None:
        return self.get_image(scale=self._default_image_scale)


## OpenAI API Request / Response Models ##


class OpenAiChatMessage(BaseModel):
    role: str
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class OpenAiResponseChoice(BaseModel):
    index: int
    message: OpenAiChatMessage
    finish_reason: str | None


class OpenAiResponseUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class OpenAiApiResponse(BaseModel):
    model_config = ConfigDict(
        protected_namespaces=(),
    )

    id: str
    model: str | None = None  # returned by openai
    choices: list[OpenAiResponseChoice]
    created: int
    usage: OpenAiResponseUsage | None = None


# Create a type alias for score values
ScoreValue = float


class QualityGrade(str, Enum):
    POOR = "poor"
    FAIR = "fair"
    GOOD = "good"
    EXCELLENT = "excellent"
    UNSPECIFIED = "unspecified"


class PageConfidenceScores(BaseModel):
    parse_score: ScoreValue = np.nan
    layout_score: ScoreValue = np.nan
    table_score: ScoreValue = np.nan
    ocr_score: ScoreValue = np.nan

    # Accept null/None or string "NaN" values on input and coerce to np.nan
    @field_validator(
        "parse_score", "layout_score", "table_score", "ocr_score", mode="before"
    )
    @classmethod
    def _coerce_none_or_nan_str(cls, v):
        if v is None:
            return np.nan
        if isinstance(v, str) and v.strip().lower() in {"nan", "null", "none", ""}:
            return np.nan
        return v

    def _score_to_grade(self, score: ScoreValue) -> QualityGrade:
        if score < 0.5:
            return QualityGrade.POOR
        elif score < 0.8:
            return QualityGrade.FAIR
        elif score < 0.9:
            return QualityGrade.GOOD
        elif score >= 0.9:
            return QualityGrade.EXCELLENT

        return QualityGrade.UNSPECIFIED

    @computed_field  # type: ignore
    @property
    def mean_grade(self) -> QualityGrade:
        return self._score_to_grade(self.mean_score)

    @computed_field  # type: ignore
    @property
    def low_grade(self) -> QualityGrade:
        return self._score_to_grade(self.low_score)

    @computed_field  # type: ignore
    @property
    def mean_score(self) -> ScoreValue:
        return ScoreValue(
            np.nanmean(
                [
                    self.ocr_score,
                    self.table_score,
                    self.layout_score,
                    self.parse_score,
                ]
            )
        )

    @computed_field  # type: ignore
    @property
    def low_score(self) -> ScoreValue:
        return ScoreValue(
            np.nanquantile(
                [
                    self.ocr_score,
                    self.table_score,
                    self.layout_score,
                    self.parse_score,
                ],
                q=0.05,
            )
        )


class ConfidenceReport(PageConfidenceScores):
    pages: dict[int, PageConfidenceScores] = Field(
        default_factory=lambda: defaultdict(PageConfidenceScores)
    )
    _mean_score_override: ScoreValue = PrivateAttr(default=np.nan)
    _low_score_override: ScoreValue = PrivateAttr(default=np.nan)

    @staticmethod
    def _coerce_override_score(value: Any) -> ScoreValue:
        if value is None:
            return ScoreValue(np.nan)
        if isinstance(value, str) and value.strip().lower() in {
            "nan",
            "null",
            "none",
            "",
        }:
            return ScoreValue(np.nan)
        return ScoreValue(value)

    @model_validator(mode="wrap")
    @classmethod
    def _accept_flat_confidence_scores(cls, value, handler):
        mean_override = ScoreValue(np.nan)
        low_override = ScoreValue(np.nan)

        if isinstance(value, dict):
            mean_override = cls._coerce_override_score(value.get("mean_score"))
            low_override = cls._coerce_override_score(value.get("low_score"))
            value = dict(value)
            value.pop("mean_score", None)
            value.pop("low_score", None)
            value.pop("mean_grade", None)
            value.pop("low_grade", None)

        model = handler(value)
        if not model.pages:
            model._mean_score_override = mean_override
            model._low_score_override = low_override
        return model

    @computed_field  # type: ignore
    @property
    def mean_score(self) -> ScoreValue:
        if not np.isnan(self._mean_score_override):
            return self._mean_score_override
        if not self.pages:
            return super().mean_score
        return ScoreValue(
            np.nanmean(
                [c.mean_score for c in self.pages.values()],
            )
        )

    @computed_field  # type: ignore
    @property
    def low_score(self) -> ScoreValue:
        if not np.isnan(self._low_score_override):
            return self._low_score_override
        if not self.pages:
            return super().low_score
        return ScoreValue(
            np.nanmean(
                [c.low_score for c in self.pages.values()],
            )
        )
