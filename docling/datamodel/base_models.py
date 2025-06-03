import math
from collections import defaultdict
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Dict, List, Literal, Optional, Union

import numpy as np
from docling_core.types.doc import (
    BoundingBox,
    DocItemLabel,
    NodeItem,
    PictureDataType,
    Size,
    TableCell,
)
from docling_core.types.doc.page import SegmentedPdfPage, TextCell
from docling_core.types.io import (
    DocumentStream,
)

# DO NOT REMOVE; explicitly exposed from this location
from PIL.Image import Image
from pydantic import BaseModel, ConfigDict, Field, computed_field

if TYPE_CHECKING:
    from docling.backend.pdf_backend import PdfPageBackend


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
    JSON_DOCLING = "json_docling"


class OutputFormat(str, Enum):
    MARKDOWN = "md"
    JSON = "json"
    HTML = "html"
    HTML_SPLIT_PAGE = "html_split_page"
    TEXT = "text"
    DOCTAGS = "doctags"


FormatToExtensions: Dict[InputFormat, List[str]] = {
    InputFormat.DOCX: ["docx", "dotx", "docm", "dotm"],
    InputFormat.PPTX: ["pptx", "potx", "ppsx", "pptm", "potm", "ppsm"],
    InputFormat.PDF: ["pdf"],
    InputFormat.MD: ["md"],
    InputFormat.HTML: ["html", "htm", "xhtml"],
    InputFormat.XML_JATS: ["xml", "nxml"],
    InputFormat.IMAGE: ["jpg", "jpeg", "png", "tif", "tiff", "bmp"],
    InputFormat.ASCIIDOC: ["adoc", "asciidoc", "asc"],
    InputFormat.CSV: ["csv"],
    InputFormat.XLSX: ["xlsx"],
    InputFormat.XML_USPTO: ["xml", "txt"],
    InputFormat.JSON_DOCLING: ["json"],
}

FormatToMimeType: Dict[InputFormat, List[str]] = {
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
    InputFormat.MD: ["text/markdown", "text/x-markdown"],
    InputFormat.CSV: ["text/csv"],
    InputFormat.XLSX: [
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    ],
    InputFormat.XML_USPTO: ["application/xml", "text/plain"],
    InputFormat.JSON_DOCLING: ["application/json"],
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


class ErrorItem(BaseModel):
    component_type: DoclingComponentType
    module_name: str
    error_message: str


class Cluster(BaseModel):
    id: int
    label: DocItemLabel
    bbox: BoundingBox
    confidence: float = 1.0
    cells: List[TextCell] = []
    children: List["Cluster"] = []  # Add child cluster support


class BasePageElement(BaseModel):
    label: DocItemLabel
    id: int
    page_no: int
    cluster: Cluster
    text: Optional[str] = None


class LayoutPrediction(BaseModel):
    clusters: List[Cluster] = []


class VlmPredictionToken(BaseModel):
    text: str = ""
    token: int = -1
    logprob: float = -1


class VlmPrediction(BaseModel):
    text: str = ""
    generated_tokens: list[VlmPredictionToken] = []
    generation_time: float = -1


class ContainerElement(
    BasePageElement
):  # Used for Form and Key-Value-Regions, only for typing.
    pass


class Table(BasePageElement):
    otsl_seq: List[str]
    num_rows: int = 0
    num_cols: int = 0
    table_cells: List[TableCell]


class TableStructurePrediction(BaseModel):
    table_map: Dict[int, Table] = {}


class TextElement(BasePageElement):
    text: str


class FigureElement(BasePageElement):
    annotations: List[PictureDataType] = []
    provenance: Optional[str] = None
    predicted_class: Optional[str] = None
    confidence: Optional[float] = None


class FigureClassificationPrediction(BaseModel):
    figure_count: int = 0
    figure_map: Dict[int, FigureElement] = {}


class EquationPrediction(BaseModel):
    equation_count: int = 0
    equation_map: Dict[int, TextElement] = {}


class PagePredictions(BaseModel):
    layout: Optional[LayoutPrediction] = None
    tablestructure: Optional[TableStructurePrediction] = None
    figures_classification: Optional[FigureClassificationPrediction] = None
    equations_prediction: Optional[EquationPrediction] = None
    vlm_response: Optional[VlmPrediction] = None


PageElement = Union[TextElement, Table, FigureElement, ContainerElement]


class AssembledUnit(BaseModel):
    elements: List[PageElement] = []
    body: List[PageElement] = []
    headers: List[PageElement] = []


class ItemAndImageEnrichmentElement(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    item: NodeItem
    image: Image


class Page(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    page_no: int
    # page_hash: Optional[str] = None
    size: Optional[Size] = None
    cells: List[TextCell] = []
    parsed_page: Optional[SegmentedPdfPage] = None
    predictions: PagePredictions = PagePredictions()
    assembled: Optional[AssembledUnit] = None

    _backend: Optional["PdfPageBackend"] = (
        None  # Internal PDF backend. By default it is cleared during assembling.
    )
    _default_image_scale: float = 1.0  # Default image scale for external usage.
    _image_cache: Dict[
        float, Image
    ] = {}  # Cache of images in different scales. By default it is cleared during assembling.

    def get_image(
        self, scale: float = 1.0, cropbox: Optional[BoundingBox] = None
    ) -> Optional[Image]:
        if self._backend is None:
            return self._image_cache.get(scale, None)

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
    def image(self) -> Optional[Image]:
        return self.get_image(scale=self._default_image_scale)


## OpenAI API Request / Response Models ##


class OpenAiChatMessage(BaseModel):
    role: str
    content: str


class OpenAiResponseChoice(BaseModel):
    index: int
    message: OpenAiChatMessage
    finish_reason: str


class OpenAiResponseUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class OpenAiApiResponse(BaseModel):
    model_config = ConfigDict(
        protected_namespaces=(),
    )

    id: str
    model: Optional[str] = None  # returned by openai
    choices: List[OpenAiResponseChoice]
    created: int
    usage: OpenAiResponseUsage


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
    pages: Dict[int, PageConfidenceScores] = Field(
        default_factory=lambda: defaultdict(PageConfidenceScores)
    )

    @computed_field  # type: ignore
    @property
    def mean_score(self) -> ScoreValue:
        return ScoreValue(
            np.nanmean(
                [c.mean_score for c in self.pages.values()],
            )
        )

    @computed_field  # type: ignore
    @property
    def low_score(self) -> ScoreValue:
        return ScoreValue(
            np.nanmean(
                [c.low_score for c in self.pages.values()],
            )
        )
