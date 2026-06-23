import enum
import math
import warnings
from datetime import datetime
from typing import TYPE_CHECKING, Annotated, Literal, Optional

from docling_core.types.doc.document import DoclingDocument
from pydantic import AliasChoices, AnyUrl, BaseModel, ConfigDict, Field

from docling.datamodel.base_models import (
    ConversionStatus,
    ErrorItem,
    FailureCategory,
    QualityGrade,
)
from docling.datamodel.service.tasks import TaskProcessingMeta, TaskType
from docling.utils.profiling import ProfilingItem

if TYPE_CHECKING:
    from docling.datamodel.base_models import ConfidenceReport, PageConfidenceScores


def _nan_to_none(value: float) -> Optional[float]:
    return None if value is None or math.isnan(value) else float(value)


class ConfidenceScores(BaseModel):
    """JSON-safe snapshot of docling confidence scores.

    Plain stored floats (NaN coerced to null) rather than a reuse of
    docling's ``ConfidenceReport``, so the wire schema carries no numpy
    computed fields and deserializing clients don't recompute anything.

    Document-level only. A per-page breakdown can be added later as an
    optional ``pages`` field without breaking this shape (adding an optional
    field is non-breaking on its own).
    """

    parse_score: Optional[float] = None
    layout_score: Optional[float] = None
    table_score: Optional[float] = None
    ocr_score: Optional[float] = None
    mean_score: Optional[float] = None
    low_score: Optional[float] = None
    mean_grade: QualityGrade = QualityGrade.UNSPECIFIED
    low_grade: QualityGrade = QualityGrade.UNSPECIFIED

    @classmethod
    def from_scores(
        cls, scores: "PageConfidenceScores | ConfidenceReport"
    ) -> "ConfidenceScores":
        return cls(
            parse_score=_nan_to_none(scores.parse_score),
            layout_score=_nan_to_none(scores.layout_score),
            table_score=_nan_to_none(scores.table_score),
            ocr_score=_nan_to_none(scores.ocr_score),
            mean_score=_nan_to_none(scores.mean_score),
            low_score=_nan_to_none(scores.low_score),
            mean_grade=scores.mean_grade,
            low_grade=scores.low_grade,
        )


class ExportDocumentResponse(BaseModel):
    filename: str
    md_content: Optional[str] = None
    json_content: Optional[DoclingDocument] = None
    html_content: Optional[str] = None
    text_content: Optional[str] = None
    doctags_content: Optional[str] = None
    doclang_content: Optional[str] = None


class DocumentResultItem(BaseModel):
    """Canonical document-level result with legacy ExportResult wire compatibility."""

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    kind: Literal["ExportResult"] = "ExportResult"
    document: ExportDocumentResponse = Field(
        validation_alias=AliasChoices("document", "content"),
        serialization_alias="content",
    )
    status: ConversionStatus
    errors: list[ErrorItem] = []
    timings: dict[str, ProfilingItem] = {}
    confidence: Optional[ConfidenceScores] = None

    @property
    def content(self) -> ExportDocumentResponse:
        warnings.warn(
            "DocumentResultItem.content is deprecated; use .document instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.document

    @content.setter
    def content(self, value: ExportDocumentResponse) -> None:
        warnings.warn(
            "DocumentResultItem.content is deprecated; use .document instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.document = value


ExportResult = DocumentResultItem


class ZipArchiveResult(BaseModel):
    """Container for a zip archive of the conversion."""

    kind: Literal["ZipArchiveResult"] = "ZipArchiveResult"
    content: bytes


class RemoteTargetResult(BaseModel):
    """No content, the result has been pushed to a remote target."""

    kind: Literal["RemoteTargetResult"] = "RemoteTargetResult"


class ArtifactRef(BaseModel):
    artifact_type: Literal[
        "json", "html", "markdown", "text", "doctags", "doclang", "resource_bundle"
    ]
    mime_type: str
    uri: AnyUrl
    url_expires_at: datetime | None = None


class DocumentArtifactItem(BaseModel):
    """Per-document result item for PresignedUrlTarget responses."""

    source_index: int
    source_uri: str
    filename: str
    status: ConversionStatus
    errors: list[ErrorItem] = []
    timings: dict[str, ProfilingItem] = {}
    artifacts: list[ArtifactRef] = []
    confidence: Optional[ConfidenceScores] = None


class ChunkedDocumentResultItem(BaseModel):
    """A single chunk of a document with its metadata and content."""

    filename: str
    chunk_index: int
    text: Annotated[
        str,
        Field(
            description="The chunk text with structural context (headers, formatting)"
        ),
    ]
    raw_text: Annotated[
        str | None,
        Field(
            description="Raw chunk text without additional formatting or context",
        ),
    ] = None
    num_tokens: Annotated[
        int | None,
        Field(
            description="Number of tokens in the text, if the chunker is aware of tokens"
        ),
    ] = None
    headings: Annotated[
        list[str] | None, Field(description="List of headings for this chunk")
    ] = None
    captions: Annotated[
        list[str] | None,
        Field(
            description="List of captions for this chunk (e.g. for pictures and tables)",
        ),
    ] = None
    doc_items: Annotated[list[str], Field(description="List of doc items references")]
    page_numbers: Annotated[
        list[int] | None,
        Field(description="Page numbers where this chunk content appears"),
    ] = None
    metadata: Annotated[
        dict | None, Field(description="Additional metadata associated with this chunk")
    ] = None


class ChunkedDocumentResult(BaseModel):
    kind: Literal["ChunkedDocumentResponse"] = "ChunkedDocumentResponse"
    chunks: list[ChunkedDocumentResultItem]
    documents: list[ExportResult]
    chunking_info: Optional[dict] = None


class PresignedArtifactResult(BaseModel):
    """Internal DoclingTaskResult.result union member for PresignedUrlTarget."""

    kind: Literal["PresignedArtifactResult"] = "PresignedArtifactResult"
    documents: list[DocumentArtifactItem]


class ConvertedOutcomeCountsMixin(BaseModel):
    num_converted: int
    num_succeeded: int
    num_partially_succeeded: int = 0
    num_failed: int


# FailureCategory lives in base_models; re-exported here for back-compat imports.


class FailurePhase(str, enum.Enum):
    ADMISSION = "admission"
    SOURCE_ENUMERATION = "source_enumeration"
    EXECUTION = "execution"
    ORCHESTRATION = "orchestration"


class PublicFailureInfo(BaseModel):
    category: FailureCategory
    message: str
    retryable: bool
    phase: FailurePhase
    details: dict[str, str] = Field(default_factory=dict)


class TaskFailureResult(BaseModel):
    kind: Literal["TaskFailureResult"] = "TaskFailureResult"
    failure: PublicFailureInfo


ResultType = Annotated[
    ExportResult
    | ZipArchiveResult
    | RemoteTargetResult
    | ChunkedDocumentResult
    | PresignedArtifactResult,
    Field(discriminator="kind"),
]


class DoclingTaskResult(ConvertedOutcomeCountsMixin):
    result: ResultType
    processing_time: float


class ConvertDocumentResult(DoclingTaskResult):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "ConvertDocumentResult is deprecated and will be removed in a future version. "
            "Use DoclingTaskResult instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


class HealthCheckResponse(BaseModel):
    status: str = "ok"


class ReadinessResponse(BaseModel):
    status: str = "ok"


class ClearResponse(BaseModel):
    status: str = "ok"


class ConvertDocumentResponse(BaseModel):
    """Single-document inline response with task-level timing flattened in."""

    document: ExportDocumentResponse
    status: ConversionStatus
    errors: list[ErrorItem] = []
    # Inline convert responses have no outer DoclingTaskResult envelope, so the
    # task-level elapsed time is flattened onto this response model.
    processing_time: float
    timings: dict[str, ProfilingItem] = {}
    confidence: Optional[ConfidenceScores] = None


class PresignedUrlConvertDocumentResponse(ConvertedOutcomeCountsMixin):
    """Counts-only response model for remote targets without per-document artifacts."""

    processing_time: float


class PresignedUrlConvertResponse(PresignedUrlConvertDocumentResponse):
    documents: list[DocumentArtifactItem]


class ConvertDocumentErrorResponse(BaseModel):
    status: ConversionStatus


class UsageLimitExceededDetails(BaseModel):
    currentUsage: int
    limit: int


class UsageLimitExceededResponse(BaseModel):
    error: Literal["usage_limit_exceeded"]
    message: str
    details: UsageLimitExceededDetails


class ChunkDocumentResponse(BaseModel):
    chunks: list[ChunkedDocumentResultItem]
    documents: list[ExportResult]
    processing_time: float


class TaskStatusResponse(BaseModel):
    task_id: str
    task_type: TaskType
    task_status: ConversionStatus
    task_position: Optional[int] = None
    task_meta: Optional[TaskProcessingMeta] = None
    error_message: Optional[str] = None
    failure: Optional[PublicFailureInfo] = None


class MessageKind(str, enum.Enum):
    CONNECTION = "connection"
    UPDATE = "update"
    ERROR = "error"


class WebsocketMessage(BaseModel):
    message: MessageKind
    task: Optional[TaskStatusResponse] = None
    error: Optional[str] = None
