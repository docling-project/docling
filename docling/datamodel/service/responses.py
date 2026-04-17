import enum
import warnings
from typing import Annotated, Literal, Optional

from docling_core.types.doc.document import DoclingDocument
from pydantic import BaseModel, Field

from docling.datamodel.base_models import ConversionStatus, ErrorItem
from docling.datamodel.service.tasks import TaskProcessingMeta, TaskType
from docling.utils.profiling import ProfilingItem


class ExportDocumentResponse(BaseModel):
    filename: str
    md_content: Optional[str] = None
    json_content: Optional[DoclingDocument] = None
    html_content: Optional[str] = None
    text_content: Optional[str] = None
    doctags_content: Optional[str] = None


class ExportResult(BaseModel):
    """Container of all exported content."""

    kind: Literal["ExportResult"] = "ExportResult"
    content: ExportDocumentResponse
    status: ConversionStatus
    errors: list[ErrorItem] = []
    timings: dict[str, ProfilingItem] = {}


class ZipArchiveResult(BaseModel):
    """Container for a zip archive of the conversion."""

    kind: Literal["ZipArchiveResult"] = "ZipArchiveResult"
    content: bytes


class RemoteTargetResult(BaseModel):
    """No content, the result has been pushed to a remote target."""

    kind: Literal["RemoteTargetResult"] = "RemoteTargetResult"


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


ResultType = Annotated[
    ExportResult | ZipArchiveResult | RemoteTargetResult | ChunkedDocumentResult,
    Field(discriminator="kind"),
]


class DoclingTaskResult(BaseModel):
    result: ResultType
    processing_time: float
    num_converted: int
    num_succeeded: int
    num_failed: int


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
    document: ExportDocumentResponse
    status: ConversionStatus
    errors: list[ErrorItem] = []
    processing_time: float
    timings: dict[str, ProfilingItem] = {}


class PresignedUrlConvertDocumentResponse(BaseModel):
    processing_time: float
    num_converted: int
    num_succeeded: int
    num_failed: int


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
    task_status: str
    task_position: Optional[int] = None
    task_meta: Optional[TaskProcessingMeta] = None
    error_message: Optional[str] = None


class MessageKind(str, enum.Enum):
    CONNECTION = "connection"
    UPDATE = "update"
    ERROR = "error"


class WebsocketMessage(BaseModel):
    message: MessageKind
    task: Optional[TaskStatusResponse] = None
    error: Optional[str] = None
