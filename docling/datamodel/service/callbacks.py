import enum
from typing import Annotated, Literal

from pydantic import AnyUrl, BaseModel, Field

from docling.datamodel.base_models import ConversionStatus, InputFormat


class CallbackSpec(BaseModel):
    url: AnyUrl
    headers: dict[str, str] = {}
    ca_cert: str = ""


class ProgressKind(str, enum.Enum):
    SET_NUM_DOCS = "set_num_docs"
    UPDATE_PROCESSED = "update_processed"
    DOCUMENT_COMPLETED = "document_completed"


class BaseProgress(BaseModel):
    kind: ProgressKind


class ProgressSetNumDocs(BaseProgress):
    kind: Literal[ProgressKind.SET_NUM_DOCS] = ProgressKind.SET_NUM_DOCS

    num_docs: int


class ProcessedDocsItem(BaseModel):
    source: str
    status: ConversionStatus
    error: str | None = None


class ProgressUpdateProcessed(BaseProgress):
    kind: Literal[ProgressKind.UPDATE_PROCESSED] = ProgressKind.UPDATE_PROCESSED

    num_processed: int
    num_succeeded: int
    num_partial_success: int = 0
    num_failed: int

    docs: list[ProcessedDocsItem]


class DocumentCompletedItem(BaseModel):
    """Detailed information about a completed document conversion."""

    source: str
    status: ConversionStatus
    document_type: InputFormat | None = None
    num_pages: int | None = None
    num_characters: int | None = None
    num_tables: int | None = None
    num_pictures: int | None = None
    processing_time: float | None = None  # in seconds
    doc_hash: str | None = None
    error: str | None = None


class ProgressDocumentCompleted(BaseProgress):
    """Progress update sent after each document is converted."""

    kind: Literal[ProgressKind.DOCUMENT_COMPLETED] = ProgressKind.DOCUMENT_COMPLETED

    document: DocumentCompletedItem
    # Context about overall task progress
    total_processed: int  # How many docs processed so far
    total_docs: int | None = None  # Total docs in task (if known)


class ProgressCallbackRequest(BaseModel):
    task_id: str
    progress: Annotated[
        ProgressSetNumDocs | ProgressUpdateProcessed | ProgressDocumentCompleted,
        Field(discriminator="kind"),
    ]


class ProgressCallbackResponse(BaseModel):
    status: Literal["ack"] = "ack"
