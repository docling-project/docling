# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import enum
from typing import Annotated, Any, Literal

from pydantic import AnyUrl, BaseModel, Field, model_serializer, model_validator

from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.service.responses import PublicFailureInfo


class CallbackSpec(BaseModel):
    url: AnyUrl
    headers: dict[str, str] = {}
    ca_cert: str = ""


class ProgressKind(str, enum.Enum):
    SET_NUM_DOCS = "set_num_docs"
    UPDATE_PROCESSED = "update_processed"
    DOCUMENT_COMPLETED = "document_completed"
    TASK_COMPLETED = "task_completed"


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
    num_partially_succeeded: int
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


class ProgressTaskCompleted(BaseProgress):
    """Terminal task outcome, independent of document outcomes."""

    kind: Literal[ProgressKind.TASK_COMPLETED] = ProgressKind.TASK_COMPLETED
    task_status: Literal["success", "failure"]
    failure: PublicFailureInfo | None = None

    @model_validator(mode="after")
    def validate_failure(self) -> "ProgressTaskCompleted":
        if (self.task_status == "failure") != (self.failure is not None):
            raise ValueError(
                "failure must be present exactly when task_status is failure"
            )
        return self

    @model_serializer(mode="wrap")
    def serialize_without_empty_failure(self, handler: Any) -> dict[str, Any]:
        data = handler(self)
        if self.failure is None:
            data.pop("failure", None)
        return data


class ProgressCallbackRequest(BaseModel):
    task_id: str
    progress: Annotated[
        ProgressSetNumDocs
        | ProgressUpdateProcessed
        | ProgressDocumentCompleted
        | ProgressTaskCompleted,
        Field(discriminator="kind"),
    ]


class ProgressCallbackResponse(BaseModel):
    status: Literal["ack"] = "ack"
