"""Progress event types for reporting conversion progress."""

from enum import Enum

from pydantic import BaseModel, ConfigDict


class ProgressEventType(str, Enum):
    """Types of progress events emitted during conversion."""

    DOCUMENT_START = "document_start"
    DOCUMENT_COMPLETE = "document_complete"
    PHASE_START = "phase_start"
    PHASE_COMPLETE = "phase_complete"
    PAGE_COMPLETE = "page_complete"


class ConversionPhase(str, Enum):
    """Phases of the document conversion pipeline."""

    BUILD = "build"
    ASSEMBLE = "assemble"
    ENRICH = "enrich"


class ProgressEvent(BaseModel):
    """Base progress event with event type and document name."""

    model_config = ConfigDict(frozen=True)

    event_type: ProgressEventType
    document_name: str


class DocumentProgressEvent(ProgressEvent):
    """Event emitted at document start/complete, includes page count."""

    page_count: int | None = None


class PhaseProgressEvent(ProgressEvent):
    """Event emitted at phase start/complete."""

    phase: ConversionPhase


class PageProgressEvent(ProgressEvent):
    """Event emitted when a page completes processing."""

    page_no: int
    total_pages: int
