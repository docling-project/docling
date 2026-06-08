"""In-process progress events for document conversion.

These models describe page-level progress emitted by `DocumentConverter`
during conversion, so callers can render progress bars or stream updates
without polling. They mirror the discriminated-union style already used by
`docling.datamodel.service.callbacks` for the HTTP service progress webhook,
but are scoped to a single in-process conversion rather than a batch task.
The names are deliberately distinct from that module to avoid confusing the
in-process events with the HTTP-webhook ones.
"""

import enum
from typing import Callable, Literal, Union

from pydantic import BaseModel

from docling.datamodel.base_models import ConversionStatus


class ConversionProgressKind(str, enum.Enum):
    PAGE_STARTED = "page_started"
    PAGE_COMPLETED = "page_completed"
    DOCUMENT_COMPLETED = "document_completed"


class BaseConversionProgress(BaseModel):
    kind: ConversionProgressKind


class PageStartedProgress(BaseConversionProgress):
    """Emitted when a page enters the conversion pipeline."""

    kind: Literal[ConversionProgressKind.PAGE_STARTED] = (
        ConversionProgressKind.PAGE_STARTED
    )

    page_no: int  # 1-based page number
    # Number of pages being processed. When a ``page_range`` restricts the
    # conversion, this is the count of pages in that range, not the physical
    # page count of the document.
    total_pages: int


class PageCompletedProgress(BaseConversionProgress):
    """Emitted when a page finishes processing (successfully or not)."""

    kind: Literal[ConversionProgressKind.PAGE_COMPLETED] = (
        ConversionProgressKind.PAGE_COMPLETED
    )

    page_no: int  # 1-based page number
    total_pages: int  # see PageStartedProgress.total_pages
    success: bool = True


class DocumentCompletedProgress(BaseConversionProgress):
    """Emitted once after a whole document finishes converting."""

    kind: Literal[ConversionProgressKind.DOCUMENT_COMPLETED] = (
        ConversionProgressKind.DOCUMENT_COMPLETED
    )

    num_pages: int
    status: ConversionStatus


ConversionProgressEvent = Union[
    PageStartedProgress, PageCompletedProgress, DocumentCompletedProgress
]

# A user-supplied callback receiving conversion progress events.
#
# Thread-safety: ``StandardPdfPipeline`` (the default PDF pipeline) processes
# pages on a background producer thread, so for PDFs the callback is always
# invoked from more than one thread, even for a single document. Concurrent
# document conversions (``settings.perf.doc_batch_concurrency > 1``) add
# further concurrency. The callback must therefore be thread-safe. For the
# threaded PDF pipeline no per-page ordering is guaranteed between
# ``page_started`` and ``page_completed`` events. The terminal
# ``document_completed`` event is always emitted last for a given document.
ProgressCallback = Callable[[ConversionProgressEvent], None]
