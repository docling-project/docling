# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""In-process progress events emitted while documents are converted.

Pass a `ProgressCallback` to `DocumentConverter.convert(progress_callback=...)`
(or `convert_all`) to receive the events of that call, or to the
`DocumentConverter` constructor to receive the events of every call. The names differ from `docling.datamodel.service.callbacks`,
which describes the batch-level webhook of the HTTP service.
"""

import enum
import logging
from collections.abc import Iterable, Sequence
from typing import Annotated, Callable, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

from docling.datamodel.base_models import ConversionStatus

_log = logging.getLogger(__name__)


class ConversionProgressKind(str, enum.Enum):
    DOCUMENT_STARTED = "document_started"
    PHASE_STARTED = "phase_started"
    PAGE_COMPLETED = "page_completed"
    ENRICHMENT_PROGRESS = "enrichment_progress"
    DOCUMENT_COMPLETED = "document_completed"


class ConversionPhase(str, enum.Enum):
    """The steps every converted document goes through, in this order.

    `INITIALIZE` gets the pipeline ready. It takes long only for the first
    document of a pipeline, while its models are downloaded and loaded.
    """

    INITIALIZE = "initialize"
    BUILD = "build"
    ASSEMBLE = "assemble"
    ENRICH = "enrich"


class EnrichmentStep(str, enum.Enum):
    """Stable identifier of an enrichment step, safe to match on.

    Enrichment models that docling does not know about report `OTHER`.
    """

    CODE_FORMULA = "code_formula"
    PICTURE_CLASSIFICATION = "picture_classification"
    PICTURE_DESCRIPTION = "picture_description"
    CHART_EXTRACTION = "chart_extraction"
    OTHER = "other"


ENRICHMENT_STEP_LABELS: dict[EnrichmentStep, str] = {
    EnrichmentStep.CODE_FORMULA: "code and formulas",
    EnrichmentStep.PICTURE_CLASSIFICATION: "picture classification",
    EnrichmentStep.PICTURE_DESCRIPTION: "picture description",
    EnrichmentStep.CHART_EXTRACTION: "chart extraction",
}


class BaseConversionProgress(BaseModel):
    """Fields shared by all events.

    `document_index` is the 1-based position of the document in the
    `convert_all` call (always 1 for `convert`). It tells apart documents with
    the same name and documents converted concurrently.
    """

    model_config = ConfigDict(frozen=True)

    kind: ConversionProgressKind
    document_index: int
    document_name: str


class DocumentStartedProgress(BaseConversionProgress):
    """Emitted once per input document, also for inputs that are skipped."""

    kind: Literal[ConversionProgressKind.DOCUMENT_STARTED] = (
        ConversionProgressKind.DOCUMENT_STARTED
    )


class PhaseStartedProgress(BaseConversionProgress):
    """Emitted when the pipeline enters a phase, for every input format."""

    kind: Literal[ConversionProgressKind.PHASE_STARTED] = (
        ConversionProgressKind.PHASE_STARTED
    )

    phase: ConversionPhase


class PageCompletedProgress(BaseConversionProgress):
    """Emitted once per page, after all page-level models (including tables).

    Only page-based pipelines emit it. Pages may finish out of order, so use
    `completed_pages` rather than `page_no` to drive a progress bar.
    `total_pages` counts the pages selected by `page_range`.
    """

    kind: Literal[ConversionProgressKind.PAGE_COMPLETED] = (
        ConversionProgressKind.PAGE_COMPLETED
    )

    page_no: int
    success: bool
    completed_pages: int
    total_pages: int


class EnrichmentProgress(BaseConversionProgress):
    """Item-level progress of one enrichment step, e.g. picture description.

    Enrichment runs on the assembled document, after the last page completed.
    Each step that has work to do first reports `completed_items=0` and then
    reports again after every batch, ending with
    `completed_items == total_items`. Match on `step`; `label` is a readable
    name for display and may change.
    """

    kind: Literal[ConversionProgressKind.ENRICHMENT_PROGRESS] = (
        ConversionProgressKind.ENRICHMENT_PROGRESS
    )

    step: EnrichmentStep
    label: str
    completed_items: int
    total_items: int


class DocumentCompletedProgress(BaseConversionProgress):
    """Emitted once as the last event for a document, also when it failed."""

    kind: Literal[ConversionProgressKind.DOCUMENT_COMPLETED] = (
        ConversionProgressKind.DOCUMENT_COMPLETED
    )

    status: ConversionStatus


ConversionProgressEvent = Annotated[
    Union[
        DocumentStartedProgress,
        PhaseStartedProgress,
        PageCompletedProgress,
        EnrichmentProgress,
        DocumentCompletedProgress,
    ],
    Field(discriminator="kind"),
]

ProgressCallback = Callable[[ConversionProgressEvent], None]


class ProgressReporter:
    """Per-document sink between the converter, the pipeline and the callbacks.

    A misbehaving callback must not break the conversion, so its exceptions
    are logged and swallowed. All events of one document are emitted from the
    thread that converts it; with `doc_batch_concurrency > 1` several
    documents report concurrently.
    """

    def __init__(
        self,
        callbacks: Sequence[ProgressCallback] = (),
        document_name: str = "",
        document_index: int = 1,
    ) -> None:
        self.callbacks = tuple(callbacks)
        self.document_name = document_name
        self.document_index = document_index
        self._finished_page_nos: set[int] = set()

    @property
    def enabled(self) -> bool:
        return bool(self.callbacks)

    def _emit(self, event: ConversionProgressEvent) -> None:
        for callback in self.callbacks:
            try:
                callback(event)
            except Exception:
                _log.warning("progress callback raised an exception", exc_info=True)

    def document_started(self) -> None:
        if self.enabled:
            self._emit(
                DocumentStartedProgress(
                    document_index=self.document_index,
                    document_name=self.document_name,
                )
            )

    def phase_started(self, phase: ConversionPhase) -> None:
        if self.enabled:
            self._emit(
                PhaseStartedProgress(
                    document_index=self.document_index,
                    document_name=self.document_name,
                    phase=phase,
                )
            )

    def page_completed(self, page_no: int, total_pages: int, success: bool) -> None:
        if not self.enabled or page_no < 1 or page_no in self._finished_page_nos:
            return
        self._finished_page_nos.add(page_no)
        self._emit(
            PageCompletedProgress(
                document_index=self.document_index,
                document_name=self.document_name,
                page_no=page_no,
                success=success,
                completed_pages=len(self._finished_page_nos),
                total_pages=total_pages,
            )
        )

    def fail_unfinished_pages(self, page_nos: Iterable[int], total_pages: int) -> None:
        """Report pages that never completed (timeout, early stop) as failed."""
        for page_no in sorted(page_nos):
            self.page_completed(page_no, total_pages=total_pages, success=False)

    def enrichment_progress(
        self, step: EnrichmentStep, label: str, completed_items: int, total_items: int
    ) -> None:
        if self.enabled:
            self._emit(
                EnrichmentProgress(
                    document_index=self.document_index,
                    document_name=self.document_name,
                    step=step,
                    label=label,
                    completed_items=completed_items,
                    total_items=total_items,
                )
            )

    def document_completed(self, status: ConversionStatus) -> None:
        if self.enabled:
            self._emit(
                DocumentCompletedProgress(
                    document_index=self.document_index,
                    document_name=self.document_name,
                    status=status,
                )
            )
