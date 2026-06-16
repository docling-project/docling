"""Conversion job handle for asynchronous docling-serve tasks."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime
from typing import Generic, TypeVar

from docling.datamodel.service.responses import TaskStatusResponse
from docling.service_client.watchers import is_terminal_task_status

T_Result = TypeVar("T_Result")


@dataclass(slots=True)
class _JobHandlers(Generic[T_Result]):
    poll: Callable[[str, float], TaskStatusResponse]
    watch: Callable[[str, float | None], Iterator[TaskStatusResponse]]
    wait: Callable[[str, float | None], TaskStatusResponse]
    fetch_result: Callable[[str, TaskStatusResponse | None], T_Result]


class ConversionJob(Generic[T_Result]):
    """Long-lived handle for a submitted docling-serve task."""

    def __init__(
        self,
        task_id: str,
        submitted_at: datetime,
        handlers: _JobHandlers[T_Result],
        initial_status: TaskStatusResponse | None = None,
    ) -> None:
        self.task_id = task_id
        self.submitted_at = submitted_at
        self._handlers = handlers
        self._last_status = initial_status

    @property
    def status(self) -> str:
        if self._last_status is None:
            return "pending"
        return self._last_status.task_status

    @property
    def queue_position(self) -> int | None:
        if self._last_status is None:
            return None
        return self._last_status.task_position

    @property
    def done(self) -> bool:
        return self._last_status is not None and is_terminal_task_status(
            self._last_status
        )

    def poll(self, wait: float = 0.0) -> TaskStatusResponse:
        update = self._handlers.poll(self.task_id, wait)
        self._last_status = update
        return update

    def watch(self, timeout: float | None = None) -> Iterator[TaskStatusResponse]:
        for update in self._handlers.watch(self.task_id, timeout):
            self._last_status = update
            yield update

    def result(self, timeout: float | None = None) -> T_Result:
        if not self.done:
            self._last_status = self._handlers.wait(self.task_id, timeout)
        return self._handlers.fetch_result(self.task_id, self._last_status)
