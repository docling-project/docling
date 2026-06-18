"""Conversion job handles for asynchronous docling-serve tasks."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
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


@dataclass(slots=True)
class _AsyncJobHandlers(Generic[T_Result]):
    poll: Callable[[str, float], Awaitable[TaskStatusResponse]]
    watch: Callable[[str, float | None], AsyncIterator[TaskStatusResponse]]
    wait: Callable[[str, float | None], Awaitable[TaskStatusResponse]]
    fetch_result: Callable[[str, TaskStatusResponse | None], Awaitable[T_Result]]


class _ConversionJobBase(Generic[T_Result]):
    """Shared identity and cached status for sync and async job handles."""

    def __init__(
        self,
        task_id: str,
        submitted_at: datetime,
        initial_status: TaskStatusResponse | None = None,
    ) -> None:
        self.task_id = task_id
        self.submitted_at = submitted_at
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


class ConversionJob(_ConversionJobBase[T_Result]):
    """Long-lived handle for a submitted docling-serve task."""

    def __init__(
        self,
        task_id: str,
        submitted_at: datetime,
        handlers: _JobHandlers[T_Result],
        initial_status: TaskStatusResponse | None = None,
    ) -> None:
        super().__init__(task_id, submitted_at, initial_status)
        self._handlers = handlers

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


class AsyncConversionJob(_ConversionJobBase[T_Result]):
    """Long-lived async handle for a submitted docling-serve task."""

    def __init__(
        self,
        task_id: str,
        submitted_at: datetime,
        handlers: _AsyncJobHandlers[T_Result],
        initial_status: TaskStatusResponse | None = None,
    ) -> None:
        super().__init__(task_id, submitted_at, initial_status)
        self._handlers = handlers

    async def poll(self, wait: float = 0.0) -> TaskStatusResponse:
        update = await self._handlers.poll(self.task_id, wait)
        self._last_status = update
        return update

    async def watch(
        self, timeout: float | None = None
    ) -> AsyncIterator[TaskStatusResponse]:  # type: ignore[override]
        async for update in self._handlers.watch(self.task_id, timeout):
            self._last_status = update
            yield update

    async def result(self, timeout: float | None = None) -> T_Result:
        if not self.done:
            self._last_status = await self._handlers.wait(self.task_id, timeout)
        return await self._handlers.fetch_result(self.task_id, self._last_status)
