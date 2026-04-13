"""Task status watchers for docling-serve client jobs."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from typing import Protocol

from websockets.sync.client import connect

from docling.datamodel.service.responses import (
    MessageKind,
    TaskStatusResponse,
    WebsocketMessage,
)
from docling.service_client.exceptions import (
    ServiceUnavailableError,
    TaskNotFoundError,
    TaskTimeoutError,
)

TERMINAL_TASK_STATUSES: set[str] = {"success", "failure"}


def is_terminal_task_status(status: TaskStatusResponse) -> bool:
    return status.task_status in TERMINAL_TASK_STATUSES


def _poll_sleep_duration(
    poll_started: float, poll_interval: float, deadline: float
) -> float:
    elapsed = time.monotonic() - poll_started
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return 0.0
    return max(0.0, min(poll_interval, remaining) - elapsed)


class StatusWatcher(Protocol):
    """Protocol for job status watchers."""

    def iter_updates(
        self, task_id: str, timeout: float | None
    ) -> Iterator[TaskStatusResponse]: ...

    def wait_for_terminal(
        self, task_id: str, timeout: float | None
    ) -> TaskStatusResponse: ...


class PollingWatcher:
    """Status watcher using `GET /v1/status/poll/{task_id}` with server-side wait."""

    def __init__(
        self,
        poll_status: Callable[[str, float], TaskStatusResponse],
        poll_server_wait: float,
        poll_client_interval: float | None,
        default_timeout: float,
    ) -> None:
        self._poll_status = poll_status
        self._poll_server_wait = poll_server_wait
        self._poll_client_interval = (
            poll_server_wait if poll_client_interval is None else poll_client_interval
        )
        self._default_timeout = default_timeout

    def iter_updates(
        self, task_id: str, timeout: float | None = None
    ) -> Iterator[TaskStatusResponse]:
        wait_timeout = self._default_timeout if timeout is None else timeout
        deadline = time.monotonic() + wait_timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TaskTimeoutError(
                    f"Timed out waiting for task {task_id} after {wait_timeout:.2f}s."
                )

            poll_wait = min(self._poll_server_wait, remaining)
            poll_started = time.monotonic()
            update = self._poll_status(task_id, poll_wait)
            yield update
            if is_terminal_task_status(update):
                return

            # Keep a minimum client-side poll cadence when server-side wait is ignored.
            sleep_for = _poll_sleep_duration(
                poll_started=poll_started,
                poll_interval=self._poll_client_interval,
                deadline=deadline,
            )
            if sleep_for > 0:
                time.sleep(sleep_for)

    def wait_for_terminal(
        self, task_id: str, timeout: float | None = None
    ) -> TaskStatusResponse:
        final_status: TaskStatusResponse | None = None
        for update in self.iter_updates(task_id=task_id, timeout=timeout):
            final_status = update

        if final_status is None:
            raise TaskTimeoutError(
                f"Timed out waiting for task {task_id} to emit status updates."
            )
        return final_status


class WebSocketWatcher:
    """Status watcher using `WS /v1/status/ws/{task_id}` with poll fallback."""

    def __init__(
        self,
        ws_url_for_task: Callable[[str], str],
        poll_fallback: PollingWatcher | None,
        fallback_to_poll: bool,
        connect_timeout: float,
        default_timeout: float,
        additional_headers: dict[str, str] | None = None,
    ) -> None:
        self._ws_url_for_task = ws_url_for_task
        self._poll_fallback = poll_fallback
        self._fallback_to_poll = fallback_to_poll
        self._connect_timeout = connect_timeout
        self._default_timeout = default_timeout
        self._additional_headers = additional_headers or {}

    def iter_updates(
        self, task_id: str, timeout: float | None = None
    ) -> Iterator[TaskStatusResponse]:
        wait_timeout = self._default_timeout if timeout is None else timeout
        try:
            yield from self._iter_ws_updates(task_id=task_id, timeout=wait_timeout)
        except ServiceUnavailableError:
            if self._fallback_to_poll and self._poll_fallback is not None:
                yield from self._poll_fallback.iter_updates(
                    task_id=task_id, timeout=wait_timeout
                )
                return
            raise

    def wait_for_terminal(
        self, task_id: str, timeout: float | None = None
    ) -> TaskStatusResponse:
        final_status: TaskStatusResponse | None = None
        for update in self.iter_updates(task_id=task_id, timeout=timeout):
            final_status = update

        if final_status is None:
            raise TaskTimeoutError(
                f"Timed out waiting for task {task_id} to emit status updates."
            )
        return final_status

    def _iter_ws_updates(
        self, task_id: str, timeout: float
    ) -> Iterator[TaskStatusResponse]:
        ws_url = self._ws_url_for_task(task_id)
        deadline = time.monotonic() + timeout
        try:
            with connect(
                ws_url,
                open_timeout=self._connect_timeout,
                close_timeout=self._connect_timeout,
                additional_headers=self._additional_headers,
            ) as websocket:
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TaskTimeoutError(
                            f"Timed out waiting for task {task_id} after {timeout:.2f}s."
                        )

                    raw_message = websocket.recv(timeout=remaining)
                    envelope = WebsocketMessage.model_validate_json(raw_message)

                    if envelope.error:
                        if envelope.error == "Task not found.":
                            raise TaskNotFoundError(f"Task {task_id} was not found.")
                        raise ServiceUnavailableError(
                            "WebSocket status stream failed.",
                            detail=envelope.error,
                        )

                    if envelope.task is None:
                        continue

                    yield envelope.task
                    if is_terminal_task_status(envelope.task):
                        return

                    # Only send "next" for UPDATE messages.  The server sends
                    # CONNECTION once before the update loop begins; sending
                    # "next" in response to it would queue an extra token that
                    # the server later consumes as a request for a post-terminal
                    # UPDATE, causing a send-after-close RuntimeError.
                    if envelope.message == MessageKind.UPDATE:
                        websocket.send("next")

        except TaskTimeoutError:
            raise
        except TaskNotFoundError:
            raise
        except Exception as exc:
            raise ServiceUnavailableError(
                "WebSocket status stream is unavailable.", detail=str(exc)
            ) from exc
