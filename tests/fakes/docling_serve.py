"""A docling-serve route pack.

Paths and response models mirror docling-serve's own ``app.py`` so the two can
be diffed. Verified against it: ``/health``, ``/version``,
``/v1/convert/source/async``, ``/v1/convert/file/async``,
``/v1/convert/source/batch``, ``/v1/chunk/{path_name}/source/async``,
``/v1/chunk/{path_name}/file/async``, ``/v1/status/poll/{task_id}`` and
``/v1/result/{task_id}`` -- every path the service client can construct except
the WebSocket status stream (``/v1/status/ws/{task_id}``), which is not
implemented here. Clients must therefore use ``StatusWatcherKind.POLLING``;
the WebSocket watcher is not exercised.

``/artifacts/...`` is not a docling-serve route. It stands in for the external
storage a presigned URL points at, so the client's artifact download and its
SSRF check run against a real endpoint.

Submitting returns ``pending``; each poll advances the task one step along
``pending -> started -> success``, so the polling loop and the watcher
genuinely iterate rather than short-circuiting on a canned terminal status.
Every response is built from this repo's own response models, so the fake
cannot drift into being a second copy of the API.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from itertools import count
from typing import Any

from docling_core.types.doc import DoclingDocument
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

from docling.datamodel.base_models import ConversionStatus
from docling.datamodel.service.responses import (
    ArtifactRef,
    ConvertDocumentResponse,
    DocumentArtifactItem,
    ExportDocumentResponse,
    HealthCheckResponse,
    PresignedUrlConvertResponse,
    TaskStatusResponse,
)
from docling.datamodel.service.tasks import TaskType

DEFAULT_MARKDOWN = "# Fake service result\n\nConverted by the in-process fake.\n"


def _as_wire(model: BaseModel) -> Any:
    """Serialise a response model exactly as the service would send it."""
    return json.loads(model.model_dump_json())


def _fake_document(name: str) -> DoclingDocument:
    """A small but genuine DoclingDocument, as the real service would return."""
    doc = DoclingDocument(name=name)
    doc.add_title(text="Fake service result")
    doc.add_text(label="text", text="Converted by the in-process fake.")
    return doc


@dataclass
class FakeTask:
    task_id: str
    task_type: str = "convert"
    polls_before_success: int = 1
    terminal_status: ConversionStatus = ConversionStatus.SUCCESS
    filename: str = "sample.pdf"
    markdown: str = DEFAULT_MARKDOWN
    errors: list[dict[str, Any]] = field(default_factory=list)
    polls: int = 0
    target_kind: str = "inbody"
    source_uri: str = "https://example.com/sample.pdf"

    def status(self) -> ConversionStatus:
        if self.polls == 0:
            return ConversionStatus.PENDING
        if self.polls <= self.polls_before_success:
            return ConversionStatus.STARTED
        return self.terminal_status


class FakeDoclingServe:
    """State plus an ``APIRouter`` mirroring docling-serve."""

    def __init__(self, base_url: str = "") -> None:
        self.base_url = base_url.rstrip("/")
        self.tasks: dict[str, FakeTask] = {}
        self._ids = count(1)
        # Applied to tasks created by the submit routes; a test changes these
        # before submitting to script a slow or failing task.
        self.polls_before_success = 1
        self.terminal_status = ConversionStatus.SUCCESS
        # Set by the fixture once the server is bound, so tests can reach it.
        self.service: Any = None
        self.router = self._build_router()

    # -- task helpers ----------------------------------------------------

    def new_task(
        self, task_type: str = "convert", target_kind: str = "inbody"
    ) -> FakeTask:
        task = FakeTask(
            task_id=f"task-{next(self._ids)}",
            task_type=task_type,
            polls_before_success=self.polls_before_success,
            terminal_status=self.terminal_status,
            target_kind=target_kind,
        )
        self.tasks[task.task_id] = task
        return task

    @staticmethod
    async def _requested_target(request: Request) -> str:
        """The target kind the caller asked for, from JSON body or form data."""
        body = await request.body()
        if request.headers.get("content-type", "").startswith("application/json"):
            target = json.loads(body).get("target") or {}
            return target.get("kind", "inbody")
        # File uploads send options as multipart form fields.
        match = re.search(rb'name="target_type"\r\n\r\n([^\r]+)', body)
        return match.group(1).decode() if match else "inbody"

    def _status(self, task: FakeTask) -> TaskStatusResponse:
        return TaskStatusResponse(
            task_id=task.task_id,
            task_type=TaskType(task.task_type),
            task_status=task.status(),
            task_position=0,
            error_message=(
                "conversion failed in the fake service"
                if task.status() is ConversionStatus.FAILURE
                else None
            ),
        )

    def _result(self, task: FakeTask) -> BaseModel:
        """The result envelope the client expects for the requested target."""
        if task.target_kind == "presigned_url":
            failed = task.terminal_status is ConversionStatus.FAILURE
            return PresignedUrlConvertResponse(
                num_converted=1,
                num_succeeded=0 if failed else 1,
                num_failed=1 if failed else 0,
                processing_time=0.25,
                documents=[
                    DocumentArtifactItem(
                        source_index=0,
                        source_uri=task.source_uri,
                        filename=task.filename,
                        status=task.terminal_status,
                        artifacts=[
                            ArtifactRef(
                                artifact_type="json",
                                mime_type="application/json",
                                uri=f"{self.base_url}/artifacts/{task.task_id}/json",
                            ),
                            ArtifactRef(
                                artifact_type="markdown",
                                mime_type="text/markdown",
                                uri=f"{self.base_url}/artifacts/{task.task_id}/md",
                            ),
                        ],
                    )
                ],
            )
        return ConvertDocumentResponse(
            document=ExportDocumentResponse(
                filename=task.filename,
                md_content=task.markdown,
                json_content=_fake_document(task.filename),
            ),
            status=task.terminal_status,
            processing_time=0.25,
        )

    # -- routes ----------------------------------------------------------

    def _build_router(self) -> APIRouter:
        router = APIRouter()

        @router.get("/health", response_model=HealthCheckResponse)
        async def health() -> HealthCheckResponse:
            return HealthCheckResponse()

        @router.get("/version")
        async def version() -> dict[str, str]:
            return {"version": "0.0.0-fake"}

        @router.post("/v1/convert/source/async", response_model=TaskStatusResponse)
        async def convert_source_async(request: Request) -> TaskStatusResponse:
            kind = await self._requested_target(request)
            return self._status(self.new_task(target_kind=kind))

        @router.post("/v1/convert/file/async", response_model=TaskStatusResponse)
        async def convert_file_async(request: Request) -> TaskStatusResponse:
            kind = await self._requested_target(request)
            return self._status(self.new_task(target_kind=kind))

        @router.post("/v1/convert/source/batch", response_model=TaskStatusResponse)
        async def convert_source_batch(request: Request) -> TaskStatusResponse:
            kind = await self._requested_target(request)
            return self._status(self.new_task(target_kind=kind))

        @router.post(
            "/v1/chunk/{path_name}/source/async", response_model=TaskStatusResponse
        )
        async def chunk_source_async(path_name: str) -> TaskStatusResponse:
            return self._status(self.new_task("chunk"))

        @router.post(
            "/v1/chunk/{path_name}/file/async", response_model=TaskStatusResponse
        )
        async def chunk_file_async(path_name: str) -> TaskStatusResponse:
            return self._status(self.new_task("chunk"))

        @router.get("/v1/status/poll/{task_id}", response_model=TaskStatusResponse)
        async def poll(task_id: str) -> Any:
            task = self.tasks.get(task_id)
            if task is None:
                return JSONResponse({"detail": "task not found"}, status_code=404)
            task.polls += 1
            return self._status(task)

        # The real route returns a union of result envelopes; serialising the
        # chosen model directly avoids FastAPI filtering fields against a
        # response_model that cannot describe every branch.
        @router.get("/v1/result/{task_id}")
        async def result(task_id: str) -> JSONResponse:
            task = self.tasks.get(task_id)
            if task is None:
                return JSONResponse({"detail": "task not found"}, status_code=404)
            return JSONResponse(_as_wire(self._result(task)))

        @router.get("/artifacts/{task_id}/{kind}")
        async def artifact(task_id: str, kind: str) -> Any:
            task = self.tasks.get(task_id)
            if task is None:
                return JSONResponse({"detail": "task not found"}, status_code=404)
            if kind == "md":
                return PlainTextResponse(task.markdown, media_type="text/markdown")
            return JSONResponse(_fake_document(task.filename).export_to_dict())

        return router
