"""A docling-serve route pack for :class:`FakeHttpService`.

Submitting a task returns ``pending``; each poll advances the task one step
along ``pending -> started -> success``, so the client's polling loop and the
watcher actually iterate instead of short-circuiting on a canned terminal
status. Tests can override the step count, force a failure, or inject faults
per route.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from itertools import count
from typing import Any

from docling_core.types.doc import DoclingDocument
from pydantic import BaseModel

from docling.datamodel.base_models import ConversionStatus
from docling.datamodel.service.responses import (
    ArtifactRef,
    ConvertDocumentResponse,
    DocumentArtifactItem,
    ExportDocumentResponse,
    PresignedUrlConvertResponse,
    TaskStatusResponse,
)
from docling.datamodel.service.tasks import TaskType
from tests.fakes.http_service import FakeHttpService, RecordedRequest, Response

DEFAULT_MARKDOWN = "# Fake service result\n\nConverted by the in-process fake.\n"


def _as_wire(model: BaseModel) -> dict[str, Any]:
    """Serialise a response model exactly as the service would put it on the wire.

    Building every response from this repo's own response models is what keeps
    the fake from becoming a second, drifting copy of the docling-serve API: a
    change to the models either flows through here or fails loudly.
    """
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

    def is_terminal(self) -> bool:
        return self.status() in (
            ConversionStatus.SUCCESS,
            ConversionStatus.PARTIAL_SUCCESS,
            ConversionStatus.FAILURE,
        )


class FakeDoclingServe:
    """Registers docling-serve routes onto a :class:`FakeHttpService`."""

    def __init__(self, service: FakeHttpService, base_url: str = "") -> None:
        self.service = service
        self.base_url = base_url.rstrip("/")
        self.tasks: dict[str, FakeTask] = {}
        self._ids = count(1)
        # Defaults applied to tasks created by the submit routes; a test can
        # change these before submitting to script a slow or failing task.
        self.polls_before_success = 1
        self.terminal_status = ConversionStatus.SUCCESS
        self._register()

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
    def _requested_target(request: RecordedRequest) -> str:
        """The target kind the client asked for, from JSON body or form data."""
        if request.headers.get("content-type", "").startswith("application/json"):
            target = request.json().get("target") or {}
            return target.get("kind", "inbody")
        # File uploads send options as multipart form fields.
        match = re.search(rb'name="target_type"\r\n\r\n([^\r]+)', request.body)
        return match.group(1).decode() if match else "inbody"

    def _status_payload(self, task: FakeTask) -> dict[str, Any]:
        status = TaskStatusResponse(
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
        return _as_wire(status)

    def _result_payload(self, task: FakeTask) -> dict[str, Any]:
        """The result envelope the client expects for the requested target."""
        if task.target_kind == "presigned_url":
            failed = task.terminal_status is ConversionStatus.FAILURE
            response = PresignedUrlConvertResponse(
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
            return _as_wire(response)

        return _as_wire(
            ConvertDocumentResponse(
                document=ExportDocumentResponse(
                    filename=task.filename,
                    md_content=task.markdown,
                    json_content=_fake_document(task.filename),
                ),
                status=task.terminal_status,
                processing_time=0.25,
            )
        )

    # -- routes ----------------------------------------------------------

    def _register(self) -> None:
        service = self.service

        @service.route("GET", r"/health")
        def _health(request: RecordedRequest, match: re.Match[str]) -> Response:
            return Response(body={"status": "ok"})

        @service.route("GET", r"/version")
        def _version(request: RecordedRequest, match: re.Match[str]) -> Response:
            return Response(body={"version": "0.0.0-fake"})

        @service.route("POST", r"/v1/convert/source/async")
        def _submit_source(request: RecordedRequest, match: re.Match[str]) -> Response:
            task = self.new_task(target_kind=self._requested_target(request))
            return Response(body=self._status_payload(task))

        @service.route("POST", r"/v1/convert/file/async")
        def _submit_file(request: RecordedRequest, match: re.Match[str]) -> Response:
            task = self.new_task(target_kind=self._requested_target(request))
            return Response(body=self._status_payload(task))

        @service.route("POST", r"/v1/chunk/[^/]+/source/async")
        def _submit_chunk_source(
            request: RecordedRequest, match: re.Match[str]
        ) -> Response:
            return Response(body=self._status_payload(self.new_task("chunk")))

        @service.route("POST", r"/v1/chunk/[^/]+/file/async")
        def _submit_chunk_file(
            request: RecordedRequest, match: re.Match[str]
        ) -> Response:
            return Response(body=self._status_payload(self.new_task("chunk")))

        @service.route("GET", r"/v1/status/poll/(?P<task_id>[^/]+)")
        def _poll(request: RecordedRequest, match: re.Match[str]) -> Response:
            task = self.tasks.get(match.group("task_id"))
            if task is None:
                return Response(status=404, body={"detail": "task not found"})
            task.polls += 1
            return Response(body=self._status_payload(task))

        @service.route("GET", r"/v1/result/(?P<task_id>[^/]+)")
        def _result(request: RecordedRequest, match: re.Match[str]) -> Response:
            task = self.tasks.get(match.group("task_id"))
            if task is None:
                return Response(status=404, body={"detail": "task not found"})
            return Response(body=self._result_payload(task))

        # Presigned artifacts are served from this same host, so the client's
        # artifact download and URL validation run for real.
        @service.route("GET", r"/artifacts/(?P<task_id>[^/]+)/(?P<kind>json|md)")
        def _artifact(request: RecordedRequest, match: re.Match[str]) -> Response:
            task = self.tasks.get(match.group("task_id"))
            if task is None:
                return Response(status=404, body={"detail": "task not found"})
            if match.group("kind") == "md":
                return Response(
                    body=task.markdown, headers={"Content-Type": "text/markdown"}
                )
            document = _fake_document(task.filename)
            return Response(body=document.export_to_dict())
