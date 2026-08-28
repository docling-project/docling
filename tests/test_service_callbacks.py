# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from docling.datamodel.service import (
    FailureCategory,
    FailurePhase,
    ProgressCallbackRequest,
    ProgressKind,
    ProgressTaskCompleted,
    PublicFailureInfo,
)


def _failure() -> PublicFailureInfo:
    return PublicFailureInfo(
        category=FailureCategory.INTERNAL,
        message="Internal processing error.",
        retryable=False,
        phase=FailurePhase.EXECUTION,
    )


def test_progress_task_completed_round_trip_and_discrimination() -> None:
    success = ProgressCallbackRequest(
        task_id="task-1",
        progress=ProgressTaskCompleted(task_status="success"),
    )
    assert success.model_dump(mode="json") == {
        "task_id": "task-1",
        "progress": {
            "kind": "task_completed",
            "task_status": "success",
            "failure": None,
        },
    }

    failure = ProgressCallbackRequest.model_validate(
        {
            "task_id": "task-2",
            "progress": {
                "kind": "task_completed",
                "task_status": "failure",
                "failure": _failure().model_dump(mode="json"),
            },
        }
    )
    assert isinstance(failure.progress, ProgressTaskCompleted)
    assert failure.progress.kind == ProgressKind.TASK_COMPLETED
    assert failure.progress.failure == _failure()
