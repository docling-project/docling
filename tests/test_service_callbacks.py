# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import pytest
from pydantic import ValidationError

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


@pytest.mark.parametrize(
    ("task_status", "failure"),
    [("success", _failure()), ("failure", None)],
)
def test_progress_task_completed_rejects_invalid_failure_combinations(
    task_status: str, failure: PublicFailureInfo | None
) -> None:
    with pytest.raises(ValidationError):
        ProgressTaskCompleted(task_status=task_status, failure=failure)  # type: ignore[arg-type]
