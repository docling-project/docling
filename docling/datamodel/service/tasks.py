# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import enum

from pydantic import BaseModel


class TaskType(str, enum.Enum):
    CONVERT = "convert"
    CHUNK = "chunk"


class TaskStatus(str, enum.Enum):
    PENDING = "pending"
    STARTED = "started"
    SUCCESS = "success"
    FAILURE = "failure"
    CANCELLED = "cancelled"


class TaskProcessingMeta(BaseModel):
    num_docs: int
    num_processed: int = 0
    num_succeeded: int = 0
    num_partially_succeeded: int = 0
    num_failed: int = 0
