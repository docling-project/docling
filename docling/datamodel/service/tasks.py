import enum

from pydantic import BaseModel


class TaskType(str, enum.Enum):
    CONVERT = "convert"
    CHUNK = "chunk"


class TaskProcessingMeta(BaseModel):
    num_docs: int
    num_processed: int = 0
    num_succeeded: int = 0
    num_failed: int = 0
