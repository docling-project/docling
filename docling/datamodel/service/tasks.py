import enum

from pydantic import AliasChoices, BaseModel, Field


class TaskType(str, enum.Enum):
    CONVERT = "convert"
    CHUNK = "chunk"


class TaskProcessingMeta(BaseModel):
    num_docs: int
    num_processed: int = 0
    num_succeeded: int = 0
    num_partially_succeeded: int = Field(
        default=0,
        validation_alias=AliasChoices(
            "num_partially_succeeded",
            "num_partial_success",
        ),
    )
    num_failed: int = 0
