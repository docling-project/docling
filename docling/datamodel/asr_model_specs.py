import logging
from enum import Enum

from pydantic import (
    AnyUrl,
)

from docling.datamodel.accelerator_options import AcceleratorDevice
from docling.datamodel.pipeline_options_asr_model import (
    # ApiAsrOptions,
    InferenceFramework,
    InlineAsrOptions,
    AsrResponseFormat,
    TransformersModelType,
)

_log = logging.getLogger(__name__)

# SmolDocling
WHISPER_TINY = InlineAsrOptions(
    repo_id="openai/whisper-tiny",
    inference_framework=InferenceFramework.TRANSFORMERS,
    response_format = AsrResponseFormat.WHISPER,
)

class AsrModelType(str, Enum):
    WHISPER_TINY = "whisper_tiny"
