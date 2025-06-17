import logging
import os
import re
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Union, cast

import whisper  # type: ignore

# import librosa
# import numpy as np
# import soundfile as sf  # type: ignore
from docling_core.types.doc.labels import DocItemLabel
from pydantic import BaseModel, Field, validator

# from pydub import AudioSegment  # type: ignore
# from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline
from docling.backend.abstract_backend import AbstractDocumentBackend
from docling.backend.audio_backend import AudioBackend
from docling.datamodel.base_models import (
    ConversionStatus,
)
from docling.datamodel.document import ConversionResult, InputDocument
from docling.datamodel.pipeline_options import (
    AsrPipelineOptions,
)
from docling.datamodel.pipeline_options_asr_model import (
    AsrResponseFormat,
    InlineAsrOptions,
)
from docling.datamodel.pipeline_options_vlm_model import (
    InferenceFramework,
)
from docling.datamodel.settings import settings
from docling.pipeline.base_pipeline import BasePipeline
from docling.utils.profiling import ProfilingScope, TimeRecorder

_log = logging.getLogger(__name__)


class _ConversationWord(BaseModel):
    text: str
    start_time: Optional[float] = Field(
        None, description="Start time in seconds from video start"
    )
    end_time: Optional[float] = Field(
        None, ge=0, description="End time in seconds from video start"
    )


class _ConversationItem(BaseModel):
    text: str
    start_time: Optional[float] = Field(
        None, description="Start time in seconds from video start"
    )
    end_time: Optional[float] = Field(
        None, ge=0, description="End time in seconds from video start"
    )
    speaker_id: Optional[int] = Field(None, description="Numeric speaker identifier")
    speaker: Optional[str] = Field(
        None, description="Speaker name, defaults to speaker-{speaker_id}"
    )
    words: Optional[list[_ConversationWord]] = Field(
        None, description="Individual words with time-stamps"
    )

    def __lt__(self, other):
        if not isinstance(other, _ConversationItem):
            return NotImplemented
        return self.start_time < other.start_time

    def __eq__(self, other):
        if not isinstance(other, _ConversationItem):
            return NotImplemented
        return self.start_time == other.start_time

    def to_string(self) -> str:
        """Format the conversation entry as a string"""
        result = ""
        if (self.start_time is not None) and (self.end_time is not None):
            result += f"[time: {self.start_time}-{self.end_time}] "

        if self.speaker is not None:
            result += f"[speaker:{self.speaker}] "

        result += self.text
        return result


class _NativeWhisperModel:
    def __init__(self, model_name: str = "medium"):
        """
        Transcriber using native Whisper.
        """

        self.model = whisper.load_model(model_name)

        self.verbose = True
        self.word_timestamps = True

    def run(self, conv_res: ConversionResult) -> ConversionResult:
        audio_path: Path = Path(conv_res.input.file).resolve()

        try:
            conversation = self.transcribe(audio_path)

            for _ in conversation:
                conv_res.document.add_text(label=DocItemLabel.TEXT, text=_.to_string())

            conv_res.status = ConversionStatus.SUCCESS
            return conv_res

        except Exception as exc:
            _log.error(f"Audio tranciption has an error: {exc}")

        conv_res.status = ConversionStatus.FAILURE
        return conv_res

    def transcribe(self, fpath: Path) -> list[_ConversationItem]:
        result = self.model.transcribe(
            str(fpath), verbose=self.verbose, word_timestamps=self.word_timestamps
        )

        convo: list[_ConversationItem] = []
        for _ in result["segments"]:
            item = _ConversationItem(
                start_time=_["start"], end_time=_["end"], text=_["text"], words=[]
            )
            item.words = []
            for __ in _["words"]:
                item.words.append(
                    _ConversationWord(
                        start_time=__["start"],
                        end_time=__["end"],
                        text=__["word"],
                    )
                )
            convo.append(item)

        return convo


class AsrPipeline(BasePipeline):
    def __init__(self, pipeline_options: AsrPipelineOptions):
        super().__init__(pipeline_options)
        self.keep_backend = True

        self.pipeline_options: AsrPipelineOptions = pipeline_options

        artifacts_path: Optional[Path] = None
        if pipeline_options.artifacts_path is not None:
            artifacts_path = Path(pipeline_options.artifacts_path).expanduser()
        elif settings.artifacts_path is not None:
            artifacts_path = Path(settings.artifacts_path).expanduser()

        if artifacts_path is not None and not artifacts_path.is_dir():
            raise RuntimeError(
                f"The value of {artifacts_path=} is not valid. "
                "When defined, it must point to a folder containing all models required by the pipeline."
            )

        # self._model = _WhisperModel()
        self._model = _NativeWhisperModel()

    def _determine_status(self, conv_res: ConversionResult) -> ConversionStatus:
        status = ConversionStatus.SUCCESS
        return status

    @classmethod
    def get_default_options(cls) -> AsrPipelineOptions:
        return AsrPipelineOptions()

    def _build_document(self, conv_res: ConversionResult) -> ConversionResult:
        with TimeRecorder(conv_res, "doc_build", scope=ProfilingScope.DOCUMENT):
            _log.info(f"do something: {conv_res.input.file}")
            self._model.run(conv_res=conv_res)
            _log.info(f"finished doing something: {conv_res.input.file}")

        return conv_res

    @classmethod
    def is_backend_supported(cls, backend: AbstractDocumentBackend):
        return isinstance(backend, AudioBackend)
