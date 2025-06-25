import logging
import os
import re
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Union, cast

from docling_core.types.doc import DoclingDocument, DocumentOrigin

# import whisper  # type: ignore
# import librosa
# import numpy as np
# import soundfile as sf  # type: ignore
from docling_core.types.doc.labels import DocItemLabel
from pydantic import BaseModel, Field, validator

from docling.backend.abstract_backend import AbstractDocumentBackend
from docling.backend.noop_backend import NoOpBackend

# from pydub import AudioSegment  # type: ignore
# from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline
from docling.datamodel.accelerator_options import (
    AcceleratorOptions,
)
from docling.datamodel.base_models import (
    ConversionStatus,
    FormatToMimeType,
)
from docling.datamodel.document import ConversionResult, InputDocument
from docling.datamodel.pipeline_options import (
    AsrPipelineOptions,
)
from docling.datamodel.pipeline_options_asr_model import (
    InlineAsrNativeWhisperOptions,
    # AsrResponseFormat,
    InlineAsrOptions,
)
from docling.datamodel.pipeline_options_vlm_model import (
    InferenceFramework,
)
from docling.datamodel.settings import settings
from docling.pipeline.base_pipeline import BasePipeline
from docling.utils.accelerator_utils import decide_device
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
    def __init__(
        self,
        enabled: bool,
        artifacts_path: Optional[Path],
        accelerator_options: AcceleratorOptions,
        asr_options: InlineAsrNativeWhisperOptions,
    ):
        """
        Transcriber using native Whisper.
        """
        self.enabled = enabled

        _log.info(f"artifacts-path: {artifacts_path}")
        _log.info(f"accelerator_options: {accelerator_options}")

        if self.enabled:
            try:
                import whisper  # type: ignore
            except ImportError:
                raise ImportError(
                    "whisper is not installed. Please install it via `pip install openai-whisper` or do `uv sync --extra asr`."
                )
            self.asr_options = asr_options
            self.max_tokens = asr_options.max_new_tokens
            self.temperature = asr_options.temperature

            self.device = decide_device(
                accelerator_options.device,
                supported_devices=asr_options.supported_devices,
            )
            _log.info(f"Available device for Whisper: {self.device}")

            self.model_name = asr_options.repo_id
            _log.info(f"loading _NativeWhisperModel({self.model_name})")
            if artifacts_path is not None:
                _log.info(f"loading {self.model_name} from {artifacts_path}")
                self.model = whisper.load_model(
                    name=self.model_name,
                    device=self.device,
                    download_root=str(artifacts_path),
                )
            else:
                self.model = whisper.load_model(
                    name=self.model_name, device=self.device
                )

            self.verbose = asr_options.verbose
            self.timestamps = asr_options.timestamps
            self.word_timestamps = asr_options.word_timestamps

    def run(self, conv_res: ConversionResult) -> ConversionResult:
        audio_path: Path = Path(conv_res.input.file).resolve()

        try:
            conversation = self.transcribe(audio_path)

            # Ensure we have a proper DoclingDocument
            origin = DocumentOrigin(
                filename=conv_res.input.file.name or "audio.wav",
                mimetype="audio/x-wav",
                binary_hash=conv_res.input.document_hash,
            )
            conv_res.document = DoclingDocument(
                name=conv_res.input.file.stem or "audio.wav", origin=origin
            )

            for citem in conversation:
                conv_res.document.add_text(
                    label=DocItemLabel.TEXT, text=citem.to_string()
                )

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
            if "words" in _ and self.word_timestamps:
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

        if isinstance(self.pipeline_options.asr_options, InlineAsrNativeWhisperOptions):
            asr_options: InlineAsrNativeWhisperOptions = (
                self.pipeline_options.asr_options
            )
            self._model = _NativeWhisperModel(
                enabled=True,  # must be always enabled for this pipeline to make sense.
                artifacts_path=artifacts_path,
                accelerator_options=pipeline_options.accelerator_options,
                asr_options=asr_options,
            )
        else:
            _log.error(f"No model support for {self.pipeline_options.asr_options}")

    def _determine_status(self, conv_res: ConversionResult) -> ConversionStatus:
        status = ConversionStatus.SUCCESS
        return status

    @classmethod
    def get_default_options(cls) -> AsrPipelineOptions:
        return AsrPipelineOptions()

    def _build_document(self, conv_res: ConversionResult) -> ConversionResult:
        _log.info(f"start _build_document in AsrPipeline: {conv_res.input.file}")
        with TimeRecorder(conv_res, "doc_build", scope=ProfilingScope.DOCUMENT):
            self._model.run(conv_res=conv_res)

        return conv_res

    @classmethod
    def is_backend_supported(cls, backend: AbstractDocumentBackend):
        return isinstance(backend, NoOpBackend)
