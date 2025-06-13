import logging
import os
import re
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Union, cast

import soundfile as sf
from docling_core.types.doc.labels import DocItemLabel
from pydantic import BaseModel, Field, validator

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

import librosa

_log = logging.getLogger(__name__)


class ConversationEntry(BaseModel):
    text: str
    start_time: float = Field(
        ..., ge=0, description="Start time in seconds from video start"
    )
    end_time: float = Field(
        ..., ge=0, description="End time in seconds from video start"
    )
    speaker_id: int = Field(..., ge=0, description="Numeric speaker identifier")
    speaker: Optional[str] = Field(
        None, description="Speaker name, defaults to speaker-{speaker_id}"
    )

    @validator("end_time")
    def end_time_must_be_after_start(cls, v, values):
        if "start_time" in values and v <= values["start_time"]:
            raise ValueError("end_time must be greater than start_time")
        return v

    @validator("speaker", always=True)
    def set_default_speaker_name(cls, v, values):
        if v is None and "speaker_id" in values:
            return f"speaker-{values['speaker_id']}"
        return v

    def __lt__(self, other):
        if not isinstance(other, ConversationEntry):
            return NotImplemented
        return self.start_time < other.start_time

    def __eq__(self, other):
        if not isinstance(other, ConversationEntry):
            return NotImplemented
        return self.start_time == other.start_time

    def to_string(self) -> str:
        """Format the conversation entry as a string"""
        return f"[time: {self.start_time}-{self.end_time}] [speaker:{self.speaker}] {self.text}"


class _WhisperModel:
    def __init__(self):
        _log.info("initialisation `_WhisperModel`")

        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.model_repo = "openai/whisper-tiny"

        self.processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        self.model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-tiny"
        )

        # FIXME
        self.max_new_tokens = 256
        
        _log.info(f"model is loaded: {self.model_repo}")

    def run(self, conv_res: ConversionResult):
        # fpath = Path(conv_res.input.file)
        # _log.info(f"`_WhisperModel::run: {conv_res}`")
        _log.info(f"`_WhisperModel::run: {conv_res.input}`")
        _log.info(f"`_WhisperModel::run: {conv_res.input.file}`")

        if os.path.exists(str(conv_res.input.file)):
            print("file exists")
        else:
            print("file does not exist")
        #

        _log.info(f"sampling-rate: {self.processor.feature_extractor.sampling_rate}")

        try:
            fpath = conv_res.input.file
            # array, sampling_rate = sf.read(fpath)#, samplerate=processor.feature_extractor.sampling_rate)
            array, sampling_rate = sf.read(
                fpath
            )  # , samplerate=self.processor.feature_extractor.sampling_rate)

            _log.info(
                f"read the file .. (sampling-rate: {sampling_rate}, array: {array.shape})"
            )

            array, sampling_rate = librosa.load(fpath, sr=16000)

            _log.info(
                f"read the file .. (sampling-rate: {sampling_rate}, array: {array.shape})"
            )

            
            processed_input = self.processor(
                array,
                sampling_rate=self.processor.feature_extractor.sampling_rate,  # sampling_rate,
                return_tensors="pt",
            )
            print(processed_input)

            # pre-process to get the input features
            input_features = self.processor(
                array, sampling_rate=sampling_rate, return_tensors="pt"
            ).input_features

            _log.info(f"got input-features: {input_features.shape}")
            _log.info(f"max new tokens: {self.max_new_tokens}")

            # generate token ids by running model forward sequentially
            predicted_ids = self.model.generate(
                input_features, max_new_tokens=self.max_new_tokens, return_timestamps=True
            )

            _log.info("ran model ..")

            """
            transcription = self.processor.batch_decode(predicted_ids,
                                                        skip_special_tokens=False,
                                                        decode_with_timestamps=True)

            _log.info("decoded output ..")
            
            print(f"Transcription: {transcription}")
            """

            conversation = []

            print("Timestamp info:")
            for pidi, pid in enumerate(predicted_ids):
                # timestamps = processor.tokenizer.decode(pid, decode_with_timestamps=True)
                timestamps = self.processor.tokenizer.decode(pid, output_offsets=True)
                print(f"Predicted id [{pidi}]: {timestamps['text']}")
                for offset in timestamps["offsets"]:
                    print(f" => {offset['timestamp']}: {offset['text']}")

                    item = ConversationEntry(
                        text=offset["text"],
                        speaker_id=pidi,
                        start_time=offset["timestamp"][0],
                        end_time=offset["timestamp"][1],
                    )
                    conv_res.document.add_text(
                        label=DocItemLabel.TEXT, text=item.to_string()
                    )

            conv_res.status = ConversionStatus.SUCCESS

            print("document: \n\n", conv_res.document.export_to_markdown())

        except Exception as exc:
            conv_res.status = ConversionStatus.FAILED
            print(exc)

        return conv_res


class AsrPipeline(BasePipeline):
    def __init__(self, pipeline_options: AsrPipelineOptions):
        super().__init__(pipeline_options)
        self.keep_backend = True

        self.pipeline_options: AsrPipelineOptions

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

        self._model = _WhisperModel()

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
