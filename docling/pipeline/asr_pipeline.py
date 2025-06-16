import logging
import os
import re
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Union, cast

import librosa  # type: ignore
import numpy as np
import soundfile as sf  # type: ignore
from docling_core.types.doc.labels import DocItemLabel
from pydantic import BaseModel, Field, validator
from pydub import AudioSegment  # type: ignore
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline

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


class _WhisperASR:
    def __init__(self, model_name: str = "openai/whisper-small"):
        """
        Transcriber using Hugging Face Transformers Whisper + energy-based VAD.
        """
        print(f"Loading Whisper model: {model_name}")

        self.device = "cpu"

        self.transcriber = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            return_timestamps=True,
            device=self.device,
        )

    def _energy_vad(
        self,
        y: np.ndarray,
        sr: int,
        frame_length=2048,
        hop_length=512,
        threshold_percentile=85,
    ):
        """
        Simple energy-based VAD.
        Returns list of (start_time, end_time) tuples for speech segments.
        """
        _log.debug(f"_energy_vad {sr}: ", y.shape)
        energy = np.array(
            [
                np.sum(np.abs(y[i : i + frame_length] ** 2))
                for i in range(0, len(y), hop_length)
            ]
        )
        _log.debug(f"energy: {energy}")

        threshold = np.percentile(energy, threshold_percentile) * 0.3
        _log.debug(f"threshold: {threshold}")

        speech_frames = energy > threshold
        _log.debug(f"speech_frames: {speech_frames}")

        frame_times = librosa.frames_to_time(
            np.arange(len(energy)), sr=sr, hop_length=hop_length
        )

        segments = []
        start_time = None

        for i, is_speech in enumerate(speech_frames):
            t = frame_times[i]
            if is_speech and start_time is None:
                start_time = t
            elif not is_speech and start_time is not None:
                segments.append((start_time, t))
                start_time = None

        if start_time is not None:
            segments.append((start_time, frame_times[-1]))

        return segments

    def _merge_vad_segments(self, segments, min_duration=5.0, max_gap=0.5):
        """
        Merge short/adjacent speech segments to improve transcription quality.
        """
        if not segments:
            return []

        merged = []
        current_start, current_end = segments[0]

        for start, end in segments[1:]:
            gap = start - current_end
            if gap <= max_gap or (current_end - current_start) < min_duration:
                current_end = end  # merge
            else:
                if current_end - current_start >= 1.0:  # skip ultra-short
                    merged.append((current_start, current_end))
                current_start, current_end = start, end

        if current_end - current_start >= 1.0:
            merged.append((current_start, current_end))

        return merged

    def run(self, conv_res: ConversionResult) -> ConversionResult:
        """
        Transcribe audio using custom VAD and Whisper, returning timestamped segments.
        Returns list of {"start", "end", "text"} dictionaries.
        """
        audio_path = conv_res.input.file

        _log.info(f"Loading audio and resampling: {audio_path}")
        y, sr = librosa.load(audio_path, sr=16000)

        speech_segments = self._energy_vad(y=y, sr=int(sr))
        speech_segments = self._merge_vad_segments(speech_segments)
        _log.info("#-speech: ", len(speech_segments))

        _log.info("Preparing AudioSegment for chunk slicing...")
        pcm = (y * 32767).astype(np.int16).tobytes()
        audio_seg = AudioSegment(data=pcm, sample_width=2, frame_rate=16000, channels=1)

        result = self._create_conversation_entries_v2(speech_segments, audio_seg)
        result.sort()

        for _ in result:
            conv_res.document.add_text(label=DocItemLabel.TEXT, text=_.to_string())

        conv_res.status = ConversionStatus.SUCCESS
        return conv_res

    def _create_conversation_entries_v1(
        self, speech_segments, audio_seg
    ) -> list[_ConversationItem]:
        """
        Chunk audio based on speech_segments, transcribe with Whisper,
        and return structured _ConversationItem items.
        """
        results = []
        chunk_id = 0

        for start, end in speech_segments:
            duration = end - start
            while duration > 0:
                sub_end = min(start + 30.0, end)
                chunk = audio_seg[start * 1000 : sub_end * 1000]
                samples = (
                    np.array(chunk.get_array_of_samples()).astype(np.float32) / 32768.0
                )

                try:
                    _log.debug(
                        f"Transcribing chunk {chunk_id}: {start:.2f}s - {sub_end:.2f}s [{sub_end - start:.2f}]"
                    )
                    result = self.transcriber(samples, return_timestamps=True)

                    # Adjust timestamps globally
                    for seg in result["chunks"]:
                        t0, t1 = seg["timestamp"]
                        if t0 is None or t1 is None or t1 <= t0:
                            _log.warning(f"skipping bad segment: {seg}")
                            continue

                        item = _ConversationItem(
                            text=seg["text"].strip(),
                            start_time=start + t0,
                            end_time=start + t1,
                        )
                        results.append(item)

                    start = sub_end
                    duration = end - start
                    chunk_id += 1
                except Exception as exc:
                    _log.error(f"Exception: {exc}")

        return results

    def _create_conversation_entries_v2(
        self, speech_segments, audio_seg
    ) -> list[_ConversationItem]:
        """
        Chunk audio based on speech_segments, transcribe with Whisper,
        and return structured _ConversationItem items.
        """
        results = []
        chunk_id = 0

        if len(speech_segments) == 0:
            return []

        any_valid = False
        last_valid_offset: float = speech_segments[0][0]

        for start, end in speech_segments:
            if any_valid:
                last_valid_offset = min(start, last_valid_offset)
            else:
                last_valid_offset = start

            duration = end - last_valid_offset

            if duration > 0.2:
                sub_end = min(last_valid_offset + 30.0, end)

                chunk_i0 = int(last_valid_offset * 1000)
                chunk_i1 = int(sub_end * 1000)

                chunk = audio_seg[chunk_i0:chunk_i1]
                samples = (
                    np.array(chunk.get_array_of_samples()).astype(np.float32) / 32768.0
                )
                chunk_id += 1

                try:
                    result = self.transcriber(samples, return_timestamps=True)

                    any_valid = False

                    last_valid_offset_ = last_valid_offset

                    for seg in result["chunks"]:
                        t0, t1 = seg["timestamp"]
                        if t0 is None or t1 is None or t1 <= t0:
                            _log.warning(f" => skipping bad segment: {seg}")
                            continue

                        global_start = round(last_valid_offset_ + t0, 2)
                        global_end = round(last_valid_offset_ + t1, 2)
                        text = seg["text"].strip()

                        results.append(
                            _ConversationItem(
                                start_time=global_start, end_time=global_end, text=text
                            )
                        )
                        last_valid_offset = max(global_end, last_valid_offset)
                        any_valid = True

                    if not any_valid:
                        _log.warning(
                            "No valid transcription in chunk, nudging forward 1s."
                        )
                        last_valid_offset += 1.0

                except Exception as e:
                    _log.error(f"Whisper failed: {e}")
                    last_valid_offset += 1.0

                duration = end - last_valid_offset
            else:
                any_valid = False

        return results


class _WhisperModel:
    def __init__(self):
        _log.info("initialisation `_WhisperModel`")

        self.device = "cpu"
        self.chunk_length = 30

        self.batch_size = 8

        # self.model_repo = "openai/whisper-tiny"
        # self.model_repo = "openai/whisper-small"
        self.model_repo = "openai/whisper-medium"
        # self.model_repo = "openai/whisper-large"

        self.processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        self.model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-tiny"
        )

        # FIXME
        self.max_new_tokens = 256

        _log.info(f"model is loaded: {self.model_repo}")

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model_repo,
            chunk_length_s=self.chunk_length,
            device=self.device,
        )

    def run(self, conv_res: ConversionResult) -> ConversionResult:
        return self._run_pipeline(conv_res=conv_res)

    def _run_pipeline(self, conv_res: ConversionResult) -> ConversionResult:
        try:
            fpath = conv_res.input.file

            array, sampling_rate = librosa.load(fpath, sr=16000)

            prediction = self.pipe(
                inputs=array, batch_size=self.batch_size, return_timestamps=True
            )  # ["chunks"]

            for _ in prediction["chunks"]:
                item = _ConversationItem(
                    text=_["text"],
                    start_time=_["timestamp"][0],
                    end_time=_["timestamp"][1],
                )
                conv_res.document.add_text(
                    label=DocItemLabel.TEXT, text=item.to_string()
                )

            conv_res.status = ConversionStatus.SUCCESS
        except Exception as exc:
            conv_res.status = ConversionStatus.FAILURE
            _log.error(f"Failed to convert with {self.model_repo}: {exc}")

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

        # self._model = _WhisperModel()
        self._model = _WhisperASR()

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
