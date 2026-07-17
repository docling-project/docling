"""Video conversion pipeline.

Orchestrates three steps:
1. Extract audio from the video via ffmpeg and transcribe it using the
   shared ASR transcriber.
2. Sample representative frames using either fixed-interval or
   scene-change sampling.
3. Merge transcript segments and frames by timestamp into a single
   DoclingDocument.
"""

import logging
import shutil
import subprocess
import tempfile
from io import BytesIO
from pathlib import Path

from docling_core.types.doc import (
    ContentLayer,
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
    ImageRef,
    TrackSource,
)

from docling.backend.abstract_backend import AbstractDocumentBackend
from docling.backend.noop_backend import NoOpBackend
from docling.datamodel.base_models import (
    ConversionStatus,
    DoclingComponentType,
    ErrorItem,
)
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import VideoPipelineOptions
from docling.pipeline.asr_transcriber import (
    _AsrModelFactory,
    _merge_into_sentences,
)
from docling.pipeline.base_pipeline import BasePipeline
from docling.utils.profiling import ProfilingScope, TimeRecorder
from docling.utils.speaker_diarization import (
    assign_speakers,
    diarize,
)
from docling.utils.video_frame_sampling import (
    FixedIntervalFrameSampler,
    SimpleSceneChangeFrameSampler,
    VideoFrame,
    VideoFrameSamplingMode,
)

_log = logging.getLogger(__name__)

MISSING_FFMPEG_MESSAGE = (
    "FFmpeg is required for video processing but was not found on PATH. "
    "Install it with your system package manager (e.g., 'brew install ffmpeg' "
    "on macOS, 'apt-get install ffmpeg' on Linux, 'winget install ffmpeg' on "
    "Windows)."
)

_VIDEO_SUFFIX_TO_MIMETYPE = {
    ".mp4": "video/mp4",
    ".avi": "video/x-msvideo",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
}


def _video_mimetype(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    return _VIDEO_SUFFIX_TO_MIMETYPE.get(suffix, "video/mp4")


def _extract_audio(video_path: Path, wav_path: Path) -> bool:
    """Extract audio track from video to a 16kHz mono WAV. Returns True on success."""
    result = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-y",
            str(wav_path),
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        _log.debug(
            "Audio extraction failed (rc=%s): %s",
            result.returncode,
            result.stderr.decode("utf-8", "replace")[-300:],
        )
        return False
    return True


class VideoPipeline(BasePipeline):
    """Pipeline that transcribes a video's audio and samples its frames,
    producing a DoclingDocument with transcript text and frame images
    interleaved by timestamp.
    """

    def __init__(self, pipeline_options: VideoPipelineOptions):
        super().__init__(pipeline_options)
        self.keep_backend = True
        self.pipeline_options: VideoPipelineOptions = pipeline_options
        self._asr_model = _AsrModelFactory.create(
            asr_options=self.pipeline_options.asr_options,
            artifacts_path=self.artifacts_path,
            accelerator_options=pipeline_options.accelerator_options,
        )

    @classmethod
    def get_default_options(cls) -> VideoPipelineOptions:
        return VideoPipelineOptions()

    @classmethod
    def is_backend_supported(cls, backend: AbstractDocumentBackend) -> bool:
        return isinstance(backend, NoOpBackend)

    def _determine_status(self, conv_res: ConversionResult) -> ConversionStatus:
        if conv_res.status == ConversionStatus.FAILURE or conv_res.errors:
            return ConversionStatus.FAILURE
        has_text = conv_res.document and any(
            t.text and t.text.strip() for t in (conv_res.document.texts or [])
        )
        has_pictures = conv_res.document and conv_res.document.pictures
        if not has_text and not has_pictures:
            _log.warning(
                "Video conversion produced an empty document. File: %s",
                conv_res.input.file.name,
            )
            return ConversionStatus.PARTIAL_SUCCESS
        return ConversionStatus.SUCCESS

    def _build_document(self, conv_res: ConversionResult) -> ConversionResult:
        _log.info("Processing video document %s.", conv_res.input.file.name)
        with TimeRecorder(conv_res, "doc_build", scope=ProfilingScope.DOCUMENT):
            self._process_video(conv_res)
        return conv_res

    def _process_video(self, conv_res: ConversionResult) -> None:
        # 1. Resolve input to a local path
        path_or_stream = conv_res.input._backend.path_or_stream
        temp_video: Path | None = None

        if isinstance(path_or_stream, BytesIO):
            suffix = Path(conv_res.input.file.name).suffix or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                f.write(path_or_stream.getvalue())
                temp_video = Path(f.name)
            video_path = temp_video
        elif isinstance(path_or_stream, Path):
            video_path = path_or_stream
        else:
            conv_res.errors.append(
                ErrorItem(
                    component_type=DoclingComponentType.PIPELINE,
                    module_name="VideoPipeline",
                    error_message=f"Unsupported input type: {type(path_or_stream)}",
                )
            )
            conv_res.status = ConversionStatus.FAILURE
            return

        try:
            # 2. Validate ffmpeg
            if shutil.which("ffmpeg") is None:
                conv_res.errors.append(
                    ErrorItem(
                        component_type=DoclingComponentType.PIPELINE,
                        module_name="VideoPipeline",
                        error_message=MISSING_FFMPEG_MESSAGE,
                    )
                )
                conv_res.status = ConversionStatus.FAILURE
                return

            # 3. Extract audio and transcribe
            transcript_items = []
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
                wav_path = Path(wf.name)

            try:
                audio_ok = _extract_audio(video_path, wav_path)
                if audio_ok and wav_path.exists() and wav_path.stat().st_size > 0:
                    transcript_items = self._asr_model.transcribe(wav_path)
                    transcript_items = _merge_into_sentences(transcript_items)
                    # Run diarization while WAV is still available
                    if self.pipeline_options.enable_diarization:
                        try:
                            diarization = diarize(
                                wav_path,
                                accelerator_device=self.pipeline_options.accelerator_options.device,
                            )
                            transcript_items = assign_speakers(
                                transcript_items, diarization
                            )
                            _log.info(
                                "Diarization: %d speakers detected",
                                diarization.num_speakers,
                            )
                        except Exception as exc:
                            _log.warning("Speaker diarization failed: %s", exc)
                else:
                    _log.warning(
                        "Audio extraction produced no output for %s; "
                        "document will contain frames only.",
                        video_path.name,
                    )
            finally:
                if wav_path.exists():
                    wav_path.unlink()

            # 4. Sample frames
            opts = self.pipeline_options
            frames: list[VideoFrame] = []
            if opts.generate_frame_images:
                if opts.frame_sampling_mode == VideoFrameSamplingMode.SCENE_CHANGE:
                    sampler = SimpleSceneChangeFrameSampler(
                        prominence=opts.scene_change_prominence,
                        cuts_per_minute=opts.cuts_per_minute,
                        probe_fps=opts.scene_change_probe_fps,
                        min_scene_duration_seconds=opts.min_scene_duration_seconds,
                        max_frames=opts.max_sampled_frames,
                        smooth_window=opts.scene_change_smooth_window,
                    )
                else:
                    sampler = FixedIntervalFrameSampler(
                        interval_seconds=opts.frame_interval_seconds,
                        max_frames=opts.max_sampled_frames,
                    )
                try:
                    frames = sampler.sample(video_path)
                except Exception as exc:
                    _log.warning("Frame sampling failed: %s", exc)

            # 5. Build DoclingDocument
            filename = conv_res.input.file.name or "video.mp4"
            origin = DocumentOrigin(
                filename=filename,
                mimetype=_video_mimetype(filename),
                binary_hash=conv_res.input.document_hash,
            )
            conv_res.document = DoclingDocument(
                name=conv_res.input.file.stem or "video", origin=origin
            )

            # 6. Merge transcript + frames by timestamp
            events: list[tuple[float, int, object]] = []
            for item in transcript_items:
                if (
                    item.start_time is not None
                    and item.end_time is not None
                    and item.text.strip()
                ):
                    events.append((item.start_time, 1, item))
            for frame in frames:
                events.append((frame.timestamp, 0, frame))

            events.sort(key=lambda e: (e[0], e[1]))

            for _, event_type, payload in events:
                if event_type == 0:
                    frame = payload  # type: ignore[assignment]
                    try:
                        picture = conv_res.document.add_picture(
                            image=ImageRef.from_pil(frame.image, dpi=72),
                            content_layer=ContentLayer.BODY,
                        )
                        picture.source = TrackSource(
                            start_time=frame.timestamp,
                            end_time=frame.timestamp + 0.001,
                            identifier=(
                                f"scene:{frame.scene_id}"
                                if frame.scene_id is not None
                                else None
                            ),
                        )
                    except Exception as exc:
                        _log.warning(
                            "Failed to add frame at %.3fs: %s",
                            frame.timestamp,
                            exc,
                        )
                else:
                    item = payload  # type: ignore[assignment]
                    try:
                        conv_res.document.add_text(
                            label=DocItemLabel.TEXT,
                            text=item.text,
                            content_layer=ContentLayer.BODY,
                            source=TrackSource(
                                start_time=item.start_time,
                                end_time=item.end_time,
                                voice=item.speaker,
                            ),
                        )
                    except Exception as exc:
                        _log.warning(
                            "Failed to add transcript at %.3fs: %s",
                            item.start_time,
                            exc,
                        )

        finally:
            if temp_video is not None and temp_video.exists():
                try:
                    temp_video.unlink()
                except Exception:
                    pass
