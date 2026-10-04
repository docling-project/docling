# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Video frame sampling utilities.

This module is intentionally free of any docling imports. It provides two
frame samplers over a video file, using ffmpeg as the only hard runtime
dependency (ffmpeg is already required by the ASR path).

- ``FixedIntervalFrameSampler`` extracts one frame every N seconds.
- ``SimpleSceneChangeFrameSampler`` probes low-resolution frames and emits a
  representative frame per detected scene using a mean-absolute-difference
  heuristic.

Both return ``VideoFrame`` objects carrying the frame image and its timestamp.
"""

import logging
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field


class VideoFrameSamplingMode(str, Enum):
    """Frame sampling strategy for the video pipeline."""

    FIXED_INTERVAL = "fixed_interval"
    SCENE_CHANGE = "scene_change"


_log = logging.getLogger(__name__)

MISSING_FFMPEG_MESSAGE: Final[str] = (
    "FFmpeg is required for video processing but was not found on PATH. "
    "Install it with your system package manager (e.g., 'brew install ffmpeg' "
    "on macOS, 'apt-get install ffmpeg' on Linux, 'winget install ffmpeg' on "
    "Windows)."
)

# Container demuxer forced for each accepted file extension. Without ``-f``,
# ffmpeg/ffprobe pick the demuxer from the file content, so a text playlist or
# concat script saved as ``.mp4`` would be followed to other files. Forcing the
# demuxer that matches the extension makes such inputs fail to open instead.
# Only the extensions accepted for ``InputFormat.VIDEO`` are mapped; any other
# suffix is refused rather than falling back to content detection.
FFMPEG_DEMUXER_BY_SUFFIX: Final[dict[str, str]] = {
    ".mp4": "mov",
    ".mov": "mov",
    ".mkv": "matroska",
    ".webm": "matroska",
    ".avi": "avi",
}

# Default wall-clock limit for one ffmpeg/ffprobe call when no document budget
# applies. Full-decode calls get at least the media duration (see
# ``FfmpegRunner.scale_to_duration``).
FFMPEG_CALL_TIMEOUT_SECONDS: Final[float] = 300.0

# Upper bound on the encoded bytes accepted for one full-resolution PNG frame.
_MAX_PNG_FRAME_BYTES: Final[int] = 128 * 1024 * 1024

# Only the tail of ffmpeg's diagnostics is kept for logging.
_STDERR_TAIL_BYTES: Final[int] = 4096

_READ_CHUNK_BYTES: Final[int] = 1024 * 1024


def ffmpeg_demuxer_for(video_path: Path) -> str | None:
    """Return the ffmpeg demuxer forced for ``video_path``, or None if unsupported."""
    return FFMPEG_DEMUXER_BY_SUFFIX.get(video_path.suffix.lower())


def unsupported_container_message(video_path: Path) -> str:
    """Describe why ``video_path`` cannot be read as a video container."""
    return (
        f"Unsupported video container extension {video_path.suffix!r}; "
        f"expected one of {sorted(FFMPEG_DEMUXER_BY_SUFFIX)}."
    )


def ffmpeg_input_args(video_path: Path) -> list[str]:
    """Build the input options shared by every ffmpeg/ffprobe call.

    Only the ``file`` protocol is allowed, the demuxer is forced from the file
    extension, and the path is passed as a ``file:`` URL so that a ``:`` in the
    name is never read as a protocol prefix.

    Raises:
        ValueError: If the file extension has no known container demuxer.
    """
    demuxer = ffmpeg_demuxer_for(video_path)
    if demuxer is None:
        raise ValueError(unsupported_container_message(video_path))
    return [
        "-protocol_whitelist",
        "file",
        "-f",
        demuxer,
        "-i",
        f"file:{video_path.resolve()}",
    ]


class FfmpegRunner:
    """Run ffmpeg/ffprobe calls for one document under a shared time budget.

    Each call is limited to ``call_timeout`` seconds and, when a document
    deadline is set, to the time left until that deadline. Calls that run out
    of time are killed and recorded in ``timeouts`` so the caller can report
    them; the sampling helpers then return what they collected so far.
    Captured output is bounded: stdout is streamed to a caller-provided sink
    with a byte cap, and only the tail of stderr is kept for logging.
    """

    def __init__(
        self,
        document_timeout: float | None = None,
        call_timeout: float = FFMPEG_CALL_TIMEOUT_SECONDS,
    ):
        self._deadline = (
            None if document_timeout is None else time.monotonic() + document_timeout
        )
        self.call_timeout = call_timeout
        self.timeouts: list[str] = []

    def scale_to_duration(self, duration: float) -> None:
        """Allow each call at least real-time decoding of ``duration`` seconds."""
        self.call_timeout = max(self.call_timeout, duration)

    def _call_budget(self) -> float:
        if self._deadline is None:
            return self.call_timeout
        return min(self.call_timeout, self._deadline - time.monotonic())

    def run(
        self,
        argv: list[str],
        what: str,
        sink: Callable[[bytes], None] | None = None,
        max_output_bytes: int | None = None,
        chunk_size: int = _READ_CHUNK_BYTES,
    ) -> bool:
        """Run ``argv``, streaming stdout chunks to ``sink``.

        Args:
            argv: Command line to execute.
            what: Short description used in log and timeout messages.
            sink: Receives stdout in chunks; stdout is discarded when None.
            max_output_bytes: Stop and fail once stdout exceeds this size.
            chunk_size: Read size for stdout chunks.

        Returns:
            True if the process exited with status 0 within its time budget
            and output limit.
        """
        budget = self._call_budget()
        if budget <= 0:
            self._record_timeout(what, 0.0)
            return False

        timed_out = threading.Event()
        with tempfile.TemporaryFile() as stderr_file:
            try:
                proc = subprocess.Popen(
                    argv,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE if sink is not None else subprocess.DEVNULL,
                    stderr=stderr_file,
                )
            except OSError as exc:
                _log.warning("%s could not start %s: %s", what, argv[0], exc)
                return False

            def _kill() -> None:
                timed_out.set()
                proc.kill()

            watchdog = threading.Timer(budget, _kill)
            watchdog.start()
            over_limit = False
            try:
                if sink is not None and proc.stdout is not None:
                    total = 0
                    while chunk := proc.stdout.read(chunk_size):
                        total += len(chunk)
                        if max_output_bytes is not None and total > max_output_bytes:
                            over_limit = True
                            proc.kill()
                            break
                        sink(chunk)
                proc.wait()
            finally:
                watchdog.cancel()
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
                if proc.stdout is not None:
                    proc.stdout.close()

            if timed_out.is_set():
                self._record_timeout(what, budget)
                return False
            if over_limit:
                _log.warning(
                    "%s stopped: output exceeded %d bytes", what, max_output_bytes
                )
                return False
            if proc.returncode != 0:
                size = stderr_file.seek(0, 2)
                stderr_file.seek(max(0, size - _STDERR_TAIL_BYTES))
                _log.debug(
                    "%s failed (rc=%s): %s",
                    what,
                    proc.returncode,
                    stderr_file.read().decode("utf-8", "replace"),
                )
                return False
        return True

    def _record_timeout(self, what: str, budget: float) -> None:
        if budget <= 0:
            message = f"{what} skipped: document time budget exhausted"
        else:
            message = f"{what} stopped after its time limit of {budget:.3g}s"
        _log.warning(message)
        self.timeouts.append(message)


def ffmpeg_base_args() -> list[str]:
    """Leading ffmpeg options: no stdin, and diagnostics limited to errors."""
    return ["ffmpeg", "-nostdin", "-hide_banner", "-nostats", "-v", "error"]


class VideoFrame(BaseModel):
    """A single sampled video frame with its timestamp."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    timestamp: float = Field(..., ge=0, description="Seconds from video start.")
    image: Image.Image = Field(..., description="The decoded frame image.")
    scene_id: int | None = Field(
        None, description="Scene index if produced by a scene sampler."
    )


class VideoScene(BaseModel):
    """A contiguous time window treated as one scene."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    scene_id: int
    start_time: float = Field(..., ge=0)
    end_time: float = Field(..., ge=0)
    representative_frame: VideoFrame | None = None


def _require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(MISSING_FFMPEG_MESSAGE)


# Auto-prominence calibration. The frame-diff signal is mostly ambient motion
# (near-zero for static screen shares, elevated for podcasts/vlogs where people
# move constantly) with sparse spikes at genuine scene cuts.
_AUTO_PROMINENCE_FLOOR: Final[float] = 0.012
"""Minimum auto threshold. Keeps static footage sensitive to subtle cuts while
staying above codec noise (~0.005-0.01). Below this, tiny diffs are ignored."""

_AUTO_PROMINENCE_K: Final[float] = 5.0
"""Robust sigmas above ambient motion a peak must clear to count as a cut.
Higher = stricter (fewer scenes on busy video); lower = more sensitive."""


def _auto_prominence(diffs: NDArray[np.float64]) -> float:
    """Adapt the scene-cut threshold to how busy the video is.

    Uses the median frame difference as the ambient-motion floor and the
    (robust) median absolute deviation as its spread, so the threshold rises
    automatically for high-motion footage — ignoring hand-waving and body
    movement — and drops toward the floor for static screens, catching subtle
    cuts. Robust statistics are used so the cut spikes themselves do not inflate
    the estimate (unlike a plain standard deviation).

    Args:
        diffs: Frame-to-frame difference signal.

    Returns:
        The calibrated prominence threshold for peak detection.
    """
    median = float(np.median(diffs))
    mad = float(np.median(np.abs(diffs - median))) * 1.4826  # ~= std for normal noise
    return max(_AUTO_PROMINENCE_FLOOR, median + _AUTO_PROMINENCE_K * mad)


def probe_duration(video_path: Path, runner: FfmpegRunner) -> float:
    """Return the video duration in seconds using ffprobe, or 0.0 if unknown."""
    if shutil.which("ffprobe") is None:
        return 0.0
    out = bytearray()
    argv = [
        "ffprobe",
        "-v",
        "error",
        *ffmpeg_input_args(video_path),
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
    ]
    if not runner.run(argv, "ffprobe duration", out.extend, max_output_bytes=4096):
        return 0.0
    try:
        return float(out.decode("ascii", "replace").strip())
    except ValueError:
        return 0.0


def _decode_png(data: bytes, what: str) -> Image.Image | None:
    try:
        return Image.open(BytesIO(data)).convert("RGB")
    except Exception as exc:  # pragma: no cover - defensive
        _log.debug("Failed to decode %s: %s", what, exc)
        return None


def _extract_frame(
    video_path: Path, timestamp: float, runner: FfmpegRunner
) -> Image.Image | None:
    """Extract a single frame at ``timestamp`` as a PIL image via ffmpeg.

    Returns None if ffmpeg produced no output (e.g. timestamp past end).
    """
    what = f"Frame extraction at {timestamp:.3f}s"
    out = bytearray()
    argv = [
        *ffmpeg_base_args(),
        "-ss",
        f"{timestamp:.3f}",
        *ffmpeg_input_args(video_path),
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "-",
    ]
    if not runner.run(argv, what, out.extend, _MAX_PNG_FRAME_BYTES) or not out:
        return None
    return _decode_png(bytes(out), what)


def _extract_frames_range(
    video_path: Path, start: float, duration: float, fps: float, runner: FfmpegRunner
) -> list[tuple[float, Image.Image]]:
    """Decode ``[start, start + duration]`` once at ``fps``, full resolution.

    Single ffmpeg spawn per call, seeking to ``start`` before decoding
    (fast input seek) rather than spawning one process per timestamp.
    """
    max_frames = int(duration * fps) + 1
    out = bytearray()
    argv = [
        *ffmpeg_base_args(),
        "-ss",
        f"{start:.3f}",
        *ffmpeg_input_args(video_path),
        "-t",
        f"{duration:.3f}",
        "-vf",
        f"fps={fps}",
        "-frames:v",
        str(max_frames),
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "-",
    ]
    what = f"Range frame probe at {start:.3f}s"
    if not runner.run(argv, what, out.extend, max_frames * _MAX_PNG_FRAME_BYTES):
        return []

    frames: list[tuple[float, Image.Image]] = []
    buf = bytes(out)
    # PNGs concatenated in the image2pipe stream; split on the PNG signature.
    sig = b"\x89PNG\r\n\x1a\n"
    offsets: list[int] = []
    pos = buf.find(sig)
    while pos != -1:
        offsets.append(pos)
        pos = buf.find(sig, pos + 1)
    for idx, off in enumerate(offsets):
        end = offsets[idx + 1] if idx + 1 < len(offsets) else len(buf)
        img = _decode_png(buf[off:end], f"frame {idx} of {what}")
        if img is not None:
            frames.append((start + idx / fps, img))
    return frames


def _iter_grid_frames(
    video_path: Path,
    fps: float,
    size: int,
    runner: FfmpegRunner,
    on_frame: Callable[[Image.Image], None],
) -> None:
    """Decode the whole video once at ``fps`` and pass each thumbnail to ``on_frame``.

    Uses a single ffmpeg pass emitting downscaled raw RGB frames, which is far
    cheaper than spawning one ffmpeg process per timestamp (each spawn re-opens
    and re-seeks the file). Frames are square ``size`` x ``size`` thumbnails;
    the timestamp of frame ``i`` is ``i / fps``. Frames are streamed, so memory
    use does not grow with the video length.
    """
    frame_bytes = size * size * 3
    pending = bytearray()

    def _sink(chunk: bytes) -> None:
        pending.extend(chunk)
        while len(pending) >= frame_bytes:
            on_frame(Image.frombytes("RGB", (size, size), bytes(pending[:frame_bytes])))
            del pending[:frame_bytes]

    argv = [
        *ffmpeg_base_args(),
        *ffmpeg_input_args(video_path),
        "-vf",
        f"fps={fps},scale={size}:{size}",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    runner.run(argv, "Scene-change probe decode", _sink, chunk_size=frame_bytes)


class FixedIntervalFrameSampler:
    """Sample one frame every ``interval_seconds`` from time zero."""

    def __init__(
        self,
        interval_seconds: float = 10.0,
        max_frames: int | None = None,
        runner: FfmpegRunner | None = None,
    ):
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be > 0")
        if max_frames is not None and max_frames <= 0:
            raise ValueError("max_frames must be > 0 when set")
        self.interval_seconds = interval_seconds
        self.max_frames = max_frames
        self.runner = runner if runner is not None else FfmpegRunner()

    def sample(self, video_path: Path) -> list[VideoFrame]:
        _require_ffmpeg()
        duration = probe_duration(video_path, self.runner)

        frames: list[VideoFrame] = []
        t = 0.0
        # If duration is unknown (0.0), rely on extraction returning None at EOF.
        while duration == 0.0 or t < duration:
            if self.max_frames is not None and len(frames) >= self.max_frames:
                _log.info("Stopped frame sampling at max_frames=%d", self.max_frames)
                break
            image = _extract_frame(video_path, t, self.runner)
            if image is None:
                break
            frames.append(VideoFrame(timestamp=t, image=image))
            t += self.interval_seconds
        return frames


class SimpleSceneChangeFrameSampler:
    """Detect scenes via local peak detection on the frame-difference signal.

    No global threshold required. The sampler:
    1. Probes the video at ``probe_fps`` (small RGB thumbnails).
    2. Computes mean-absolute pixel difference between consecutive frames.
    3. Smooths the resulting 1-D signal with a moving average.
    4. Detects scene boundaries as local peaks using scipy.signal.find_peaks
       with a prominence criterion — self-calibrating per video, no manual
       threshold needed.
    5. Selects the sharpest frame in a window around each scene midpoint
       as the representative keyframe, avoiding motion-blurred frames.
    """

    def __init__(
        self,
        probe_fps: float = 1.0,
        prominence: float | None = None,
        cuts_per_minute: float | None = None,
        min_scene_duration_seconds: float = 2.0,
        max_frames: int | None = None,
        probe_size: int = 64,
        smooth_window: int = 1,
        sharpness_candidates: int = 5,
        runner: FfmpegRunner | None = None,
    ):
        if probe_fps <= 0:
            raise ValueError("probe_fps must be > 0")
        if prominence is not None and prominence < 0:
            raise ValueError("prominence must be >= 0")
        if min_scene_duration_seconds < 0:
            raise ValueError("min_scene_duration_seconds must be >= 0")
        if max_frames is not None and max_frames <= 0:
            raise ValueError("max_frames must be > 0 when set")
        self.probe_fps = probe_fps
        self.prominence = prominence
        self.cuts_per_minute = cuts_per_minute
        self.min_scene_duration_seconds = min_scene_duration_seconds
        self.max_frames = max_frames
        self.probe_size = probe_size
        self.smooth_window = smooth_window
        self.sharpness_candidates = sharpness_candidates
        self.runner = runner if runner is not None else FfmpegRunner()

    def _probe_diffs(self, video_path: Path) -> tuple[list[float], NDArray[np.float64]]:
        """Decode probe thumbnails at ``probe_fps`` and diff consecutive frames.

        Returns:
            The probe timestamps and the difference between each pair of
            consecutive probe frames (one fewer entry than timestamps).
        """
        diffs: list[float] = []
        previous: Image.Image | None = None
        count = 0

        def _on_frame(image: Image.Image) -> None:
            nonlocal previous, count
            if previous is not None:
                diffs.append(self._mean_abs_diff(previous, image))
            previous = image
            count += 1

        _iter_grid_frames(
            video_path, self.probe_fps, self.probe_size, self.runner, _on_frame
        )
        timestamps = [i / self.probe_fps for i in range(count)]
        return timestamps, np.array(diffs, dtype=np.float64)

    @staticmethod
    def _mean_abs_diff(a: Image.Image, b: Image.Image) -> float:
        """Normalized mean absolute difference of two images in [0, 1]."""
        arr_a = np.asarray(a, dtype=np.int16)
        arr_b = np.asarray(b, dtype=np.int16)
        if arr_a.shape != arr_b.shape or arr_a.size == 0:
            return 0.0
        return float(np.abs(arr_a - arr_b).mean()) / 255.0

    @staticmethod
    def _sharpness(image: Image.Image) -> float:
        """Laplacian variance — higher = sharper, used to avoid blurry keyframes."""
        gray = np.asarray(image.convert("L"), dtype=np.float32)
        lap = (
            gray[:-2, 1:-1]
            + gray[2:, 1:-1]
            + gray[1:-1, :-2]
            + gray[1:-1, 2:]
            - 4 * gray[1:-1, 1:-1]
        )
        return float(np.var(lap))

    def _best_frame(
        self, video_path: Path, start: float, end: float, scene_id: int
    ) -> VideoFrame | None:
        """Pick the sharpest frame in a window centred on the scene midpoint.

        Decodes the whole candidate window in a single ffmpeg spawn (full
        resolution, sampled at ``sharpness_candidates`` evenly spaced points)
        instead of spawning one ffmpeg process per candidate timestamp.

        Args:
            video_path: Path to the source video.
            start: Scene start time, in seconds.
            end: Scene end time, in seconds.
            scene_id: Index of the scene this frame represents.

        Returns:
            The sharpest candidate frame, or None if none could be decoded.
        """
        mid = (start + end) / 2.0
        half = (end - start) / 2.0 * 0.4
        window_start = max(start, mid - half)
        window_end = min(end, mid + half)
        window_duration = max(window_end - window_start, 0.0)
        n = self.sharpness_candidates

        if window_duration == 0.0 or n <= 1:
            img = _extract_frame(video_path, mid, self.runner)
            return (
                VideoFrame(timestamp=mid, image=img, scene_id=scene_id) if img else None
            )

        # fps chosen so the range decode yields ~n evenly spaced frames.
        fps = (n - 1) / window_duration
        candidates = _extract_frames_range(
            video_path, window_start, window_duration, fps, self.runner
        )

        best_frame: VideoFrame | None = None
        best_score = -1.0
        for t, img in candidates:
            score = self._sharpness(img)
            if score > best_score:
                best_score = score
                best_frame = VideoFrame(timestamp=t, image=img, scene_id=scene_id)
        return best_frame

    def detect_scenes(self, video_path: Path) -> list[VideoScene]:
        """Detect scene boundaries using local peak detection on frame diffs."""
        timestamps, diffs = self._probe_diffs(video_path)
        if len(timestamps) < 2:
            return []

        w = max(1, self.smooth_window)
        smoothed = np.convolve(diffs, np.ones(w) / w, mode="same")

        min_dist = max(1, int(self.min_scene_duration_seconds * self.probe_fps))
        if self.cuts_per_minute is not None:
            target_interval = max(
                min_dist, int((60.0 / self.cuts_per_minute) * self.probe_fps)
            )
            noise_floor = float(np.percentile(smoothed, 75))
            from scipy.signal import (
                find_peaks,  # guarded: not available in slim installations
            )

            peaks, _ = find_peaks(
                smoothed, distance=target_interval, prominence=noise_floor
            )
            _log.debug(
                "Cuts/min mode: interval=%d frames, noise_floor=%.4f, peaks=%d",
                target_interval,
                noise_floor,
                len(peaks),
            )
        else:
            if self.prominence is not None:
                prominence = self.prominence
            else:
                prominence = _auto_prominence(diffs)
            _log.debug("Prominence mode: prominence=%.4f", prominence)
            from scipy.signal import (
                find_peaks,  # guarded: not available in slim installations
            )

            peaks, _ = find_peaks(smoothed, prominence=prominence, distance=min_dist)

        # Filter peaks too close to video start
        valid_peaks = [
            p for p in peaks if timestamps[p] >= self.min_scene_duration_seconds
        ]
        boundaries = [timestamps[0]] + [timestamps[p] for p in valid_peaks]
        end_time = timestamps[-1]

        scenes: list[VideoScene] = []
        for idx, start in enumerate(boundaries):
            stop = boundaries[idx + 1] if idx + 1 < len(boundaries) else end_time
            scenes.append(VideoScene(scene_id=idx, start_time=start, end_time=stop))
        return scenes

    def sample(self, video_path: Path) -> list[VideoFrame]:
        """Sample one sharp representative frame per detected scene."""
        _require_ffmpeg()
        scenes = self.detect_scenes(video_path)
        frames: list[VideoFrame] = []
        for scene in scenes:
            if self.max_frames is not None and len(frames) >= self.max_frames:
                _log.info("Stopped frame sampling at max_frames=%d", self.max_frames)
                break
            frame = self._best_frame(
                video_path, scene.start_time, scene.end_time, scene.scene_id
            )
            if frame is not None:
                scene.representative_frame = frame
                frames.append(frame)
        return frames
