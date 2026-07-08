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
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Final

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field
from scipy.signal import find_peaks


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


def _probe_duration(video_path: Path) -> float:
    """Return the video duration in seconds using ffprobe, or 0.0 on failure."""
    if shutil.which("ffprobe") is None:
        return 0.0
    try:
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(out.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        return 0.0


def _extract_frame(video_path: Path, timestamp: float) -> Image.Image | None:
    """Extract a single frame at ``timestamp`` as a PIL image via ffmpeg.

    Returns None if ffmpeg produced no output (e.g. timestamp past end).
    """
    proc = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "-",
        ],
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout:
        _log.debug(
            "Frame extraction at %.3fs produced no output (rc=%s): %s",
            timestamp,
            proc.returncode,
            proc.stderr.decode("utf-8", "replace")[-200:],
        )
        return None
    try:
        return Image.open(BytesIO(proc.stdout)).convert("RGB")
    except Exception as exc:  # pragma: no cover - defensive
        _log.debug("Failed to decode extracted frame at %.3fs: %s", timestamp, exc)
        return None


class FixedIntervalFrameSampler:
    """Sample one frame every ``interval_seconds`` from time zero."""

    def __init__(
        self,
        interval_seconds: float = 10.0,
        max_frames: int | None = None,
    ):
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be > 0")
        if max_frames is not None and max_frames <= 0:
            raise ValueError("max_frames must be > 0 when set")
        self.interval_seconds = interval_seconds
        self.max_frames = max_frames

    def sample(self, video_path: Path) -> list[VideoFrame]:
        _require_ffmpeg()
        duration = _probe_duration(video_path)

        frames: list[VideoFrame] = []
        t = 0.0
        # If duration is unknown (0.0), rely on extraction returning None at EOF.
        while duration == 0.0 or t < duration:
            if self.max_frames is not None and len(frames) >= self.max_frames:
                break
            image = _extract_frame(video_path, t)
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

    def _probe_frames(self, video_path: Path) -> list[tuple[float, Image.Image]]:
        """Extract downscaled RGB probe frames at probe_fps."""
        duration = _probe_duration(video_path)
        step = 1.0 / self.probe_fps
        probes: list[tuple[float, Image.Image]] = []
        t = 0.0
        while duration == 0.0 or t < duration:
            img = _extract_frame(video_path, t)
            if img is None:
                break
            small = img.convert("RGB").resize((self.probe_size, self.probe_size))
            probes.append((t, small))
            t += step
            if duration == 0.0 and len(probes) > 100_000:
                break
        return probes

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
        """Pick the sharpest frame in a window centred on the scene midpoint."""
        mid = (start + end) / 2.0
        half = (end - start) / 2.0 * 0.4
        n = self.sharpness_candidates
        candidates = [
            max(start, min(end, mid + half * (i - n // 2) / max(n // 2, 1)))
            for i in range(n)
        ]
        best_frame: VideoFrame | None = None
        best_score = -1.0
        for t in candidates:
            img = _extract_frame(video_path, t)
            if img is None:
                continue
            score = self._sharpness(img)
            if score > best_score:
                best_score = score
                best_frame = VideoFrame(timestamp=t, image=img, scene_id=scene_id)
        return best_frame

    def detect_scenes(self, video_path: Path) -> list[VideoScene]:
        """Detect scene boundaries using local peak detection on frame diffs."""
        probes = self._probe_frames(video_path)
        if len(probes) < 2:
            return []

        timestamps = [p[0] for p in probes]
        diffs = np.array(
            [
                self._mean_abs_diff(probes[i][1], probes[i + 1][1])
                for i in range(len(probes) - 1)
            ]
        )

        w = max(1, self.smooth_window)
        smoothed = np.convolve(diffs, np.ones(w) / w, mode="same")

        min_dist = max(1, int(self.min_scene_duration_seconds * self.probe_fps))
        if self.cuts_per_minute is not None:
            target_interval = max(
                min_dist, int((60.0 / self.cuts_per_minute) * self.probe_fps)
            )
            noise_floor = float(np.percentile(smoothed, 75))
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
            prominence = (
                self.prominence
                if self.prominence is not None
                else float(np.std(smoothed) * 2)
            )
            _log.debug("Prominence mode: prominence=%.4f", prominence)
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
        scenes = self.detect_scenes(video_path)
        frames: list[VideoFrame] = []
        for scene in scenes:
            if self.max_frames is not None and len(frames) >= self.max_frames:
                break
            frame = self._best_frame(
                video_path, scene.start_time, scene.end_time, scene.scene_id
            )
            if frame is not None:
                scene.representative_frame = frame
                frames.append(frame)
        return frames
