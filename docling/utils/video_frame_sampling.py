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
from io import BytesIO
from pathlib import Path
from typing import Final

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

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
    """Detect scenes via mean-absolute pixel difference of probe frames.

    Probes the video at ``probe_fps`` (downscaled RGB), starts a new scene when
    the normalized frame difference exceeds ``threshold`` and enough time has
    elapsed since the last boundary, and emits one representative frame per
    scene (taken at the scene midpoint).
    """

    def __init__(
        self,
        threshold: float = 0.35,
        probe_fps: float = 1.0,
        min_scene_duration_seconds: float = 2.0,
        max_frames: int | None = None,
        probe_size: int = 64,
    ):
        if threshold < 0:
            raise ValueError("threshold must be >= 0")
        if probe_fps <= 0:
            raise ValueError("probe_fps must be > 0")
        if min_scene_duration_seconds < 0:
            raise ValueError("min_scene_duration_seconds must be >= 0")
        if max_frames is not None and max_frames <= 0:
            raise ValueError("max_frames must be > 0 when set")
        self.threshold = threshold
        self.probe_fps = probe_fps
        self.min_scene_duration_seconds = min_scene_duration_seconds
        self.max_frames = max_frames
        self.probe_size = probe_size

    def _probe_frames(self, video_path: Path) -> list[tuple[float, Image.Image]]:
        """Extract downscaled RGB probe frames at probe_fps."""
        duration = _probe_duration(video_path)
        step = 1.0 / self.probe_fps
        probes: list[tuple[float, Image.Image]] = []
        t = 0.0
        while duration == 0.0 or t < duration:
            image = _extract_frame(video_path, t)
            if image is None:
                break
            small = image.convert("RGB").resize((self.probe_size, self.probe_size))
            probes.append((t, small))
            t += step
            # Safety bound if duration is unknown.
            if duration == 0.0 and len(probes) > 100000:
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

    def detect_scenes(self, video_path: Path) -> list[VideoScene]:
        _require_ffmpeg()
        probes = self._probe_frames(video_path)
        if not probes:
            return []

        boundaries: list[float] = [probes[0][0]]
        last_boundary = probes[0][0]
        prev = probes[0][1]
        for ts, frame in probes[1:]:
            diff = self._mean_abs_diff(prev, frame)
            if (
                diff >= self.threshold
                and (ts - last_boundary) >= self.min_scene_duration_seconds
            ):
                boundaries.append(ts)
                last_boundary = ts
            prev = frame

        end_time = probes[-1][0]
        scenes: list[VideoScene] = []
        for idx, start in enumerate(boundaries):
            stop = boundaries[idx + 1] if idx + 1 < len(boundaries) else end_time
            scenes.append(VideoScene(scene_id=idx, start_time=start, end_time=stop))
        return scenes

    def sample(self, video_path: Path) -> list[VideoFrame]:
        scenes = self.detect_scenes(video_path)
        frames: list[VideoFrame] = []
        for scene in scenes:
            if self.max_frames is not None and len(frames) >= self.max_frames:
                break
            midpoint = (scene.start_time + scene.end_time) / 2.0
            image = _extract_frame(video_path, midpoint)
            if image is None:
                image = _extract_frame(video_path, scene.start_time)
            if image is None:
                continue
            frame = VideoFrame(timestamp=midpoint, image=image, scene_id=scene.scene_id)
            scene.representative_frame = frame
            frames.append(frame)
        return frames
