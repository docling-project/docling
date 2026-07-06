"""Tests for docling.utils.video_frame_sampling.

The scene/frame extraction tests build a tiny synthetic video with ffmpeg when
it is available on PATH; those are skipped otherwise. The validation and
pixel-diff tests run without ffmpeg using synthetic PIL images.
"""

import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from docling.utils.video_frame_sampling import (
    FixedIntervalFrameSampler,
    SimpleSceneChangeFrameSampler,
    VideoFrame,
    VideoScene,
)

_HAS_FFMPEG = shutil.which("ffmpeg") is not None


def _make_three_scene_video(path: Path) -> None:
    """Render a 12s video: 4s red, 4s green, 4s blue (three hard cuts)."""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=160x120:d=4",
            "-f",
            "lavfi",
            "-i",
            "color=c=green:s=160x120:d=4",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=160x120:d=4",
            "-filter_complex",
            "[0:v][1:v][2:v]concat=n=3:v=1:a=0",
            str(path),
        ],
        capture_output=True,
        check=True,
    )


@pytest.fixture
def three_scene_video(tmp_path: Path) -> Path:
    if not _HAS_FFMPEG:
        pytest.skip("ffmpeg not available")
    out = tmp_path / "scenes.mp4"
    _make_three_scene_video(out)
    return out


# --------------------------------------------------------------------------- #
# Model tests (no ffmpeg)
# --------------------------------------------------------------------------- #


def test_video_frame_model_holds_image():
    img = Image.new("RGB", (4, 4))
    f = VideoFrame(timestamp=1.5, image=img, scene_id=2)
    assert f.timestamp == 1.5
    assert f.scene_id == 2
    assert f.image.size == (4, 4)


def test_video_scene_model():
    s = VideoScene(scene_id=0, start_time=0.0, end_time=4.0)
    assert s.end_time == 4.0
    assert s.representative_frame is None


# --------------------------------------------------------------------------- #
# Validation guards (no ffmpeg)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "kwargs",
    [
        {"interval_seconds": 0},
        {"interval_seconds": -1},
        {"interval_seconds": 5, "max_frames": 0},
    ],
)
def test_fixed_interval_rejects_bad_args(kwargs):
    with pytest.raises(ValueError):
        FixedIntervalFrameSampler(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"threshold": -1},
        {"probe_fps": 0},
        {"min_scene_duration_seconds": -1},
        {"max_frames": 0},
    ],
)
def test_scene_sampler_rejects_bad_args(kwargs):
    with pytest.raises(ValueError):
        SimpleSceneChangeFrameSampler(**kwargs)


# --------------------------------------------------------------------------- #
# Pixel-diff heuristic (no ffmpeg)
# --------------------------------------------------------------------------- #


def test_mean_abs_diff_identical_is_zero():
    a = Image.new("RGB", (8, 8), (100, 100, 100))
    b = Image.new("RGB", (8, 8), (100, 100, 100))
    assert SimpleSceneChangeFrameSampler._mean_abs_diff(a, b) == 0.0


def test_mean_abs_diff_black_vs_white_is_one():
    a = Image.new("RGB", (8, 8), (0, 0, 0))
    b = Image.new("RGB", (8, 8), (255, 255, 255))
    assert SimpleSceneChangeFrameSampler._mean_abs_diff(a, b) == pytest.approx(1.0)


def test_mean_abs_diff_red_vs_green_is_significant():
    a = Image.new("RGB", (8, 8), (255, 0, 0))
    b = Image.new("RGB", (8, 8), (0, 255, 0))
    # red->green differs on two channels: mean over RGB = (255+255+0)/3/255
    assert SimpleSceneChangeFrameSampler._mean_abs_diff(a, b) == pytest.approx(
        2 / 3, abs=1e-6
    )


# --------------------------------------------------------------------------- #
# End-to-end sampling (requires ffmpeg)
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not available")
def test_fixed_interval_timestamps(three_scene_video: Path):
    frames = FixedIntervalFrameSampler(interval_seconds=3.0).sample(three_scene_video)
    ts = [round(f.timestamp, 1) for f in frames]
    assert ts == [0.0, 3.0, 6.0, 9.0]
    for f in frames:
        assert f.image.mode == "RGB"
        assert f.image.size == (160, 120)


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not available")
def test_fixed_interval_respects_max_frames(three_scene_video: Path):
    frames = FixedIntervalFrameSampler(interval_seconds=1.0, max_frames=2).sample(
        three_scene_video
    )
    assert len(frames) == 2


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not available")
def test_scene_change_detects_three_scenes(three_scene_video: Path):
    sampler = SimpleSceneChangeFrameSampler(
        threshold=0.2, probe_fps=2.0, min_scene_duration_seconds=1.0
    )
    scenes = sampler.detect_scenes(three_scene_video)
    assert len(scenes) == 3
    # boundaries near 0, 4, 8
    starts = [round(s.start_time) for s in scenes]
    assert starts == [0, 4, 8]


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not available")
def test_scene_change_representative_frames_are_correct_colors(
    three_scene_video: Path,
):
    sampler = SimpleSceneChangeFrameSampler(
        threshold=0.2, probe_fps=2.0, min_scene_duration_seconds=1.0
    )
    frames = sampler.sample(three_scene_video)
    assert len(frames) == 3
    colors = []
    for f in frames:
        cx, cy = f.image.size[0] // 2, f.image.size[1] // 2
        r, g, b = f.image.getpixel((cx, cy))[:3]
        if r > 100:
            colors.append("red")
        elif g > 100:
            colors.append("green")
        elif b > 100:
            colors.append("blue")
        else:
            colors.append("?")
    assert colors == ["red", "green", "blue"]


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not available")
def test_scene_change_respects_min_duration(three_scene_video: Path):
    # A very large min duration collapses everything into one scene.
    sampler = SimpleSceneChangeFrameSampler(
        threshold=0.2, probe_fps=2.0, min_scene_duration_seconds=100.0
    )
    scenes = sampler.detect_scenes(three_scene_video)
    assert len(scenes) == 1
