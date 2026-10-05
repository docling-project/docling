# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tests for docling.utils.video_frame_sampling.

The scene/frame extraction tests build a tiny synthetic video with ffmpeg when
it is available on PATH; those are skipped otherwise. The validation and
pixel-diff tests run without ffmpeg using synthetic PIL images.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from docling.utils.video_frame_sampling import (
    FfmpegRunner,
    FixedIntervalFrameSampler,
    SimpleSceneChangeFrameSampler,
    VideoFrame,
    VideoScene,
    probe_duration,
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
        {"prominence": -1},
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
        probe_fps=2.0, min_scene_duration_seconds=1.0
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
        probe_fps=2.0, min_scene_duration_seconds=1.0
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
        probe_fps=2.0, min_scene_duration_seconds=100.0
    )
    scenes = sampler.detect_scenes(three_scene_video)
    assert len(scenes) == 1


# --- ffmpeg input handling and time limits (requires ffmpeg) -----------------


def _encode_clip(path: Path, video_codec: str) -> None:
    """Render a 2s test pattern with a sine audio track into ``path``."""
    proc = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=s=96x64:d=2:r=10",
            "-f",
            "lavfi",
            "-i",
            "sine=d=2",
            "-c:v",
            video_codec,
            "-shortest",
            str(path),
        ],
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        pytest.skip(f"ffmpeg cannot encode {video_codec} into {path.suffix}")


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not available")
@pytest.mark.parametrize(
    ("filename", "video_codec"),
    [
        ("clip.mp4", "mpeg4"),
        ("clip.mov", "mpeg4"),
        ("clip.mkv", "mpeg4"),
        ("clip.webm", "libvpx"),
        ("clip.avi", "mpeg4"),
        pytest.param(
            "take:1.mp4",
            "mpeg4",
            marks=pytest.mark.skipif(
                sys.platform == "win32", reason="':' not allowed in file names"
            ),
        ),
    ],
)
def test_every_container_samples_frames(tmp_path: Path, filename: str, video_codec):
    video = tmp_path / filename
    _encode_clip(video, video_codec)

    frames = FixedIntervalFrameSampler(interval_seconds=0.5).sample(video)
    assert len(frames) >= 4
    assert frames[0].image.size == (96, 64)
    assert len(SimpleSceneChangeFrameSampler().sample(video)) >= 1


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not available")
@pytest.mark.parametrize(
    "script",
    [
        "ffconcat version 1.0\nfile real.mkv\n",
        "#EXTM3U\n#EXT-X-TARGETDURATION:2\n#EXTINF:2.0,\nreal.mkv\n#EXT-X-ENDLIST\n",
    ],
    ids=["concat", "hls"],
)
def test_text_script_named_mp4_is_not_followed(tmp_path: Path, script: str):
    """A script saved as .mp4 must not be read as a list of other media files."""
    _encode_clip(tmp_path / "real.mkv", "mpeg4")
    disguised = tmp_path / "clip.mp4"
    disguised.write_text(script)

    runner = FfmpegRunner()
    assert probe_duration(disguised, runner) == 0.0
    assert FixedIntervalFrameSampler(runner=runner).sample(disguised) == []
    assert SimpleSceneChangeFrameSampler(runner=runner).sample(disguised) == []


def test_unknown_container_extension_is_refused(tmp_path: Path):
    with pytest.raises(ValueError, match="Unsupported video container"):
        FixedIntervalFrameSampler().sample(tmp_path / "clip.ts")


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not available")
def test_runner_stops_decode_at_time_limit():
    """An endless decode is killed at the call limit and reported."""
    runner = FfmpegRunner(call_timeout=0.5)
    received = bytearray()
    argv = ["ffmpeg", "-nostdin", "-f", "lavfi", "-i", "testsrc", "-f", "rawvideo", "-"]

    assert runner.run(argv, "Endless decode", received.extend) is False
    assert len(runner.timeouts) == 1
    assert received  # output produced before the limit was kept


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not available")
def test_runner_stops_decode_at_output_limit():
    runner = FfmpegRunner()
    received = bytearray()
    argv = ["ffmpeg", "-nostdin", "-f", "lavfi", "-i", "testsrc", "-f", "rawvideo", "-"]

    assert (
        runner.run(argv, "Endless decode", received.extend, max_output_bytes=10**6)
        is False
    )
    assert len(received) <= 10**6
    assert runner.timeouts == []
