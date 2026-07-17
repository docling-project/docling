"""Unit tests for the torch-free ``NumpyImageProcessor`` preprocessing fallback."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from docling.models.inference_engines.common.hf_vision_base import NumpyImageProcessor

# Default rescale factor used by transformers image processors (1/255).
_DEFAULT_RESCALE = 1.0 / 255.0


def _uniform_rgb(width: int, height: int, color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (width, height), color)


def test_call_returns_channel_first_batch_and_resizes() -> None:
    """A single image is resized and returned as a (N, C, H, W) float/uint batch."""
    proc = NumpyImageProcessor({"do_resize": True, "size": {"height": 32, "width": 48}})
    # PIL size is (width, height); start from a non-target shape.
    out = proc(_uniform_rgb(20, 10, (10, 20, 30)))

    pixel_values = out["pixel_values"]
    assert pixel_values.shape == (1, 3, 32, 48)
    assert pixel_values.dtype == np.uint8
    # Uniform input stays uniform through resize; channels preserved per-plane.
    assert np.array_equal(np.unique(pixel_values[0, 0]), np.array([10]))
    assert np.array_equal(np.unique(pixel_values[0, 1]), np.array([20]))
    assert np.array_equal(np.unique(pixel_values[0, 2]), np.array([30]))


def test_call_without_resize_preserves_dimensions() -> None:
    proc = NumpyImageProcessor({"do_resize": False})
    out = proc(_uniform_rgb(20, 10, (1, 2, 3)))
    # (width=20, height=10) -> (N, C, H, W) = (1, 3, 10, 20)
    assert out["pixel_values"].shape == (1, 3, 10, 20)


def test_do_resize_without_size_is_noop() -> None:
    """do_resize is honored only when a resolvable target size is present."""
    proc = NumpyImageProcessor({"do_resize": True})
    out = proc(_uniform_rgb(16, 8, (0, 0, 0)))
    assert out["pixel_values"].shape == (1, 3, 8, 16)


def test_list_and_tuple_inputs_are_stacked_into_batch() -> None:
    proc = NumpyImageProcessor({"do_resize": True, "size": {"height": 4, "width": 4}})
    images = [_uniform_rgb(6, 6, (5, 5, 5)) for _ in range(3)]
    assert proc(images)["pixel_values"].shape == (3, 3, 4, 4)
    assert proc(tuple(images))["pixel_values"].shape == (3, 3, 4, 4)


@pytest.mark.parametrize("mode", ["L", "RGBA", "P", "CMYK"])
def test_non_rgb_inputs_are_converted_to_three_channels(mode: str) -> None:
    proc = NumpyImageProcessor({"do_resize": False})
    out = proc(Image.new(mode, (4, 3)))
    assert out["pixel_values"].shape == (1, 3, 3, 4)


def test_rescale_produces_float32_and_scales_values() -> None:
    proc = NumpyImageProcessor({"do_rescale": True, "rescale_factor": _DEFAULT_RESCALE})
    out = proc(_uniform_rgb(4, 4, (255, 255, 255)))
    pixel_values = out["pixel_values"]
    assert pixel_values.dtype == np.float32
    np.testing.assert_allclose(pixel_values, 1.0, rtol=0, atol=1e-6)


def test_normalize_applies_mean_and_std_per_channel() -> None:
    proc = NumpyImageProcessor(
        {
            "do_rescale": True,
            "rescale_factor": _DEFAULT_RESCALE,
            "do_normalize": True,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
        }
    )
    # 255 -> rescale 1.0 -> normalize (1.0 - 0.5) / 0.5 = 1.0
    white = proc(_uniform_rgb(2, 2, (255, 255, 255)))["pixel_values"]
    np.testing.assert_allclose(white, 1.0, rtol=0, atol=1e-6)
    # 0 -> rescale 0.0 -> normalize (0.0 - 0.5) / 0.5 = -1.0
    black = proc(_uniform_rgb(2, 2, (0, 0, 0)))["pixel_values"]
    np.testing.assert_allclose(black, -1.0, rtol=0, atol=1e-6)


def test_normalize_uses_distinct_per_channel_statistics() -> None:
    proc = NumpyImageProcessor(
        {
            "do_rescale": False,
            "do_normalize": True,
            "image_mean": [10.0, 20.0, 30.0],
            "image_std": [2.0, 4.0, 5.0],
        }
    )
    out = proc(_uniform_rgb(2, 2, (10, 40, 5)))["pixel_values"]
    # channel 0: (10-10)/2 = 0 ; channel 1: (40-20)/4 = 5 ; channel 2: (5-30)/5 = -5
    np.testing.assert_allclose(out[0, 0], 0.0, atol=1e-6)
    np.testing.assert_allclose(out[0, 1], 5.0, atol=1e-6)
    np.testing.assert_allclose(out[0, 2], -5.0, atol=1e-6)


def test_normalize_skipped_when_mean_or_std_missing() -> None:
    """Normalization requires both mean and std; otherwise it is skipped."""
    proc = NumpyImageProcessor(
        {"do_rescale": True, "do_normalize": True, "image_mean": [0.5, 0.5, 0.5]}
    )
    out = proc(_uniform_rgb(2, 2, (255, 255, 255)))["pixel_values"]
    # Only rescale applied (1.0), no normalization since image_std is absent.
    np.testing.assert_allclose(out, 1.0, atol=1e-6)


def test_no_rescale_or_normalize_keeps_uint8_original_values() -> None:
    proc = NumpyImageProcessor({"do_resize": False})
    out = proc(_uniform_rgb(3, 3, (7, 8, 9)))["pixel_values"]
    assert out.dtype == np.uint8
    assert out[0, 0].tolist() == [[7, 7, 7], [7, 7, 7], [7, 7, 7]]
    assert out[0, 1].tolist() == [[8, 8, 8], [8, 8, 8], [8, 8, 8]]
    assert out[0, 2].tolist() == [[9, 9, 9], [9, 9, 9], [9, 9, 9]]


def test_pixel_values_are_contiguous() -> None:
    proc = NumpyImageProcessor({"do_resize": False})
    out = proc(_uniform_rgb(4, 5, (1, 1, 1)))["pixel_values"]
    assert out.flags["C_CONTIGUOUS"]


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        ({"height": 32, "width": 48}, (32, 48)),
        ({"shortest_edge": 64}, (64, 64)),
        ({"longest_edge": 100}, (100, 100)),
        ({"shortest_edge": 64, "longest_edge": 100}, (64, 64)),
        ({}, None),
        (None, None),
    ],
)
def test_target_hw_resolution(
    size: dict[str, int] | None, expected: tuple[int, int] | None
) -> None:
    proc = NumpyImageProcessor({"size": size} if size is not None else {})
    assert proc._target_hw() == expected


def test_defaults_when_config_is_empty() -> None:
    proc = NumpyImageProcessor({})
    assert proc.do_resize is False
    assert proc.do_rescale is False
    assert proc.do_normalize is False
    assert proc.rescale_factor == pytest.approx(_DEFAULT_RESCALE)
    assert proc.resample is Image.Resampling.BILINEAR


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (0, Image.Resampling.NEAREST),
        (1, Image.Resampling.LANCZOS),
        (2, Image.Resampling.BILINEAR),
        (3, Image.Resampling.BICUBIC),
        (4, Image.Resampling.BOX),
        (5, Image.Resampling.HAMMING),
        (99, Image.Resampling.BILINEAR),  # unknown code falls back to bilinear
    ],
)
def test_resample_code_mapping(code: int, expected: Image.Resampling) -> None:
    assert NumpyImageProcessor({"resample": code}).resample is expected


def test_from_config_file_reads_json(tmp_path: Path) -> None:
    config: dict[str, Any] = {
        "do_resize": True,
        "size": {"height": 640, "width": 640},
        "do_rescale": True,
        "rescale_factor": _DEFAULT_RESCALE,
        "resample": 3,
    }
    config_path = tmp_path / "preprocessor_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    proc = NumpyImageProcessor.from_config_file(config_path)
    assert proc.do_resize is True
    assert proc._target_hw() == (640, 640)
    assert proc.do_rescale is True
    assert proc.resample is Image.Resampling.BICUBIC
