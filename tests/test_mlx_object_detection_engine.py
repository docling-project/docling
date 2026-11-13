# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Unit tests for the MLX object-detection engine.

These tests do not load a model. They exercise the platform guard, checkpoint
metadata handling, the sanitize bypass, and output mapping with a stub
predictor, so they run on every platform.
"""

import json
import platform
import struct
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.object_detection_engine_options import (
    MlxObjectDetectionEngineOptions,
)
from docling.datamodel.stage_model_specs import EngineModelConfig
from docling.models.inference_engines.object_detection.base import (
    ObjectDetectionEngineInput,
)
from docling.models.inference_engines.object_detection.mlx_engine import (
    MlxObjectDetectionEngine,
    _bypass_mlx_sanitizer,
    _checkpoint_metadata,
    _is_apple_silicon,
    _safetensors_metadata,
)


def _engine() -> MlxObjectDetectionEngine:
    return MlxObjectDetectionEngine(
        options=MlxObjectDetectionEngineOptions(),
        model_config=EngineModelConfig(repo_id="example/model", revision="main"),
        accelerator_options=AcceleratorOptions(),
    )


class _FakePredictor:
    def __init__(self, results=None) -> None:
        self.images = []
        self._results = results

    def predict_batch(self, images):
        self.images = list(images)
        if self._results is not None:
            return self._results
        return [
            SimpleNamespace(
                labels=np.array([1, 2]),
                scores=np.array([0.9, 0.8]),
                boxes=np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]),
            )
            for _ in self.images
        ]


@pytest.mark.parametrize(
    ("platform_name", "machine", "expected"),
    [
        ("darwin", "arm64", True),
        ("darwin", "x86_64", False),
        ("linux", "arm64", False),
    ],
)
def test_is_apple_silicon(monkeypatch, platform_name, machine, expected):
    monkeypatch.setattr(sys, "platform", platform_name)
    monkeypatch.setattr(platform, "machine", lambda: machine)

    assert _is_apple_silicon() is expected


def test_initialize_rejects_non_apple_silicon(monkeypatch):
    engine = _engine()
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")

    with pytest.raises(RuntimeError, match="Apple Silicon"):
        engine.initialize()


def test_initialize_reports_missing_mlx_vlm(monkeypatch):
    engine = _engine()
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    monkeypatch.setitem(sys.modules, "mlx_vlm", None)
    for name in [name for name in sys.modules if name.startswith("mlx_vlm.")]:
        monkeypatch.delitem(sys.modules, name)

    with pytest.raises(ImportError, match="mlx-vlm"):
        engine.initialize()


def test_predict_batch_maps_results_and_copies_metadata():
    engine = _engine()
    predictor = _FakePredictor()
    engine._predictor = predictor
    engine._initialized = True
    inputs = [
        ObjectDetectionEngineInput(
            image=Image.new("RGB", (10, 10)), metadata={"page_no": 1}
        ),
        ObjectDetectionEngineInput(
            image=Image.new("L", (10, 10)), metadata={"page_no": 2}
        ),
    ]

    outputs = engine.predict_batch(inputs)

    assert [output.label_ids for output in outputs] == [[1, 2], [1, 2]]
    assert outputs[0].scores == pytest.approx([0.9, 0.8], abs=1e-6)
    assert outputs[0].bboxes == [
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
    ]
    assert outputs[0].metadata == {"page_no": 1}
    assert outputs[0].metadata is not inputs[0].metadata
    assert predictor.images[1].mode == "RGB"


def test_predict_batch_returns_empty_for_empty_input():
    engine = _engine()

    assert engine.predict_batch([]) == []


def test_predict_batch_requires_initialization():
    engine = _engine()

    with pytest.raises(RuntimeError, match="Call initialize"):
        engine.predict_batch(
            [ObjectDetectionEngineInput(image=Image.new("RGB", (10, 10)))]
        )


def test_predict_batch_rejects_result_count_mismatch():
    engine = _engine()
    engine._predictor = _FakePredictor(results=[])
    engine._initialized = True

    with pytest.raises(RuntimeError, match="0 results for 1 inputs"):
        engine.predict_batch(
            [ObjectDetectionEngineInput(image=Image.new("RGB", (10, 10)))]
        )


def _write_safetensors(path, metadata):
    header = json.dumps({"__metadata__": metadata}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header)


def test_safetensors_metadata_reads_format(tmp_path):
    path = tmp_path / "model.safetensors"
    _write_safetensors(path, {"format": "mlx"})

    assert _safetensors_metadata(path) == {"format": "mlx"}


def test_safetensors_metadata_handles_bad_files(tmp_path):
    assert _safetensors_metadata(tmp_path / "missing.safetensors") == {}

    short = tmp_path / "short.safetensors"
    short.write_bytes(b"\x00")
    assert _safetensors_metadata(short) == {}

    malformed = tmp_path / "malformed.safetensors"
    malformed.write_bytes(struct.pack("<Q", 4) + b"nope")
    assert _safetensors_metadata(malformed) == {}


def test_checkpoint_metadata_reads_single_file(tmp_path):
    _write_safetensors(tmp_path / "model.safetensors", {"format": "mlx"})

    assert _checkpoint_metadata(tmp_path) == {"format": "mlx"}


def test_checkpoint_metadata_reads_shards(tmp_path):
    shard = tmp_path / "model-00001-of-00001.safetensors"
    _write_safetensors(shard, {"format": "mlx"})
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer": shard.name}})
    )

    assert _checkpoint_metadata(tmp_path) == {"format": "mlx"}


def test_checkpoint_metadata_rejects_missing_shard(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer": "missing.safetensors"}})
    )

    with pytest.raises(FileNotFoundError, match="missing shard"):
        _checkpoint_metadata(tmp_path)


def test_checkpoint_metadata_requires_a_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError, match="No safetensors checkpoint"):
        _checkpoint_metadata(tmp_path)


class _StaticModel:
    @staticmethod
    def sanitize(weights):
        return {"sanitized": True}


def test_bypass_mlx_sanitizer_restores_staticmethod():
    with _bypass_mlx_sanitizer(_StaticModel):
        assert _StaticModel.sanitize({"w": 1}) == {"w": 1}

    assert _StaticModel.sanitize({"w": 1}) == {"sanitized": True}


class _BaseModel:
    @staticmethod
    def sanitize(weights):
        return {"base": True}


class _ChildModel(_BaseModel):
    pass


def test_bypass_mlx_sanitizer_restores_inherited_method():
    with _bypass_mlx_sanitizer(_ChildModel):
        assert _ChildModel.sanitize({"w": 1}) == {"w": 1}

    assert _ChildModel.sanitize({"w": 1}) == {"base": True}
    assert "sanitize" not in _ChildModel.__dict__


class _NoSanitizeModel:
    pass


def test_bypass_mlx_sanitizer_requires_a_sanitize_method():
    with pytest.raises(RuntimeError, match="no sanitize method"):
        with _bypass_mlx_sanitizer(_NoSanitizeModel):
            pass
