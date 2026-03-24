from pathlib import Path

import pytest

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.models.stages.chart_extraction.granite_vision import (
    ChartExtractionModelGraniteVision,
    ChartExtractionModelOptions,
)


class _DummyTokenizer:
    model_max_length = 512


class _DummyProcessor:
    tokenizer = _DummyTokenizer()


class _DummyModel:
    def eval(self):
        pass


def _make_model(monkeypatch, artifacts_path):
    """Instantiate ChartExtractionModelGraniteVision with mocked from_pretrained."""
    transformers = pytest.importorskip("transformers")

    called_with_paths = []

    def fake_processor_from_pretrained(*args, **kwargs):
        called_with_paths.append(args[0])
        return _DummyProcessor()

    def fake_model_from_pretrained(*args, **kwargs):
        called_with_paths.append(args[0])
        return _DummyModel()

    monkeypatch.setattr(
        transformers.AutoProcessor,
        "from_pretrained",
        fake_processor_from_pretrained,
    )
    monkeypatch.setattr(
        transformers.AutoModelForImageTextToText,
        "from_pretrained",
        fake_model_from_pretrained,
    )

    model = ChartExtractionModelGraniteVision(
        enabled=True,
        artifacts_path=artifacts_path,
        options=ChartExtractionModelOptions(),
        accelerator_options=AcceleratorOptions(device="cpu"),
    )
    return model, called_with_paths


def test_artifacts_path_with_existing_subfolder(monkeypatch, tmp_path):
    """When artifacts_path contains the model subfolder, use it directly."""
    subfolder = tmp_path / ChartExtractionModelGraniteVision._model_repo_folder
    subfolder.mkdir()

    model, called_with_paths = _make_model(monkeypatch, tmp_path)

    # Both from_pretrained calls should receive the subfolder path
    assert all(p == subfolder for p in called_with_paths)


def test_artifacts_path_without_subfolder_downloads(monkeypatch, tmp_path):
    """When artifacts_path is provided but subfolder missing, download the model."""
    download_path = tmp_path / "downloaded"
    download_path.mkdir()

    monkeypatch.setattr(
        ChartExtractionModelGraniteVision,
        "download_models",
        staticmethod(lambda **kwargs: download_path),
    )

    model, called_with_paths = _make_model(monkeypatch, tmp_path)

    # from_pretrained should receive the download path, NOT the raw tmp_path
    assert all(p == download_path for p in called_with_paths)


def test_artifacts_path_none_downloads(monkeypatch, tmp_path):
    """When artifacts_path is None, download the model."""
    download_path = tmp_path / "downloaded"
    download_path.mkdir()

    monkeypatch.setattr(
        ChartExtractionModelGraniteVision,
        "download_models",
        staticmethod(lambda **kwargs: download_path),
    )

    model, called_with_paths = _make_model(monkeypatch, None)

    # from_pretrained should receive the download path
    assert all(p == download_path for p in called_with_paths)
