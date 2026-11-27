# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.pipeline_options import GraniteVisionTableStructureOptions
from docling.models.stages.table_structure.table_structure_model_granite_vision import (
    GraniteVisionTableStructureModel,
)

pytestmark = pytest.mark.ml_vlm


def test_missing_model_folder_in_artifacts_path_triggers_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """An artifacts_path without the model folder must load the downloaded model."""
    downloaded = tmp_path / "downloaded-model"
    loaded: list[Path] = []

    def fake_download_models(*args: object, **kwargs: object) -> Path:
        return downloaded

    def fake_load_model(self: GraniteVisionTableStructureModel, path: Path) -> None:
        loaded.append(path)

    monkeypatch.setattr(
        GraniteVisionTableStructureModel, "download_models", fake_download_models
    )
    monkeypatch.setattr(
        GraniteVisionTableStructureModel, "_load_model", fake_load_model
    )

    artifacts_path = tmp_path / "artifacts"
    artifacts_path.mkdir()

    GraniteVisionTableStructureModel(
        enabled=True,
        artifacts_path=artifacts_path,
        options=GraniteVisionTableStructureOptions(),
        accelerator_options=AcceleratorOptions(device="cpu"),
    )

    assert loaded == [downloaded]
