# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest

from docling.utils import model_downloader

MLX_REPO = "mlx-community/docling-layout-heron-101-mlx-bf16"


def test_layout_mlx_download_uses_the_mlx_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    downloads: list[dict] = []
    monkeypatch.setattr(
        model_downloader,
        "download_hf_model",
        lambda **kwargs: downloads.append(kwargs),
    )

    model_downloader.download_models(
        output_dir=tmp_path,
        progress=True,
        with_layout=False,
        with_layout_mlx=True,
        with_tableformer=False,
        with_code_formula=False,
        with_picture_classifier=False,
        with_rapidocr=False,
    )

    assert len(downloads) == 1
    assert downloads[0]["repo_id"] == MLX_REPO
    assert downloads[0]["revision"] == "main"
    assert downloads[0]["local_dir"] == tmp_path / MLX_REPO.replace("/", "--")
    assert downloads[0]["progress"] is True
    assert downloads[0]["force"] is False
