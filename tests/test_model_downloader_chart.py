# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest

from docling.utils import model_downloader


def test_legacy_chart_download_uses_registered_checkpoint(
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
        with_layout=False,
        with_tableformer=False,
        with_code_formula=False,
        with_picture_classifier=False,
        with_rapidocr=False,
        with_granite_chart_extraction=True,
    )

    assert len(downloads) == 1
    assert downloads[0]["repo_id"] == (
        "ibm-granite/granite-vision-3.3-2b-chart2csv-preview"
    )
    assert downloads[0]["local_dir"] == (
        tmp_path / "ibm-granite--granite-vision-3.3-2b-chart2csv-preview"
    )
