# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest

from docling.utils import model_downloader


@pytest.mark.parametrize(
    ("download_option", "repo_id"),
    [
        ("with_granite_chart_extraction", "ibm-granite/granite-vision-4.1-4b"),
        (
            "with_granite_chart_extraction_v3_3",
            "ibm-granite/granite-vision-3.3-2b-chart2csv-preview",
        ),
    ],
)
def test_chart_download_uses_requested_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    download_option: str,
    repo_id: str,
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
        **{download_option: True},
    )

    assert len(downloads) == 1
    assert downloads[0]["repo_id"] == repo_id
    assert downloads[0]["local_dir"] == tmp_path / repo_id.replace("/", "--")
