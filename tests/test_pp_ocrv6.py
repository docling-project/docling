# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import cast

import numpy as np
import pytest
from docling_core.types.doc import BoundingBox, CoordOrigin
from PIL import Image

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import OcrMode, PpOcrv6Options
from docling.exceptions import OcrLanguageNotSupportedError
from docling.models.plugins.defaults import ocr_engines
from docling.models.stages.ocr import pp_ocrv6
from docling.models.stages.ocr.pp_ocrv6 import PpOcrv6Model


def test_pp_ocrv6_factory_and_language_contract() -> None:
    assert PpOcrv6Model in ocr_engines()["ocr_engines"]
    assert PpOcrv6Options().lang == ["auto"]
    model = PpOcrv6Model(
        enabled=False,
        artifacts_path=None,
        options=PpOcrv6Options(lang=["iso:de"]),
        accelerator_options=AcceleratorOptions(),
    )
    assert model.resolve_ocr_languages() == ["de"]

    unsupported = PpOcrv6Model(
        enabled=False,
        artifacts_path=None,
        options=PpOcrv6Options(lang=["iso:ja"]),
        accelerator_options=AcceleratorOptions(),
    )
    with pytest.raises(OcrLanguageNotSupportedError, match="ja"):
        unsupported.resolve_ocr_languages()


def test_pp_ocrv6_uses_safetensors_and_maps_crop_coordinates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_root = tmp_path / "PaddleOCR"
    for name in ("det", "rec"):
        folder = model_root / f"PP-OCRv6_tiny_{name}_safetensors"
        folder.mkdir(parents=True)
        (folder / "model.safetensors").touch()

    captured: dict[str, object] = {}

    class FakePaddleOCR:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def predict(self, image: np.ndarray) -> list[SimpleNamespace]:
            assert image.shape == (60, 120, 3)
            return [
                SimpleNamespace(
                    json={
                        "res": {
                            "rec_texts": ["Hello"],
                            "rec_scores": [0.9],
                            "rec_polys": [[[6, 12], [30, 12], [30, 24], [6, 24]]],
                        }
                    }
                )
            ]

    paddleocr_module = ModuleType("paddleocr")
    paddleocr_module.__dict__["PaddleOCR"] = FakePaddleOCR
    monkeypatch.setitem(sys.modules, "paddleocr", paddleocr_module)
    monkeypatch.setattr(
        pp_ocrv6, "decide_device", lambda device, supported_devices: "cpu"
    )

    model = PpOcrv6Model(
        enabled=True,
        artifacts_path=tmp_path,
        options=PpOcrv6Options(mode=OcrMode.FULL_PAGE, scale=3.0),
        accelerator_options=AcceleratorOptions(),
    )
    assert captured["engine"] == "transformers"
    assert captured["text_detection_model_dir"] == str(
        model_root / "PP-OCRv6_tiny_det_safetensors"
    )
    assert captured["text_recognition_model_dir"] == str(
        model_root / "PP-OCRv6_tiny_rec_safetensors"
    )

    rect = BoundingBox(l=10, t=20, r=50, b=40, coord_origin=CoordOrigin.TOPLEFT)
    backend = SimpleNamespace(
        is_valid=lambda: True,
        get_page_image=lambda **kwargs: Image.new("RGB", (120, 60)),
    )
    page = SimpleNamespace(_backend=backend)
    monkeypatch.setattr(model, "get_ocr_rects", lambda page: [rect])
    cells = []
    monkeypatch.setattr(
        model, "post_process_cells", lambda values, page, result: cells.extend(values)
    )
    assert list(
        model(cast(ConversionResult, SimpleNamespace()), [cast(Page, page)])
    ) == [page]
    assert len(cells) == 1
    assert cells[0].text == "Hello"
    assert cells[0].confidence == pytest.approx(0.9)
    assert cells[0].rect.r_x0 == pytest.approx(12)
    assert cells[0].rect.r_y0 == pytest.approx(24)
    assert cells[0].rect.r_x2 == pytest.approx(20)
    assert cells[0].rect.r_y2 == pytest.approx(28)


def test_pp_ocrv6_requires_local_safetensors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"model\.safetensors"):
        pp_ocrv6._model_dirs(tmp_path)
