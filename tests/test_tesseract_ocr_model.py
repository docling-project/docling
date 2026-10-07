# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import sys
from types import SimpleNamespace

import pytest
from PIL import Image

from docling.backend.image_backend import ImageDocumentBackend
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import InputFormat, Page
from docling.datamodel.document import ConversionResult, InputDocument
from docling.datamodel.pipeline_options import OcrMode, TesseractOcrOptions
from docling.models.stages.ocr.tesseract_ocr_model import TesseractOcrModel


class _Reader:
    def End(self):
        pass


class _OcrReader(_Reader):
    """Reads one text line with tesserocr's 0-100 mean confidence."""

    def SetImage(self, image):
        pass

    def DetectOrientationScript(self):
        return {"orient_deg": 0, "orient_conf": 10.0, "script_name": "Latin"}

    def GetComponentImages(self, level, text_only):
        return [(None, {"x": 30, "y": 30, "w": 300, "h": 60}, 0, 0)]

    def SetRectangle(self, left, top, width, height):
        pass

    def GetUTF8Text(self):
        return "Hello world\n"

    def MeanTextConf(self):
        return 42


def _fake_tesserocr(get_languages, reader=_Reader):
    return SimpleNamespace(
        OEM=SimpleNamespace(DEFAULT=0),
        PSM=SimpleNamespace(AUTO=3, OSD_ONLY=0),
        RIL=SimpleNamespace(TEXTLINE=2),
        PyTessBaseAPI=lambda **kwargs: reader(),
        get_languages=get_languages,
        tesseract_version=lambda: "test",
    )


def test_language_discovery_uses_configured_tessdata_path(monkeypatch):
    calls = []

    def get_languages(*args, **kwargs):
        calls.append((args, kwargs))
        return "/custom/tessdata", ["eng", "osd"]

    monkeypatch.setitem(sys.modules, "tesserocr", _fake_tesserocr(get_languages))

    model = TesseractOcrModel(
        enabled=True,
        artifacts_path=None,
        options=TesseractOcrOptions(path="/custom/tessdata", lang=["eng"]),
        accelerator_options=AcceleratorOptions(),
    )

    assert calls == [((), {"path": "/custom/tessdata"})]
    del model


def test_language_discovery_without_path_uses_default(monkeypatch):
    calls = []

    def get_languages(*args, **kwargs):
        calls.append((args, kwargs))
        return "/default/tessdata", ["eng", "osd"]

    monkeypatch.setitem(sys.modules, "tesserocr", _fake_tesserocr(get_languages))

    model = TesseractOcrModel(
        enabled=True,
        artifacts_path=None,
        options=TesseractOcrOptions(lang=["eng"]),
        accelerator_options=AcceleratorOptions(),
    )

    assert calls == [((), {})]
    del model


def test_cell_confidence_is_scaled_to_unit_range(monkeypatch, tmp_path):
    monkeypatch.setitem(
        sys.modules,
        "tesserocr",
        _fake_tesserocr(lambda **kwargs: ("/tessdata", ["eng", "osd"]), _OcrReader),
    )
    model = TesseractOcrModel(
        enabled=True,
        artifacts_path=None,
        options=TesseractOcrOptions(lang=["eng"], mode=OcrMode.FULL_PAGE),
        accelerator_options=AcceleratorOptions(),
    )

    image_path = tmp_path / "page.png"
    Image.new("RGB", (200, 100), "white").save(image_path)
    in_doc = InputDocument(
        path_or_stream=image_path,
        format=InputFormat.IMAGE,
        backend=ImageDocumentBackend,
    )
    conv_res = ConversionResult(input=in_doc)
    page_backend = in_doc._backend.load_page(0)
    page = Page(page_no=0, size=page_backend.get_size())
    page._backend = page_backend

    list(model(conv_res, [page]))

    assert page.parsed_page is not None
    cells = page.parsed_page.textline_cells
    assert [cell.confidence for cell in cells] == [pytest.approx(0.42)]
    assert conv_res.confidence.pages[0].ocr_score == pytest.approx(0.42)
    del model
