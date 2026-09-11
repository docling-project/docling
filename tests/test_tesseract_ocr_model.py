# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import sys
from types import SimpleNamespace

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.pipeline_options import TesseractOcrOptions
from docling.models.stages.ocr.tesseract_ocr_model import TesseractOcrModel


class _Reader:
    def End(self):
        pass


def _fake_tesserocr(get_languages):
    return SimpleNamespace(
        OEM=SimpleNamespace(DEFAULT=0),
        PSM=SimpleNamespace(AUTO=3, OSD_ONLY=0),
        RIL=SimpleNamespace(),
        PyTessBaseAPI=lambda **kwargs: _Reader(),
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
