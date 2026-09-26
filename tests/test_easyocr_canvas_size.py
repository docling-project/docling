# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""EasyOCR detector sizing without downloading model weights or running inference."""

import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image
from pydantic import ValidationError

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.pipeline_options import EasyOcrOptions
from docling.models.stages.ocr.easyocr_model import EasyOcrModel


@pytest.mark.parametrize("canvas_size", [0, -1])
def test_easyocr_canvas_size_must_be_positive(canvas_size: int) -> None:
    with pytest.raises(ValidationError):
        EasyOcrOptions(canvas_size=canvas_size)


@pytest.mark.parametrize("canvas_size", [None, 4096])
def test_easyocr_canvas_size_survives_options_round_trip(
    canvas_size: int | None,
) -> None:
    """Settings remain usable when pipeline options are serialized and rebuilt."""
    options = EasyOcrOptions(canvas_size=canvas_size)
    restored = EasyOcrOptions.model_validate(options.model_dump())

    assert restored.canvas_size == canvas_size


@pytest.mark.parametrize(
    ("canvas_size", "expected_kwargs"),
    [
        (None, {}),
        (4096, {"canvas_size": 4096}),
    ],
)
def test_easyocr_canvas_size_reaches_detector(
    monkeypatch: pytest.MonkeyPatch,
    canvas_size: int | None,
    expected_kwargs: dict[str, int],
) -> None:
    """Forward a user-supplied limit; do not alter the engine default otherwise."""
    reader = Mock()
    reader.readtext.return_value = []
    fake_easyocr = ModuleType("easyocr")
    fake_easyocr.__dict__["Reader"] = Mock(return_value=reader)
    monkeypatch.setitem(sys.modules, "easyocr", fake_easyocr)

    monkeypatch.setattr(
        "docling.models.stages.ocr.easyocr_model.decide_device",
        lambda _device: "cpu",
    )
    monkeypatch.setattr(
        "docling.models.stages.ocr.easyocr_model.TimeRecorder",
        lambda *_args: nullcontext(),
    )
    rect = SimpleNamespace(l=0.0, t=0.0, area=lambda: 1.0)
    monkeypatch.setattr(EasyOcrModel, "get_ocr_rects", lambda *_args: [rect])
    monkeypatch.setattr(EasyOcrModel, "post_process_cells", lambda *_args: None)

    model = EasyOcrModel(
        enabled=True,
        artifacts_path=None,
        options=EasyOcrOptions(lang=[], canvas_size=canvas_size),
        accelerator_options=AcceleratorOptions(),
    )
    backend = SimpleNamespace(
        is_valid=lambda: True,
        get_page_image=lambda **_kwargs: Image.new("RGB", (4, 4)),
    )
    page = SimpleNamespace(_backend=backend)

    assert list(model(SimpleNamespace(), [page])) == [page]
    assert reader.readtext.call_count == 1
    args, kwargs = reader.readtext.call_args
    assert args[0].shape == (4, 4, 3)
    assert kwargs == expected_kwargs
