# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import subprocess
from unittest.mock import patch

import pytest
from PIL import Image

from docling.backend.image_backend import ImageDocumentBackend
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import FailureCategory, InputFormat, Page
from docling.datamodel.document import ConversionResult, InputDocument
from docling.datamodel.pipeline_options import OcrMode, TesseractCliOcrOptions
from docling.models.stages.ocr.tesseract_ocr_cli_model import TesseractOcrCliModel

_MODULE = "docling.models.stages.ocr.tesseract_ocr_cli_model"

_OSD_STDOUT = (
    "Page number: 0\n"
    "Orientation in degrees: 90\n"
    "Rotate: 270\n"
    "Orientation confidence: {confidence}\n"
    "Script: Latin\n"
    "Script confidence: 1.84\n"
)

_TSV_HEADER = (
    "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num"
    "\tleft\ttop\twidth\theight\tconf\ttext"
)


class _FakeCompletedProcess:
    def __init__(self, stdout: bytes) -> None:
        self.stdout = stdout


def _tsv(words: list[str]) -> str:
    """Tesseract's `stdout tsv` output: a page row without text, then one row per word."""
    rows = [_TSV_HEADER, "1\t1\t0\t0\t0\t0\t0\t0\t400\t100\t-1\t"]
    for ix, word in enumerate(words, start=1):
        rows.append(f"5\t1\t1\t1\t1\t{ix}\t{ix * 50}\t10\t40\t12\t90\t{word}")
    return "\n".join(rows) + "\n"


@pytest.mark.parametrize(
    "words",
    [
        ["Price", "N/A", "NA", "None", "null", "nan"],
        ["007", "2024", "12"],
    ],
    ids=["na-like-words", "numeric-words"],
)
def test_recognized_words_are_kept_verbatim(words: list[str]) -> None:
    model = TesseractOcrCliModel.__new__(TesseractOcrCliModel)
    model.options = TesseractCliOcrOptions()
    model._safe_tesseract_cmd = "tesseract"
    model._safe_tessdata_path = None
    model._auto_script = False
    model._native_codes = ["eng"]
    model._dpi = 216

    with patch(
        f"{_MODULE}.subprocess.run",
        return_value=_FakeCompletedProcess(_tsv(words).encode("utf-8")),
    ):
        df = model._run_tesseract("page.png", osd=None)

    assert list(df["text"]) == words


class _FakeTesseract:
    def __init__(self, osd_confidence=8.0, ocr_stderr=None):
        self.osd_confidence = osd_confidence
        self.ocr_stderr = ocr_stderr
        self.commands: list[list[str]] = []
        self.ocr_images: list[tuple[str, tuple[int, int]]] = []

    def __call__(self, cmd, **kwargs):
        self.commands.append(cmd)
        if "osd" in cmd:
            stdout = _OSD_STDOUT.format(confidence=self.osd_confidence)
            return _FakeCompletedProcess(stdout.encode("utf-8"))
        with Image.open(next(arg for arg in cmd if arg.endswith(".png"))) as image:
            self.ocr_images.append((image.mode, image.size))
        if self.ocr_stderr is not None:
            raise subprocess.CalledProcessError(1, cmd, b"", self.ocr_stderr)
        return _FakeCompletedProcess(_tsv(["Hello"]).encode("utf-8"))


def _convert_page(monkeypatch, tmp_path, tesseract, **options):
    monkeypatch.setattr(f"{_MODULE}.subprocess.run", tesseract)
    monkeypatch.setattr(
        TesseractOcrCliModel,
        "_get_name_and_version",
        lambda self: ("tesseract", "5.5.0"),
    )
    monkeypatch.setattr(
        TesseractOcrCliModel,
        "_set_languages",
        lambda self: setattr(self, "_tesseract_vocabulary", ["eng", "osd"]),
    )
    model = TesseractOcrCliModel(
        enabled=True,
        artifacts_path=None,
        options=TesseractCliOcrOptions(mode=OcrMode.FULL_PAGE, **options),
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
    page = Page(page_no=1, size=page_backend.get_size())
    page._backend = page_backend
    assert list(model(conv_res, [page])) == [page]
    return conv_res


@pytest.mark.parametrize(
    ("confidence", "ocr_size"),
    [(0.4, (600, 300)), (8.0, (300, 600))],
    ids=["close-call", "confident"],
)
def test_region_is_rotated_only_on_a_confident_orientation(
    monkeypatch, tmp_path, confidence, ocr_size
):
    tesseract = _FakeTesseract(osd_confidence=confidence)

    _convert_page(monkeypatch, tmp_path, tesseract, lang=["eng"])

    assert tesseract.ocr_images == [("L", ocr_size)]


def test_tesseract_is_told_the_render_resolution(monkeypatch, tmp_path):
    tesseract = _FakeTesseract()

    _convert_page(monkeypatch, tmp_path, tesseract, lang=["eng"], scale=2.0)

    assert [cmd[cmd.index("--dpi") + 1] for cmd in tesseract.commands] == [
        "144",
        "144",
    ]


def test_failed_region_is_recorded_on_the_page(monkeypatch, tmp_path):
    tesseract = _FakeTesseract(ocr_stderr=b"Error in pixReadStream: simulated")

    conv_res = _convert_page(monkeypatch, tmp_path, tesseract, lang=["eng"])

    assert [
        (error.page_no, error.category, error.module_name) for error in conv_res.errors
    ] == [(1, FailureCategory.INFERENCE_FAILURE, "TesseractOcrCliModel")]
    assert "pixReadStream: simulated" in conv_res.errors[0].error_message
