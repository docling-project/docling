# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from unittest.mock import patch

import pytest

from docling.datamodel.pipeline_options import TesseractCliOcrOptions
from docling.models.stages.ocr.tesseract_ocr_cli_model import TesseractOcrCliModel

_MODULE = "docling.models.stages.ocr.tesseract_ocr_cli_model"

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

    with patch(
        f"{_MODULE}.subprocess.run",
        return_value=_FakeCompletedProcess(_tsv(words).encode("utf-8")),
    ):
        df = model._run_tesseract("page.png", osd=None)

    assert list(df["text"]) == words
