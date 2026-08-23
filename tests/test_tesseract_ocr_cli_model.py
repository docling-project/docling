from unittest.mock import patch

from docling.models.stages.ocr.tesseract_ocr_cli_model import TesseractOcrCliModel


def test_set_languages_and_prefix_normalizes_windows_script_separator():
    model = TesseractOcrCliModel.__new__(TesseractOcrCliModel)
    model._safe_tesseract_cmd = "tesseract"

    completed = type(
        "Completed",
        (),
        {
            "stdout": b"List of available languages in tessdata:\n"
            + b"script"
            + bytes([92])
            + b"Arabic\neng\n",
        },
    )()
    with patch("subprocess.run", return_value=completed):
        model._set_languages_and_prefix()

    assert model._tesseract_languages == ["script/Arabic", "eng"]
    assert model._script_prefix == "script/"


def test_set_languages_and_prefix_preserves_forward_slash_separator():
    model = TesseractOcrCliModel.__new__(TesseractOcrCliModel)
    model._safe_tesseract_cmd = "tesseract"

    completed = type(
        "Completed",
        (),
        {"stdout": b"List of available languages in tessdata:\nscript/Latin\neng\n"},
    )()
    with patch("subprocess.run", return_value=completed):
        model._set_languages_and_prefix()

    assert model._tesseract_languages == ["script/Latin", "eng"]
    assert model._script_prefix == "script/"
