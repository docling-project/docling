# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from pathlib import Path
from unittest.mock import patch

import pytest

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.pipeline_options import TesseractCliOcrOptions
from docling.exceptions import OcrLanguageNotSupportedError
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


@pytest.fixture
def custom_tessdata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Expose different installed models in the default and configured directories."""
    custom_path = str(tmp_path / "custom tessdata")

    def list_languages(cmd: list[str], **kwargs: object) -> _FakeCompletedProcess:
        assert "--list-langs" in cmd
        if "--tessdata-dir" in cmd:
            assert cmd[cmd.index("--tessdata-dir") + 1] == custom_path
            languages = ["custom", "osd", "script/Latin"]
        else:
            languages = ["eng"]
        listing = f"List of available languages ({len(languages)}):\n"
        return _FakeCompletedProcess(
            (listing + "\n".join(languages) + "\n").encode("utf-8")
        )

    monkeypatch.setattr(f"{_MODULE}.subprocess.run", list_languages)
    monkeypatch.setattr(
        TesseractOcrCliModel,
        "_get_name_and_version",
        lambda self: ("tesseract", "5.5.1"),
    )
    return custom_path


def _model_with_tessdata(path: str | None, lang: list[str]) -> TesseractOcrCliModel:
    return TesseractOcrCliModel(
        enabled=True,
        artifacts_path=None,
        options=TesseractCliOcrOptions(path=path, lang=lang),
        accelerator_options=AcceleratorOptions(),
    )


def test_custom_tessdata_language_is_available(custom_tessdata: str) -> None:
    model = _model_with_tessdata(custom_tessdata, ["custom"])

    assert model._native_codes == ["custom"]
    assert "custom" in model.supported_ocr_languages().native


def test_custom_tessdata_does_not_advertise_default_languages(
    custom_tessdata: str,
) -> None:
    with pytest.raises(OcrLanguageNotSupportedError, match="eng"):
        _model_with_tessdata(custom_tessdata, ["eng"])


def test_omitted_tessdata_path_uses_default_languages(custom_tessdata: str) -> None:
    model = _model_with_tessdata(None, ["eng"])

    assert model._native_codes == ["eng"]
    assert model.supported_ocr_languages().bcp47 == ["en"]


def test_custom_tessdata_supplies_osd_for_auto_detection(custom_tessdata: str) -> None:
    model = _model_with_tessdata(custom_tessdata, [])

    assert model._auto_script is True
    assert model._native_codes == []


def test_auto_detection_requires_osd_in_the_selected_directory(
    custom_tessdata: str,
) -> None:
    def list_languages(cmd: list[str], **kwargs: object) -> _FakeCompletedProcess:
        languages = ["custom"] if "--tessdata-dir" in cmd else ["eng", "osd"]
        listing = f"List of available languages ({len(languages)}):\n"
        return _FakeCompletedProcess(
            (listing + "\n".join(languages) + "\n").encode("utf-8")
        )

    with patch(f"{_MODULE}.subprocess.run", side_effect=list_languages):
        assert _model_with_tessdata(None, [])._auto_script is True
        with pytest.raises(ImportError, match="osd"):
            _model_with_tessdata(custom_tessdata, [])


def test_detected_script_is_read_from_the_custom_directory(
    custom_tessdata: str, tmp_path: Path
) -> None:
    model = _model_with_tessdata(custom_tessdata, [])
    filename = str(tmp_path / "page.png")
    with patch(
        f"{_MODULE}.subprocess.run",
        side_effect=[
            _FakeCompletedProcess(b"Orientation in degrees: 0\nScript: Latin\n"),
            _FakeCompletedProcess(_tsv(["Custom", "script"]).encode("utf-8")),
        ],
    ) as run:
        osd = model._perform_osd(filename)
        result = model._run_tesseract(filename, osd)

    osd_cmd, recognition_cmd = [call.args[0] for call in run.call_args_list]
    for cmd in (osd_cmd, recognition_cmd):
        assert cmd[cmd.index("--tessdata-dir") + 1] == custom_tessdata
    assert osd_cmd[osd_cmd.index("-l") + 1] == "osd"
    assert recognition_cmd[recognition_cmd.index("-l") + 1] == "script/Latin"
    assert list(result["text"]) == ["Custom", "script"]


@pytest.mark.parametrize("use_custom_path", [False, True], ids=["default", "custom"])
def test_orientation_detection_uses_configured_tessdata(
    tmp_path: Path, use_custom_path: bool
) -> None:
    path = str(tmp_path / "custom tessdata") if use_custom_path else None
    model = TesseractOcrCliModel(
        enabled=False,
        artifacts_path=None,
        options=TesseractCliOcrOptions(path=path, lang=["eng"]),
        accelerator_options=AcceleratorOptions(),
    )
    with patch(
        f"{_MODULE}.subprocess.run",
        return_value=_FakeCompletedProcess(
            b"Orientation in degrees: 90\nScript: Latin\n"
        ),
    ) as run:
        result = model._perform_osd(str(tmp_path / "page.png"))

    cmd = run.call_args.args[0]
    if path is None:
        assert "--tessdata-dir" not in cmd
    else:
        assert "--tessdata-dir" in cmd
        assert cmd[cmd.index("--tessdata-dir") + 1] == path
    assert result.loc[result["key"] == "Script", "value"].iloc[0].strip() == "Latin"
