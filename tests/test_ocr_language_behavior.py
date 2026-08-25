"""Auto-engine selection, driven by the requested OCR language.

`OcrAutoOptions` is the one place where a language tag changes *which engine
runs*, not merely which recognizer it loads, and deciding that means probing the
installed engines for real. The parsing rules that settle before any engine
loads are in `test_ocr_language.py`.
"""

import logging
import sys

import pytest

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.pipeline_options import OcrAutoOptions
from docling.exceptions import OcrLanguageNotSupportedError
from docling.models.stages.ocr.auto_ocr_model import OcrAutoModel

pytestmark = pytest.mark.ml_ocr

# Amharic: a valid tag written in a script none of docling's engines recognize.
_UNSERVABLE_TAG = "am"


def _auto_model(lang: list[str]) -> OcrAutoModel:
    return OcrAutoModel(
        enabled=True,
        artifacts_path=None,
        options=OcrAutoOptions(lang=lang),
        accelerator_options=AcceleratorOptions(),
    )


def test_auto_gives_the_delegate_the_users_language() -> None:
    model = _auto_model(["zh-Hant"])

    assert model._engine is not None
    assert model._engine.options.lang == ["zh-Hant"]


@pytest.mark.skipif(sys.platform != "darwin", reason="ocrmac is macOS-only")
def test_auto_falls_through_an_engine_that_cannot_serve_the_language(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Apple Vision ships no Devanagari recognizer, so auto must move on rather
    than fail -- picking an *available* engine is the whole contract of `auto`."""
    pytest.importorskip("ocrmac")

    with caplog.at_level(logging.INFO):
        model = _auto_model(["hi"])

    assert "skipping ocrmac" in caplog.text
    assert model._engine is not None
    assert not isinstance(model._engine, type(model))


def test_auto_reports_every_candidate_when_none_can_serve_the_language() -> None:
    """The aggregated error replaces a bare "No OCR engine found." warning."""
    with pytest.raises(OcrLanguageNotSupportedError) as excinfo:
        _auto_model([_UNSERVABLE_TAG])

    message = str(excinfo.value)
    assert "am-Ethi" in message
    # Every candidate is named with the reason it was passed over.
    assert "No installed engine can serve it" in message
