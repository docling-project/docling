"""Tests for OcrAutoModel when no OCR engine is installed."""

import logging
import sys
from typing import Iterable
from unittest.mock import MagicMock, patch

import pytest

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import OcrAutoOptions
from docling.models.stages.ocr.auto_ocr_model import OcrAutoModel

# Build a list of patches that make every supported OCR engine appear missing.
# The imports are performed inside OcrAutoModel.__init__ try/except blocks, so we
# patch the names as they would be resolved inside that module's namespace.
_ENGINE_PATCHES = [
    patch("onnxruntime.__name__", side_effect=ImportError("onnxruntime not installed")),
    patch("easyocr.__name__", side_effect=ImportError("easyocr not installed")),
    patch("torch.__name__", side_effect=ImportError("torch not installed")),
]

# The real patch targets: builtins.__import__ would be too broad, so we patch
# the specific sys.modules entries to be absent and override __import__ at the
# module level.  The cleanest approach for nested-import probes is to patch
# builtins.__import__ selectively.
_BLOCKED_MODULES = frozenset(
    {
        "ocrmac",
        "ocrmac.ocrmac",
        "onnxruntime",
        "rapidocr",
        "easyocr",
        "torch",
    }
)


def _make_selective_import(real_import):
    """Return a replacement for builtins.__import__ that raises ImportError for
    the OCR engine modules we want to pretend are absent."""

    def _import(name, *args, **kwargs):
        if name in _BLOCKED_MODULES:
            raise ImportError(f"Mocked: {name} is not installed")
        return real_import(name, *args, **kwargs)

    return _import


@pytest.fixture()
def no_ocr_engines(monkeypatch):
    """Monkeypatch builtins.__import__ so all OCR engines appear absent."""
    import builtins

    real_import = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", _make_selective_import(real_import))
    yield


def _make_model(enabled: bool = True) -> OcrAutoModel:
    return OcrAutoModel(
        enabled=enabled,
        artifacts_path=None,
        options=OcrAutoOptions(),
        accelerator_options=AcceleratorOptions(),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_no_ocr_engine_engine_is_none(no_ocr_engines):
    """When no OCR engine is available _engine must be None."""
    model = _make_model(enabled=True)
    assert model._engine is None


def test_no_ocr_engine_warning_logged(no_ocr_engines, caplog):
    """When no OCR engine is available a WARNING must be emitted with install hints."""
    expected_fragment = (
        "No OCR engine is available. Install one of the supported extras to "
        "enable OCR: `pip install docling[rapidocr]` (recommended) "
        "or `pip install docling[easyocr]`."
    )

    with caplog.at_level(logging.WARNING, logger="docling.models.stages.ocr.auto_ocr_model"):
        _make_model(enabled=True)

    warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any(expected_fragment in msg for msg in warning_messages), (
        f"Expected warning not found. Captured warnings: {warning_messages}"
    )


def test_no_ocr_engine_pages_pass_through_unchanged(no_ocr_engines):
    """When _engine is None, __call__ must yield pages unchanged."""
    model = _make_model(enabled=True)
    assert model._engine is None

    # Build lightweight mock pages – we only need identity equality.
    mock_pages = [MagicMock(spec=Page), MagicMock(spec=Page)]
    mock_conv_res = MagicMock(spec=ConversionResult)

    result = list(model(mock_conv_res, iter(mock_pages)))

    assert result == mock_pages, (
        "Pages should pass through unchanged when no OCR engine is available."
    )


def test_no_ocr_engine_disabled_model_passes_through(no_ocr_engines):
    """When enabled=False the model must pass pages through regardless of engine state."""
    model = _make_model(enabled=False)
    # _engine stays None because enabled=False skips the engine-selection block
    assert model._engine is None

    mock_pages = [MagicMock(spec=Page), MagicMock(spec=Page)]
    mock_conv_res = MagicMock(spec=ConversionResult)

    result = list(model(mock_conv_res, iter(mock_pages)))

    assert result == mock_pages
