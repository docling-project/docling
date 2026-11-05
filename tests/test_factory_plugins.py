# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from docling.models.factories.ocr_factory import OcrFactory

EXTERNAL_PLUGIN_MODULE = "docling_test_external_ocr_plugin"
EXTERNAL_PLUGIN_NAME = "docling_test_external_ocr"

EXTERNAL_PLUGIN_SOURCE = """
from typing import ClassVar, Literal

from docling.datamodel.pipeline_options import OcrOptions


class ExternalOcrOptions(OcrOptions):
    kind: ClassVar[Literal["docling_test_external_ocr"]] = "docling_test_external_ocr"


class ExternalOcrModel:
    @classmethod
    def get_options_type(cls):
        return ExternalOcrOptions


def ocr_engines():
    return {"ocr_engines": [ExternalOcrModel]}
"""


@pytest.fixture
def external_plugin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Install a third-party distribution exposing a docling plugin entry point."""
    (tmp_path / f"{EXTERNAL_PLUGIN_MODULE}.py").write_text(EXTERNAL_PLUGIN_SOURCE)

    dist_info = tmp_path / "docling_test_external_ocr_plugin-0.1.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: docling-test-external-ocr-plugin\nVersion: 0.1.0\n"
    )
    (dist_info / "entry_points.txt").write_text(
        f"[docling]\n{EXTERNAL_PLUGIN_NAME} = {EXTERNAL_PLUGIN_MODULE}\n"
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop(EXTERNAL_PLUGIN_MODULE, None)
    yield
    sys.modules.pop(EXTERNAL_PLUGIN_MODULE, None)


def _load_ocr_factory(allow_external_plugins: bool) -> OcrFactory:
    factory = OcrFactory()
    factory.load_from_plugins(allow_external_plugins=allow_external_plugins)
    return factory


@pytest.mark.usefixtures("external_plugin")
def test_external_plugin_not_imported_when_disallowed():
    factory = _load_ocr_factory(allow_external_plugins=False)
    plugin_names = {meta.plugin_name for meta in factory.registered_meta.values()}

    assert EXTERNAL_PLUGIN_MODULE not in sys.modules
    assert EXTERNAL_PLUGIN_NAME not in plugin_names
    assert "docling_defaults" in plugin_names


@pytest.mark.usefixtures("external_plugin")
def test_external_plugin_loaded_when_allowed():
    factory = _load_ocr_factory(allow_external_plugins=True)

    assert EXTERNAL_PLUGIN_MODULE in sys.modules
    assert "docling_test_external_ocr" in factory.registered_kind
    meta_by_kind = {meta.kind: meta for meta in factory.registered_meta.values()}
    assert meta_by_kind["docling_test_external_ocr"].plugin_name == (
        EXTERNAL_PLUGIN_NAME
    )
    assert meta_by_kind["docling_test_external_ocr"].module == EXTERNAL_PLUGIN_MODULE
