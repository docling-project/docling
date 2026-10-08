# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import os
import subprocess
import sys
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path

import pytest

from docling.datamodel.pipeline_options import OcrAutoOptions
from docling.models.base_ocr_model import BaseOcrModel
from docling.models.factories.ocr_factory import OcrFactory
from docling.models.stages.ocr.auto_ocr_model import OcrAutoModel

EXTERNAL_PLUGIN_MODULE = "docling_test_external_ocr_plugin"
EXTERNAL_PLUGIN_NAME = "docling_test_external_ocr"

EXTERNAL_PLUGIN_SOURCE = """
from pathlib import Path
from typing import ClassVar, Literal

from docling.datamodel.pipeline_options import OcrOptions, PictureDescriptionBaseOptions

Path(__file__).with_suffix(".loaded").write_text("imported", encoding="utf-8")


class ExternalOcrOptions(OcrOptions):
    kind: ClassVar[Literal["docling_test_external_ocr"]] = "docling_test_external_ocr"


class ExternalOcrModel:
    @classmethod
    def get_options_type(cls) -> type[ExternalOcrOptions]:
        return ExternalOcrOptions


def ocr_engines() -> dict[str, list[type[ExternalOcrModel]]]:
    return {"ocr_engines": [ExternalOcrModel]}


class ExternalPictureOptions(PictureDescriptionBaseOptions):
    kind: ClassVar[Literal["docling_test_external_picture"]] = "docling_test_external_picture"


class ExternalPictureModel:
    @classmethod
    def get_options_type(cls) -> type[ExternalPictureOptions]:
        return ExternalPictureOptions


def picture_description() -> dict[str, list[type[ExternalPictureModel]]]:
    return {"picture_description": [ExternalPictureModel]}
"""


@pytest.fixture
def external_plugin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Install a third-party distribution exposing a docling plugin entry point."""
    (tmp_path / f"{EXTERNAL_PLUGIN_MODULE}.py").write_text(
        EXTERNAL_PLUGIN_SOURCE, encoding="utf-8"
    )

    dist_info = tmp_path / "docling_test_external_ocr_plugin-0.1.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: docling-test-external-ocr-plugin\nVersion: 0.1.0\n",
        encoding="utf-8",
    )
    (dist_info / "entry_points.txt").write_text(
        f"[docling]\n{EXTERNAL_PLUGIN_NAME} = {EXTERNAL_PLUGIN_MODULE}\n",
        encoding="utf-8",
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop(EXTERNAL_PLUGIN_MODULE, None)
    yield tmp_path
    sys.modules.pop(EXTERNAL_PLUGIN_MODULE, None)


def _load_ocr_factory(allow_external_plugins: bool) -> OcrFactory:
    factory = OcrFactory()
    factory.load_from_plugins(allow_external_plugins=allow_external_plugins)
    return factory


def _run_cli(
    arguments: list[str], plugin_path: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "docling.cli.main", *arguments],
        env={**os.environ, "PYTHONPATH": str(plugin_path), "COLUMNS": "200"},
        capture_output=True,
        text=True,
        timeout=60,
    )


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
    assert meta_by_kind["docling_test_external_ocr"].distribution == (
        "docling-test-external-ocr-plugin"
    )


@pytest.mark.usefixtures("external_plugin")
def test_factory_overrides_preserve_registration_and_provenance() -> None:
    class CustomFactory(OcrFactory):
        def register(
            self, cls: type[BaseOcrModel], plugin_name: str, plugin_module_name: str
        ) -> None:
            super().register(cls, plugin_name, plugin_module_name)

        def process_plugin(
            self,
            config: Mapping[str, Iterable[type[BaseOcrModel]]],
            plugin_name: str,
            plugin_module_name: str,
        ) -> None:
            super().process_plugin(config, plugin_name, plugin_module_name)

    factory = CustomFactory()
    factory.register(OcrAutoModel, "docling_defaults", "manual")
    factory.load_from_plugins(allow_external_plugins=True)

    meta_by_kind = {meta.kind: meta for meta in factory.registered_meta.values()}
    assert meta_by_kind[EXTERNAL_PLUGIN_NAME].distribution == (
        "docling-test-external-ocr-plugin"
    )
    assert factory.registered_meta[OcrAutoOptions].module == "manual"
    assert factory.registered_meta[OcrAutoOptions].distribution is None


@pytest.mark.parametrize(
    "arguments, source_name, plugin_loaded, warning_expected",
    [
        (["--help"], "input.md", False, False),
        (["convert", "--help", "--allow-external-plugins"], "input.md", False, False),
        (["convert"], "input.md", False, True),
        (["convert", "--allow-external-plugins"], "input.md", True, False),
        (["convert", "--allow-external-plugins"], "input.csv", True, False),
    ],
)
def test_cli_plugin_discovery(
    external_plugin: Path,
    arguments: list[str],
    source_name: str,
    plugin_loaded: bool,
    warning_expected: bool,
) -> None:
    source = external_plugin / source_name
    source.write_text(
        "title\nPlugin discovery\n"
        if source.suffix == ".csv"
        else "# Plugin discovery\n",
        encoding="utf-8",
    )
    output = external_plugin / "output"
    if "--help" not in arguments:
        arguments = [*arguments, str(source), "--output", str(output)]

    result = _run_cli(arguments, external_plugin)

    assert result.returncode == 0, result.stdout + result.stderr
    assert ("will not be loaded" in result.stderr) is warning_expected
    assert (external_plugin / f"{EXTERNAL_PLUGIN_MODULE}.loaded").exists() is (
        plugin_loaded
    )
    if "--help" not in arguments:
        assert "Plugin discovery" in (output / "input.md").read_text(encoding="utf-8")


def test_cli_lists_external_plugins(external_plugin: Path) -> None:
    result = _run_cli(["convert", "--show-external-plugins"], external_plugin)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Available picture description engines" in result.stdout
    assert "docling_test_external_picture" in result.stdout
    assert "docling_test_external_ocr" in result.stdout
    assert EXTERNAL_PLUGIN_NAME in result.stdout
    assert "docling_defaults" not in result.stdout
    assert "will not be loaded" not in result.stderr
    assert result.stdout.count("docling-test-external-ocr-plugin") == 2
    assert EXTERNAL_PLUGIN_MODULE not in result.stdout
