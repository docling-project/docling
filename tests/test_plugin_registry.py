from collections.abc import Iterator
from dataclasses import dataclass
from importlib import metadata
from types import ModuleType
from typing import Callable

import pytest

from docling.models.factories import (
    get_layout_factory,
    get_ocr_factory,
    get_picture_description_factory,
    get_table_structure_factory,
)
from docling.models.factories.plugin_registry import (
    PluginDiscoveryError,
    load_plugin_modules,
)


@dataclass(frozen=True)
class _FakeEntryPoint:
    name: str
    module: str
    loader: Callable[[], ModuleType]
    group: str = "docling"

    def load(self) -> ModuleType:
        return self.loader()


@dataclass(frozen=True)
class _FakeDistribution:
    entry_points: tuple[_FakeEntryPoint, ...]


@pytest.fixture(autouse=True)
def _clear_factory_cache() -> Iterator[None]:
    factory_getters = (
        get_layout_factory,
        get_ocr_factory,
        get_picture_description_factory,
        get_table_structure_factory,
    )
    for get_factory in factory_getters:
        get_factory.cache_clear()
    load_plugin_modules.cache_clear()
    yield
    for get_factory in factory_getters:
        get_factory.cache_clear()
    load_plugin_modules.cache_clear()


def test_disabled_external_plugins_are_not_imported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported = False

    def load_external_plugin() -> ModuleType:
        nonlocal imported
        imported = True
        return ModuleType("third_party_docling_plugin")

    distribution = _FakeDistribution(
        entry_points=(
            _FakeEntryPoint(
                name="third-party",
                module="third_party_docling_plugin",
                loader=load_external_plugin,
            ),
        )
    )
    monkeypatch.setattr(metadata, "distributions", lambda: [distribution])

    get_ocr_factory(allow_external_plugins=False)

    assert imported is False


def test_plugin_entry_points_are_loaded_once_across_factories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load_count = 0

    def load_external_plugin() -> ModuleType:
        nonlocal load_count
        load_count += 1
        return ModuleType("third_party_docling_plugin")

    distribution = _FakeDistribution(
        entry_points=(
            _FakeEntryPoint(
                name="third-party",
                module="third_party_docling_plugin",
                loader=load_external_plugin,
            ),
        )
    )
    monkeypatch.setattr(metadata, "distributions", lambda: [distribution])

    get_ocr_factory(allow_external_plugins=True)
    get_layout_factory(allow_external_plugins=True)
    get_table_structure_factory(allow_external_plugins=True)
    get_picture_description_factory(allow_external_plugins=True)

    assert load_count == 1


def test_conflicting_entry_point_names_fail_before_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported_modules: list[str] = []

    def load_plugin(module_name: str) -> ModuleType:
        imported_modules.append(module_name)
        return ModuleType(module_name)

    distributions = (
        _FakeDistribution(
            entry_points=(
                _FakeEntryPoint(
                    name="duplicate-name",
                    module="first_docling_plugin",
                    loader=lambda: load_plugin("first_docling_plugin"),
                ),
            )
        ),
        _FakeDistribution(
            entry_points=(
                _FakeEntryPoint(
                    name="duplicate-name",
                    module="second_docling_plugin",
                    loader=lambda: load_plugin("second_docling_plugin"),
                ),
            )
        ),
    )
    monkeypatch.setattr(metadata, "distributions", lambda: distributions)

    with pytest.raises(
        PluginDiscoveryError,
        match=(r"duplicate-name.*first_docling_plugin.*second_docling_plugin"),
    ):
        get_ocr_factory(allow_external_plugins=True)

    assert imported_modules == []
