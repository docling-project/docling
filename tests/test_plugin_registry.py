from collections.abc import Iterator
from dataclasses import dataclass
from importlib import metadata
from types import ModuleType
from typing import Callable

import pytest

from docling.models.factories import get_ocr_factory


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
    get_ocr_factory.cache_clear()
    yield
    get_ocr_factory.cache_clear()


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
