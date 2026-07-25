from collections.abc import Iterator
from dataclasses import dataclass
from importlib import metadata
from types import ModuleType
from typing import Callable, ClassVar

import pytest

from docling.datamodel.pipeline_options import BaseOptions
from docling.models.factories import (
    get_layout_factory,
    get_ocr_factory,
    get_picture_description_factory,
    get_table_structure_factory,
)
from docling.models.factories.base_factory import (
    BaseFactory,
    PluginConfigurationError,
    PluginHookError,
)
from docling.models.factories.plugin_registry import (
    PluginDiscoveryError,
    PluginLoadError,
    load_plugin_modules,
)


@dataclass(frozen=True)
class _FakeEntryPoint:
    name: str
    module: str
    loader: Callable[[], ModuleType]
    group: str = "docling"

    @property
    def value(self) -> str:
        return self.module

    def load(self) -> ModuleType:
        return self.loader()


@dataclass(frozen=True)
class _FakeDistribution:
    entry_points: tuple[_FakeEntryPoint, ...]
    name: str = "third-party-package"

    @property
    def metadata(self) -> dict[str, str]:
        return {"Name": self.name}


class _PluginModelBase:
    def __init__(self, *, options: BaseOptions, **kwargs: object) -> None:
        self.options = options

    @classmethod
    def get_options_type(cls) -> type[BaseOptions]:
        raise NotImplementedError


class _FirstOptions(BaseOptions):
    kind: ClassVar[str] = "shared-kind"


class _SecondOptions(BaseOptions):
    kind: ClassVar[str] = "shared-kind"


class _FirstModel(_PluginModelBase):
    @classmethod
    def get_options_type(cls) -> type[BaseOptions]:
        return _FirstOptions


class _SecondModel(_PluginModelBase):
    @classmethod
    def get_options_type(cls) -> type[BaseOptions]:
        return _SecondOptions


class _ConstructorKeyErrorModel(_PluginModelBase):
    def __init__(self, *, options: BaseOptions, **kwargs: object) -> None:
        raise KeyError("raised inside model constructor")

    @classmethod
    def get_options_type(cls) -> type[BaseOptions]:
        return _FirstOptions


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


def test_external_distribution_cannot_spoof_docling_module_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported = False

    def load_spoofed_plugin() -> ModuleType:
        nonlocal imported
        imported = True
        return ModuleType("docling.spoofed_plugin")

    distribution = _FakeDistribution(
        entry_points=(
            _FakeEntryPoint(
                name="spoofed-plugin",
                module="docling.spoofed_plugin",
                loader=load_spoofed_plugin,
            ),
        ),
        name="malicious-package",
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


def test_duplicate_entry_point_declarations_are_always_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported = False

    def load_plugin() -> ModuleType:
        nonlocal imported
        imported = True
        return ModuleType("shared_docling_plugin")

    distributions = (
        _FakeDistribution(
            entry_points=(
                _FakeEntryPoint(
                    name="duplicate-name",
                    module="shared_docling_plugin",
                    loader=load_plugin,
                ),
            ),
            name="z-provider",
        ),
        _FakeDistribution(
            entry_points=(
                _FakeEntryPoint(
                    name="duplicate-name",
                    module="shared_docling_plugin",
                    loader=load_plugin,
                ),
            ),
            name="a-provider",
        ),
    )
    monkeypatch.setattr(metadata, "distributions", lambda: distributions)

    with pytest.raises(
        PluginDiscoveryError,
        match=r"duplicate-name.*a-provider.*z-provider",
    ):
        get_ocr_factory(allow_external_plugins=True)

    assert imported is False


def test_plugin_import_failure_identifies_the_entry_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import_error = ImportError("missing optional runtime")

    def fail_to_load() -> ModuleType:
        raise import_error

    distribution = _FakeDistribution(
        entry_points=(
            _FakeEntryPoint(
                name="broken-plugin",
                module="broken_docling_plugin",
                loader=fail_to_load,
            ),
        )
    )
    monkeypatch.setattr(metadata, "distributions", lambda: [distribution])

    with pytest.raises(
        PluginLoadError,
        match=r"broken-plugin.*broken_docling_plugin",
    ) as exc_info:
        get_ocr_factory(allow_external_plugins=True)

    assert exc_info.value.__cause__ is import_error


def test_malformed_plugin_configuration_identifies_the_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin_module = ModuleType("malformed_docling_plugin")

    def ocr_engines() -> object:
        return {"ocr_engines": ["not-a-model-class"]}

    setattr(plugin_module, "ocr_engines", ocr_engines)
    distribution = _FakeDistribution(
        entry_points=(
            _FakeEntryPoint(
                name="malformed-plugin",
                module=plugin_module.__name__,
                loader=lambda: plugin_module,
            ),
        )
    )
    monkeypatch.setattr(metadata, "distributions", lambda: [distribution])

    with pytest.raises(
        PluginConfigurationError,
        match=r"malformed-plugin.*ocr_engines.*model class",
    ):
        get_ocr_factory(allow_external_plugins=True)


def test_plugin_model_kinds_must_be_unique(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin_module = ModuleType("duplicate_kind_docling_plugin")

    def ocr_engines() -> object:
        return {"ocr_engines": [_FirstModel, _SecondModel]}

    setattr(plugin_module, "ocr_engines", ocr_engines)
    distribution = _FakeDistribution(
        entry_points=(
            _FakeEntryPoint(
                name="duplicate-kind-plugin",
                module=plugin_module.__name__,
                loader=lambda: plugin_module,
            ),
        )
    )
    monkeypatch.setattr(metadata, "distributions", lambda: [distribution])
    factory = BaseFactory("ocr_engines", _PluginModelBase)

    with pytest.raises(
        PluginConfigurationError,
        match=r"duplicate-kind-plugin.*shared-kind.*_FirstModel.*_SecondModel",
    ):
        factory.load_from_plugins(allow_external_plugins=True)

    assert factory.registered_kind == []


def test_model_constructor_key_errors_are_not_reported_as_missing_models() -> None:
    factory = BaseFactory("ocr_engines", _PluginModelBase)
    factory.register(
        _ConstructorKeyErrorModel,
        plugin_name="test-plugin",
        plugin_package_name="test-package",
        plugin_module_name="test_plugin",
    )

    with pytest.raises(KeyError, match="raised inside model constructor"):
        factory.create_instance(_FirstOptions())


def test_plugin_hook_failures_identify_the_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hook_error = RuntimeError("plugin setup failed")
    plugin_module = ModuleType("failing_hook_docling_plugin")

    def ocr_engines() -> object:
        raise hook_error

    setattr(plugin_module, "ocr_engines", ocr_engines)
    distribution = _FakeDistribution(
        entry_points=(
            _FakeEntryPoint(
                name="failing-hook-plugin",
                module=plugin_module.__name__,
                loader=lambda: plugin_module,
            ),
        )
    )
    monkeypatch.setattr(metadata, "distributions", lambda: [distribution])

    with pytest.raises(
        PluginHookError,
        match=r"failing-hook-plugin.*ocr_engines.*plugin setup failed",
    ) as exc_info:
        get_ocr_factory(allow_external_plugins=True)

    assert exc_info.value.__cause__ is hook_error
