from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from importlib import metadata

_INTERNAL_MODULE = "docling"


class PluginDiscoveryError(RuntimeError):
    """Installed plugin metadata does not define a deterministic registry."""


@dataclass(frozen=True, slots=True)
class _PluginEntryPoint:
    name: str
    module_name: str
    load: Callable[[], object]


@dataclass(frozen=True, slots=True)
class PluginModule:
    name: str
    module_name: str
    module: object


@cache
def load_plugin_modules(
    group: str,
    *,
    allow_external_plugins: bool,
) -> tuple[PluginModule, ...]:
    """Load each allowed plugin entry point once, preserving discovery order."""
    return tuple(
        PluginModule(
            name=entry_point.name,
            module_name=entry_point.module_name,
            module=entry_point.load(),
        )
        for entry_point in _discover_entry_points(
            group,
            allow_external_plugins=allow_external_plugins,
        )
    )


def _discover_entry_points(
    group: str,
    *,
    allow_external_plugins: bool,
) -> tuple[_PluginEntryPoint, ...]:
    entry_points_by_name: dict[str, _PluginEntryPoint] = {}

    for distribution in metadata.distributions():
        for entry_point in distribution.entry_points:
            if entry_point.group != group:
                continue
            if not allow_external_plugins and not _is_internal(entry_point.module):
                continue

            discovered = _PluginEntryPoint(
                name=entry_point.name,
                module_name=entry_point.module,
                load=entry_point.load,
            )
            registered = entry_points_by_name.get(discovered.name)
            if registered is None:
                entry_points_by_name[discovered.name] = discovered
            elif registered.module_name != discovered.module_name:
                raise PluginDiscoveryError(
                    f"Plugin entry point {discovered.name!r} is provided by both "
                    f"{registered.module_name!r} and {discovered.module_name!r}. "
                    "Entry point names must be unique."
                )

    return tuple(entry_points_by_name.values())


def _is_internal(module_name: str) -> bool:
    return module_name == _INTERNAL_MODULE or module_name.startswith(
        f"{_INTERNAL_MODULE}."
    )
