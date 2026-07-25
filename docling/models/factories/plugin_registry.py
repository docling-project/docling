from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from importlib import metadata

_INTERNAL_DISTRIBUTIONS = frozenset({"docling-slim"})


class PluginDiscoveryError(RuntimeError):
    """Installed plugin metadata does not define a deterministic registry."""


class PluginLoadError(RuntimeError):
    """An allowed plugin entry point could not be imported."""


@dataclass(frozen=True, slots=True)
class _PluginEntryPoint:
    name: str
    module_name: str
    value: str
    distribution_name: str
    load: Callable[[], object]


@dataclass(frozen=True, slots=True)
class PluginModule:
    name: str
    module_name: str
    distribution_name: str
    module: object


@cache
def load_plugin_modules(
    group: str,
    *,
    allow_external_plugins: bool,
) -> tuple[PluginModule, ...]:
    """Load each allowed plugin entry point once, preserving discovery order."""
    return tuple(
        _load_plugin(entry_point)
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
    discovered_entry_points: list[_PluginEntryPoint] = []

    for distribution in metadata.distributions():
        distribution_name = distribution.metadata["Name"]
        for entry_point in distribution.entry_points:
            if entry_point.group != group:
                continue
            if not allow_external_plugins and not is_internal_plugin_distribution(
                distribution_name
            ):
                continue

            discovered_entry_points.append(
                _PluginEntryPoint(
                    name=entry_point.name,
                    module_name=entry_point.module,
                    value=entry_point.value,
                    distribution_name=distribution_name,
                    load=entry_point.load,
                )
            )

    discovered_entry_points.sort(
        key=lambda entry_point: (
            entry_point.name,
            entry_point.distribution_name,
            entry_point.value,
        )
    )
    entry_points_by_name: dict[str, _PluginEntryPoint] = {}
    for discovered in discovered_entry_points:
        registered = entry_points_by_name.get(discovered.name)
        if registered is not None:
            raise PluginDiscoveryError(
                f"Plugin entry point {discovered.name!r} is provided by both "
                f"{registered.distribution_name!r} ({registered.value!r}) and "
                f"{discovered.distribution_name!r} ({discovered.value!r}). "
                "Entry point names must be unique."
            )
        entry_points_by_name[discovered.name] = discovered

    return tuple(entry_points_by_name.values())


def _load_plugin(entry_point: _PluginEntryPoint) -> PluginModule:
    try:
        plugin_module = entry_point.load()
    except Exception as exc:
        raise PluginLoadError(
            f"Could not load plugin entry point {entry_point.name!r} "
            f"from {entry_point.module_name!r}: {exc}"
        ) from exc

    return PluginModule(
        name=entry_point.name,
        module_name=entry_point.module_name,
        distribution_name=entry_point.distribution_name,
        module=plugin_module,
    )


def is_internal_plugin_distribution(distribution_name: str) -> bool:
    """Return whether a plugin entry point comes from Docling's own package."""
    normalized_name = distribution_name.casefold().replace("_", "-")
    return normalized_name in _INTERNAL_DISTRIBUTIONS
