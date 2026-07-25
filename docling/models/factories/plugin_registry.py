from dataclasses import dataclass
from functools import cache
from importlib import metadata

_INTERNAL_MODULE = "docling"


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
    loaded_names: set[str] = set()
    plugins: list[PluginModule] = []

    for distribution in metadata.distributions():
        for entry_point in distribution.entry_points:
            if entry_point.group != group or entry_point.name in loaded_names:
                continue
            if not allow_external_plugins and not _is_internal(entry_point.module):
                continue

            loaded_names.add(entry_point.name)
            plugins.append(
                PluginModule(
                    name=entry_point.name,
                    module_name=entry_point.module,
                    module=entry_point.load(),
                )
            )

    return tuple(plugins)


def _is_internal(module_name: str) -> bool:
    return module_name == _INTERNAL_MODULE or module_name.startswith(
        f"{_INTERNAL_MODULE}."
    )
