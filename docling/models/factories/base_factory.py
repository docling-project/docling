# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import enum
import logging
from abc import ABCMeta
from collections.abc import Iterable, Mapping
from importlib.metadata import entry_points
from typing import Generic, Optional, Type, TypeVar

from pluggy import PluginManager
from pydantic import BaseModel

from docling.datamodel.pipeline_options import BaseOptions
from docling.models.base_model import BaseModelWithOptions

A = TypeVar("A", bound=BaseModelWithOptions)


logger = logging.getLogger(__name__)


class FactoryMeta(BaseModel):
    kind: str
    plugin_name: str
    module: str
    distribution: str | None = None


class BaseFactory(Generic[A], metaclass=ABCMeta):
    default_plugin_name = "docling"

    def __init__(self, plugin_attr_name: str, plugin_name=default_plugin_name):
        self.plugin_name = plugin_name
        self.plugin_attr_name = plugin_attr_name

        self._classes: dict[Type[BaseOptions], Type[A]] = {}
        self._meta: dict[Type[BaseOptions], FactoryMeta] = {}

    @property
    def registered_kind(self) -> list[str]:
        return [opt.kind for opt in self._classes.keys()]

    def get_enum(self) -> enum.Enum:
        return enum.Enum(
            self.plugin_attr_name + "_enum",
            names={kind: kind for kind in self.registered_kind},
            type=str,
            module=__name__,
        )

    @property
    def classes(self):
        return self._classes

    @property
    def registered_meta(self) -> dict[Type[BaseOptions], FactoryMeta]:
        return self._meta

    def create_instance(self, options: BaseOptions, **kwargs) -> A:
        try:
            _cls = self._classes[type(options)]
            return _cls(options=options, **kwargs)
        except KeyError:
            raise RuntimeError(self._err_msg_on_class_not_found(options.kind))

    def create_options(self, kind: str, *args, **kwargs) -> BaseOptions:
        for opt_cls, _ in self._classes.items():
            if opt_cls.kind == kind:
                return opt_cls(*args, **kwargs)
        raise RuntimeError(self._err_msg_on_class_not_found(kind))

    def _err_msg_on_class_not_found(self, kind: str):
        msg = []

        for opt, cls in self._classes.items():
            msg.append(f"\t{opt.kind!r} => {cls!r}")

        msg_str = "\n".join(msg)

        return f"No class found with the name {kind!r}, known classes are:\n{msg_str}"

    def register(
        self,
        cls: Type[A],
        plugin_name: str,
        plugin_module_name: str,
        *,
        distribution: str | None = None,
    ) -> None:
        opt_type = cls.get_options_type()

        if opt_type in self._classes:
            raise ValueError(
                f"{opt_type.kind!r} already registered to class {self._classes[opt_type]!r}"
            )

        self._classes[opt_type] = cls
        self._meta[opt_type] = FactoryMeta(
            kind=opt_type.kind,
            plugin_name=plugin_name,
            module=plugin_module_name,
            distribution=distribution,
        )

    def load_from_plugins(
        self, plugin_name: Optional[str] = None, allow_external_plugins: bool = False
    ) -> None:
        plugin_name = plugin_name or self.plugin_name

        plugin_manager = PluginManager(plugin_name)
        plugin_distributions: dict[str, str | None] = {}

        # Decide from the entry point metadata whether a plugin is allowed
        # before importing it, so that disallowed plugin modules are never
        # imported.
        for entry_point in entry_points(group=plugin_name):
            if plugin_manager.get_plugin(entry_point.name) is not None:
                continue

            if not allow_external_plugins and not entry_point.module.startswith(
                "docling."
            ):
                logger.warning(
                    f"The plugin {entry_point.name} will not be loaded because Docling is being executed with allow_external_plugins=false."
                )
                continue

            plugin_manager.register(entry_point.load(), name=entry_point.name)
            plugin_distributions[entry_point.name] = (
                entry_point.dist.name if entry_point.dist is not None else None
            )

        for plugin_name, plugin_module in plugin_manager.list_name_plugin():
            plugin_module_name = str(plugin_module.__name__)  # type: ignore

            attr = getattr(plugin_module, self.plugin_attr_name, None)

            if callable(attr):
                logger.info("Loading plugin %r", plugin_name)

                config = attr()
                self.process_plugin(
                    config,
                    plugin_name,
                    plugin_module_name,
                    distribution=plugin_distributions[plugin_name],
                )

    def process_plugin(
        self,
        config: Mapping[str, Iterable[Type[A]]],
        plugin_name: str,
        plugin_module_name: str,
        *,
        distribution: str | None = None,
    ) -> None:
        for item in config[self.plugin_attr_name]:
            try:
                self.register(
                    item, plugin_name, plugin_module_name, distribution=distribution
                )
            except ValueError:
                logger.warning("%r already registered", item)
