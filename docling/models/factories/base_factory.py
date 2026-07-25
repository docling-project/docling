import enum
import logging
from abc import ABCMeta
from collections.abc import Mapping, Sequence
from typing import Generic, Literal, TypeVar, cast

from pydantic import BaseModel

from docling.datamodel.pipeline_options import BaseOptions
from docling.models.base_model import BaseModelWithOptions
from docling.models.factories.plugin_registry import load_plugin_modules

A = TypeVar("A", bound=BaseModelWithOptions)
PluginCapability = Literal[
    "layout_engines",
    "ocr_engines",
    "picture_description",
    "table_structure_engines",
]


logger = logging.getLogger(__name__)


class PluginConfigurationError(RuntimeError):
    """A plugin hook returned data outside Docling's plugin contract."""


class FactoryMeta(BaseModel):
    kind: str
    plugin_name: str
    module: str


class BaseFactory(Generic[A], metaclass=ABCMeta):
    default_plugin_name = "docling"

    def __init__(
        self,
        plugin_attr_name: PluginCapability,
        model_type: type[A],
        plugin_name: str = default_plugin_name,
    ) -> None:
        self.plugin_name = plugin_name
        self.plugin_attr_name = plugin_attr_name
        self.model_type = model_type

        self._classes: dict[type[BaseOptions], type[A]] = {}
        self._options_by_kind: dict[str, type[BaseOptions]] = {}
        self._meta: dict[type[BaseOptions], FactoryMeta] = {}

    @property
    def registered_kind(self) -> list[str]:
        return list(self._options_by_kind)

    def get_enum(self) -> type[enum.Enum]:
        return enum.Enum(
            self.plugin_attr_name + "_enum",
            names={kind: kind for kind in self.registered_kind},
            type=str,
            module=__name__,
        )

    @property
    def classes(self) -> Mapping[type[BaseOptions], type[A]]:
        return self._classes

    @property
    def registered_meta(self) -> Mapping[type[BaseOptions], FactoryMeta]:
        return self._meta

    def create_instance(self, options: BaseOptions, **kwargs) -> A:
        try:
            model_class = self._classes[type(options)]
        except KeyError:
            raise RuntimeError(self._err_msg_on_class_not_found(options.kind)) from None
        return model_class(options=options, **kwargs)

    def create_options(self, kind: str, *args, **kwargs) -> BaseOptions:
        try:
            options_type = self._options_by_kind[kind]
        except KeyError:
            raise RuntimeError(self._err_msg_on_class_not_found(kind)) from None
        return options_type(*args, **kwargs)

    def _err_msg_on_class_not_found(self, kind: str):
        msg = []

        for opt, cls in self._classes.items():
            msg.append(f"\t{opt.kind!r} => {cls!r}")

        msg_str = "\n".join(msg)

        return f"No class found with the name {kind!r}, known classes are:\n{msg_str}"

    def register(self, cls: type[A], plugin_name: str, plugin_module_name: str) -> None:
        self._register_models(
            (cls,),
            plugin_name=plugin_name,
            plugin_module_name=plugin_module_name,
        )

    def load_from_plugins(
        self, plugin_name: str | None = None, allow_external_plugins: bool = False
    ) -> None:
        plugin_name = plugin_name or self.plugin_name

        for plugin in load_plugin_modules(
            plugin_name,
            allow_external_plugins=allow_external_plugins,
        ):
            # Plugin hook names are the documented third-party interface.
            hook = getattr(plugin.module, self.plugin_attr_name, None)

            if hook is None:
                continue
            if not callable(hook):
                raise self._configuration_error(
                    plugin.name,
                    f"the {self.plugin_attr_name!r} hook must be callable",
                )

            logger.info("Loading plugin %r", plugin.name)
            config = hook()
            self.process_plugin(config, plugin.name, plugin.module_name)

    def process_plugin(
        self, config: object, plugin_name: str, plugin_module_name: str
    ) -> None:
        if not isinstance(config, Mapping):
            raise self._configuration_error(
                plugin_name, "the hook must return a mapping"
            )

        plugin_config = cast(Mapping[object, object], config)
        models = plugin_config.get(self.plugin_attr_name)
        if not isinstance(models, list):
            raise self._configuration_error(
                plugin_name,
                f"the {self.plugin_attr_name!r} value must be a list of model classes",
            )

        validated_models: list[type[A]] = []
        for index, model in enumerate(models):
            if not isinstance(model, type) or not issubclass(model, self.model_type):
                raise self._configuration_error(
                    plugin_name,
                    f"{self.plugin_attr_name!r} item {index} must be a "
                    f"{self.model_type.__name__} model class",
                )
            validated_models.append(cast(type[A], model))

        self._register_models(
            validated_models,
            plugin_name=plugin_name,
            plugin_module_name=plugin_module_name,
        )

    def _register_models(
        self,
        models: Sequence[type[A]],
        *,
        plugin_name: str,
        plugin_module_name: str,
    ) -> None:
        classes = self._classes.copy()
        options_by_kind = self._options_by_kind.copy()
        registrations: list[tuple[type[BaseOptions], type[A]]] = []

        for model in models:
            options_type = model.get_options_type()
            registered_model = classes.get(options_type)
            if registered_model is not None:
                raise self._configuration_error(
                    plugin_name,
                    f"{options_type.__name__} is already registered to "
                    f"{registered_model.__name__}, so it cannot also register "
                    f"{model.__name__}",
                )

            registered_options = options_by_kind.get(options_type.kind)
            if registered_options is not None:
                registered_model = classes[registered_options]
                raise self._configuration_error(
                    plugin_name,
                    f"model kind {options_type.kind!r} is already registered to "
                    f"{registered_model.__name__}, so it cannot also register "
                    f"{model.__name__}",
                )

            classes[options_type] = model
            options_by_kind[options_type.kind] = options_type
            registrations.append((options_type, model))

        for options_type, model in registrations:
            self._classes[options_type] = model
            self._options_by_kind[options_type.kind] = options_type
            self._meta[options_type] = FactoryMeta(
                kind=options_type.kind,
                plugin_name=plugin_name,
                module=plugin_module_name,
            )

    def _configuration_error(
        self, plugin_name: str, problem: str
    ) -> PluginConfigurationError:
        return PluginConfigurationError(
            f"Plugin {plugin_name!r} has an invalid {self.plugin_attr_name!r} "
            f"contract: {problem}."
        )
