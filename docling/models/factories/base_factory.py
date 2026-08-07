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


class PluginHookError(RuntimeError):
    """A plugin hook failed while declaring its models."""


class FactoryMeta(BaseModel):
    kind: str
    plugin_name: str
    module: str


class BaseFactory(Generic[A], metaclass=ABCMeta):
    default_plugin_name = "docling"
    model_type: type[BaseModelWithOptions] | None = None

    def __init__(
        self,
        plugin_attr_name: PluginCapability,
        plugin_name: str = default_plugin_name,
    ) -> None:
        self.plugin_name = plugin_name
        self.plugin_attr_name = plugin_attr_name

        self._classes: dict[type[BaseOptions], type[A]] = {}
        self._options_by_kind: dict[str, type[BaseOptions]] = {}
        self._meta: dict[type[BaseOptions], FactoryMeta] = {}
        self._model_contracts: dict[type[A], tuple[type[BaseOptions], str]] = {}

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

    def register(
        self,
        cls: type[A],
        plugin_name: str,
        plugin_module_name: str,
    ) -> None:
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
            try:
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
                try:
                    config = hook()
                except Exception as exc:
                    raise PluginHookError(
                        f"Plugin {plugin.name!r} failed while running its "
                        f"{self.plugin_attr_name!r} hook: {exc}"
                    ) from exc
                self.process_plugin(
                    config,
                    plugin.name,
                    plugin.module_name,
                )
            except PluginConfigurationError as exc:
                logger.warning("%s Skipping this plugin.", exc)

    def process_plugin(
        self,
        config: object,
        plugin_name: str,
        plugin_module_name: str,
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
            if not isinstance(model, type) or (
                self.model_type is not None and not issubclass(model, self.model_type)
            ):
                expected_model = (
                    self.model_type.__name__
                    if self.model_type is not None
                    else "BaseModelWithOptions"
                )
                raise self._configuration_error(
                    plugin_name,
                    f"{self.plugin_attr_name!r} item {index} must be a "
                    f"{expected_model} model class",
                )
            validated_models.append(cast(type[A], model))

        self._validate_registrations(validated_models, plugin_name)
        for model in validated_models:
            self.register(model, plugin_name, plugin_module_name)

    def _register_models(
        self,
        models: Sequence[type[A]],
        *,
        plugin_name: str,
        plugin_module_name: str,
    ) -> None:
        self._validate_registrations(models, plugin_name)
        for model in models:
            options_type, kind = self._validate_options_type(model, plugin_name)
            self._classes[options_type] = model
            self._options_by_kind[kind] = options_type
            self._meta[options_type] = FactoryMeta(
                kind=kind,
                plugin_name=plugin_name,
                module=plugin_module_name,
            )

    def _validate_registrations(
        self, models: Sequence[type[A]], plugin_name: str
    ) -> None:
        classes = self._classes.copy()
        options_by_kind = self._options_by_kind.copy()

        for model in models:
            options_type, kind = self._validate_options_type(model, plugin_name)
            registered_model = classes.get(options_type)
            if registered_model is not None:
                raise self._configuration_error(
                    plugin_name,
                    f"{options_type.__name__} is already registered to "
                    f"{registered_model.__name__}, so it cannot also register "
                    f"{model.__name__}",
                )

            registered_options = options_by_kind.get(kind)
            if registered_options is not None:
                registered_model = classes[registered_options]
                raise self._configuration_error(
                    plugin_name,
                    f"model kind {kind!r} is already registered to "
                    f"{registered_model.__name__}, so it cannot also register "
                    f"{model.__name__}",
                )

            classes[options_type] = model
            options_by_kind[kind] = options_type

    def _validate_options_type(
        self, model: type[A], plugin_name: str
    ) -> tuple[type[BaseOptions], str]:
        cached_contract = self._model_contracts.get(model)
        if cached_contract is not None:
            return cached_contract

        try:
            options_type = model.get_options_type()
        except Exception as exc:
            raise self._configuration_error(
                plugin_name,
                f"{model.__name__}.get_options_type() failed: {exc}",
            ) from exc

        if not isinstance(options_type, type) or not issubclass(
            options_type, BaseOptions
        ):
            raise self._configuration_error(
                plugin_name,
                f"{model.__name__}.get_options_type() must return a "
                "BaseOptions subclass",
            )

        kind = vars(options_type).get("kind")
        if not isinstance(kind, str) or not kind:
            raise self._configuration_error(
                plugin_name,
                f"{options_type.__name__} must declare a non-empty string kind",
            )
        contract = (options_type, kind)
        self._model_contracts[model] = contract
        return contract

    def _configuration_error(
        self, plugin_name: str, problem: str
    ) -> PluginConfigurationError:
        return PluginConfigurationError(
            f"Plugin {plugin_name!r} has an invalid {self.plugin_attr_name!r} "
            f"contract: {problem}."
        )
