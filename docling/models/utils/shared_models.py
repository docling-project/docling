# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Share one loaded model between stages that load the same weights.

Two stages can load the same model folder with the same load options. Each stage
then holds its own copy of the weights in memory. ``load_shared_model`` returns the
copy that is already loaded. The registry holds weak references, so a model is freed
when its last user drops it.

Callers must not change the returned model. Do not call ``.to()``, do not change the
dtype and do not write to ``generation_config``. Call ``.eval()`` only, because it is
idempotent. Keep the processor and the generation config per user.
"""

import logging
import threading
import weakref
from pathlib import Path
from typing import Any

import torch
from transformers import PreTrainedModel

_log = logging.getLogger(__name__)

_DTYPE_KWARGS = ("dtype", "torch_dtype")

_SharedKey = tuple[type[Any], Path, tuple[tuple[str, bool | int | float | str], ...]]

_registry: weakref.WeakValueDictionary[_SharedKey, PreTrainedModel] = (
    weakref.WeakValueDictionary()
)
_registry_lock = threading.Lock()


def _normalize_dtype(value: Any) -> Any:
    """Give ``torch.bfloat16``, ``"torch.bfloat16"`` and ``"bfloat16"`` one value."""
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    if isinstance(value, str):
        return value.removeprefix("torch.")
    return value


def _build_key(
    model_cls: type[Any], folder: Path, kwargs: dict[str, Any]
) -> _SharedKey | None:
    """Build the registry key, or return None when the load must not be shared.

    ``revision`` is not in the key: the local folder pins the weights.
    ``None`` values are not in the key: ``from_pretrained`` treats them as omitted.
    """
    normalized: dict[str, Any] = {}
    for name, value in kwargs.items():
        if value is None or name == "revision":
            continue
        if name in _DTYPE_KWARGS:
            name = "dtype"
            value = _normalize_dtype(value)
        if not isinstance(value, (bool, int, float, str)):
            return None
        if name in normalized and normalized[name] != value:
            return None
        normalized[name] = value
    return (model_cls, folder, tuple(sorted(normalized.items())))


def load_shared_model(
    model_cls: type[Any], model_path: Path | str, **kwargs: Any
) -> PreTrainedModel:
    """Load a model like ``model_cls.from_pretrained(model_path, **kwargs)``.

    Return the model that is already loaded when the model class, the resolved local
    folder and the keyword arguments all match. Load without sharing when
    ``model_path`` is not an existing local folder, or when a keyword argument value
    is not a bool, int, float or str (for example a ``BitsAndBytesConfig``).
    """
    folder = Path(model_path).resolve()
    key = _build_key(model_cls, folder, kwargs) if folder.is_dir() else None
    if key is None:
        return model_cls.from_pretrained(model_path, **kwargs)

    with _registry_lock:
        model = _registry.get(key)
        if model is not None:
            _log.debug(f"Reusing the loaded {model_cls.__name__} from {folder}")
            return model
        model = model_cls.from_pretrained(model_path, **kwargs)
        _registry[key] = model
        return model
