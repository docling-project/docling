# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Regression tests for version-aware dtype kwargs in Transformers model loaders."""

import importlib
import importlib.machinery
import importlib.metadata
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from packaging import version

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.extraction_options import ExtractionPromptStyle
from docling.datamodel.pipeline_options import (
    GraniteVisionTableStructureOptions,
    PictureDescriptionVlmOptions,
)
from docling.datamodel.pipeline_options_vlm_model import (
    InferenceFramework,
    InlineVlmOptions,
    ResponseFormat,
    TransformersModelType,
)
from docling.datamodel.vlm_model_specs import (
    GRANITE_VISION_4_1_TRANSFORMERS,
    NU_EXTRACT_2B_TRANSFORMERS,
)
from docling.utils.transformers_utils import transformers_dtype_kwargs


class _DummyTokenizer:
    def __init__(self) -> None:
        self.padding_side = "right"
        self.model_max_length = 4096


class _DummyProcessor:
    def __init__(self) -> None:
        self.tokenizer = _DummyTokenizer()


class _DummyModel:
    def eval(self) -> "_DummyModel":
        return self


def _ensure_torch(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, bool]:
    """Return ``(torch_module, is_stub)``, installing a minimal stub in slim envs."""
    if importlib.util.find_spec("torch") is not None:
        import torch

        monkeypatch.setattr(torch, "compile", lambda model, *args, **kwargs: model)
        return torch, False

    # Let transformers finish its package-level torch probe before installing
    # a stub module in sys.modules (find_spec raises if __spec__ is None).
    importlib.import_module("transformers")

    fake_torch = types.ModuleType("torch")
    fake_torch.__spec__ = importlib.machinery.ModuleSpec("torch", loader=None)
    fake_torch.bfloat16 = "bfloat16_sentinel"  # type: ignore[attr-defined]
    fake_torch.BoolTensor = Any  # type: ignore[attr-defined]
    fake_torch.compile = lambda model, *args, **kwargs: model  # type: ignore[attr-defined]
    fake_torch.backends = types.SimpleNamespace(  # type: ignore[attr-defined]
        cuda=types.SimpleNamespace(is_built=lambda: False),
        mps=types.SimpleNamespace(is_built=lambda: False, is_available=lambda: False),
    )
    fake_torch.cuda = types.SimpleNamespace(  # type: ignore[attr-defined]
        is_available=lambda: False,
        device_count=lambda: 0,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    return fake_torch, True


def _import_with_torch(
    monkeypatch: pytest.MonkeyPatch, module_name: str
) -> tuple[Any, Any]:
    """Import ``module_name`` with real or stub torch without leaking stubbed modules."""
    torch_mod, is_stub = _ensure_torch(monkeypatch)
    if not is_stub:
        return torch_mod, importlib.import_module(module_name)

    before = set(sys.modules)
    mod = importlib.import_module(module_name)
    for added in set(sys.modules) - before:
        if added.startswith("docling.models."):
            monkeypatch.delitem(sys.modules, added, raising=False)
    return torch_mod, mod


def _make_strict_model_cls(
    transformers_version: str,
    captured_kwargs: dict[str, Any],
) -> type:
    """Build a fake model class that mimics transformers v4/v5 dtype kwargs."""
    is_v5 = version.parse(transformers_version).major >= 5

    class _FakeModelCls:
        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any) -> _DummyModel:
            if not is_v5 and "dtype" in kwargs:
                raise TypeError(
                    "Qwen2VLForConditionalGeneration.__init__() got an unexpected "
                    "keyword argument 'dtype'"
                )
            if is_v5 and "torch_dtype" in kwargs:
                raise TypeError(
                    "Unexpected deprecated keyword argument 'torch_dtype' on "
                    "transformers>=5"
                )
            captured_kwargs.update(kwargs)
            return _DummyModel()

    return _FakeModelCls


def _mock_transformers_version(
    monkeypatch: pytest.MonkeyPatch, transformers_version: str
) -> None:
    real_version = importlib.metadata.version

    def _fake_version(package: str) -> str:
        if package == "transformers":
            return transformers_version
        return real_version(package)

    monkeypatch.setattr(importlib.metadata, "version", _fake_version)


@pytest.mark.parametrize(
    ("transformers_version", "expected_kwarg"),
    [
        ("4.42.0", "torch_dtype"),
        ("4.48.0", "torch_dtype"),
        ("4.55.4", "torch_dtype"),
        ("4.57.3", "torch_dtype"),
        ("5.0.0rc1", "dtype"),
        ("5.0.0", "dtype"),
        ("5.16.1", "dtype"),
    ],
)
def test_transformers_dtype_kwargs_helper(
    transformers_version: str,
    expected_kwarg: str,
) -> None:
    result = transformers_dtype_kwargs(
        "bfloat16", transformers_version=transformers_version
    )
    assert result == {expected_kwarg: "bfloat16"}


def test_transformers_dtype_kwargs_omits_none() -> None:
    assert transformers_dtype_kwargs(None, transformers_version="4.48.0") == {}
    assert transformers_dtype_kwargs(None, transformers_version="5.16.1") == {}


def test_transformers_dtype_kwargs_reads_installed_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_transformers_version(monkeypatch, "4.48.0")
    assert transformers_dtype_kwargs("float16") == {"torch_dtype": "float16"}

    _mock_transformers_version(monkeypatch, "5.16.1")
    assert transformers_dtype_kwargs("float16") == {"dtype": "float16"}


@pytest.mark.parametrize(
    ("transformers_version", "expected_kwarg"),
    [("4.48.0", "torch_dtype"), ("5.16.1", "dtype")],
)
def test_nuextract_model_uses_versioned_dtype_kwarg(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    transformers_version: str,
    expected_kwarg: str,
) -> None:
    _mock_transformers_version(monkeypatch, transformers_version)
    _, nuextract_mod = _import_with_torch(
        monkeypatch, "docling.models.extraction.nuextract_transformers_model"
    )

    captured_kwargs: dict[str, Any] = {}
    fake_model_cls = _make_strict_model_cls(transformers_version, captured_kwargs)

    monkeypatch.setattr(
        nuextract_mod.AutoProcessor,
        "from_pretrained",
        lambda *args, **kwargs: _DummyProcessor(),
    )
    monkeypatch.setattr(nuextract_mod, "AutoModelForImageTextToText", fake_model_cls)
    monkeypatch.setattr(
        nuextract_mod.GenerationConfig,
        "from_pretrained",
        lambda *args, **kwargs: object(),
    )

    nuextract_mod.NuExtractTransformersModel(
        enabled=True,
        artifacts_path=tmp_path,
        accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CPU),
        vlm_options=NU_EXTRACT_2B_TRANSFORMERS,
    )

    assert captured_kwargs[expected_kwarg] == "bfloat16"
    other_kwarg = "dtype" if expected_kwarg == "torch_dtype" else "torch_dtype"
    assert other_kwarg not in captured_kwargs


def test_nuextract_model_omits_dtype_when_none(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _mock_transformers_version(monkeypatch, "4.48.0")
    _, nuextract_mod = _import_with_torch(
        monkeypatch, "docling.models.extraction.nuextract_transformers_model"
    )

    captured_kwargs: dict[str, Any] = {}
    fake_model_cls = _make_strict_model_cls("4.48.0", captured_kwargs)

    monkeypatch.setattr(
        nuextract_mod.AutoProcessor,
        "from_pretrained",
        lambda *args, **kwargs: _DummyProcessor(),
    )
    monkeypatch.setattr(nuextract_mod, "AutoModelForImageTextToText", fake_model_cls)
    monkeypatch.setattr(
        nuextract_mod.GenerationConfig,
        "from_pretrained",
        lambda *args, **kwargs: object(),
    )

    vlm_options = NU_EXTRACT_2B_TRANSFORMERS.model_copy(update={"torch_dtype": None})
    nuextract_mod.NuExtractTransformersModel(
        enabled=True,
        artifacts_path=tmp_path,
        accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CPU),
        vlm_options=vlm_options,
    )

    assert "dtype" not in captured_kwargs
    assert "torch_dtype" not in captured_kwargs


@pytest.mark.parametrize(
    ("transformers_version", "expected_kwarg"),
    [("4.48.0", "torch_dtype"), ("5.16.1", "dtype")],
)
def test_transformers_extraction_model_uses_versioned_dtype_kwarg(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    transformers_version: str,
    expected_kwarg: str,
) -> None:
    _mock_transformers_version(monkeypatch, transformers_version)
    fake_torch, ext_mod = _import_with_torch(
        monkeypatch, "docling.models.extraction.transformers_extraction_model"
    )

    captured_kwargs: dict[str, Any] = {}
    fake_model_cls = _make_strict_model_cls(transformers_version, captured_kwargs)

    monkeypatch.setattr(
        ext_mod.AutoProcessor,
        "from_pretrained",
        lambda *args, **kwargs: _DummyProcessor(),
    )
    monkeypatch.setattr(ext_mod, "AutoModelForImageTextToText", fake_model_cls)
    monkeypatch.setattr(
        ext_mod.GenerationConfig,
        "from_pretrained",
        lambda *args, **kwargs: object(),
    )

    ext_mod.TransformersExtractionModel(
        enabled=True,
        artifacts_path=tmp_path,
        accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CPU),
        vlm_options=GRANITE_VISION_4_1_TRANSFORMERS.model_copy(
            update={"torch_dtype": None}
        ),
        prompt_style=ExtractionPromptStyle.GRANITE_VISION,
    )

    assert captured_kwargs[expected_kwarg] == fake_torch.bfloat16
    other_kwarg = "dtype" if expected_kwarg == "torch_dtype" else "torch_dtype"
    assert other_kwarg not in captured_kwargs


def test_transformers_extraction_model_nuextract_explicit_dtype_on_v4(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _mock_transformers_version(monkeypatch, "4.48.0")
    _, ext_mod = _import_with_torch(
        monkeypatch, "docling.models.extraction.transformers_extraction_model"
    )

    captured_kwargs: dict[str, Any] = {}
    fake_model_cls = _make_strict_model_cls("4.48.0", captured_kwargs)

    monkeypatch.setattr(
        ext_mod.AutoProcessor,
        "from_pretrained",
        lambda *args, **kwargs: _DummyProcessor(),
    )
    monkeypatch.setattr(ext_mod, "AutoModelForImageTextToText", fake_model_cls)
    monkeypatch.setattr(
        ext_mod.GenerationConfig,
        "from_pretrained",
        lambda *args, **kwargs: object(),
    )

    ext_mod.TransformersExtractionModel(
        enabled=True,
        artifacts_path=tmp_path,
        accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CPU),
        vlm_options=NU_EXTRACT_2B_TRANSFORMERS.model_copy(
            update={"torch_dtype": "float16"}
        ),
        prompt_style=ExtractionPromptStyle.NUEXTRACT,
    )

    assert captured_kwargs["torch_dtype"] == "float16"
    assert "dtype" not in captured_kwargs


@pytest.mark.parametrize(
    ("transformers_version", "expected_kwarg"),
    [("4.51.3", "torch_dtype"), ("5.16.1", "dtype")],
)
def test_hf_transformers_vlm_model_uses_versioned_dtype_kwarg(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    transformers_version: str,
    expected_kwarg: str,
) -> None:
    _mock_transformers_version(monkeypatch, transformers_version)
    transformers = pytest.importorskip("transformers")
    _, hf_vlm_mod = _import_with_torch(
        monkeypatch, "docling.models.vlm_pipeline_models.hf_transformers_model"
    )

    captured_kwargs: dict[str, Any] = {}
    fake_model_cls = _make_strict_model_cls(transformers_version, captured_kwargs)

    vlm_options = InlineVlmOptions(
        repo_id="org/test-vlm",
        prompt="Convert this page.",
        torch_dtype="bfloat16",
        inference_framework=InferenceFramework.TRANSFORMERS,
        transformers_model_type=TransformersModelType.AUTOMODEL_IMAGETEXTTOTEXT,
        response_format=ResponseFormat.PLAINTEXT,
    )
    (tmp_path / "org--test-vlm").mkdir()

    monkeypatch.setattr(
        transformers.AutoProcessor,
        "from_pretrained",
        lambda *args, **kwargs: _DummyProcessor(),
    )
    monkeypatch.setattr(
        transformers,
        "AutoModelForImageTextToText",
        fake_model_cls,
    )
    monkeypatch.setattr(
        transformers.GenerationConfig,
        "from_pretrained",
        lambda *args, **kwargs: object(),
    )

    hf_vlm_mod.HuggingFaceTransformersVlmModel(
        enabled=True,
        artifacts_path=tmp_path,
        accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CPU),
        vlm_options=vlm_options,
    )

    assert captured_kwargs[expected_kwarg] == "bfloat16"
    other_kwarg = "dtype" if expected_kwarg == "torch_dtype" else "torch_dtype"
    assert other_kwarg not in captured_kwargs


def test_hf_transformers_vlm_model_omits_dtype_when_none(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _mock_transformers_version(monkeypatch, "5.16.1")
    transformers = pytest.importorskip("transformers")
    _, hf_vlm_mod = _import_with_torch(
        monkeypatch, "docling.models.vlm_pipeline_models.hf_transformers_model"
    )

    captured_kwargs: dict[str, Any] = {}
    fake_model_cls = _make_strict_model_cls("5.16.1", captured_kwargs)

    vlm_options = InlineVlmOptions(
        repo_id="org/test-vlm",
        prompt="Convert this page.",
        torch_dtype=None,
        inference_framework=InferenceFramework.TRANSFORMERS,
        transformers_model_type=TransformersModelType.AUTOMODEL_IMAGETEXTTOTEXT,
        response_format=ResponseFormat.PLAINTEXT,
    )
    (tmp_path / "org--test-vlm").mkdir()

    monkeypatch.setattr(
        transformers.AutoProcessor,
        "from_pretrained",
        lambda *args, **kwargs: _DummyProcessor(),
    )
    monkeypatch.setattr(
        transformers,
        "AutoModelForImageTextToText",
        fake_model_cls,
    )
    monkeypatch.setattr(
        transformers.GenerationConfig,
        "from_pretrained",
        lambda *args, **kwargs: object(),
    )

    hf_vlm_mod.HuggingFaceTransformersVlmModel(
        enabled=True,
        artifacts_path=tmp_path,
        accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CPU),
        vlm_options=vlm_options,
    )

    assert "dtype" not in captured_kwargs
    assert "torch_dtype" not in captured_kwargs


@pytest.mark.parametrize(
    ("transformers_version", "expected_kwarg"),
    [("4.48.0", "torch_dtype"), ("5.16.1", "dtype")],
)
def test_picture_description_vlm_model_uses_versioned_dtype_kwarg(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    transformers_version: str,
    expected_kwarg: str,
) -> None:
    _mock_transformers_version(monkeypatch, transformers_version)
    transformers = pytest.importorskip("transformers")
    fake_torch, pic_mod = _import_with_torch(
        monkeypatch,
        "docling.models.stages.picture_description.picture_description_vlm_model",
    )
    PictureDescriptionVlmModel = pic_mod.PictureDescriptionVlmModel

    captured_kwargs: dict[str, Any] = {}
    fake_model_cls = _make_strict_model_cls(transformers_version, captured_kwargs)

    monkeypatch.setattr(
        transformers.AutoProcessor,
        "from_pretrained",
        lambda *args, **kwargs: _DummyProcessor(),
    )
    monkeypatch.setattr(
        transformers,
        "AutoModelForImageTextToText",
        fake_model_cls,
    )

    PictureDescriptionVlmModel(
        enabled=True,
        enable_remote_services=False,
        artifacts_path=tmp_path,
        options=PictureDescriptionVlmOptions(repo_id="org/pic-vlm"),
        accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CPU),
    )

    assert captured_kwargs[expected_kwarg] == fake_torch.bfloat16
    other_kwarg = "dtype" if expected_kwarg == "torch_dtype" else "torch_dtype"
    assert other_kwarg not in captured_kwargs


@pytest.mark.parametrize(
    ("transformers_version", "expected_kwarg"),
    [("4.48.0", "torch_dtype"), ("5.16.1", "dtype")],
)
def test_granite_vision_table_model_uses_versioned_dtype_kwarg(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    transformers_version: str,
    expected_kwarg: str,
) -> None:
    _mock_transformers_version(monkeypatch, transformers_version)
    fake_torch, gv_mod = _import_with_torch(
        monkeypatch,
        "docling.models.stages.table_structure.table_structure_model_granite_vision",
    )

    captured_kwargs: dict[str, Any] = {}
    fake_model_cls = _make_strict_model_cls(transformers_version, captured_kwargs)

    monkeypatch.setattr(
        gv_mod.AutoProcessor,
        "from_pretrained",
        lambda *args, **kwargs: _DummyProcessor(),
    )
    monkeypatch.setattr(gv_mod, "AutoModelForImageTextToText", fake_model_cls)

    gv_mod.GraniteVisionTableStructureModel(
        enabled=True,
        artifacts_path=tmp_path,
        options=GraniteVisionTableStructureOptions(),
        accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CPU),
    )

    assert captured_kwargs[expected_kwarg] == fake_torch.bfloat16
    other_kwarg = "dtype" if expected_kwarg == "torch_dtype" else "torch_dtype"
    assert other_kwarg not in captured_kwargs
