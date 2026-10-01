import pytest

from docling.utils import torch_dtype as td
from docling.utils.torch_dtype import resolve_torch_dtype


@pytest.fixture
def no_native_bf16(monkeypatch):
    monkeypatch.setattr(td, "cpu_has_native_bfloat16", lambda: False)


@pytest.fixture
def native_bf16(monkeypatch):
    monkeypatch.setattr(td, "cpu_has_native_bfloat16", lambda: True)


def test_supported_dtype_is_kept(native_bf16):
    assert resolve_torch_dtype("bfloat16", "float32", "cpu") == "bfloat16"
    assert resolve_torch_dtype("float32", "float16", "cpu") == "float32"


def test_accelerators_keep_dtype(no_native_bf16):
    assert resolve_torch_dtype("bfloat16", "float32", "cuda:0") == "bfloat16"
    assert resolve_torch_dtype("float16", "float32", "mps") == "float16"


def test_unsupported_bf16_on_cpu_falls_back(no_native_bf16):
    assert resolve_torch_dtype("bfloat16", "float32", "cpu") == "float32"


def test_float16_on_cpu_falls_back(native_bf16):
    assert resolve_torch_dtype("float16", "float32", "cpu") == "float32"


def test_explicit_fallback_is_honoured(no_native_bf16):
    assert resolve_torch_dtype("bfloat16", "float16", "cpu") == "float16"


def test_no_fallback_keeps_configured_dtype(no_native_bf16):
    assert resolve_torch_dtype("bfloat16", None, "cpu") == "bfloat16"
    assert resolve_torch_dtype(None, "float32", "cpu") is None
