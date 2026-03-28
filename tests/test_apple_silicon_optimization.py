"""Tests for Apple Silicon optimization: TableFormer MPS and VLM auto-selection.

Validates:
1. TableFormer models no longer override MPS to CPU
2. VLM auto-selecting constants pick MLX on Apple Silicon, Transformers otherwise
3. _has_apple_silicon_mlx() helper detects hardware correctly
"""

import sys

import pytest

from docling.datamodel.accelerator_options import AcceleratorDevice
from docling.datamodel.pipeline_options_vlm_model import InferenceFramework
from docling.datamodel.vlm_model_specs import (
    GRANITEDOCLING,
    GRANITEDOCLING_MLX,
    GRANITEDOCLING_TRANSFORMERS,
    SMOLDOCLING,
    SMOLDOCLING_MLX,
    SMOLDOCLING_TRANSFORMERS,
)
from docling.utils.accelerator_utils import decide_device


class TestTableFormerMpsSupport:
    """Verify TableFormer models support MPS device selection."""

    def test_decide_device_allows_mps(self):
        """decide_device with MPS in supported_devices returns 'mps' when available."""
        import torch

        if not (torch.backends.mps.is_built() and torch.backends.mps.is_available()):
            pytest.skip("MPS not available on this machine")

        device = decide_device(
            AcceleratorDevice.AUTO,
            supported_devices=[
                AcceleratorDevice.CPU,
                AcceleratorDevice.CUDA,
                AcceleratorDevice.MPS,
                AcceleratorDevice.XPU,
            ],
        )
        # On Apple Silicon without CUDA, AUTO should resolve to mps
        assert device == "mps"

    def test_decide_device_mps_explicit(self):
        """Explicitly requesting MPS with MPS in supported_devices returns 'mps'."""
        import torch

        if not (torch.backends.mps.is_built() and torch.backends.mps.is_available()):
            pytest.skip("MPS not available on this machine")

        device = decide_device(
            AcceleratorDevice.MPS,
            supported_devices=[
                AcceleratorDevice.CPU,
                AcceleratorDevice.CUDA,
                AcceleratorDevice.MPS,
                AcceleratorDevice.XPU,
            ],
        )
        assert device == "mps"

    def test_decide_device_cpu_fallback(self):
        """CPU is always a valid fallback."""
        device = decide_device(
            AcceleratorDevice.CPU,
            supported_devices=[
                AcceleratorDevice.CPU,
                AcceleratorDevice.CUDA,
                AcceleratorDevice.MPS,
                AcceleratorDevice.XPU,
            ],
        )
        assert device == "cpu"


class TestVlmAutoSelection:
    """Verify VLM auto-selecting constants and factory functions."""

    def test_auto_selecting_constants_exist(self):
        """GRANITEDOCLING and SMOLDOCLING auto-selecting constants are importable."""
        assert GRANITEDOCLING is not None
        assert SMOLDOCLING is not None
        assert hasattr(GRANITEDOCLING, "inference_framework")
        assert hasattr(SMOLDOCLING, "inference_framework")

    def test_explicit_constants_unchanged(self):
        """Explicit MLX and Transformers constants remain intact."""
        assert GRANITEDOCLING_TRANSFORMERS.inference_framework == InferenceFramework.TRANSFORMERS
        assert GRANITEDOCLING_MLX.inference_framework == InferenceFramework.MLX
        assert SMOLDOCLING_TRANSFORMERS.inference_framework == InferenceFramework.TRANSFORMERS
        assert SMOLDOCLING_MLX.inference_framework == InferenceFramework.MLX

    def test_explicit_constants_repo_ids(self):
        """Explicit constants have correct repo IDs."""
        assert GRANITEDOCLING_TRANSFORMERS.repo_id == "ibm-granite/granite-docling-258M"
        assert GRANITEDOCLING_MLX.repo_id == "ibm-granite/granite-docling-258M-mlx"
        assert "SmolDocling" in SMOLDOCLING_TRANSFORMERS.repo_id
        assert "SmolDocling" in SMOLDOCLING_MLX.repo_id

    def test_selectors_mlx_path(self, monkeypatch):
        """Factory functions return MLX variants when MPS and mlx-vlm are available."""
        from docling.datamodel import vlm_model_specs as specs

        class _Mps:
            def is_built(self):
                return True

            def is_available(self):
                return True

        class _Torch:
            class backends:
                mps = _Mps()

        monkeypatch.setitem(sys.modules, "torch", _Torch())
        monkeypatch.setitem(sys.modules, "mlx_vlm", object())

        granite = specs._get_granitedocling_model()
        smol = specs._get_smoldocling_model()

        assert granite.inference_framework == InferenceFramework.MLX
        assert granite.repo_id == "ibm-granite/granite-docling-258M-mlx"
        assert smol.inference_framework == InferenceFramework.MLX
        assert "mlx" in smol.repo_id

    def test_selectors_transformers_fallback(self, monkeypatch):
        """Factory functions return Transformers variants when MPS is unavailable."""
        from docling.datamodel import vlm_model_specs as specs

        class _MpsOff:
            def is_built(self):
                return False

            def is_available(self):
                return False

        class _TorchOff:
            class backends:
                mps = _MpsOff()

        monkeypatch.setitem(sys.modules, "torch", _TorchOff())
        if "mlx_vlm" in sys.modules:
            del sys.modules["mlx_vlm"]

        granite = specs._get_granitedocling_model()
        smol = specs._get_smoldocling_model()

        assert granite.inference_framework == InferenceFramework.TRANSFORMERS
        assert granite.repo_id == "ibm-granite/granite-docling-258M"
        assert smol.inference_framework == InferenceFramework.TRANSFORMERS
        assert "preview" in smol.repo_id
        assert "mlx" not in smol.repo_id

    def test_selectors_no_mlx_vlm_installed(self, monkeypatch):
        """Factory functions fall back to Transformers when mlx-vlm is not installed."""
        from docling.datamodel import vlm_model_specs as specs

        # MPS available but mlx-vlm not installed
        class _Mps:
            def is_built(self):
                return True

            def is_available(self):
                return True

        class _Torch:
            class backends:
                mps = _Mps()

        monkeypatch.setitem(sys.modules, "torch", _Torch())
        # Setting to None causes ImportError on `import mlx_vlm`
        monkeypatch.setitem(sys.modules, "mlx_vlm", None)

        granite = specs._get_granitedocling_model()
        assert granite.inference_framework == InferenceFramework.TRANSFORMERS

    def test_selectors_torch_import_error(self, monkeypatch):
        """If torch cannot be imported, selectors return Transformers variant."""
        from docling.datamodel import vlm_model_specs as specs

        # Remove torch to simulate ImportError
        monkeypatch.setitem(sys.modules, "torch", None)
        if "mlx_vlm" in sys.modules:
            del sys.modules["mlx_vlm"]

        granite = specs._get_granitedocling_model()
        assert granite.inference_framework == InferenceFramework.TRANSFORMERS


class TestHasAppleSiliconMlx:
    """Test the _has_apple_silicon_mlx() shared helper."""

    def test_returns_true_when_both_available(self, monkeypatch):
        """Returns True when MPS is available and mlx-vlm is installed."""
        from docling.datamodel import vlm_model_specs as specs

        class _Mps:
            def is_built(self):
                return True

            def is_available(self):
                return True

        class _Torch:
            class backends:
                mps = _Mps()

        monkeypatch.setitem(sys.modules, "torch", _Torch())
        monkeypatch.setitem(sys.modules, "mlx_vlm", object())

        assert specs._has_apple_silicon_mlx() is True

    def test_returns_false_no_mps(self, monkeypatch):
        """Returns False when MPS is not available."""
        from docling.datamodel import vlm_model_specs as specs

        class _MpsOff:
            def is_built(self):
                return False

            def is_available(self):
                return False

        class _TorchOff:
            class backends:
                mps = _MpsOff()

        monkeypatch.setitem(sys.modules, "torch", _TorchOff())
        monkeypatch.setitem(sys.modules, "mlx_vlm", object())

        assert specs._has_apple_silicon_mlx() is False

    def test_returns_false_no_mlx_vlm(self, monkeypatch):
        """Returns False when mlx-vlm is not installed."""
        from docling.datamodel import vlm_model_specs as specs

        class _Mps:
            def is_built(self):
                return True

            def is_available(self):
                return True

        class _Torch:
            class backends:
                mps = _Mps()

        monkeypatch.setitem(sys.modules, "torch", _Torch())
        # Setting to None causes ImportError on `import mlx_vlm`
        monkeypatch.setitem(sys.modules, "mlx_vlm", None)

        assert specs._has_apple_silicon_mlx() is False

    def test_returns_false_no_torch(self, monkeypatch):
        """Returns False when torch is not installed."""
        from docling.datamodel import vlm_model_specs as specs

        monkeypatch.setitem(sys.modules, "torch", None)

        assert specs._has_apple_silicon_mlx() is False
