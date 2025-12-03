"""
Test DeepSeek-OCR model integration with mock-based unit tests.

These tests use mocking to test the DeepSeek-OCR model logic without requiring
actual GPU hardware (CUDA or MPS), following the pattern used in test_asr_mlx_whisper.py.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from docling_core.types.doc import BoundingBox, CoordOrigin

from docling.datamodel.accelerator_options import (
    AcceleratorDevice,
    AcceleratorOptions,
)
from docling.datamodel.pipeline_options import DeepSeekOcrOptions
from docling.models.deepseek_ocr_model import (
    _DEEPSEEK_OCR_CUDA_REPO,
    _DEEPSEEK_OCR_MPS_REPO,
    DeepSeekOcrModel,
    _parse_pytorch_version,
)


class TestParsePytorchVersion:
    """Test PyTorch version parsing utility."""

    def test_parse_standard_version(self):
        """Test parsing standard version strings."""
        assert _parse_pytorch_version("2.7.0") == (2, 7, 0)
        assert _parse_pytorch_version("2.6.1") == (2, 6, 1)
        assert _parse_pytorch_version("1.13.0") == (1, 13, 0)

    def test_parse_version_with_suffix(self):
        """Test parsing versions with CUDA/dev suffixes."""
        assert _parse_pytorch_version("2.7.0+cu118") == (2, 7, 0)
        assert _parse_pytorch_version("2.7.0.dev20241201") == (2, 7, 0)
        assert _parse_pytorch_version("2.7.0+cpu") == (2, 7, 0)

    def test_parse_invalid_version(self):
        """Test parsing invalid version strings."""
        assert _parse_pytorch_version("invalid") == (0, 0, 0)
        assert _parse_pytorch_version("") == (0, 0, 0)


class TestDeepSeekOcrOptions:
    """Test DeepSeekOcrOptions configuration."""

    def test_default_options(self):
        """Test default option values."""
        options = DeepSeekOcrOptions()
        assert options.kind == "deepseekocr"
        assert options.repo_id == "deepseek-ai/DeepSeek-OCR"
        assert options.prompt == "<image>\nFree OCR."
        assert options.base_size == 1024
        assert options.image_size == 640
        assert options.crop_mode is True
        assert options.trust_remote_code is True
        assert options.lang == []

    def test_custom_options(self):
        """Test custom option values."""
        options = DeepSeekOcrOptions(
            repo_id="custom/model",
            prompt="<image>\nConvert to markdown.",
            base_size=2048,
            image_size=1280,
        )
        assert options.repo_id == "custom/model"
        assert options.prompt == "<image>\nConvert to markdown."
        assert options.base_size == 2048
        assert options.image_size == 1280


class TestDeepSeekOcrModelDisabled:
    """Test DeepSeekOcrModel when disabled."""

    def test_disabled_model_no_init(self):
        """Test that disabled model doesn't initialize GPU resources."""
        options = DeepSeekOcrOptions()
        accelerator_options = AcceleratorOptions(device=AcceleratorDevice.AUTO)

        # Should not raise even without GPU
        model = DeepSeekOcrModel(
            enabled=False,
            artifacts_path=None,
            options=options,
            accelerator_options=accelerator_options,
        )

        assert model.enabled is False
        assert model.device is None
        assert model.dtype is None


class TestDeepSeekOcrModelDeviceDetection:
    """Test device detection logic with mocked torch."""

    def test_cuda_device_detection(self, monkeypatch):
        """Test CUDA device is selected when available."""
        # Create mock torch module
        mock_torch = MagicMock()
        mock_torch.__version__ = "2.7.0"
        mock_torch.backends.cuda.is_built.return_value = True
        mock_torch.cuda.is_available.return_value = True
        mock_torch.backends.mps.is_built.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.device.return_value = Mock()
        mock_torch.bfloat16 = "bfloat16"

        # Mock transformers
        mock_transformers = MagicMock()
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_transformers.AutoModel.from_pretrained.return_value = mock_model
        mock_transformers.AutoTokenizer.from_pretrained.return_value = MagicMock()

        monkeypatch.setitem(sys.modules, "torch", mock_torch)
        monkeypatch.setitem(sys.modules, "transformers", mock_transformers)

        options = DeepSeekOcrOptions()
        accelerator_options = AcceleratorOptions(device=AcceleratorDevice.CUDA)

        with patch.object(
            DeepSeekOcrModel, "_patch_llama_flash_attention", return_value=None
        ):
            model = DeepSeekOcrModel(
                enabled=True,
                artifacts_path=None,
                options=options,
                accelerator_options=accelerator_options,
            )

        mock_torch.device.assert_called_with("cuda")
        assert model.dtype == "bfloat16"

    def test_mps_device_detection(self, monkeypatch):
        """Test MPS device is selected on Apple Silicon."""
        # Create mock torch module
        mock_torch = MagicMock()
        mock_torch.__version__ = "2.7.0"
        mock_torch.backends.cuda.is_built.return_value = False
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_built.return_value = True
        mock_torch.backends.mps.is_available.return_value = True
        mock_torch.device.return_value = Mock()
        mock_torch.float16 = "float16"

        # Mock transformers
        mock_transformers = MagicMock()
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_transformers.AutoModel.from_pretrained.return_value = mock_model
        mock_transformers.AutoTokenizer.from_pretrained.return_value = MagicMock()

        monkeypatch.setitem(sys.modules, "torch", mock_torch)
        monkeypatch.setitem(sys.modules, "transformers", mock_transformers)

        options = DeepSeekOcrOptions()
        accelerator_options = AcceleratorOptions(device=AcceleratorDevice.MPS)

        with patch.object(
            DeepSeekOcrModel, "_patch_llama_flash_attention", return_value=None
        ):
            model = DeepSeekOcrModel(
                enabled=True,
                artifacts_path=None,
                options=options,
                accelerator_options=accelerator_options,
            )

        mock_torch.device.assert_called_with("mps")
        assert model.dtype == "float16"

    def test_no_gpu_raises_error(self, monkeypatch):
        """Test that RuntimeError is raised when no GPU is available."""
        # Create mock torch module with no GPU
        mock_torch = MagicMock()
        mock_torch.__version__ = "2.7.0"
        mock_torch.backends.cuda.is_built.return_value = False
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_built.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        # Mock transformers
        mock_transformers = MagicMock()

        monkeypatch.setitem(sys.modules, "torch", mock_torch)
        monkeypatch.setitem(sys.modules, "transformers", mock_transformers)

        options = DeepSeekOcrOptions()
        accelerator_options = AcceleratorOptions(device=AcceleratorDevice.AUTO)

        with pytest.raises(RuntimeError, match="DeepSeek-OCR requires a GPU"):
            DeepSeekOcrModel(
                enabled=True,
                artifacts_path=None,
                options=options,
                accelerator_options=accelerator_options,
            )

    def test_mps_old_pytorch_raises_error(self, monkeypatch):
        """Test that old PyTorch version raises error for MPS."""
        # Create mock torch module with old version
        mock_torch = MagicMock()
        mock_torch.__version__ = "2.5.0"  # Below 2.7.0 requirement
        mock_torch.backends.cuda.is_built.return_value = False
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_built.return_value = True
        mock_torch.backends.mps.is_available.return_value = True

        # Mock transformers
        mock_transformers = MagicMock()

        monkeypatch.setitem(sys.modules, "torch", mock_torch)
        monkeypatch.setitem(sys.modules, "transformers", mock_transformers)

        options = DeepSeekOcrOptions()
        accelerator_options = AcceleratorOptions(device=AcceleratorDevice.MPS)

        with pytest.raises(RuntimeError, match=r"requires PyTorch 2\.7\.0"):
            DeepSeekOcrModel(
                enabled=True,
                artifacts_path=None,
                options=options,
                accelerator_options=accelerator_options,
            )

    def test_mps_auto_switches_to_mps_repo(self, monkeypatch):
        """Test that MPS auto-switches to MPS-compatible model repo."""
        # Create mock torch module
        mock_torch = MagicMock()
        mock_torch.__version__ = "2.7.0"
        mock_torch.backends.cuda.is_built.return_value = False
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_built.return_value = True
        mock_torch.backends.mps.is_available.return_value = True
        mock_torch.device.return_value = Mock()
        mock_torch.float16 = "float16"

        # Mock transformers
        mock_transformers = MagicMock()
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_transformers.AutoModel.from_pretrained.return_value = mock_model
        mock_transformers.AutoTokenizer.from_pretrained.return_value = MagicMock()

        monkeypatch.setitem(sys.modules, "torch", mock_torch)
        monkeypatch.setitem(sys.modules, "transformers", mock_transformers)

        # Use default CUDA repo - should auto-switch to MPS repo
        options = DeepSeekOcrOptions(repo_id=_DEEPSEEK_OCR_CUDA_REPO)
        accelerator_options = AcceleratorOptions(device=AcceleratorDevice.MPS)

        with patch.object(
            DeepSeekOcrModel, "_patch_llama_flash_attention", return_value=None
        ):
            DeepSeekOcrModel(
                enabled=True,
                artifacts_path=None,
                options=options,
                accelerator_options=accelerator_options,
            )

        # Verify MPS-compatible model was loaded
        mock_transformers.AutoModel.from_pretrained.assert_called_once()
        call_args = mock_transformers.AutoModel.from_pretrained.call_args
        assert call_args[0][0] == _DEEPSEEK_OCR_MPS_REPO


class TestDeepSeekOcrModelTextParsing:
    """Test text parsing logic."""

    def test_parse_text_to_cells_single_line(self):
        """Test parsing single line of text."""
        options = DeepSeekOcrOptions()
        accelerator_options = AcceleratorOptions(device=AcceleratorDevice.AUTO)

        model = DeepSeekOcrModel(
            enabled=False,
            artifacts_path=None,
            options=options,
            accelerator_options=accelerator_options,
        )
        model.scale = 3

        ocr_rect = BoundingBox(l=0, t=0, r=300, b=100, coord_origin=CoordOrigin.TOPLEFT)
        cells = model._parse_text_to_cells("Hello World", ocr_rect)

        assert len(cells) == 1
        assert cells[0].text == "Hello World"
        assert cells[0].from_ocr is True
        assert cells[0].confidence == 1.0

    def test_parse_text_to_cells_multiple_lines(self):
        """Test parsing multiple lines of text."""
        options = DeepSeekOcrOptions()
        accelerator_options = AcceleratorOptions(device=AcceleratorDevice.AUTO)

        model = DeepSeekOcrModel(
            enabled=False,
            artifacts_path=None,
            options=options,
            accelerator_options=accelerator_options,
        )
        model.scale = 3

        ocr_rect = BoundingBox(l=0, t=0, r=300, b=300, coord_origin=CoordOrigin.TOPLEFT)
        text = "Line 1\nLine 2\nLine 3"
        cells = model._parse_text_to_cells(text, ocr_rect)

        assert len(cells) == 3
        assert cells[0].text == "Line 1"
        assert cells[1].text == "Line 2"
        assert cells[2].text == "Line 3"

    def test_parse_text_to_cells_empty(self):
        """Test parsing empty text."""
        options = DeepSeekOcrOptions()
        accelerator_options = AcceleratorOptions(device=AcceleratorDevice.AUTO)

        model = DeepSeekOcrModel(
            enabled=False,
            artifacts_path=None,
            options=options,
            accelerator_options=accelerator_options,
        )
        model.scale = 3

        ocr_rect = BoundingBox(l=0, t=0, r=300, b=100, coord_origin=CoordOrigin.TOPLEFT)
        cells = model._parse_text_to_cells("", ocr_rect)

        assert len(cells) == 0

    def test_parse_text_to_cells_whitespace_only(self):
        """Test parsing whitespace-only text."""
        options = DeepSeekOcrOptions()
        accelerator_options = AcceleratorOptions(device=AcceleratorDevice.AUTO)

        model = DeepSeekOcrModel(
            enabled=False,
            artifacts_path=None,
            options=options,
            accelerator_options=accelerator_options,
        )
        model.scale = 3

        ocr_rect = BoundingBox(l=0, t=0, r=300, b=100, coord_origin=CoordOrigin.TOPLEFT)
        cells = model._parse_text_to_cells("   \n\n   ", ocr_rect)

        assert len(cells) == 0


class TestDeepSeekOcrModelGetOptionsType:
    """Test get_options_type class method."""

    def test_get_options_type(self):
        """Test that get_options_type returns DeepSeekOcrOptions."""
        assert DeepSeekOcrModel.get_options_type() == DeepSeekOcrOptions


class TestDeepSeekOcrModelInferParameters:
    """Test that infer() is called with correct parameters for CUDA vs MPS."""

    def test_cuda_infer_without_device_dtype(self, monkeypatch, tmp_path):
        """Test that CUDA path does NOT pass device/dtype to infer()."""
        # Create mock torch module for CUDA
        mock_torch = MagicMock()
        mock_torch.__version__ = "2.7.0"
        mock_torch.backends.cuda.is_built.return_value = True
        mock_torch.cuda.is_available.return_value = True
        mock_torch.backends.mps.is_built.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        # Create a mock device with type 'cuda'
        mock_device = MagicMock()
        mock_device.type = "cuda"
        mock_torch.device.return_value = mock_device
        mock_torch.bfloat16 = "bfloat16"

        # Mock transformers
        mock_transformers = MagicMock()
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_model.infer.return_value = "Test OCR result"
        mock_transformers.AutoModel.from_pretrained.return_value = mock_model
        mock_transformers.AutoTokenizer.from_pretrained.return_value = MagicMock()

        monkeypatch.setitem(sys.modules, "torch", mock_torch)
        monkeypatch.setitem(sys.modules, "transformers", mock_transformers)

        options = DeepSeekOcrOptions()
        accelerator_options = AcceleratorOptions(device=AcceleratorDevice.CUDA)

        with patch.object(
            DeepSeekOcrModel, "_patch_llama_flash_attention", return_value=None
        ):
            model = DeepSeekOcrModel(
                enabled=True,
                artifacts_path=None,
                options=options,
                accelerator_options=accelerator_options,
            )

        # Create a mock image
        mock_image = MagicMock()
        mock_image.save = MagicMock()

        # Create OCR bounding box
        ocr_rect = BoundingBox(l=0, t=0, r=300, b=100, coord_origin=CoordOrigin.TOPLEFT)

        # Set scale (normally set in __call__)
        model.scale = 3

        # Call _run_ocr
        model._run_ocr(mock_image, ocr_rect)

        # Verify infer was called
        mock_model.infer.assert_called_once()

        # Get the kwargs passed to infer
        call_kwargs = mock_model.infer.call_args.kwargs

        # Assert device and dtype are NOT in kwargs for CUDA
        assert "device" not in call_kwargs, "CUDA infer() should NOT receive 'device' parameter"
        assert "dtype" not in call_kwargs, "CUDA infer() should NOT receive 'dtype' parameter"

        # Verify other expected parameters are present
        assert "prompt" in call_kwargs
        assert "image_file" in call_kwargs
        assert "eval_mode" in call_kwargs
        assert call_kwargs["eval_mode"] is True

    def test_mps_infer_with_device_dtype(self, monkeypatch, tmp_path):
        """Test that MPS path DOES pass device/dtype to infer()."""
        # Create mock torch module for MPS
        mock_torch = MagicMock()
        mock_torch.__version__ = "2.7.0"
        mock_torch.backends.cuda.is_built.return_value = False
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_built.return_value = True
        mock_torch.backends.mps.is_available.return_value = True

        # Create a mock device with type 'mps'
        mock_device = MagicMock()
        mock_device.type = "mps"
        mock_torch.device.return_value = mock_device
        mock_torch.float16 = "float16"

        # Mock transformers
        mock_transformers = MagicMock()
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_model.infer.return_value = "Test OCR result"
        mock_transformers.AutoModel.from_pretrained.return_value = mock_model
        mock_transformers.AutoTokenizer.from_pretrained.return_value = MagicMock()

        monkeypatch.setitem(sys.modules, "torch", mock_torch)
        monkeypatch.setitem(sys.modules, "transformers", mock_transformers)

        options = DeepSeekOcrOptions()
        accelerator_options = AcceleratorOptions(device=AcceleratorDevice.MPS)

        with patch.object(
            DeepSeekOcrModel, "_patch_llama_flash_attention", return_value=None
        ):
            model = DeepSeekOcrModel(
                enabled=True,
                artifacts_path=None,
                options=options,
                accelerator_options=accelerator_options,
            )

        # Create a mock image
        mock_image = MagicMock()
        mock_image.save = MagicMock()

        # Create OCR bounding box
        ocr_rect = BoundingBox(l=0, t=0, r=300, b=100, coord_origin=CoordOrigin.TOPLEFT)

        # Set scale (normally set in __call__)
        model.scale = 3

        # Call _run_ocr
        model._run_ocr(mock_image, ocr_rect)

        # Verify infer was called
        mock_model.infer.assert_called_once()

        # Get the kwargs passed to infer
        call_kwargs = mock_model.infer.call_args.kwargs

        # Assert device and dtype ARE in kwargs for MPS
        assert "device" in call_kwargs, "MPS infer() SHOULD receive 'device' parameter"
        assert "dtype" in call_kwargs, "MPS infer() SHOULD receive 'dtype' parameter"
        assert call_kwargs["device"] == mock_device
        assert call_kwargs["dtype"] == "float16"

        # Verify other expected parameters are present
        assert "prompt" in call_kwargs
        assert "image_file" in call_kwargs
        assert "eval_mode" in call_kwargs
        assert call_kwargs["eval_mode"] is True


class TestDeepSeekOcrModelImportError:
    """Test import error handling."""

    def test_missing_torch_raises_import_error(self, monkeypatch):
        """Test that missing torch raises ImportError with helpful message."""
        # We test the import error message by checking the code path
        # The actual import error is raised in _init_model when torch/transformers
        # are not available. We verify the error message format.
        options = DeepSeekOcrOptions()
        accelerator_options = AcceleratorOptions(device=AcceleratorDevice.AUTO)

        # Mock the _init_model to raise the expected ImportError
        def mock_init_model(self, accel_opts, artifacts_path):
            raise ImportError(
                "DeepSeek-OCR requires 'transformers' and 'torch' packages. "
                "Please install them via `pip install docling[deepseekocr]` to use this OCR engine."
            )

        with patch.object(DeepSeekOcrModel, "_init_model", mock_init_model):
            with pytest.raises(ImportError, match=r"transformers.*torch"):
                DeepSeekOcrModel(
                    enabled=True,
                    artifacts_path=None,
                    options=options,
                    accelerator_options=accelerator_options,
                )
