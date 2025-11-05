"""
Test MLX Whisper integration for Apple Silicon ASR pipeline.
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.asr_model_specs import (
    WHISPER_BASE,
    WHISPER_BASE_MLX,
    WHISPER_LARGE,
    WHISPER_LARGE_MLX,
    WHISPER_MEDIUM,
    WHISPER_SMALL,
    WHISPER_TINY,
    WHISPER_TURBO,
)
from docling.datamodel.pipeline_options import AsrPipelineOptions
from docling.datamodel.pipeline_options_asr_model import (
    InferenceAsrFramework,
    InlineAsrMlxWhisperOptions,
)
from docling.pipeline.asr_pipeline import AsrPipeline, _MlxWhisperModel


class TestMlxWhisperIntegration:
    """Test MLX Whisper model integration."""

    def test_mlx_whisper_options_creation(self):
        """Test that MLX Whisper options are created correctly."""
        options = InlineAsrMlxWhisperOptions(
            repo_id="mlx-community/whisper-tiny-mlx",
            language="en",
            task="transcribe",
        )

        assert options.inference_framework == InferenceAsrFramework.MLX
        assert options.repo_id == "mlx-community/whisper-tiny-mlx"
        assert options.language == "en"
        assert options.task == "transcribe"
        assert options.word_timestamps is True
        assert AcceleratorDevice.MPS in options.supported_devices

    def test_whisper_models_auto_select_mlx(self):
        """Test that Whisper models automatically select MLX when MPS and mlx-whisper are available."""
        # This test verifies that the models are correctly configured
        # In a real Apple Silicon environment with mlx-whisper installed,
        # these models would automatically use MLX

        # Check that the models exist and have the correct structure
        assert hasattr(WHISPER_TURBO, "inference_framework")
        assert hasattr(WHISPER_TURBO, "repo_id")

        assert hasattr(WHISPER_BASE, "inference_framework")
        assert hasattr(WHISPER_BASE, "repo_id")

        assert hasattr(WHISPER_SMALL, "inference_framework")
        assert hasattr(WHISPER_SMALL, "repo_id")

    def test_explicit_mlx_models_shape(self):
        """Explicit MLX options should have MLX framework and valid repos."""
        assert WHISPER_BASE_MLX.inference_framework.name == "MLX"
        assert WHISPER_LARGE_MLX.inference_framework.name == "MLX"
        assert WHISPER_BASE_MLX.repo_id.startswith("mlx-community/")

    def test_model_selectors_mlx_and_native_paths(self, monkeypatch):
        """Cover MLX/native selection branches in asr_model_specs getters."""
        from docling.datamodel import asr_model_specs as specs

        # Force MLX path
        class _Mps:
            def is_built(self):
                return True

            def is_available(self):
                return True

        class _Torch:
            class backends:
                mps = _Mps()

        monkeypatch.setitem(sys.modules, "torch", _Torch())
        monkeypatch.setitem(sys.modules, "mlx_whisper", object())

        m_tiny = specs._get_whisper_tiny_model()
        m_small = specs._get_whisper_small_model()
        m_base = specs._get_whisper_base_model()
        m_medium = specs._get_whisper_medium_model()
        m_large = specs._get_whisper_large_model()
        m_turbo = specs._get_whisper_turbo_model()
        assert (
            m_tiny.inference_framework == InferenceAsrFramework.MLX
            and m_tiny.repo_id.startswith("mlx-community/whisper-tiny")
        )
        assert (
            m_small.inference_framework == InferenceAsrFramework.MLX
            and m_small.repo_id.startswith("mlx-community/whisper-small")
        )
        assert (
            m_base.inference_framework == InferenceAsrFramework.MLX
            and m_base.repo_id.startswith("mlx-community/whisper-base")
        )
        assert (
            m_medium.inference_framework == InferenceAsrFramework.MLX
            and "medium" in m_medium.repo_id
        )
        assert (
            m_large.inference_framework == InferenceAsrFramework.MLX
            and "large" in m_large.repo_id
        )
        assert (
            m_turbo.inference_framework == InferenceAsrFramework.MLX
            and m_turbo.repo_id.endswith("whisper-turbo")
        )

        # Force native path (no mlx or no mps)
        if "mlx_whisper" in sys.modules:
            del sys.modules["mlx_whisper"]

        class _MpsOff:
            def is_built(self):
                return False

            def is_available(self):
                return False

        class _TorchOff:
            class backends:
                mps = _MpsOff()

        monkeypatch.setitem(sys.modules, "torch", _TorchOff())
        n_tiny = specs._get_whisper_tiny_model()
        n_small = specs._get_whisper_small_model()
        n_base = specs._get_whisper_base_model()
        n_medium = specs._get_whisper_medium_model()
        n_large = specs._get_whisper_large_model()
        n_turbo = specs._get_whisper_turbo_model()
        assert (
            n_tiny.inference_framework == InferenceAsrFramework.WHISPER
            and n_tiny.repo_id == "tiny"
        )
        assert (
            n_small.inference_framework == InferenceAsrFramework.WHISPER
            and n_small.repo_id == "small"
        )
        assert (
            n_base.inference_framework == InferenceAsrFramework.WHISPER
            and n_base.repo_id == "base"
        )
        assert (
            n_medium.inference_framework == InferenceAsrFramework.WHISPER
            and n_medium.repo_id == "medium"
        )
        assert (
            n_large.inference_framework == InferenceAsrFramework.WHISPER
            and n_large.repo_id == "large"
        )
        assert (
            n_turbo.inference_framework == InferenceAsrFramework.WHISPER
            and n_turbo.repo_id == "turbo"
        )

    def test_selector_import_errors_force_native(self, monkeypatch):
        """If torch import fails, selector must return native."""
        from docling.datamodel import asr_model_specs as specs

        # Simulate environment where MPS is unavailable and mlx_whisper missing
        class _MpsOff:
            def is_built(self):
                return False

            def is_available(self):
                return False

        class _TorchOff:
            class backends:
                mps = _MpsOff()

        monkeypatch.setitem(sys.modules, "torch", _TorchOff())
        if "mlx_whisper" in sys.modules:
            del sys.modules["mlx_whisper"]

        model = specs._get_whisper_base_model()
        assert model.inference_framework == InferenceAsrFramework.WHISPER

    @patch("builtins.__import__")
    def test_mlx_whisper_model_initialization(self, mock_import):
        """Test MLX Whisper model initialization."""
        # Mock the mlx_whisper import
        mock_mlx_whisper = Mock()
        mock_import.return_value = mock_mlx_whisper

        accelerator_options = AcceleratorOptions(device=AcceleratorDevice.MPS)
        asr_options = InlineAsrMlxWhisperOptions(
            repo_id="mlx-community/whisper-tiny-mlx",
            inference_framework=InferenceAsrFramework.MLX,
            language="en",
            task="transcribe",
            word_timestamps=True,
            no_speech_threshold=0.6,
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.4,
        )

        model = _MlxWhisperModel(
            enabled=True,
            artifacts_path=None,
            accelerator_options=accelerator_options,
            asr_options=asr_options,
        )

        assert model.enabled is True
        assert model.model_path == "mlx-community/whisper-tiny-mlx"
        assert model.language == "en"
        assert model.task == "transcribe"
        assert model.word_timestamps is True

    def test_mlx_whisper_model_import_error(self):
        """Test that ImportError is raised when mlx-whisper is not available."""
        accelerator_options = AcceleratorOptions(device=AcceleratorDevice.MPS)
        asr_options = InlineAsrMlxWhisperOptions(
            repo_id="mlx-community/whisper-tiny-mlx",
            inference_framework=InferenceAsrFramework.MLX,
            language="en",
            task="transcribe",
            word_timestamps=True,
            no_speech_threshold=0.6,
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.4,
        )

        with patch(
            "builtins.__import__",
            side_effect=ImportError("No module named 'mlx_whisper'"),
        ):
            with pytest.raises(ImportError, match="mlx-whisper is not installed"):
                _MlxWhisperModel(
                    enabled=True,
                    artifacts_path=None,
                    accelerator_options=accelerator_options,
                    asr_options=asr_options,
                )

    @patch("builtins.__import__")
    def test_mlx_whisper_transcribe(self, mock_import):
        """Test MLX Whisper transcription method."""
        # Mock the mlx_whisper module and its transcribe function
        mock_mlx_whisper = Mock()
        mock_import.return_value = mock_mlx_whisper

        # Mock the transcribe result
        mock_result = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 2.5,
                    "text": "Hello world",
                    "words": [
                        {"start": 0.0, "end": 0.5, "word": "Hello"},
                        {"start": 0.5, "end": 1.0, "word": "world"},
                    ],
                }
            ]
        }
        mock_mlx_whisper.transcribe.return_value = mock_result

        accelerator_options = AcceleratorOptions(device=AcceleratorDevice.MPS)
        asr_options = InlineAsrMlxWhisperOptions(
            repo_id="mlx-community/whisper-tiny-mlx",
            inference_framework=InferenceAsrFramework.MLX,
            language="en",
            task="transcribe",
            word_timestamps=True,
            no_speech_threshold=0.6,
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.4,
        )

        model = _MlxWhisperModel(
            enabled=True,
            artifacts_path=None,
            accelerator_options=accelerator_options,
            asr_options=asr_options,
        )

        # Test transcription
        audio_path = Path("test_audio.wav")
        result = model.transcribe(audio_path)

        # Verify the result
        assert len(result) == 1
        assert result[0].start_time == 0.0
        assert result[0].end_time == 2.5
        assert result[0].text == "Hello world"
        assert len(result[0].words) == 2
        assert result[0].words[0].text == "Hello"
        assert result[0].words[1].text == "world"

        # Verify mlx_whisper.transcribe was called with correct parameters
        mock_mlx_whisper.transcribe.assert_called_once_with(
            str(audio_path),
            path_or_hf_repo="mlx-community/whisper-tiny-mlx",
            language="en",
            task="transcribe",
            word_timestamps=True,
            no_speech_threshold=0.6,
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.4,
        )

    @patch("builtins.__import__")
    def test_asr_pipeline_with_mlx_whisper(self, mock_import):
        """Test that AsrPipeline can be initialized with MLX Whisper options."""
        # Mock the mlx_whisper import
        mock_mlx_whisper = Mock()
        mock_import.return_value = mock_mlx_whisper

        accelerator_options = AcceleratorOptions(device=AcceleratorDevice.MPS)
        asr_options = InlineAsrMlxWhisperOptions(
            repo_id="mlx-community/whisper-tiny-mlx",
            inference_framework=InferenceAsrFramework.MLX,
            language="en",
            task="transcribe",
            word_timestamps=True,
            no_speech_threshold=0.6,
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.4,
        )
        pipeline_options = AsrPipelineOptions(
            asr_options=asr_options,
            accelerator_options=accelerator_options,
        )

        pipeline = AsrPipeline(pipeline_options)
        assert isinstance(pipeline._model, _MlxWhisperModel)
        assert pipeline._model.model_path == "mlx-community/whisper-tiny-mlx"
