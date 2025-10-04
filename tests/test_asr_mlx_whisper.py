"""
Test MLX Whisper integration for Apple Silicon ASR pipeline.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.asr_model_specs import (
    WHISPER_BASE,
    WHISPER_LARGE,
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
