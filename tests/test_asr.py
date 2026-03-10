"""
Tests for ASR backends: MLX Whisper, Native Whisper, and WhisperS2T.
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
    InlineAsrWhisperS2TOptions,
)
from docling.pipeline.asr_pipeline import AsrPipeline, _MlxWhisperModel, _WhisperS2TModel


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

        class _Cuda:
            def is_available(self):
                return False

        class _Torch:
            class backends:
                mps = _Mps()

            cuda = _Cuda()

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

        # Force native path (no mlx, no mps, no whisper_s2t)
        if "mlx_whisper" in sys.modules:
            del sys.modules["mlx_whisper"]
        monkeypatch.setitem(sys.modules, "whisper_s2t", None)

        class _MpsOff:
            def is_built(self):
                return False

            def is_available(self):
                return False

        # additional tests
        class _CudaOff:
            def is_available(self):
                return False

        class _TorchOff:
            class backends:
                mps = _MpsOff()

            cuda = _CudaOff()

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

        class _CudaOff:
            def is_available(self):
                return False

        class _TorchOff:
            class backends:
                mps = _MpsOff()

            cuda = _CudaOff()

        monkeypatch.setitem(sys.modules, "torch", _TorchOff())
        if "mlx_whisper" in sys.modules:
            del sys.modules["mlx_whisper"]
        monkeypatch.setitem(sys.modules, "whisper_s2t", None)

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


# ===========================================================================
# WhisperS2T tests
# ===========================================================================


class TestWhisperS2TOptions:
    """Test WhisperS2T options creation and defaults."""

    def test_whisper_s2t_options_creation(self):
        """Test that WhisperS2T options are created with correct defaults."""
        options = InlineAsrWhisperS2TOptions(
            repo_id="tiny",
            language="en",
            task="transcribe",
        )

        assert options.inference_framework == InferenceAsrFramework.WHISPER_S2T
        assert options.repo_id == "tiny"
        assert options.language == "en"
        assert options.task == "transcribe"
        assert options.compute_type == "float16"
        assert options.batch_size == 8
        assert options.beam_size == 1
        assert options.word_timestamps is False
        assert options.cpu_threads == 4
        assert options.num_workers == 1
        assert options.initial_prompt is None

    def test_whisper_s2t_supported_devices(self):
        """WhisperS2T should support CPU and CUDA but not MPS."""
        options = InlineAsrWhisperS2TOptions(
            repo_id="tiny",
            language="en",
            task="transcribe",
        )
        assert AcceleratorDevice.CPU in options.supported_devices
        assert AcceleratorDevice.CUDA in options.supported_devices
        assert AcceleratorDevice.MPS not in options.supported_devices

    def test_whisper_s2t_custom_options(self):
        """Test WhisperS2T options with non-default values."""
        options = InlineAsrWhisperS2TOptions(
            repo_id="large-v3",
            language="fr",
            task="translate",
            compute_type="float32",
            batch_size=4,
            beam_size=5,
            word_timestamps=True,
            cpu_threads=8,
            num_workers=2,
            initial_prompt="Meeting transcription:",
        )

        assert options.repo_id == "large-v3"
        assert options.language == "fr"
        assert options.task == "translate"
        assert options.compute_type == "float32"
        assert options.batch_size == 4
        assert options.beam_size == 5
        assert options.word_timestamps is True
        assert options.cpu_threads == 8
        assert options.num_workers == 2
        assert options.initial_prompt == "Meeting transcription:"


class TestWhisperS2TAutoSelection:
    """Test auto-selection logic for WhisperS2T in asr_model_specs."""

    def test_auto_select_s2t_cuda(self, monkeypatch):
        """CUDA + whisper_s2t installed (no MPS) -> should select WhisperS2T with float16."""
        from docling.datamodel import asr_model_specs as specs

        class _MpsOff:
            def is_built(self):
                return False

            def is_available(self):
                return False

        class _CudaOn:
            def is_available(self):
                return True

        class _Torch:
            class backends:
                mps = _MpsOff()

            cuda = _CudaOn()

        monkeypatch.setitem(sys.modules, "torch", _Torch())
        monkeypatch.setitem(sys.modules, "whisper_s2t", object())
        if "mlx_whisper" in sys.modules:
            monkeypatch.delitem(sys.modules, "mlx_whisper")

        for getter, expected_repo in [
            (specs._get_whisper_tiny_model, "tiny"),
            (specs._get_whisper_small_model, "small"),
            (specs._get_whisper_base_model, "base"),
            (specs._get_whisper_medium_model, "medium"),
            (specs._get_whisper_large_model, "large-v3"),
            (specs._get_whisper_turbo_model, "large-v3-turbo"),
        ]:
            model = getter()
            assert model.inference_framework == InferenceAsrFramework.WHISPER_S2T, (
                f"{getter.__name__} did not select WHISPER_S2T"
            )
            assert model.repo_id == expected_repo, (
                f"{getter.__name__} repo_id={model.repo_id}, expected {expected_repo}"
            )
            assert model.compute_type == "float16", (
                f"{getter.__name__} should use float16 on CUDA"
            )

    def test_auto_select_s2t_cpu_fallback(self, monkeypatch):
        """No MPS, no CUDA, but whisper_s2t installed -> S2T with float32 (CPU)."""
        from docling.datamodel import asr_model_specs as specs

        class _MpsOff:
            def is_built(self):
                return False

            def is_available(self):
                return False

        class _CudaOff:
            def is_available(self):
                return False

        class _Torch:
            class backends:
                mps = _MpsOff()

            cuda = _CudaOff()

        monkeypatch.setitem(sys.modules, "torch", _Torch())
        monkeypatch.setitem(sys.modules, "whisper_s2t", object())
        if "mlx_whisper" in sys.modules:
            monkeypatch.delitem(sys.modules, "mlx_whisper")

        for getter in [
            specs._get_whisper_tiny_model,
            specs._get_whisper_small_model,
            specs._get_whisper_base_model,
            specs._get_whisper_medium_model,
            specs._get_whisper_large_model,
            specs._get_whisper_turbo_model,
        ]:
            model = getter()
            assert model.inference_framework == InferenceAsrFramework.WHISPER_S2T, (
                f"{getter.__name__} did not select WHISPER_S2T on CPU"
            )
            assert model.compute_type == "float32", (
                f"{getter.__name__} should use float32 on CPU"
            )

    def test_auto_select_native_fallback_no_s2t(self, monkeypatch):
        """No MPS, no CUDA, no whisper_s2t -> native Whisper fallback."""
        from docling.datamodel import asr_model_specs as specs

        class _MpsOff:
            def is_built(self):
                return False

            def is_available(self):
                return False

        class _CudaOff:
            def is_available(self):
                return False

        class _Torch:
            class backends:
                mps = _MpsOff()

            cuda = _CudaOff()

        monkeypatch.setitem(sys.modules, "torch", _Torch())
        if "mlx_whisper" in sys.modules:
            monkeypatch.delitem(sys.modules, "mlx_whisper")
        monkeypatch.setitem(sys.modules, "whisper_s2t", None)

        for getter in [
            specs._get_whisper_tiny_model,
            specs._get_whisper_small_model,
            specs._get_whisper_base_model,
            specs._get_whisper_medium_model,
            specs._get_whisper_large_model,
            specs._get_whisper_turbo_model,
        ]:
            model = getter()
            assert model.inference_framework == InferenceAsrFramework.WHISPER, (
                f"{getter.__name__} did not fall back to native WHISPER"
            )

    def test_mlx_takes_priority_over_s2t(self, monkeypatch):
        """MPS + mlx_whisper + whisper_s2t all present -> MLX wins (priority 1)."""
        from docling.datamodel import asr_model_specs as specs

        class _MpsOn:
            def is_built(self):
                return True

            def is_available(self):
                return True

        class _CudaOn:
            def is_available(self):
                return True

        class _Torch:
            class backends:
                mps = _MpsOn()

            cuda = _CudaOn()

        monkeypatch.setitem(sys.modules, "torch", _Torch())
        monkeypatch.setitem(sys.modules, "mlx_whisper", object())
        monkeypatch.setitem(sys.modules, "whisper_s2t", object())

        model = specs._get_whisper_tiny_model()
        assert model.inference_framework == InferenceAsrFramework.MLX


class TestWhisperS2TModel:
    """Test _WhisperS2TModel initialization, transcription, and error handling."""

    def test_whisper_s2t_model_initialization(self):
        """Test _WhisperS2TModel initializes with correct attributes."""
        mock_whisper_s2t = Mock()
        mock_model = Mock()
        mock_whisper_s2t.load_model.return_value = mock_model

        with patch.dict("sys.modules", {"whisper_s2t": mock_whisper_s2t}):
            asr_options = InlineAsrWhisperS2TOptions(
                repo_id="tiny",
                inference_framework=InferenceAsrFramework.WHISPER_S2T,
                language="en",
                task="transcribe",
                compute_type="float16",
                batch_size=16,
                beam_size=1,
            )
            model = _WhisperS2TModel(
                enabled=True,
                artifacts_path=None,
                accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CUDA),
                asr_options=asr_options,
            )

            assert model.enabled is True
            assert model.model_identifier == "tiny"
            assert model.language == "en"
            assert model.task == "transcribe"
            assert model.batch_size == 16
            assert model.word_timestamps is False
            mock_whisper_s2t.load_model.assert_called_once()

    def test_whisper_s2t_import_error(self):
        """ImportError raised when whisper_s2t is not installed."""
        asr_options = InlineAsrWhisperS2TOptions(
            repo_id="tiny",
            inference_framework=InferenceAsrFramework.WHISPER_S2T,
            language="en",
            task="transcribe",
        )

        with patch.dict("sys.modules", {"whisper_s2t": None}):
            with pytest.raises(ImportError, match="whisper_s2t is not installed"):
                _WhisperS2TModel(
                    enabled=True,
                    artifacts_path=None,
                    accelerator_options=AcceleratorOptions(
                        device=AcceleratorDevice.CPU
                    ),
                    asr_options=asr_options,
                )

    def test_whisper_s2t_parse_device(self):
        """Test _parse_device correctly splits device strings."""
        mock_whisper_s2t = Mock()
        mock_whisper_s2t.load_model.return_value = Mock()

        with patch.dict("sys.modules", {"whisper_s2t": mock_whisper_s2t}):
            asr_options = InlineAsrWhisperS2TOptions(
                repo_id="tiny",
                inference_framework=InferenceAsrFramework.WHISPER_S2T,
                language="en",
                task="transcribe",
            )
            model = _WhisperS2TModel(
                enabled=True,
                artifacts_path=None,
                accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CPU),
                asr_options=asr_options,
            )

            # Test parsing
            assert model._parse_device("cuda:0") == ("cuda", 0)
            assert model._parse_device("cuda:1") == ("cuda", 1)
            assert model._parse_device("cpu") == ("cpu", 0)
            assert model._parse_device("cuda:abc") == ("cuda", 0)  # invalid index

    def test_whisper_s2t_transcribe(self):
        """Test transcription returns correct _ConversationItem list."""
        mock_whisper_s2t = Mock()
        mock_model_instance = Mock()
        mock_whisper_s2t.load_model.return_value = mock_model_instance

        # Mock transcribe_with_vad output
        mock_model_instance.transcribe_with_vad.return_value = [
            [
                {"start_time": 0.0, "end_time": 2.5, "text": "Hello world"},
                {"start_time": 3.0, "end_time": 5.0, "text": "How are you"},
            ]
        ]

        with patch.dict("sys.modules", {"whisper_s2t": mock_whisper_s2t}):
            asr_options = InlineAsrWhisperS2TOptions(
                repo_id="tiny",
                inference_framework=InferenceAsrFramework.WHISPER_S2T,
                language="en",
                task="transcribe",
                batch_size=16,
            )
            model = _WhisperS2TModel(
                enabled=True,
                artifacts_path=None,
                accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CPU),
                asr_options=asr_options,
            )

            result = model.transcribe(Path("dummy.wav"))

            assert len(result) == 2
            assert result[0].start_time == 0.0
            assert result[0].end_time == 2.5
            assert result[0].text == "Hello world"
            assert result[1].start_time == 3.0
            assert result[1].end_time == 5.0
            assert result[1].text == "How are you"

            mock_model_instance.transcribe_with_vad.assert_called_once_with(
                [str(Path("dummy.wav"))],
                lang_codes=["en"],
                tasks=["transcribe"],
                initial_prompts=[None],
                batch_size=16,
            )

    def test_whisper_s2t_transcribe_with_word_timestamps(self):
        """Test transcription with word-level timestamps."""
        mock_whisper_s2t = Mock()
        mock_model_instance = Mock()
        mock_whisper_s2t.load_model.return_value = mock_model_instance

        mock_model_instance.transcribe_with_vad.return_value = [
            [
                {
                    "start_time": 0.0,
                    "end_time": 2.5,
                    "text": "Hello world",
                    "word_timestamps": [
                        {"start": 0.0, "end": 1.0, "word": "Hello"},
                        {"start": 1.0, "end": 2.5, "word": "world"},
                    ],
                },
            ]
        ]

        with patch.dict("sys.modules", {"whisper_s2t": mock_whisper_s2t}):
            asr_options = InlineAsrWhisperS2TOptions(
                repo_id="tiny",
                inference_framework=InferenceAsrFramework.WHISPER_S2T,
                language="en",
                task="transcribe",
                word_timestamps=True,
            )
            model = _WhisperS2TModel(
                enabled=True,
                artifacts_path=None,
                accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CPU),
                asr_options=asr_options,
            )

            result = model.transcribe(Path("dummy.wav"))

            assert len(result) == 1
            assert result[0].words is not None
            assert len(result[0].words) == 2
            assert result[0].words[0].text == "Hello"
            assert result[0].words[0].start_time == 0.0
            assert result[0].words[1].text == "world"
            assert result[0].words[1].end_time == 2.5

    def test_whisper_s2t_transcribe_empty_output(self):
        """Test transcription handles empty output gracefully."""
        mock_whisper_s2t = Mock()
        mock_model_instance = Mock()
        mock_whisper_s2t.load_model.return_value = mock_model_instance
        mock_model_instance.transcribe_with_vad.return_value = []

        with patch.dict("sys.modules", {"whisper_s2t": mock_whisper_s2t}):
            asr_options = InlineAsrWhisperS2TOptions(
                repo_id="tiny",
                inference_framework=InferenceAsrFramework.WHISPER_S2T,
                language="en",
                task="transcribe",
            )
            model = _WhisperS2TModel(
                enabled=True,
                artifacts_path=None,
                accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CPU),
                asr_options=asr_options,
            )

            result = model.transcribe(Path("dummy.wav"))
            assert result == []

    def test_whisper_s2t_run_success(self, tmp_path):
        """Test _WhisperS2TModel.run success path with file input."""
        from docling.backend.noop_backend import NoOpBackend
        from docling.datamodel.base_models import ConversionStatus, InputFormat
        from docling.datamodel.document import ConversionResult, InputDocument

        # Create a real file so backend initializes
        audio_path = tmp_path / "test.wav"
        audio_path.write_bytes(b"RIFF....WAVE")
        input_doc = InputDocument(
            path_or_stream=audio_path, format=InputFormat.AUDIO, backend=NoOpBackend
        )
        conv_res = ConversionResult(input=input_doc)

        mock_whisper_s2t = Mock()
        mock_model_instance = Mock()
        mock_whisper_s2t.load_model.return_value = mock_model_instance
        mock_model_instance.transcribe_with_vad.return_value = [
            [{"start_time": 0.0, "end_time": 1.0, "text": "test transcription"}]
        ]

        with patch.dict("sys.modules", {"whisper_s2t": mock_whisper_s2t}):
            asr_options = InlineAsrWhisperS2TOptions(
                repo_id="tiny",
                inference_framework=InferenceAsrFramework.WHISPER_S2T,
                language="en",
                task="transcribe",
            )
            model = _WhisperS2TModel(
                enabled=True,
                artifacts_path=None,
                accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CPU),
                asr_options=asr_options,
            )

            out = model.run(conv_res)
            assert out.status == ConversionStatus.SUCCESS
            assert out.document is not None
            assert len(out.document.texts) == 1

    def test_whisper_s2t_run_failure(self, tmp_path):
        """Test _WhisperS2TModel.run failure path when transcribe raises."""
        from docling.backend.noop_backend import NoOpBackend
        from docling.datamodel.base_models import ConversionStatus, InputFormat
        from docling.datamodel.document import ConversionResult, InputDocument

        audio_path = tmp_path / "test.wav"
        audio_path.write_bytes(b"RIFF....WAVE")
        input_doc = InputDocument(
            path_or_stream=audio_path, format=InputFormat.AUDIO, backend=NoOpBackend
        )
        conv_res = ConversionResult(input=input_doc)

        mock_whisper_s2t = Mock()
        mock_model_instance = Mock()
        mock_whisper_s2t.load_model.return_value = mock_model_instance
        mock_model_instance.transcribe_with_vad.side_effect = RuntimeError("boom")

        with patch.dict("sys.modules", {"whisper_s2t": mock_whisper_s2t}):
            asr_options = InlineAsrWhisperS2TOptions(
                repo_id="tiny",
                inference_framework=InferenceAsrFramework.WHISPER_S2T,
                language="en",
                task="transcribe",
            )
            model = _WhisperS2TModel(
                enabled=True,
                artifacts_path=None,
                accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CPU),
                asr_options=asr_options,
            )

            out = model.run(conv_res)
            assert out.status == ConversionStatus.FAILURE

    def test_whisper_s2t_run_bytesio_input(self, tmp_path):
        """Test _WhisperS2TModel.run with BytesIO input (temp file handling)."""
        from io import BytesIO

        from docling.backend.noop_backend import NoOpBackend
        from docling.datamodel.base_models import ConversionStatus, InputFormat
        from docling.datamodel.document import ConversionResult, InputDocument

        audio_bytes = BytesIO(b"RIFF....WAVE")
        input_doc = InputDocument(
            path_or_stream=audio_bytes,
            format=InputFormat.AUDIO,
            backend=NoOpBackend,
            filename="test.wav",
        )
        conv_res = ConversionResult(input=input_doc)

        mock_whisper_s2t = Mock()
        mock_model_instance = Mock()
        mock_whisper_s2t.load_model.return_value = mock_model_instance
        mock_model_instance.transcribe_with_vad.return_value = [
            [{"start_time": 0.0, "end_time": 1.0, "text": "from bytes"}]
        ]

        with patch.dict("sys.modules", {"whisper_s2t": mock_whisper_s2t}):
            asr_options = InlineAsrWhisperS2TOptions(
                repo_id="tiny",
                inference_framework=InferenceAsrFramework.WHISPER_S2T,
                language="en",
                task="transcribe",
            )
            model = _WhisperS2TModel(
                enabled=True,
                artifacts_path=None,
                accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CPU),
                asr_options=asr_options,
            )

            out = model.run(conv_res)
            assert out.status == ConversionStatus.SUCCESS
            assert out.document is not None

    def test_whisper_s2t_large_v3_sets_n_mels(self):
        """Test that large-v3, distil-large-v3, and large-v3-turbo pass n_mels=128."""
        mock_whisper_s2t = Mock()
        mock_whisper_s2t.load_model.return_value = Mock()

        with patch.dict("sys.modules", {"whisper_s2t": mock_whisper_s2t}):
            for repo_id in ["large-v3", "distil-large-v3", "large-v3-turbo"]:
                asr_options = InlineAsrWhisperS2TOptions(
                    repo_id=repo_id,
                    inference_framework=InferenceAsrFramework.WHISPER_S2T,
                    language="en",
                    task="transcribe",
                )
                _WhisperS2TModel(
                    enabled=True,
                    artifacts_path=None,
                    accelerator_options=AcceleratorOptions(
                        device=AcceleratorDevice.CPU
                    ),
                    asr_options=asr_options,
                )

                # Verify n_mels=128 was passed to load_model
                call_kwargs = mock_whisper_s2t.load_model.call_args
                assert call_kwargs[1].get("n_mels") == 128, (
                    f"n_mels should be 128 for {repo_id}"
                )
                mock_whisper_s2t.load_model.reset_mock()


class TestWhisperS2TPipelineIntegration:
    """Test AsrPipeline integration with WhisperS2T backend."""

    def test_asr_pipeline_with_whisper_s2t(self):
        """Test that AsrPipeline can be initialized with WhisperS2T options."""
        mock_whisper_s2t = Mock()
        mock_whisper_s2t.load_model.return_value = Mock()

        with patch.dict("sys.modules", {"whisper_s2t": mock_whisper_s2t}):
            asr_options = InlineAsrWhisperS2TOptions(
                repo_id="tiny",
                inference_framework=InferenceAsrFramework.WHISPER_S2T,
                language="en",
                task="transcribe",
            )
            pipeline_options = AsrPipelineOptions(
                asr_options=asr_options,
                accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CPU),
            )

            pipeline = AsrPipeline(pipeline_options)
            assert isinstance(pipeline._model, _WhisperS2TModel)
            assert pipeline._model.model_identifier == "tiny"

