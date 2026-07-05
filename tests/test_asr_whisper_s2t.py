"""
Tests for the WhisperS2T (CTranslate2-based) ASR backend.
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.pipeline_options import AsrPipelineOptions
from docling.datamodel.pipeline_options_asr_model import (
    InferenceAsrFramework,
    InlineAsrWhisperS2TOptions,
)
from docling.pipeline.asr_pipeline import AsrPipeline, _WhisperS2TModel

pytestmark = pytest.mark.ml_asr


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
        assert options.torch_dtype == "float16"
        assert options.batch_size == 8
        assert options.beam_size == 1
        assert options.word_timestamps is False
        assert options.num_threads == AcceleratorOptions().num_threads
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
            torch_dtype="float32",
            batch_size=4,
            beam_size=5,
            word_timestamps=True,
            num_threads=8,
            initial_prompt="Meeting transcription:",
        )

        assert options.repo_id == "large-v3"
        assert options.language == "fr"
        assert options.task == "translate"
        assert options.torch_dtype == "float32"
        assert options.batch_size == 4
        assert options.beam_size == 5
        assert options.word_timestamps is True
        assert options.num_threads == 8
        assert options.initial_prompt == "Meeting transcription:"


class TestWhisperS2TAutoSelection:
    """Test auto-selection logic for WhisperS2T in asr_model_specs."""

    def test_auto_select_never_uses_s2t(self, monkeypatch):
        """WhisperS2T must never be auto-selected, even when it is installed
        alongside CUDA: the auto-selecting WHISPER_* models stay on native
        Whisper. WhisperS2T is opt-in only, via the explicit *_S2T options."""
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
                f"{getter.__name__} must not auto-select WhisperS2T "
                f"(got {model.inference_framework})"
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
                torch_dtype="float16",
                batch_size=16,
                beam_size=1,
            )
            model = _WhisperS2TModel(
                enabled=True,
                artifacts_path=None,
                accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CPU),
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

    def test_whisper_s2t_cpu_coerces_float16_compute_type(self):
        """Regression test: compute_type='float16' must be coerced to 'float32'
        when running on CPU (CTranslate2 does not support float16 on CPU)."""
        mock_whisper_s2t = Mock()
        mock_whisper_s2t.load_model.return_value = Mock()

        with patch.dict("sys.modules", {"whisper_s2t": mock_whisper_s2t}):
            asr_options = InlineAsrWhisperS2TOptions(
                repo_id="tiny",
                inference_framework=InferenceAsrFramework.WHISPER_S2T,
                language="en",
                task="transcribe",
                torch_dtype="float16",
            )
            _WhisperS2TModel(
                enabled=True,
                artifacts_path=None,
                accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CPU),
                asr_options=asr_options,
            )

        call_kwargs = mock_whisper_s2t.load_model.call_args.kwargs
        assert call_kwargs.get("compute_type") == "float32", (
            f"compute_type='float16' on CPU should be coerced to 'float32', "
            f"got {call_kwargs.get('compute_type')!r}"
        )

    def test_whisper_s2t_cpu_coerces_bfloat16_compute_type(self):
        """Regression test: compute_type='bfloat16' must also be coerced to
        'float32' when running on CPU (CTranslate2 supports float32/int8/
        int8_float32 on CPU only)."""
        mock_whisper_s2t = Mock()
        mock_whisper_s2t.load_model.return_value = Mock()

        with patch.dict("sys.modules", {"whisper_s2t": mock_whisper_s2t}):
            asr_options = InlineAsrWhisperS2TOptions(
                repo_id="tiny",
                inference_framework=InferenceAsrFramework.WHISPER_S2T,
                language="en",
                task="transcribe",
                torch_dtype="bfloat16",
            )
            _WhisperS2TModel(
                enabled=True,
                artifacts_path=None,
                accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CPU),
                asr_options=asr_options,
            )

        call_kwargs = mock_whisper_s2t.load_model.call_args.kwargs
        assert call_kwargs.get("compute_type") == "float32", (
            f"compute_type='bfloat16' on CPU should be coerced to 'float32', "
            f"got {call_kwargs.get('compute_type')!r}"
        )

    def test_whisper_s2t_cpu_preserves_cpu_compatible_compute_types(self):
        """Regression test: compute_type values already supported by CTranslate2
        on CPU (float32, int8, int8_float32) must be passed through unchanged."""
        mock_whisper_s2t = Mock()
        mock_whisper_s2t.load_model.return_value = Mock()

        with patch.dict("sys.modules", {"whisper_s2t": mock_whisper_s2t}):
            for compute_type in ("float32", "int8", "int8_float32"):
                asr_options = InlineAsrWhisperS2TOptions(
                    repo_id="tiny",
                    inference_framework=InferenceAsrFramework.WHISPER_S2T,
                    language="en",
                    task="transcribe",
                    torch_dtype=compute_type,
                )
                _WhisperS2TModel(
                    enabled=True,
                    artifacts_path=None,
                    accelerator_options=AcceleratorOptions(
                        device=AcceleratorDevice.CPU
                    ),
                    asr_options=asr_options,
                )

                call_kwargs = mock_whisper_s2t.load_model.call_args.kwargs
                assert call_kwargs.get("compute_type") == compute_type, (
                    f"compute_type={compute_type!r} on CPU should pass through "
                    f"unchanged, got {call_kwargs.get('compute_type')!r}"
                )
                mock_whisper_s2t.load_model.reset_mock()

    def test_whisper_s2t_run_zero_duration_segment_does_not_fail(self, tmp_path):
        """Regression test for the S2T zero-duration handling regression: a
        segment with non-empty text and start_time == end_time must not abort
        the whole conversion. This matches native/MLX behavior via the shared
        _process_conversation() helper."""
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
        # Single zero-duration segment with non-empty text.
        mock_model_instance.transcribe_with_vad.return_value = [
            [{"start_time": 1.0, "end_time": 1.0, "text": "zero duration text"}]
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
            assert out.status == ConversionStatus.SUCCESS, (
                f"Conversion must not fail on zero-duration segment, "
                f"got status={out.status}"
            )
            assert out.document is not None
            assert len(out.document.texts) == 1, (
                "Zero-duration segment text must be preserved after normalization"
            )
            assert out.document.texts[0].text == "zero duration text"


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
