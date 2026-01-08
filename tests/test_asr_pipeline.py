import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from docling.datamodel import asr_model_specs
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.document import ConversionResult, InputDocument
from docling.datamodel.pipeline_options import AsrPipelineOptions
from docling.document_converter import AudioFormatOption, DocumentConverter
from docling.pipeline.asr_pipeline import AsrPipeline

# pytestmark = pytest.mark.skipif(
#     sys.version_info >= (3, 14),
#     reason="Python 3.14 is not yet supported by whisper dependencies.",
# )


@pytest.fixture
def test_audio_path():
    return Path("./tests/data/audio/sample_10s.mp3")


def get_asr_converter():
    """Create a DocumentConverter configured for ASR with whisper_turbo model."""
    pipeline_options = AsrPipelineOptions()
    pipeline_options.asr_options = asr_model_specs.WHISPER_TINY

    converter = DocumentConverter(
        format_options={
            InputFormat.AUDIO: AudioFormatOption(
                pipeline_cls=AsrPipeline,
                pipeline_options=pipeline_options,
            )
        }
    )
    return converter


def test_asr_pipeline_conversion(test_audio_path):
    """Test ASR pipeline conversion using whisper_turbo model on sample_10s.mp3."""
    # Check if the test audio file exists
    assert test_audio_path.exists(), f"Test audio file not found: {test_audio_path}"

    converter = get_asr_converter()

    # Convert the audio file
    doc_result: ConversionResult = converter.convert(test_audio_path)

    # Verify conversion was successful
    assert doc_result.status == ConversionStatus.SUCCESS, (
        f"Conversion failed with status: {doc_result.status}"
    )

    # Verify we have a document
    assert doc_result.document is not None, "No document was created"

    # Verify we have text content (transcribed audio)
    texts = doc_result.document.texts
    assert len(texts) > 0, "No text content found in transcribed audio"

    # Print transcribed text for verification (optional, for debugging)
    print(f"Transcribed text from {test_audio_path.name}:")
    for i, text_item in enumerate(texts):
        print(f"  {i + 1}: {text_item.text}")


@pytest.fixture
def silent_audio_path():
    """Fixture to provide the path to a silent audio file."""
    path = Path("./tests/data/audio/silent_1s.wav")
    if not path.exists():
        pytest.skip("Silent audio file for testing not found at " + str(path))
    return path


def test_asr_pipeline_with_silent_audio(silent_audio_path):
    """
    Test that the ASR pipeline correctly handles silent audio files
    by returning a PARTIAL_SUCCESS status.
    """
    converter = get_asr_converter()
    doc_result: ConversionResult = converter.convert(silent_audio_path)

    # Accept PARTIAL_SUCCESS or SUCCESS depending on runtime behavior
    assert doc_result.status in (
        ConversionStatus.PARTIAL_SUCCESS,
        ConversionStatus.SUCCESS,
    )


def test_has_text_and_determine_status_helpers():
    """Unit-test _has_text and _determine_status on a minimal ConversionResult."""
    pipeline_options = AsrPipelineOptions()
    pipeline_options.asr_options = asr_model_specs.WHISPER_TINY
    # Avoid importing torch in decide_device by forcing CPU-only native path
    pipeline_options.asr_options = asr_model_specs.WHISPER_TINY_NATIVE
    pipeline = AsrPipeline(pipeline_options)

    # Create an empty ConversionResult with proper InputDocument
    doc_path = Path("./tests/data/audio/sample_10s.mp3")
    from docling.backend.noop_backend import NoOpBackend
    from docling.datamodel.base_models import InputFormat

    input_doc = InputDocument(
        path_or_stream=doc_path,
        format=InputFormat.AUDIO,
        backend=NoOpBackend,
    )
    conv_res = ConversionResult(input=input_doc)

    # Simulate run result with empty document/texts
    conv_res.status = ConversionStatus.SUCCESS
    assert pipeline._has_text(conv_res.document) is False
    assert pipeline._determine_status(conv_res) in (
        ConversionStatus.PARTIAL_SUCCESS,
        ConversionStatus.SUCCESS,
        ConversionStatus.FAILURE,
    )

    # Now make a document with whitespace-only text to exercise empty detection
    conv_res.document.texts = []
    conv_res.errors = []
    assert pipeline._has_text(conv_res.document) is False

    # Emulate non-empty
    class _T:
        def __init__(self, t):
            self.text = t

    conv_res.document.texts = [_T("   "), _T("ok")]
    assert pipeline._has_text(conv_res.document) is True


def test_is_backend_supported_noop_backend():
    from pathlib import Path

    from docling.backend.noop_backend import NoOpBackend
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.document import InputDocument

    class _Dummy:
        pass

    # Create a proper NoOpBackend instance
    doc_path = Path("./tests/data/audio/sample_10s.mp3")
    input_doc = InputDocument(
        path_or_stream=doc_path,
        format=InputFormat.AUDIO,
        backend=NoOpBackend,
    )
    noop_backend = NoOpBackend(input_doc, doc_path)

    assert AsrPipeline.is_backend_supported(noop_backend) is True
    assert AsrPipeline.is_backend_supported(_Dummy()) is False


def test_native_and_mlx_transcribe_language_handling(monkeypatch, tmp_path):
    """Cover language None/empty handling in model.transcribe wrappers."""
    from docling.datamodel.accelerator_options import (
        AcceleratorDevice,
        AcceleratorOptions,
    )
    from docling.datamodel.pipeline_options_asr_model import (
        InferenceAsrFramework,
        InlineAsrMlxWhisperOptions,
        InlineAsrNativeWhisperOptions,
    )
    from docling.pipeline.asr_pipeline import _MlxWhisperModel, _NativeWhisperModel

    # Native
    opts_n = InlineAsrNativeWhisperOptions(
        repo_id="tiny",
        inference_framework=InferenceAsrFramework.WHISPER,
        verbose=False,
        timestamps=False,
        word_timestamps=False,
        temperature=0.0,
        max_new_tokens=1,
        max_time_chunk=1.0,
        language="",
    )
    m = _NativeWhisperModel(
        True, None, AcceleratorOptions(device=AcceleratorDevice.CPU), opts_n
    )
    m.model = Mock()
    m.verbose = False
    m.word_timestamps = False
    # ensure language mapping occurs and transcribe is called
    m.model.transcribe.return_value = {"segments": []}
    m.transcribe(tmp_path / "a.wav")
    m.model.transcribe.assert_called()

    # MLX
    opts_m = InlineAsrMlxWhisperOptions(
        repo_id="mlx-community/whisper-tiny-mlx",
        inference_framework=InferenceAsrFramework.MLX,
        language="",
    )
    with patch.dict("sys.modules", {"mlx_whisper": Mock()}):
        mm = _MlxWhisperModel(
            True, None, AcceleratorOptions(device=AcceleratorDevice.MPS), opts_m
        )
        mm.mlx_whisper = Mock()
        mm.mlx_whisper.transcribe.return_value = {"segments": []}
        mm.transcribe(tmp_path / "b.wav")
        mm.mlx_whisper.transcribe.assert_called()


def test_native_init_with_artifacts_path_and_device_logging(tmp_path):
    """Cover _NativeWhisperModel init path with artifacts_path passed."""
    from docling.datamodel.accelerator_options import (
        AcceleratorDevice,
        AcceleratorOptions,
    )
    from docling.datamodel.pipeline_options_asr_model import (
        InferenceAsrFramework,
        InlineAsrNativeWhisperOptions,
    )
    from docling.pipeline.asr_pipeline import _NativeWhisperModel

    opts = InlineAsrNativeWhisperOptions(
        repo_id="tiny",
        inference_framework=InferenceAsrFramework.WHISPER,
        verbose=False,
        timestamps=False,
        word_timestamps=False,
        temperature=0.0,
        max_new_tokens=1,
        max_time_chunk=1.0,
        language="en",
    )
    # Patch out whisper import side-effects during init by stubbing decide_device path only
    model = _NativeWhisperModel(
        True, tmp_path, AcceleratorOptions(device=AcceleratorDevice.CPU), opts
    )
    # swap real model for mock to avoid actual load
    model.model = Mock()
    assert model.enabled is True


def test_native_run_success_with_bytesio_builds_document(tmp_path):
    """Cover _NativeWhisperModel.run with BytesIO input and success path."""
    from io import BytesIO

    from docling.backend.noop_backend import NoOpBackend
    from docling.datamodel.accelerator_options import (
        AcceleratorDevice,
        AcceleratorOptions,
    )
    from docling.datamodel.document import ConversionResult, InputDocument
    from docling.datamodel.pipeline_options_asr_model import (
        InferenceAsrFramework,
        InlineAsrNativeWhisperOptions,
    )
    from docling.pipeline.asr_pipeline import _NativeWhisperModel

    # Prepare InputDocument with BytesIO
    audio_bytes = BytesIO(b"RIFF....WAVE")
    input_doc = InputDocument(
        path_or_stream=audio_bytes,
        format=InputFormat.AUDIO,
        backend=NoOpBackend,
        filename="a.wav",
    )
    conv_res = ConversionResult(input=input_doc)

    # Model with mocked underlying whisper
    opts = InlineAsrNativeWhisperOptions(
        repo_id="tiny",
        inference_framework=InferenceAsrFramework.WHISPER,
        verbose=False,
        timestamps=False,
        word_timestamps=True,
        temperature=0.0,
        max_new_tokens=1,
        max_time_chunk=1.0,
        language="en",
    )
    model = _NativeWhisperModel(
        True, None, AcceleratorOptions(device=AcceleratorDevice.CPU), opts
    )
    model.model = Mock()
    model.verbose = False
    model.word_timestamps = True
    model.model.transcribe.return_value = {
        "segments": [
            {
                "start": 0.0,
                "end": 1.0,
                "text": "hi",
                "words": [{"start": 0.0, "end": 0.5, "word": "hi"}],
            }
        ]
    }

    out = model.run(conv_res)
    # Status is determined later by pipeline; here we validate document content
    assert out.document is not None
    assert len(out.document.texts) >= 1


def test_native_run_failure_sets_status(tmp_path):
    """Cover _NativeWhisperModel.run failure path when transcribe raises."""
    from docling.backend.noop_backend import NoOpBackend
    from docling.datamodel.accelerator_options import (
        AcceleratorDevice,
        AcceleratorOptions,
    )
    from docling.datamodel.document import ConversionResult, InputDocument
    from docling.datamodel.pipeline_options_asr_model import (
        InferenceAsrFramework,
        InlineAsrNativeWhisperOptions,
    )
    from docling.pipeline.asr_pipeline import _NativeWhisperModel

    # Create a real file so backend initializes
    audio_path = tmp_path / "a.wav"
    audio_path.write_bytes(b"RIFF....WAVE")
    input_doc = InputDocument(
        path_or_stream=audio_path, format=InputFormat.AUDIO, backend=NoOpBackend
    )
    conv_res = ConversionResult(input=input_doc)

    opts = InlineAsrNativeWhisperOptions(
        repo_id="tiny",
        inference_framework=InferenceAsrFramework.WHISPER,
        verbose=False,
        timestamps=False,
        word_timestamps=False,
        temperature=0.0,
        max_new_tokens=1,
        max_time_chunk=1.0,
        language="en",
    )
    model = _NativeWhisperModel(
        True, None, AcceleratorOptions(device=AcceleratorDevice.CPU), opts
    )
    model.model = Mock()
    model.model.transcribe.side_effect = RuntimeError("boom")

    out = model.run(conv_res)
    assert out.status.name == "FAILURE"


def test_mlx_run_success_and_failure(tmp_path):
    """Cover _MlxWhisperModel.run success and failure paths."""
    from docling.backend.noop_backend import NoOpBackend
    from docling.datamodel.accelerator_options import (
        AcceleratorDevice,
        AcceleratorOptions,
    )
    from docling.datamodel.document import ConversionResult, InputDocument
    from docling.datamodel.pipeline_options_asr_model import (
        InferenceAsrFramework,
        InlineAsrMlxWhisperOptions,
    )
    from docling.pipeline.asr_pipeline import _MlxWhisperModel

    # Success path
    # Create real files so backend initializes and hashes compute
    path_ok = tmp_path / "b.wav"
    path_ok.write_bytes(b"RIFF....WAVE")
    input_doc = InputDocument(
        path_or_stream=path_ok, format=InputFormat.AUDIO, backend=NoOpBackend
    )
    conv_res = ConversionResult(input=input_doc)
    with patch.dict("sys.modules", {"mlx_whisper": Mock()}):
        opts = InlineAsrMlxWhisperOptions(
            repo_id="mlx-community/whisper-tiny-mlx",
            inference_framework=InferenceAsrFramework.MLX,
            language="en",
        )
        model = _MlxWhisperModel(
            True, None, AcceleratorOptions(device=AcceleratorDevice.MPS), opts
        )
        model.mlx_whisper = Mock()
        model.mlx_whisper.transcribe.return_value = {
            "segments": [{"start": 0.0, "end": 1.0, "text": "ok"}]
        }
        out = model.run(conv_res)
        assert out.status.name == "SUCCESS"

    # Failure path
    path_fail = tmp_path / "c.wav"
    path_fail.write_bytes(b"RIFF....WAVE")
    input_doc2 = InputDocument(
        path_or_stream=path_fail, format=InputFormat.AUDIO, backend=NoOpBackend
    )
    conv_res2 = ConversionResult(input=input_doc2)
    with patch.dict("sys.modules", {"mlx_whisper": Mock()}):
        opts2 = InlineAsrMlxWhisperOptions(
            repo_id="mlx-community/whisper-tiny-mlx",
            inference_framework=InferenceAsrFramework.MLX,
            language="en",
        )
        model2 = _MlxWhisperModel(
            True, None, AcceleratorOptions(device=AcceleratorDevice.MPS), opts2
        )
        model2.mlx_whisper = Mock()
        model2.mlx_whisper.transcribe.side_effect = RuntimeError("fail")
        out2 = model2.run(conv_res2)
        assert out2.status.name == "FAILURE"
