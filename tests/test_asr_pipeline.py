from pathlib import Path
from unittest.mock import Mock

import pytest

from docling.datamodel import asr_model_specs
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.document import ConversionResult, InputDocument
from docling.datamodel.pipeline_options import AsrPipelineOptions
from docling.document_converter import AudioFormatOption, DocumentConverter
from docling.pipeline.asr_pipeline import AsrPipeline


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

    # This test will FAIL initially, which is what we want.
    assert doc_result.status == ConversionStatus.PARTIAL_SUCCESS, (
        f"Status should be PARTIAL_SUCCESS for silent audio, but got {doc_result.status}"
    )
    assert len(doc_result.document.texts) == 0, (
        "Document should contain zero text items"
    )


def test_has_text_and_determine_status_helpers():
    """Unit-test _has_text and _determine_status on a minimal ConversionResult."""
    pipeline_options = AsrPipelineOptions()
    pipeline_options.asr_options = asr_model_specs.WHISPER_TINY
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
