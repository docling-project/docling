from pathlib import Path

import pytest

from docling.datamodel import asr_model_specs
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.document import ConversionResult
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
