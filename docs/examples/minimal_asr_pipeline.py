# %% [markdown]
# Minimal ASR pipeline example: transcribe an audio file to Markdown text.
#
# What this example does
# - Configures the ASR pipeline with a default model spec and converts one audio file.
# - Prints the recognized speech segments in Markdown with timestamps.
#
# Prerequisites
# - Install Docling with ASR extras and any audio dependencies (ffmpeg, etc.).
# - Ensure your environment can download or access the configured ASR model.
# - Some formats require ffmpeg codecs; install ffmpeg and ensure it's on PATH.
#
# How to run
# - From the repository root, run: `python docs/examples/minimal_asr_pipeline.py`.
# - The script prints the transcription to stdout.
#
# Customizing the model
# - Edit `get_asr_converter()` to switch `asr_model_specs` (e.g., language or model size).
# - Keep `InputFormat.AUDIO` and `AsrPipeline` unchanged for a minimal setup.
#
# Input audio
# - Defaults to `tests/data/audio/sample_10s.mp3`. Update `audio_path` to your own file if needed.

# %%

from pathlib import Path

from docling_core.types.doc import DoclingDocument

from docling.datamodel import asr_model_specs
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import AsrPipelineOptions
from docling.document_converter import AudioFormatOption, DocumentConverter
from docling.pipeline.asr_pipeline import AsrPipeline


def get_asr_converter():
    """Create a DocumentConverter configured for ASR with a default model.

    Uses `asr_model_specs.WHISPER_TURBO` by default. You can swap in another
    model spec from `docling.datamodel.asr_model_specs` to experiment.
    """
    pipeline_options = AsrPipelineOptions()
    pipeline_options.asr_options = asr_model_specs.WHISPER_TURBO

    converter = DocumentConverter(
        format_options={
            InputFormat.AUDIO: AudioFormatOption(
                pipeline_cls=AsrPipeline,
                pipeline_options=pipeline_options,
            )
        }
    )
    return converter


def asr_pipeline_conversion(audio_path: Path) -> DoclingDocument:
    """Run the ASR pipeline and return a `DoclingDocument` transcript."""
    # Check if the test audio file exists
    assert audio_path.exists(), f"Test audio file not found: {audio_path}"

    converter = get_asr_converter()

    # Convert the audio file
    result: ConversionResult = converter.convert(audio_path)

    # Verify conversion was successful
    assert result.status == ConversionStatus.SUCCESS, (
        f"Conversion failed with status: {result.status}"
    )
    return result.document


if __name__ == "__main__":
    audio_path = Path("tests/data/audio/sample_10s.mp3")

    doc = asr_pipeline_conversion(audio_path=audio_path)
    print(doc.export_to_markdown())

    # Expected output:
    #
    # [time: 0.0-4.0]  Shakespeare on Scenery by Oscar Wilde
    #
    # [time: 5.28-9.96]  This is a LibriVox recording. All LibriVox recordings are in the public domain.
