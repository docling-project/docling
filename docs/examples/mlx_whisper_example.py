#!/usr/bin/env python3
"""
Example script demonstrating MLX Whisper integration for Apple Silicon.

This script shows how to use the MLX Whisper models for speech recognition
on Apple Silicon devices with optimized performance.
"""

import argparse
import sys
from pathlib import Path

# Add the repository root to the path so we can import docling
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.asr_model_specs import (
    WHISPER_BASE,
    WHISPER_LARGE,
    WHISPER_MEDIUM,
    WHISPER_SMALL,
    WHISPER_TINY,
    WHISPER_TURBO,
)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import AsrPipelineOptions
from docling.document_converter import AudioFormatOption, DocumentConverter
from docling.pipeline.asr_pipeline import AsrPipeline


def transcribe_audio_with_mlx_whisper(audio_file_path: str, model_size: str = "base"):
    """
    Transcribe audio using Whisper models with automatic MLX optimization for Apple Silicon.

    Args:
        audio_file_path: Path to the audio file to transcribe
        model_size: Size of the Whisper model to use
                  ("tiny", "base", "small", "medium", "large", "turbo")
                  Note: MLX optimization is automatically used on Apple Silicon when available

    Returns:
        The transcribed text
    """
    # Select the appropriate Whisper model (automatically uses MLX on Apple Silicon)
    model_map = {
        "tiny": WHISPER_TINY,
        "base": WHISPER_BASE,
        "small": WHISPER_SMALL,
        "medium": WHISPER_MEDIUM,
        "large": WHISPER_LARGE,
        "turbo": WHISPER_TURBO,
    }

    if model_size not in model_map:
        raise ValueError(
            f"Invalid model size: {model_size}. Choose from: {list(model_map.keys())}"
        )

    asr_options = model_map[model_size]

    # Configure accelerator options for Apple Silicon
    accelerator_options = AcceleratorOptions(device=AcceleratorDevice.MPS)

    # Create pipeline options
    pipeline_options = AsrPipelineOptions(
        asr_options=asr_options,
        accelerator_options=accelerator_options,
    )

    # Create document converter with MLX Whisper configuration
    converter = DocumentConverter(
        format_options={
            InputFormat.AUDIO: AudioFormatOption(
                pipeline_cls=AsrPipeline,
                pipeline_options=pipeline_options,
            )
        }
    )

    # Run transcription
    result = converter.convert(Path(audio_file_path))

    if result.status.value == "success":
        # Extract text from the document
        text_content = []
        for item in result.document.texts:
            text_content.append(item.text)

        return "\n".join(text_content)
    else:
        raise RuntimeError(f"Transcription failed: {result.status}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="MLX Whisper example for Apple Silicon speech recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

# Use default test audio file
python mlx_whisper_example.py

# Use your own audio file
python mlx_whisper_example.py --audio /path/to/your/audio.mp3

# Use specific model size
python mlx_whisper_example.py --audio audio.wav --model tiny

# Use default test file with specific model
python mlx_whisper_example.py --model turbo
        """,
    )

    parser.add_argument(
        "--audio",
        type=str,
        help="Path to audio file for transcription (default: tests/data/audio/sample_10s.mp3)",
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["tiny", "base", "small", "medium", "large", "turbo"],
        default="base",
        help="Whisper model size to use (default: base)",
    )

    return parser.parse_args()


def main():
    """Main function to demonstrate MLX Whisper usage."""
    args = parse_args()

    # Determine audio file path
    if args.audio:
        audio_file_path = args.audio
    else:
        # Use default test audio file if no audio file specified
        default_audio = (
            Path(__file__).parent.parent.parent
            / "tests"
            / "data"
            / "audio"
            / "sample_10s.mp3"
        )
        if default_audio.exists():
            audio_file_path = str(default_audio)
            print("No audio file specified, using default test file:")
            print(f"  Audio file: {audio_file_path}")
            print(f"  Model size: {args.model}")
            print()
        else:
            print("Error: No audio file specified and default test file not found.")
            print(
                "Please specify an audio file with --audio or ensure tests/data/audio/sample_10s.mp3 exists."
            )
            sys.exit(1)

    if not Path(audio_file_path).exists():
        print(f"Error: Audio file '{audio_file_path}' not found.")
        sys.exit(1)

    try:
        print(f"Transcribing '{audio_file_path}' using Whisper {args.model} model...")
        print(
            "Note: MLX optimization is automatically used on Apple Silicon when available."
        )
        print()

        transcribed_text = transcribe_audio_with_mlx_whisper(
            audio_file_path, args.model
        )

        print("Transcription Result:")
        print("=" * 50)
        print(transcribed_text)
        print("=" * 50)

    except ImportError as e:
        print(f"Error: {e}")
        print("Please install mlx-whisper: pip install mlx-whisper")
        print("Or install with uv: uv sync --extra asr")
        sys.exit(1)
    except Exception as e:
        print(f"Error during transcription: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
