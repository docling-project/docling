#!/usr/bin/env python3
"""
Performance comparison between CPU and MLX Whisper on Apple Silicon.

This script compares the performance of:
1. Native Whisper (forced to CPU)
2. MLX Whisper (Apple Silicon optimized)

Both use the same model size for fair comparison.
"""

import argparse
import sys
import time
from pathlib import Path

# Add the repository root to the path so we can import docling
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import AsrPipelineOptions
from docling.datamodel.pipeline_options_asr_model import (
    InferenceAsrFramework,
    InlineAsrMlxWhisperOptions,
    InlineAsrNativeWhisperOptions,
)
from docling.document_converter import AudioFormatOption, DocumentConverter
from docling.pipeline.asr_pipeline import AsrPipeline


def create_cpu_whisper_options(model_size: str = "turbo"):
    """Create native Whisper options forced to CPU."""
    return InlineAsrNativeWhisperOptions(
        repo_id=model_size,
        inference_framework=InferenceAsrFramework.WHISPER,
        verbose=True,
        timestamps=True,
        word_timestamps=True,
        temperature=0.0,
        max_new_tokens=256,
        max_time_chunk=30.0,
    )


def create_mlx_whisper_options(model_size: str = "turbo"):
    """Create MLX Whisper options for Apple Silicon."""
    model_map = {
        "tiny": "mlx-community/whisper-tiny-mlx",
        "small": "mlx-community/whisper-small-mlx",
        "base": "mlx-community/whisper-base-mlx",
        "medium": "mlx-community/whisper-medium-mlx-8bit",
        "large": "mlx-community/whisper-large-mlx-8bit",
        "turbo": "mlx-community/whisper-turbo",
    }

    return InlineAsrMlxWhisperOptions(
        repo_id=model_map[model_size],
        inference_framework=InferenceAsrFramework.MLX,
        language="en",
        task="transcribe",
        word_timestamps=True,
        no_speech_threshold=0.6,
        logprob_threshold=-1.0,
        compression_ratio_threshold=2.4,
    )


def run_transcription_test(
    audio_file: Path, asr_options, device: AcceleratorDevice, test_name: str
):
    """Run a single transcription test and return timing results."""
    print(f"\n{'=' * 60}")
    print(f"Running {test_name}")
    print(f"Device: {device}")
    print(f"Model: {asr_options.repo_id}")
    print(f"Framework: {asr_options.inference_framework}")
    print(f"{'=' * 60}")

    # Create pipeline options
    pipeline_options = AsrPipelineOptions(
        accelerator_options=AcceleratorOptions(device=device),
        asr_options=asr_options,
    )

    # Create document converter
    converter = DocumentConverter(
        format_options={
            InputFormat.AUDIO: AudioFormatOption(
                pipeline_cls=AsrPipeline,
                pipeline_options=pipeline_options,
            )
        }
    )

    # Run transcription with timing
    start_time = time.time()
    try:
        result = converter.convert(audio_file)
        end_time = time.time()

        duration = end_time - start_time

        if result.status.value == "success":
            # Extract text for verification
            text_content = []
            for item in result.document.texts:
                text_content.append(item.text)

            print(f"✅ Success! Duration: {duration:.2f} seconds")
            print(f"Transcribed text: {''.join(text_content)[:100]}...")
            return duration, True
        else:
            print(f"❌ Failed! Status: {result.status}")
            return duration, False

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"❌ Error: {e}")
        return duration, False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Performance comparison between CPU and MLX Whisper on Apple Silicon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

# Use default test audio file
python asr_pipeline_performance_comparison.py

# Use your own audio file
python asr_pipeline_performance_comparison.py --audio /path/to/your/audio.mp3

# Use a different audio file from the tests directory
python asr_pipeline_performance_comparison.py --audio tests/data/audio/another_sample.wav
        """,
    )

    parser.add_argument(
        "--audio",
        type=str,
        help="Path to audio file for testing (default: tests/data/audio/sample_10s.mp3)",
    )

    return parser.parse_args()


def main():
    """Run performance comparison between CPU and MLX Whisper."""
    args = parse_args()

    # Check if we're on Apple Silicon
    try:
        import torch

        has_mps = torch.backends.mps.is_built() and torch.backends.mps.is_available()
    except ImportError:
        has_mps = False

    try:
        import mlx_whisper

        has_mlx_whisper = True
    except ImportError:
        has_mlx_whisper = False

    print("ASR Pipeline Performance Comparison")
    print("=" * 50)
    print(f"Apple Silicon (MPS) available: {has_mps}")
    print(f"MLX Whisper available: {has_mlx_whisper}")

    if not has_mps:
        print("⚠️  Apple Silicon (MPS) not available - running CPU-only comparison")
        print("   For MLX Whisper performance benefits, run on Apple Silicon devices")
        print("   MLX Whisper is optimized for Apple Silicon devices.")

    if not has_mlx_whisper:
        print("⚠️  MLX Whisper not installed - running CPU-only comparison")
        print("   Install with: pip install mlx-whisper")
        print("   Or: uv sync --extra asr")
        print("   For MLX Whisper performance benefits, install the dependency")

    # Determine audio file path
    if args.audio:
        audio_file = Path(args.audio)
        if not audio_file.is_absolute():
            # If relative path, make it relative to the script's directory
            audio_file = Path(__file__).parent.parent.parent / audio_file
    else:
        # Use default test audio file
        audio_file = (
            Path(__file__).parent.parent.parent
            / "tests"
            / "data"
            / "audio"
            / "sample_10s.mp3"
        )

    if not audio_file.exists():
        print(f"❌ Audio file not found: {audio_file}")
        print("   Please check the path and try again.")
        sys.exit(1)

    print(f"Using test audio: {audio_file}")
    print(f"File size: {audio_file.stat().st_size / 1024:.1f} KB")

    # Test different model sizes
    model_sizes = ["tiny", "base", "turbo"]
    results = {}

    for model_size in model_sizes:
        print(f"\n{'#' * 80}")
        print(f"Testing model size: {model_size}")
        print(f"{'#' * 80}")

        model_results = {}

        # Test 1: Native Whisper (forced to CPU)
        cpu_options = create_cpu_whisper_options(model_size)
        cpu_duration, cpu_success = run_transcription_test(
            audio_file,
            cpu_options,
            AcceleratorDevice.CPU,
            f"Native Whisper {model_size} (CPU)",
        )
        model_results["cpu"] = {"duration": cpu_duration, "success": cpu_success}

        # Test 2: MLX Whisper (Apple Silicon optimized) - only if available
        if has_mps and has_mlx_whisper:
            mlx_options = create_mlx_whisper_options(model_size)
            mlx_duration, mlx_success = run_transcription_test(
                audio_file,
                mlx_options,
                AcceleratorDevice.MPS,
                f"MLX Whisper {model_size} (MPS)",
            )
            model_results["mlx"] = {"duration": mlx_duration, "success": mlx_success}
        else:
            print(f"\n{'=' * 60}")
            print(f"Skipping MLX Whisper {model_size} (MPS) - not available")
            print(f"{'=' * 60}")
            model_results["mlx"] = {"duration": 0.0, "success": False}

        results[model_size] = model_results

    # Print summary
    print(f"\n{'#' * 80}")
    print("PERFORMANCE COMPARISON SUMMARY")
    print(f"{'#' * 80}")
    print(
        f"{'Model':<10} {'CPU (sec)':<12} {'MLX (sec)':<12} {'Speedup':<12} {'Status':<10}"
    )
    print("-" * 80)

    for model_size, model_results in results.items():
        cpu_duration = model_results["cpu"]["duration"]
        mlx_duration = model_results["mlx"]["duration"]
        cpu_success = model_results["cpu"]["success"]
        mlx_success = model_results["mlx"]["success"]

        if cpu_success and mlx_success:
            speedup = cpu_duration / mlx_duration
            status = "✅ Both OK"
        elif cpu_success:
            speedup = float("inf")
            status = "❌ MLX Failed"
        elif mlx_success:
            speedup = 0
            status = "❌ CPU Failed"
        else:
            speedup = 0
            status = "❌ Both Failed"

        print(
            f"{model_size:<10} {cpu_duration:<12.2f} {mlx_duration:<12.2f} {speedup:<12.2f}x {status:<10}"
        )

    # Calculate overall improvement
    successful_tests = [
        (r["cpu"]["duration"], r["mlx"]["duration"])
        for r in results.values()
        if r["cpu"]["success"] and r["mlx"]["success"]
    ]

    if successful_tests:
        avg_cpu = sum(cpu for cpu, mlx in successful_tests) / len(successful_tests)
        avg_mlx = sum(mlx for cpu, mlx in successful_tests) / len(successful_tests)
        avg_speedup = avg_cpu / avg_mlx

        print("-" * 80)
        print(
            f"{'AVERAGE':<10} {avg_cpu:<12.2f} {avg_mlx:<12.2f} {avg_speedup:<12.2f}x {'Overall':<10}"
        )

        print(f"\n🎯 MLX Whisper provides {avg_speedup:.1f}x average speedup over CPU!")
    else:
        if has_mps and has_mlx_whisper:
            print("\n❌ No successful comparisons available.")
        else:
            print("\n⚠️  MLX Whisper not available - only CPU results shown.")
            print(
                "   Install MLX Whisper and run on Apple Silicon for performance comparison."
            )


if __name__ == "__main__":
    main()
