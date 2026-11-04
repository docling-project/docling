#!/usr/bin/env python3
# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Run real ASR conversions on one audio fragment, outside normal pytest runs.

uv run python scripts/validate_asr_models.py --model granite_speech_5_0_2b
uv run python scripts/validate_asr_models.py --audio recording.wav --device cuda

Each model runs in a separate process. Failures do not stop the remaining models.
The report describes decoding success, not transcription accuracy.
"""

import argparse
import hashlib
import subprocess
import sys
import time
from pathlib import Path

from pydantic import BaseModel, Field

from docling.datamodel.asr_model_specs import AsrModelType


class ModelRun(BaseModel):
    model: str
    success: bool
    seconds: float
    error: str | None = None


class RunReport(BaseModel):
    audio: str
    audio_sha256: str
    runs: list[ModelRun] = Field(default_factory=list)


def _convert(args: argparse.Namespace) -> None:
    # Defer conversion setup until a worker is requested.
    from docling.cli.main import _resolve_asr_options
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.base_models import ConversionStatus, InputFormat
    from docling.datamodel.pipeline_options import AsrPipelineOptions
    from docling.datamodel.pipeline_options_asr_model import (
        InlineAsrTransformersOptions,
    )
    from docling.document_converter import AudioFormatOption, DocumentConverter
    from docling.pipeline.asr_pipeline import AsrPipeline

    model = AsrModelType(args.model[0])
    options = _resolve_asr_options(model).model_copy(deep=True)
    if isinstance(options, InlineAsrTransformersOptions):
        options.language = args.language
        if args.revision is not None:
            options.revision = args.revision
    pipeline_options = AsrPipelineOptions(
        asr_options=options,
        artifacts_path=args.artifacts_path,
        accelerator_options=AcceleratorOptions(device=args.device),
    )
    converter = DocumentConverter(
        format_options={
            InputFormat.AUDIO: AudioFormatOption(
                pipeline_cls=AsrPipeline,
                pipeline_options=pipeline_options,
            )
        }
    )
    result = converter.convert(args.audio, raises_on_error=False)
    if result.status != ConversionStatus.SUCCESS:
        errors = "; ".join(error.error_message for error in result.errors)
        raise RuntimeError(f"Conversion status: {result.status.value}. {errors}")
    transcript = "\n".join(
        item.text for item in result.document.texts if item.text.strip()
    )
    if not transcript.strip():
        raise RuntimeError("The conversion produced an empty transcript")
    args.output.mkdir(parents=True, exist_ok=True)
    result.document.save_as_doclang_archive(args.output / "document.dclx")
    (args.output / "transcript.md").write_text(
        result.document.export_to_markdown(), encoding="utf-8"
    )
    (args.output / "transcript.txt").write_text(transcript + "\n", encoding="utf-8")
    (args.output / "options.json").write_text(
        options.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )


def run_models(args: argparse.Namespace) -> int:
    """Run every selected preset and retain each process's diagnostics."""
    models = list(dict.fromkeys(args.model or [model.value for model in AsrModelType]))
    audio = args.audio.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=True)
    report = RunReport(
        audio=str(audio), audio_sha256=hashlib.sha256(audio.read_bytes()).hexdigest()
    )
    for model in models:
        output = args.output / model
        output.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--model",
            model,
            "--audio",
            str(audio),
            "--output",
            str(output.resolve()),
            "--device",
            args.device,
            "--language",
            args.language,
        ]
        if args.revision is not None:
            command.extend(["--revision", args.revision])
        if args.artifacts_path is not None:
            command.extend(["--artifacts-path", str(args.artifacts_path.resolve())])
        print(f"Running {model} ...", flush=True)
        started = time.monotonic()
        error = None
        with (output / "run.log").open("w", encoding="utf-8") as log:
            try:
                process = subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=args.timeout,
                    check=False,
                )
                if process.returncode != 0:
                    error = f"Process exited with code {process.returncode}; see {output / 'run.log'}"
            except subprocess.TimeoutExpired:
                error = f"Exceeded {args.timeout:g} seconds; see {output / 'run.log'}"
        run = ModelRun(
            model=model,
            success=error is None,
            seconds=time.monotonic() - started,
            error=error,
        )
        report.runs.append(run)
        (args.output / "report.json").write_text(
            report.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"{'PASS' if run.success else 'FAIL'} {model} ({run.seconds:.1f}s)",
            flush=True,
        )
        if error is not None:
            print(error, flush=True)
    name_width = max(len("Name"), *(len(run.model) for run in report.runs))
    print(f"\n{'Name':<{name_width}}  {'Status':<6}  {'Time (s)':>8}")
    print(f"{'-' * name_width}  {'-' * 6}  {'-' * 8}")
    for run in report.runs:
        status = "PASS" if run.success else "FAIL"
        print(f"{run.model:<{name_width}}  {status:<6}  {run.seconds:>8.1f}")
    print(f"Report: {args.output / 'report.json'}")
    return 0 if all(run.success for run in report.runs) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audio", type=Path, default=Path("tests/data/audio/sources/sample_10s.mp3")
    )
    parser.add_argument(
        "--model",
        action="append",
        choices=[model.value for model in AsrModelType],
        help="Repeat to select several presets; defaults to all presets.",
    )
    parser.add_argument("--output", type=Path, default=Path("asr-validation"))
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--language", default="en", help="Source language of the shared fragment."
    )
    parser.add_argument(
        "--revision", help="Override the revision for selected Transformers models."
    )
    parser.add_argument(
        "--artifacts-path",
        type=Path,
        help="Use local model artifacts without downloading weights.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=900,
        help="Maximum seconds per model, including downloads (default: 900).",
    )
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.list_models:
        for model in AsrModelType:
            print(model.value)
        return 0
    if not args.audio.is_file():
        parser.error(f"Audio file does not exist: {args.audio}")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.worker:
        if args.model is None or len(args.model) != 1:
            parser.error("Worker mode requires exactly one --model")
        _convert(args)
        return 0
    return run_models(args)


if __name__ == "__main__":
    raise SystemExit(main())
