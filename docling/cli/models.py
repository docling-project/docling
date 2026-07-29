import logging
import sys
import warnings
from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

from docling.exceptions import DoclingModelDownloadError, DoclingMultiModelDownloadError

# Check for CLI dependencies
try:
    import typer
    from rich.console import Console
    from rich.logging import RichHandler
except ImportError as e:
    missing_package = str(e).split("'")[1] if "'" in str(e) else "typer or rich"
    print(
        f"Error: Missing required CLI dependency '{missing_package}'", file=sys.stderr
    )
    print("\nThe docling-tools CLI requires additional dependencies.", file=sys.stderr)
    print("Please install them using one of the following options:\n", file=sys.stderr)
    print("  1. Install the full docling package (recommended):", file=sys.stderr)
    print("     pip install docling\n", file=sys.stderr)
    print("  2. Install docling-slim with CLI support:", file=sys.stderr)
    print("     pip install docling-slim[cli]\n", file=sys.stderr)
    print("  3. Install just the missing dependencies:", file=sys.stderr)
    print("     pip install typer rich\n", file=sys.stderr)
    sys.exit(1)

from docling.datamodel.settings import settings
from docling.models.stages.ocr.easyocr_model import (
    _resolve_easyocr_recognition_models,
)
from docling.models.stages.ocr.rapid_ocr_model import _parse_rapidocr_model_spec
from docling.models.utils.hf_model_download import download_hf_model
from docling.utils.model_downloader import download_models

warnings.filterwarnings(action="ignore", category=UserWarning, module="pydantic|torch")
warnings.filterwarnings(action="ignore", category=FutureWarning, module="easyocr")

console = Console()
err_console = Console(stderr=True)

app = typer.Typer(
    name="Docling models helper",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)


class _AvailableModels(str, Enum):
    LAYOUT = "layout"
    TABLEFORMER = "tableformer"
    TABLEFORMERV2 = "tableformerv2"
    CODE_FORMULA = "code_formula"
    PICTURE_CLASSIFIER = "picture_classifier"
    SMOLVLM = "smolvlm"
    GRANITEDOCLING = "granitedocling"
    GRANITEDOCLING_MLX = "granitedocling_mlx"
    SMOLDOCLING = "smoldocling"
    SMOLDOCLING_MLX = "smoldocling_mlx"
    GRANITE_VISION = "granite_vision"
    GRANITE_CHART_EXTRACTION = "granite_chart_extraction"
    GRANITE_CHART_EXTRACTION_V4 = "granite_chart_extraction_v4"
    RAPIDOCR = "rapidocr"
    EASYOCR = "easyocr"
    NEMOTRON_OCR_V2 = "nemotron_ocr_v2"


_default_models = [
    _AvailableModels.LAYOUT,
    _AvailableModels.TABLEFORMER,
    _AvailableModels.CODE_FORMULA,
    _AvailableModels.PICTURE_CLASSIFIER,
    _AvailableModels.RAPIDOCR,
]


@app.command("download")
def download(
    output_dir: Annotated[
        Path,
        typer.Option(
            ...,
            "-o",
            "--output-dir",
            help="The directory where to download the models.",
        ),
    ] = (settings.cache_dir / "models"),
    force: Annotated[
        bool, typer.Option(..., help="If true, the download will be forced.")
    ] = False,
    models: Annotated[
        Optional[list[_AvailableModels]],
        typer.Argument(
            help="Models to download (default behavior: a predefined set of models will be downloaded).",
        ),
    ] = None,
    all: Annotated[
        bool,
        typer.Option(
            ...,
            "--all",
            help="If true, all available models will be downloaded (mutually exclusive with passing specific models).",
            show_default=True,
        ),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option(
            ...,
            "-q",
            "--quiet",
            help="No extra output is generated, the CLI prints only the directory with the cached models.",
        ),
    ] = False,
    easyocr_lang: Annotated[
        Optional[list[str]],
        typer.Option(
            ...,
            "--easyocr-lang",
            help="EasyOCR language code to prefetch. Repeat for multiple languages.",
        ),
    ] = None,
    rapidocr_backend_lang: Annotated[
        Optional[list[str]],
        typer.Option(
            ...,
            "--rapidocr-backend-lang",
            help=(
                "RapidOCR checkpoint set to prefetch, as '<backend>:<lang>' "
                "(e.g. 'onnxruntime:el', 'torch:korean'). Repeat for multiple. Replaces the default set."
            ),
    ] = None,
    hf_token: Annotated[
        str | None,
        typer.Option(
            ...,
            "--hf-token",
            envvar="HF_TOKEN",
            help="HuggingFace Token token to authenticate and accelerate model downloads. If not provided, reads from HF_TOKEN environment variable (if set). Else runs unauthenticated.",
        ),
    ] = None,
):
    if models and all:
        raise typer.BadParameter(
            "Cannot simultaneously set 'all' parameter and specify models to download."
        )
    if not quiet:
        logging.basicConfig(
            level=logging.INFO,
            format="[blue]%(message)s[/blue]",
            datefmt="[%X]",
            handlers=[RichHandler(show_level=False, show_time=False, markup=True)],
        )
    to_download = models or (list(_AvailableModels) if all else _default_models)
    if easyocr_lang is not None:
        if _AvailableModels.EASYOCR not in to_download:
            raise typer.BadParameter(
                "--easyocr-lang requires the 'easyocr' model",
                param_hint="--easyocr-lang",
            )
        try:
            _resolve_easyocr_recognition_models(easyocr_lang)
        except ValueError as error:
            raise typer.BadParameter(str(error), param_hint="--easyocr-lang") from error
    if rapidocr_backend_lang is not None:
        if _AvailableModels.RAPIDOCR not in to_download:
            raise typer.BadParameter(
                "--rapidocr-backend-lang requires the 'rapidocr' model",
                param_hint="--rapidocr-backend-lang",
            )
        try:
            for value in rapidocr_backend_lang:
                _parse_rapidocr_model_spec(value)
        except ValueError as error:
            raise typer.BadParameter(
                str(error), param_hint="--rapidocr-backend-lang"
            ) from error
    try: 
      output_dir = download_models(
          output_dir=output_dir,
          force=force,
          progress=(not quiet),
          with_layout=_AvailableModels.LAYOUT in to_download,
          with_tableformer=_AvailableModels.TABLEFORMER in to_download,
          with_tableformer_v2=_AvailableModels.TABLEFORMERV2 in to_download,
          with_code_formula=_AvailableModels.CODE_FORMULA in to_download,
          with_picture_classifier=_AvailableModels.PICTURE_CLASSIFIER in to_download,
          with_smolvlm=_AvailableModels.SMOLVLM in to_download,
          with_granitedocling=_AvailableModels.GRANITEDOCLING in to_download,
          with_granitedocling_mlx=_AvailableModels.GRANITEDOCLING_MLX in to_download,
          with_smoldocling=_AvailableModels.SMOLDOCLING in to_download,
          with_smoldocling_mlx=_AvailableModels.SMOLDOCLING_MLX in to_download,
          with_granite_vision=_AvailableModels.GRANITE_VISION in to_download,
          with_granite_chart_extraction=_AvailableModels.GRANITE_CHART_EXTRACTION
          in to_download,
          with_granite_chart_extraction_v4=_AvailableModels.GRANITE_CHART_EXTRACTION_V4
          in to_download,
          with_rapidocr=_AvailableModels.RAPIDOCR in to_download,
          rapidocr_models=rapidocr_backend_lang,
          with_easyocr=_AvailableModels.EASYOCR in to_download,
          easyocr_languages=easyocr_lang,
          with_nemotron_ocr=_AvailableModels.NEMOTRON_OCR_V2 in to_download,
          hf_token=hf_token,
      )
      if quiet:
          typer.echo(output_dir)
      else:
          typer.secho(f"\nModels downloaded into: {output_dir}.", fg="green")

          console.print(
              "\n",
              "Docling can now be configured for running offline using the local artifacts.\n\n",
              "Using the CLI:",
              f"`docling --artifacts-path={output_dir} FILE`",
              "\n",
              "Using Python: see the documentation at <https://docling-project.github.io/docling/usage>.",
       )

   except DoclingMultiModelDownloadError as e:
        if quiet:
            typer.echo(output_dir)
        else:
            typer.secho(
                "\n[bold red]❌ Model download(s) finished with errors:[/bold red]",
                err=True,
            )

            # Iterate through and neatly align the individual failures
            for model_name, exc in e.failures:
                typer.secho(
                    f"  • [bold]{model_name}:[/bold] {exc}",
                    fg=typer.colors.RED,
                    err=True,
                )

            # Add a helpful diagnostic tip
            if any(
                "token" in str(exc).lower() or "unauthorized" in str(exc).lower()
                for _, exc in e.failures
            ):
                typer.secho(
                    "\n[yellow]💡 Tip: One or more downloads failed due to authentication. "
                    "Double-check your '--hf-token' value or the 'HF_TOKEN' environment variable.[/yellow]",
                    err=True,
                )

        # Cleanly halt the CLI execution path returning exit status 1
        raise typer.Exit(code=1)


@app.command("download-hf-repo")
def download_hf_repo(
    models: Annotated[
        list[str],
        typer.Argument(
            help="Specific models to download from HuggingFace identified by their repo id. For example: docling-project/docling-models .",
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            ...,
            "-o",
            "--output-dir",
            help="The directory where to download the models.",
        ),
    ] = (settings.cache_dir / "models"),
    force: Annotated[
        bool, typer.Option(..., help="If true, the download will be forced.")
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option(
            ...,
            "-q",
            "--quiet",
            help="No extra output is generated, the CLI prints only the directory with the cached models.",
        ),
    ] = False,
    token: Annotated[
        str | None,
        typer.Option(
            ...,
            "--hf-token",
            envvar="HF_TOKEN",
            help="HuggingFace Token token to authenticate and accelerate model downloads. If not provided, will use the HF_TOKEN environment variable (if set). Else runs unauthenticated.",
        ),
    ] = None,
):
    if not quiet:
        logging.basicConfig(
            level=logging.INFO,
            format="[blue]%(message)s[/blue]",
            datefmt="[%X]",
            handlers=[RichHandler(show_level=False, show_time=False, markup=True)],
        )

    failed_models: list[tuple[str, str]] = []

    for item in models:
        typer.secho(f"\nDownloading {item} model from HuggingFace...")
        try:
            download_hf_model(
                repo_id=item,
                # would be better to reuse "repo_cache_folder" property: https://github.com/docling-project/docling/blob/main/docling/datamodel/pipeline_options_vlm_model.py#L76
                # but creating options objects seams like an overkill
                local_dir=output_dir / item.replace("/", "--"),
                force=force,
                progress=(not quiet),
                token=token,
            )
        except DoclingModelDownloadError as e:
            failed_models.append((item, str(e)))
    if quiet:
        typer.echo(output_dir)
        if failed_models:
            raise typer.Exit(code=1)
    else:
        typer.secho(f"\nModels downloaded into: {output_dir}.", fg="green")

        if failed_models:
            typer.secho(
                "\n[bold red]❌ Model download(s) finished with errors:[/bold red]",
                err=True,
            )

            for model_id, err_msg in failed_models:
                typer.secho(
                    f"  • [bold]{model_id}:[/bold] {err_msg}",
                    fg=typer.colors.RED,
                    err=True,
                )

            # Add a helpful diagnostic tip
            if any(
                "token" in str(err_msg).lower()
                or "unauthorized" in str(err_msg).lower()
                for _, err_msg in failed_models
            ):
                typer.secho(
                    "\n[yellow]💡 Tip: One or more downloads failed due to authentication. "
                    "Double-check your '--hf-token' value or the 'HF_TOKEN' environment variable.[/yellow]",
                    err=True,
                )

            # Finally, exit the execution path with code 1 to indicate a partial/full failure
            raise typer.Exit(code=1)


click_app = typer.main.get_command(app)

if __name__ == "__main__":
    app()
