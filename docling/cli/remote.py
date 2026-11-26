# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""`docling convert-remote` — convert documents through a docling-serve service.

This command drives the synchronous ``DoclingServiceClient`` internally and reuses
the same exporter as the local ``convert`` command, so the written output is
identical. Only the conversion options the service honors are exposed here;
local-execution flags (device, threads, PDF-backend internals, debug visualizers)
do not apply to remote conversion and are intentionally absent.

Installing the ``service-client`` extra provides the CLI runtime and the
client dependencies needed by this command.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Optional

import typer
from docling_core.types.doc import ImageRefMode
from rich.markup import escape

from docling.cli.export_utils import (
    _export_flags_from_formats,
    _parse_page_range,
    _split_list,
)
from docling.cli.main import FAIL_ON_HELP, ChunkerType, FailOn
from docling.datamodel.base_models import InputFormat, OutputFormat
from docling.datamodel.pipeline_options import ProcessingPipeline
from docling.datamodel.service.options import (
    ConvertDocumentsOptions as ConvertDocumentsRequestOptions,
)
from docling.service_client import (
    DEFAULT_MAX_CONCURRENCY,
    DoclingServiceClient,
    StatusWatcherKind,
)
from docling.utils.ocr_language import OcrLanguageResolver

if TYPE_CHECKING:
    from rich.console import Console

_log = logging.getLogger(__name__)


def _canonicalize_ocr_lang(raw: Optional[str]) -> Optional[list[str]]:
    """Canonicalize --ocr-lang locally, so a malformed tag fails here and not remotely.

    This command has no `--ocr-engine`, so only a request written behind the
    `iso:` prefix can be checked: a bare value is a code of whichever engine the
    service runs, and travels verbatim.
    """
    # `_split_list` returns None only when the option was not given, so an
    # explicitly empty value survives as `[]`: "let the engine choose".
    tags = _split_list(raw)
    if tags is None:
        return None
    try:
        return [
            language.tag()
            for language in OcrLanguageResolver.canonicalize_ocr_languages(tags)
        ]
    except ValueError as err:
        raise typer.BadParameter(str(err), param_hint="--ocr-lang") from err


_REMOTE_HELP = """\
Convert documents through a remote docling-serve service instead of locally.

Sources may be local files, local directories (walked and filtered by --from),
or http(s) URLs. Results are written to --output in the formats given by --to,
identical to `docling convert`. Only options the service honors are exposed here;
local-execution flags (device, threads, pdf-backend internals, debug visualizers)
do not apply to remote conversion and are intentionally absent.
"""

_REMOTE_EPILOG = """\
Authentication (precedence: flag > environment variable > env file):
  --service-url      or  DOCLING_SERVICE_URL     (required)
  --api-key          or  DOCLING_SERVICE_API_KEY (optional; omit if unauthenticated)
  A .env file is read only on request: --env-file PATH reads that file, and
  --dotenv reads .env in the working directory. Only the two variables above
  are taken from it, without ${VAR} expansion. The service address and its
  source are printed before any document is sent.

Exit codes:
  0  success; partially converted documents are written too (see --fail-on)
  1  runtime or connection failure (service unreachable, conversion error)
     or a failed document (see --fail-on)
  2  usage/config error (no service URL resolved from flag, env, or env file)

Examples:
  # Single file, credentials from the environment
  docling convert-remote report.pdf

  # Single file, credentials from ./.env
  docling convert-remote --dotenv report.pdf

  # Explicit endpoint + key, Markdown and JSON output to ./out
  docling convert-remote --service-url https://docling.example.com \\
    --api-key "$KEY" --to md --to json --output ./out report.pdf

  # Whole directory, only PDFs and DOCX, no OCR
  docling convert-remote --from pdf --from docx --no-ocr ./inbox
"""


def _collect_sources(
    source: list[str], from_formats: list[InputFormat]
) -> list[Path | str]:
    """Collect client sources: http(s) URLs stay as strings, local paths become
    ``Path`` objects, and directories are walked and filtered by ``from_formats``.

    Unlike local ``convert``, URLs are never downloaded here — the client sends
    them to the service as HTTP source requests.
    """
    # Imported here to reuse the exact source-collection helpers from `convert`.
    from docling.cli.main import (
        _is_http_url,
        _is_office_lock_file,
        _iter_input_paths_from_directory,
        err_console,
    )

    sources: list[Path | str] = []
    for src in source:
        if _is_http_url(src):
            sources.append(src)
            continue
        local_path = Path(src)
        if not local_path.exists():
            err_console.print(f"[red]Error: The input {src} does not exist.[/red]")
            raise typer.Exit(1)
        if local_path.is_dir():
            sources.extend(_iter_input_paths_from_directory(local_path, from_formats))
        elif _is_office_lock_file(local_path):
            _log.info(f"Ignoring temporary Office file: {local_path}")
        else:
            sources.append(local_path)
    return sources


def convert_remote(
    ctx: typer.Context,
    source: Annotated[
        list[str],
        typer.Argument(
            ...,
            metavar="source",
            help="Documents to convert: local file/directory paths or http(s) URLs.",
        ),
    ],
    service_url: Annotated[
        Optional[str],
        typer.Option(
            "--service-url",
            envvar="DOCLING_SERVICE_URL",
            help="Base URL of the docling-serve service (required; falls back to "
            "DOCLING_SERVICE_URL, then to --env-file or --dotenv).",
        ),
    ] = None,
    api_key: Annotated[
        Optional[str],
        typer.Option(
            "--api-key",
            envvar="DOCLING_SERVICE_API_KEY",
            help="API key for the service (optional; falls back to "
            "DOCLING_SERVICE_API_KEY, then to --env-file or --dotenv; omit if "
            "unauthenticated).",
        ),
    ] = None,
    env_file: Annotated[
        Optional[Path],
        typer.Option(
            "--env-file",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Read DOCLING_SERVICE_URL and DOCLING_SERVICE_API_KEY from this "
            "file when they are not set by flag or environment variable.",
        ),
    ] = None,
    dotenv: Annotated[
        bool,
        typer.Option(
            "--dotenv",
            help="Same as --env-file .env: read the .env file in the working "
            "directory (not in parent directories).",
        ),
    ] = False,
    from_formats: Annotated[
        Optional[list[InputFormat]],
        typer.Option(
            "--from",
            help="Input formats to accept; filters directories and is sent as the "
            "server allow-list. When omitted, all formats are accepted when "
            "filtering directories and the service applies its own default.",
        ),
    ] = None,
    to_formats: Annotated[
        Optional[list[OutputFormat]],
        typer.Option(
            "--to",
            help="Output formats to produce and write locally. Defaults to Markdown.",
        ),
    ] = None,
    chunker_type: ChunkerType = typer.Option(
        ChunkerType.HYBRID,
        "--chunks-type",
        help="Chunker type for '--to chunks'.",
    ),
    chunk_max_tokens: int | None = typer.Option(
        None,
        "--chunks-max-tokens",
        help="Max tokens per chunk. Defaults to the tokenizer's own limit.",
    ),
    chunk_tokenizer: str = typer.Option(
        "sentence-transformers/all-MiniLM-L6-v2",
        "--chunks-tokenizer",
        help="HuggingFace tokenizer model name/path. Used only with --chunks-type hybrid.",
    ),
    ocr: Annotated[
        bool,
        typer.Option(
            help="If enabled, the service processes bitmap content using OCR.",
        ),
    ] = True,
    force_ocr: Annotated[
        bool,
        typer.Option(
            help="Replace any existing text with OCR-generated text over the full "
            "content.",
        ),
    ] = False,
    tables: Annotated[
        bool,
        typer.Option(
            help="If enabled, the service extracts table structure.",
        ),
    ] = True,
    heading_hierarchy: Annotated[
        bool,
        typer.Option(
            "--heading-hierarchy/--no-heading-hierarchy",
            help="If enabled, the service infers section-header levels for PDF and "
            "image inputs.",
        ),
    ] = False,
    pipeline: Annotated[
        ProcessingPipeline,
        typer.Option(help="Pipeline the service uses to process PDF or image files."),
    ] = ProcessingPipeline.STANDARD,
    ocr_lang: Annotated[
        Optional[str],
        typer.Option(
            help=(
                "Comma-separated list of OCR languages. The OCR language can be provided in 2 ways:"
                " As a 'native' tag, which is specific to the selected OCR engine/backend, or as a"
                " canonicalized BCP-47 tag (e.g. 'en,de' or 'zh-Hant')."
                " By default the language is handled as a native tag and is passed through verbatim"
                " to the OCR engine."
                f" A BCP-47 tag must be prefixed with '{OcrLanguageResolver._ISO_PREFIX}'."
                " When an empty language is provided the OCR engine chooses the language."
                " An empty language triggers the OSD script detection for Tesseract and selects a "
                " default language for the other engines."
                " To skip OCR entirely use --no-ocr."
            ),
        ),
    ] = None,
    enrich_code: Annotated[
        bool,
        typer.Option(help="Enable the service's code enrichment model."),
    ] = False,
    enrich_formula: Annotated[
        bool,
        typer.Option(help="Enable the service's formula enrichment model."),
    ] = False,
    enrich_picture_classes: Annotated[
        bool,
        typer.Option(help="Enable the service's picture classification model."),
    ] = False,
    enrich_picture_description: Annotated[
        bool,
        typer.Option(help="Enable the service's picture description model."),
    ] = False,
    enrich_chart_extraction: Annotated[
        bool,
        typer.Option(help="Enable the service's chart data extraction."),
    ] = False,
    image_export_mode: Annotated[
        Optional[ImageRefMode],
        typer.Option(
            help="Image export mode for image-capable outputs (JSON, YAML, HTML, "
            "Markdown): embedded, placeholder, or referenced. If unset, the "
            "service default applies.",
        ),
    ] = None,
    page_range: Annotated[
        Optional[str],
        typer.Option(
            "--page-range",
            help="Only convert a range of pages, e.g. 1-4 (page numbers start at 1).",
        ),
    ] = None,
    document_timeout: Annotated[
        Optional[float],
        typer.Option(
            help="Server-side timeout for processing each document, in seconds.",
        ),
    ] = None,
    abort_on_error: Annotated[
        bool,
        typer.Option(
            "--abort-on-error/--no-abort-on-error",
            help="If enabled, the service aborts the batch on the first error.",
        ),
    ] = False,
    fail_on: Annotated[
        FailOn,
        typer.Option("--fail-on", help=FAIL_ON_HELP),
    ] = FailOn.FAILURE,
    max_concurrency: Annotated[
        int,
        typer.Option(
            help="Maximum number of documents converted concurrently against the "
            "service.",
        ),
    ] = DEFAULT_MAX_CONCURRENCY,
    timeout: Annotated[
        float,
        typer.Option(
            help="Client-side timeout waiting for each job to finish, in seconds.",
        ),
    ] = 300.0,
    watcher: Annotated[
        StatusWatcherKind,
        typer.Option(
            help="How the client tracks job status: websocket (default) or polling.",
        ),
    ] = StatusWatcherKind.WEBSOCKET,
    output: Annotated[
        Path,
        typer.Option(help="Output directory where results are saved."),
    ] = Path("."),
    verbose: Annotated[
        int,
        typer.Option(
            "--verbose",
            "-v",
            count=True,
            help="Set the verbosity level. -v for info logging, -vv for debug logging.",
        ),
    ] = 0,
) -> None:
    log_format = "%(asctime)s\t%(levelname)s\t%(name)s: %(message)s"
    if verbose == 0:
        logging.basicConfig(level=logging.WARNING, format=log_format)
    elif verbose == 1:
        logging.basicConfig(level=logging.INFO, format=log_format)
    else:
        logging.basicConfig(level=logging.DEBUG, format=log_format)

    from docling.cli.main import err_console, export_documents

    # 1. Resolve credentials: flag > environment variable (both by Typer) >
    # the env file that the user names.
    url_source = _service_url_source(ctx)
    if dotenv:
        if env_file is not None:
            err_console.print("[red]Use --env-file or --dotenv, not both.[/red]")
            raise typer.Exit(2)
        env_file = Path(".env")
        if not env_file.is_file():
            err_console.print("[red]--dotenv: no .env file in this directory.[/red]")
            raise typer.Exit(2)
    if env_file is not None:
        env_values = _read_env_file(env_file, err_console)
        if not service_url and env_values.get(_SERVICE_URL_VAR):
            service_url = env_values[_SERVICE_URL_VAR]
            url_source = str(env_file.resolve())
        if not api_key and env_values.get(_SERVICE_API_KEY_VAR):
            api_key = env_values[_SERVICE_API_KEY_VAR]

    if not service_url:
        err_console.print(
            "[red]No service URL. Pass --service-url, set DOCLING_SERVICE_URL, "
            "or read it from a file with --env-file or --dotenv.[/red]"
        )
        raise typer.Exit(2)
    # Show where the documents go before anything is sent.
    # soft_wrap keeps the line whole, and escape() keeps "[...]" in the URL or
    # path from being read as Rich markup.
    err_console.print(
        f"Docling service: {escape(service_url)} (from {escape(url_source)})",
        soft_wrap=True,
    )

    # Forward --from only when set: the default list can contain formats an
    # older service rejects.
    requested_from_formats = from_formats
    if from_formats is None:
        from_formats = list(InputFormat)
    if to_formats is None:
        to_formats = [OutputFormat.MARKDOWN]

    parsed_page_range = _parse_page_range(page_range)

    # 2. Collect sources — dirs filtered by --from, http(s) URLs stay strings.
    sources = _collect_sources(source, from_formats)

    # 3. Map flags -> ConvertDocumentsOptions (only fields the service honors).
    option_kwargs = {
        "to_formats": to_formats,
        "do_ocr": ocr,
        "force_ocr": force_ocr,
        "ocr_lang": _canonicalize_ocr_lang(ocr_lang),
        "do_table_structure": tables,
        "do_pdf_heading_hierarchy": heading_hierarchy,
        "pipeline": pipeline,
        "do_code_enrichment": enrich_code,
        "do_formula_enrichment": enrich_formula,
        "do_picture_classification": enrich_picture_classes,
        "do_picture_description": enrich_picture_description,
        "do_chart_extraction": enrich_chart_extraction,
        "document_timeout": document_timeout,
        "abort_on_error": abort_on_error,
    }
    if requested_from_formats is not None:
        option_kwargs["from_formats"] = requested_from_formats
    # Only send image_export_mode when the user set it, so the service/datamodel
    # default applies otherwise.
    if image_export_mode is not None:
        option_kwargs["image_export_mode"] = image_export_mode
    if parsed_page_range is not None:
        option_kwargs["page_range"] = parsed_page_range
    options = ConvertDocumentsRequestOptions(**option_kwargs)
    # Local export must match what the service produced: resolved value (the
    # user's override, or the options model default when unset).
    resolved_image_mode = options.image_export_mode

    # 4. Run and reuse the existing exporter.
    with DoclingServiceClient(
        url=service_url,
        api_key=api_key or "",
        status_watcher=StatusWatcherKind(watcher.value),
        job_timeout=timeout,
        max_concurrency=max_concurrency,
    ) as client:
        try:
            client.health()  # fail fast with a clear connection error
        except Exception as e:
            err_console.print(f"[red]Cannot reach service at {service_url}: {e}[/red]")
            raise typer.Exit(1)

        output.mkdir(parents=True, exist_ok=True)
        conv_results = client.convert_all(source=sources, options=options)
        try:
            summary = export_documents(
                conv_results,
                output_dir=output,
                **_export_flags_from_formats(to_formats),
                show_layout=False,
                print_timings=False,
                export_timings=False,
                image_export_mode=resolved_image_mode,
                chunker_type=chunker_type,
                chunk_max_tokens=chunk_max_tokens,
                chunk_tokenizer=chunk_tokenizer,
            )
        except typer.Exit:
            raise
        except Exception as e:
            err_console.print(f"[red]Conversion failed: {e}[/red]")
            raise typer.Exit(1)
        if summary.should_fail(fail_on):
            raise typer.Exit(1)


_SERVICE_URL_VAR = "DOCLING_SERVICE_URL"
_SERVICE_API_KEY_VAR = "DOCLING_SERVICE_API_KEY"


def _service_url_source(ctx: typer.Context) -> str:
    """Name where Typer took ``--service-url`` from, for the address line."""
    source = ctx.get_parameter_source("service_url")
    # Compare by name: Typer >= 0.26 vendors click, so the ParameterSource enum
    # class depends on the Typer version.
    if source is not None and source.name == "ENVIRONMENT":
        return _SERVICE_URL_VAR
    return "--service-url"


def _read_env_file(path: Path, err_console: "Console") -> dict[str, str]:
    """Read the service settings from an env file that the user names.

    Only ``DOCLING_SERVICE_URL`` and ``DOCLING_SERVICE_API_KEY`` are returned,
    and ``os.environ`` is not changed. ``${VAR}`` references are kept as text,
    so a value in the file cannot copy other environment variables.
    """
    try:
        from dotenv import dotenv_values
    except ImportError:
        err_console.print(
            "[red]Reading an env file needs python-dotenv. Install the `cli` "
            "extra, or pass --service-url and --api-key.[/red]"
        )
        raise typer.Exit(2)
    values = dotenv_values(path, interpolate=False)
    return {
        key: value
        for key in (_SERVICE_URL_VAR, _SERVICE_API_KEY_VAR)
        if (value := values.get(key))
    }


def _app_callback() -> None:
    """Group callback, so that the app help text shows."""


def register(app: "typer.Typer") -> None:
    """Attach the ``convert-remote`` command and the app help to ``app``.

    Called from ``docling.cli.main`` after the app and shared helpers are defined
    and before the click command is built, so registration order is deterministic
    regardless of which module is imported first.
    """
    app.callback(
        help=(
            "Convert documents to a unified representation, locally with "
            "`convert` or through a docling-serve service with `convert-remote`."
        ),
    )(_app_callback)
    app.command(
        "convert-remote",
        no_args_is_help=True,
        help=_REMOTE_HELP,
        short_help=(
            "Convert via a remote docling-serve service "
            "(see `docling convert-remote --help`)."
        ),
        epilog=_REMOTE_EPILOG,
    )(convert_remote)
