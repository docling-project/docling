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
from typing import Annotated, Optional

import typer
from docling_core.types.doc import ImageRefMode

from docling.cli.export_utils import _export_flags_from_formats, _split_list
from docling.datamodel.base_models import InputFormat, OutputFormat
from docling.datamodel.pipeline_options import ProcessingPipeline
from docling.datamodel.service.options import (
    ConvertDocumentsOptions as ConvertDocumentsRequestOptions,
)
from docling.datamodel.settings import PageRange
from docling.service_client import (
    DEFAULT_MAX_CONCURRENCY,
    DoclingServiceClient,
    StatusWatcherKind,
)

_log = logging.getLogger(__name__)

_REMOTE_HELP = """\
Convert documents through a remote docling-serve service instead of locally.

Sources may be local files, local directories (walked and filtered by --from),
or http(s) URLs. Results are written to --output in the formats given by --to,
identical to `docling convert`. Only options the service honors are exposed here;
local-execution flags (device, threads, pdf-backend internals, debug visualizers)
do not apply to remote conversion and are intentionally absent.
"""

_REMOTE_EPILOG = """\
Authentication (precedence: flag > environment variable > .env file):
  --service-url      or  DOCLING_SERVICE_URL     (required)
  --api-key          or  DOCLING_SERVICE_API_KEY (optional; omit if unauthenticated)
  A .env file in the working directory is loaded automatically when present.

Exit codes:
  0  success
  1  runtime or connection failure (service unreachable, conversion error)
  2  usage/config error (no service URL resolved from flag, env, or .env)

Examples:
  # Single file, credentials from environment / .env
  docling convert-remote report.pdf

  # Explicit endpoint + key, Markdown and JSON output to ./out
  docling convert-remote --service-url https://docling.example.com \\
    --api-key "$KEY" --to md --to json --output ./out report.pdf

  # Whole directory, only PDFs and DOCX, no OCR
  docling convert-remote --from pdf --from docx --no-ocr ./inbox
"""


def _parse_page_range(raw: Optional[str]) -> Optional[PageRange]:
    """Parse a ``--page-range`` value like ``1-4`` (or a single page ``4``).

    Page numbers start at 1. Returns ``None`` when no range is given so the
    service default (all pages) applies.
    """
    if raw is None:
        return None
    text = raw.strip()
    try:
        if "-" in text:
            start_str, end_str = text.split("-", 1)
            start, end = int(start_str), int(end_str)
        else:
            start = end = int(text)
    except ValueError:
        raise typer.BadParameter(
            f"Invalid --page-range {raw!r}. Use START-END (e.g. 1-4) or a single page.",
        )
    if start < 1 or end < start:
        raise typer.BadParameter(
            f"Invalid --page-range {raw!r}. Page numbers start at 1 and END must be "
            ">= START.",
        )
    return (start, end)


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
        _is_temporary_word_file,
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
        elif _is_temporary_word_file(local_path):
            _log.info(f"Ignoring temporary Word file: {local_path}")
        else:
            sources.append(local_path)
    return sources


def convert_remote(
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
            "DOCLING_SERVICE_URL or a .env file).",
        ),
    ] = None,
    api_key: Annotated[
        Optional[str],
        typer.Option(
            "--api-key",
            envvar="DOCLING_SERVICE_API_KEY",
            help="API key for the service (optional; falls back to "
            "DOCLING_SERVICE_API_KEY or a .env file; omit if unauthenticated).",
        ),
    ] = None,
    from_formats: Annotated[
        Optional[list[InputFormat]],
        typer.Option(
            "--from",
            help="Input formats to accept; filters directories and is sent as the "
            "server allow-list. Defaults to all supported formats.",
        ),
    ] = None,
    to_formats: Annotated[
        Optional[list[OutputFormat]],
        typer.Option(
            "--to",
            help="Output formats to produce and write locally. Defaults to Markdown.",
        ),
    ] = None,
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
    pipeline: Annotated[
        ProcessingPipeline,
        typer.Option(help="Pipeline the service uses to process PDF or image files."),
    ] = ProcessingPipeline.STANDARD,
    ocr_lang: Annotated[
        Optional[str],
        typer.Option(
            help="Comma-separated list of OCR languages (engine-specific names).",
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

    # 1. Validate credentials (flag/env/.env already merged by Typer + callback).
    if not service_url:
        err_console.print(
            "[red]No service URL. Pass --service-url, set DOCLING_SERVICE_URL, "
            "or add it to a .env file.[/red]"
        )
        raise typer.Exit(2)

    if from_formats is None:
        from_formats = list(InputFormat)
    if to_formats is None:
        to_formats = [OutputFormat.MARKDOWN]

    parsed_page_range = _parse_page_range(page_range)

    # 2. Collect sources — dirs filtered by --from, http(s) URLs stay strings.
    sources = _collect_sources(source, from_formats)

    # 3. Map flags -> ConvertDocumentsOptions (only fields the service honors).
    option_kwargs = {
        "from_formats": from_formats,
        "to_formats": to_formats,
        "do_ocr": ocr,
        "force_ocr": force_ocr,
        "ocr_lang": _split_list(ocr_lang),
        "do_table_structure": tables,
        "pipeline": pipeline,
        "do_code_enrichment": enrich_code,
        "do_formula_enrichment": enrich_formula,
        "do_picture_classification": enrich_picture_classes,
        "do_picture_description": enrich_picture_description,
        "do_chart_extraction": enrich_chart_extraction,
        "document_timeout": document_timeout,
        "abort_on_error": abort_on_error,
    }
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
        conv_results = client.convert_all(sources, options=options)
        try:
            export_documents(
                conv_results,
                output_dir=output,
                **_export_flags_from_formats(to_formats),
                show_layout=False,
                print_timings=False,
                export_timings=False,
                image_export_mode=resolved_image_mode,
            )
        except typer.Exit:
            raise
        except Exception as e:
            err_console.print(f"[red]Conversion failed: {e}[/red]")
            raise typer.Exit(1)


def _load_dotenv() -> None:
    """Load a working-directory ``.env`` before command params resolve.

    ``override=False`` keeps the intended precedence: a real environment variable
    beats ``.env``, and Typer's flag-beats-envvar handling stays on top. ``.env``
    support is best-effort — flags and environment variables still work without it.
    """
    try:
        from dotenv import find_dotenv, load_dotenv

        # usecwd=True searches from the user's working directory; the default
        # (usecwd=False) searches up from this installed package dir and never
        # finds the user's .env.
        load_dotenv(find_dotenv(usecwd=True), override=False)
    except ImportError:
        pass


def register(app: "typer.Typer") -> None:
    """Attach the ``convert-remote`` command and the ``.env`` loader to ``app``.

    Called from ``docling.cli.main`` after the app and shared helpers are defined
    and before the click command is built, so registration order is deterministic
    regardless of which module is imported first.
    """
    app.callback(
        help=(
            "Convert documents to a unified representation, locally with "
            "`convert` or through a docling-serve service with `convert-remote`."
        ),
    )(_load_dotenv)
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
