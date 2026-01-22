import datetime
import importlib
import logging
import platform
import re
import sys
import tempfile
import time
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Dict, List, Optional, Type

import rich.table
import typer
from docling_core.transforms.serializer.html import (
    HTMLDocSerializer,
    HTMLOutputStyle,
    HTMLParams,
)
from docling_core.transforms.visualizer.layout_visualizer import LayoutVisualizer
from docling_core.types.doc import ImageRefMode
from docling_core.utils.file import resolve_source_to_path
from pydantic import TypeAdapter
from rich.console import Console

from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
from docling.backend.image_backend import ImageDocumentBackend
from docling.backend.mets_gbs_backend import MetsGbsDocumentBackend
from docling.backend.pdf_backend import PdfDocumentBackend
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel import vlm_model_specs
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.asr_model_specs import (
    WHISPER_BASE,
    WHISPER_BASE_MLX,
    WHISPER_BASE_NATIVE,
    WHISPER_LARGE,
    WHISPER_LARGE_MLX,
    WHISPER_LARGE_NATIVE,
    WHISPER_MEDIUM,
    WHISPER_MEDIUM_MLX,
    WHISPER_MEDIUM_NATIVE,
    WHISPER_SMALL,
    WHISPER_SMALL_MLX,
    WHISPER_SMALL_NATIVE,
    WHISPER_TINY,
    WHISPER_TINY_MLX,
    WHISPER_TINY_NATIVE,
    WHISPER_TURBO,
    WHISPER_TURBO_MLX,
    WHISPER_TURBO_NATIVE,
    AsrModelType,
)
from docling.datamodel.backend_options import PdfBackendOptions
from docling.datamodel.base_models import (
    ConversionStatus,
    FormatToExtensions,
    InputFormat,
    OutputFormat,
)
from docling.datamodel.document import ConversionResult, DoclingVersion
from docling.datamodel.pipeline_options import (
    AsrPipelineOptions,
    ConvertPipelineOptions,
    OcrAutoOptions,
    OcrOptions,
    PaginatedPipelineOptions,
    PdfBackend,
    PdfPipelineOptions,
    PipelineOptions,
    ProcessingPipeline,
    TableFormerMode,
    TableStructureOptions,
    TesseractCliOcrOptions,
    TesseractOcrOptions,
    VlmPipelineOptions,
)
from docling.datamodel.settings import settings
from docling.datamodel.vlm_model_specs import VlmModelType
from docling.document_converter import (
    AudioFormatOption,
    DocumentConverter,
    ExcelFormatOption,
    FormatOption,
    HTMLFormatOption,
    MarkdownFormatOption,
    PdfFormatOption,
    PowerpointFormatOption,
    WordFormatOption,
)
from docling.models.factories import (
    get_layout_factory,
    get_ocr_factory,
    get_table_structure_factory,
)
from docling.models.factories.base_factory import BaseFactory
from docling.pipeline.asr_pipeline import AsrPipeline
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling.utils.profiling import ProfilingItem

warnings.filterwarnings(action="ignore", category=UserWarning, module="pydantic|torch")
warnings.filterwarnings(action="ignore", category=FutureWarning, module="easyocr")

_log = logging.getLogger(__name__)

console = Console()
err_console = Console(stderr=True)

ocr_factory_internal = get_ocr_factory(allow_external_plugins=False)
ocr_engines_enum_internal = ocr_factory_internal.get_enum()

DOCLING_ASCII_ART = r"""
                             ████ ██████
                           ███░░██░░░░░██████
                      ████████░░░░░░░░████████████
                   ████████░░░░░░░░░░░░░░░░░░████████
                 ██████░░░░░░░░░░░░░░░░░░░░░░░░░░██████
              ██████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░█████
            ██████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░█████
          ██████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░██████
         ██████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░██████
        ██████░░░░░░░   ░░░░░░░░░░░░░░░░░░░░░░   ░░░░░░░██████
       ██████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░██████
      ██████░░░░░░         ░░░░░░░░░░░░░░░          ░░░░░░██████
      ███▒██░░░░░   ████     ░░░░░░░░░░░░   ████     ░░░░░██▒███
     ███▒██░░░░░░  ████      ░░░░░░░░░░░░  ████      ░░░░░██▒████
     ███▒██░░░░░░  ██     ██ ░░░░░░░░░░░░  ██     ██ ░░░░░██▒▒███
     ███▒███░░░░░        ██  ░░░░████░░░░        ██  ░░░░░██▒▒███
    ████▒▒██░░░░░░         ░░░███▒▒▒▒███░░░        ░░░░░░░██▒▒████
    ████▒▒██░░░░░░░░░░░░░░░░░█▒▒▒▒▒▒▒▒▒▒█░░░░░░░░░░░░░░░░███▒▒████
    ████▒▒▒██░░░░░░░░░░░░█████  ▒▒▒▒▒▒  ██████░░░░░░░░░░░██▒▒▒████
     ███▒▒▒▒██░░░░░░░░███▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒███░░░░░░░░██▒▒▒▒███
     ███▒▒▒▒▒███░░░░░░██▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒██░░░░░░███▒▒▒▒▒███
     ████▒▒▒▒▒████░░░░░░██████████████████████░░░░░░████▒▒▒▒▒████
      ███▒▒▒▒▒▒▒▒████░░░░░░░░░░░░░░░░░░░░░░░░░░░████▒▒▒▒▒▒▒▒▒███
      ████▒▒▒▒▒▒▒▒███░░░░░████████████████████████▒▒▒▒▒▒▒▒▒████
       ████▒▒▒▒▒▒██░░░░░░█                   █░░░░░██▒▒▒▒▒▒████
        ████▒▒▒▒█░░░░░░░█   D O C L I N G   █░░░░░░░░██▒▒▒████
         ████▒▒██░░░░░░█                   █░░░░░░░░░░█▒▒████
          ██████░░░░░░█   D O C L I N G   █░░░░░░░░░░░██████
            ████░░░░░█                   █░░░░░░░░░░░░████
             █████░░█   D O C L I N G   █░░░░░░░░░░░█████
               █████                   █░░░░░░░░████████
                 ██   D O C L I N G   █░░░░░░░░█████
                 █                   █░░░████████
                █████████████████████████████
"""


app = typer.Typer(
    name="Docling",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)


def logo_callback(value: bool):
    if value:
        print(DOCLING_ASCII_ART)
        raise typer.Exit()


def version_callback(value: bool):
    if value:
        v = DoclingVersion()
        print(f"Docling version: {v.docling_version}")
        print(f"Docling Core version: {v.docling_core_version}")
        print(f"Docling IBM Models version: {v.docling_ibm_models_version}")
        print(f"Docling Parse version: {v.docling_parse_version}")
        print(f"Python: {v.py_impl_version} ({v.py_lang_version})")
        print(f"Platform: {v.platform_str}")
        raise typer.Exit()


def show_external_plugins_callback(value: bool):
    if value:
        ocr_factory_all = get_ocr_factory(allow_external_plugins=True)
        layout_factory_all = get_layout_factory(allow_external_plugins=True)
        table_factory_all = get_table_structure_factory(allow_external_plugins=True)

        def print_external_plugins(factory: BaseFactory, factory_name: str):
            table = rich.table.Table(title=f"Available {factory_name} engines")
            table.add_column("Name", justify="right")
            table.add_column("Plugin")
            table.add_column("Package")
            for meta in factory.registered_meta.values():
                if not meta.module.startswith("docling."):
                    table.add_row(
                        f"[bold]{meta.kind}[/bold]",
                        meta.plugin_name,
                        meta.module.split(".")[0],
                    )
            rich.print(table)

        print_external_plugins(ocr_factory_all, "OCR")
        print_external_plugins(layout_factory_all, "layout")
        print_external_plugins(table_factory_all, "table")

        raise typer.Exit()


def export_documents(
    conv_results: Iterable[ConversionResult],
    output_dir: Path,
    export_json: bool,
    export_yaml: bool,
    export_html: bool,
    export_html_split_page: bool,
    show_layout: bool,
    export_md: bool,
    export_txt: bool,
    export_doctags: bool,
    print_timings: bool,
    export_timings: bool,
    image_export_mode: ImageRefMode,
):
    success_count = 0
    failure_count = 0

    for conv_res in conv_results:
        if conv_res.status == ConversionStatus.SUCCESS:
            success_count += 1
            doc_filename = conv_res.input.file.stem

            # Export JSON format:
            if export_json:
                fname = output_dir / f"{doc_filename}.json"
                _log.info(f"writing JSON output to {fname}")
                conv_res.document.save_as_json(
                    filename=fname, image_mode=image_export_mode
                )

            # Export YAML format:
            if export_yaml:
                fname = output_dir / f"{doc_filename}.yaml"
                _log.info(f"writing YAML output to {fname}")
                conv_res.document.save_as_yaml(
                    filename=fname, image_mode=image_export_mode
                )

            # Export HTML format:
            if export_html:
                fname = output_dir / f"{doc_filename}.html"
                _log.info(f"writing HTML output to {fname}")
                conv_res.document.save_as_html(
                    filename=fname, image_mode=image_export_mode, split_page_view=False
                )

            # Export HTML format:
            if export_html_split_page:
                fname = output_dir / f"{doc_filename}.html"
                _log.info(f"writing HTML output to {fname}")
                if show_layout:
                    ser = HTMLDocSerializer(
                        doc=conv_res.document,
                        params=HTMLParams(
                            image_mode=image_export_mode,
                            output_style=HTMLOutputStyle.SPLIT_PAGE,
                        ),
                    )
                    visualizer = LayoutVisualizer()
                    visualizer.params.show_label = False
                    ser_res = ser.serialize(
                        visualizer=visualizer,
                    )
                    with open(fname, "w") as fw:
                        fw.write(ser_res.text)
                else:
                    conv_res.document.save_as_html(
                        filename=fname,
                        image_mode=image_export_mode,
                        split_page_view=True,
                    )

            # Export Text format:
            if export_txt:
                fname = output_dir / f"{doc_filename}.txt"
                _log.info(f"writing TXT output to {fname}")
                conv_res.document.save_as_markdown(
                    filename=fname,
                    strict_text=True,
                    image_mode=ImageRefMode.PLACEHOLDER,
                )

            # Export Markdown format:
            if export_md:
                fname = output_dir / f"{doc_filename}.md"
                _log.info(f"writing Markdown output to {fname}")
                conv_res.document.save_as_markdown(
                    filename=fname, image_mode=image_export_mode
                )

            # Export Document Tags format:
            if export_doctags:
                fname = output_dir / f"{doc_filename}.doctags"
                _log.info(f"writing Doc Tags output to {fname}")
                conv_res.document.save_as_doctags(filename=fname)

            # Print profiling timings
            if print_timings:
                table = rich.table.Table(title=f"Profiling Summary, {doc_filename}")
                metric_columns = [
                    "Stage",
                    "count",
                    "total",
                    "mean",
                    "median",
                    "min",
                    "max",
                    "0.1 percentile",
                    "0.9 percentile",
                ]
                for col in metric_columns:
                    table.add_column(col, style="bold")
                for stage_key, item in conv_res.timings.items():
                    col_dict = {
                        "Stage": stage_key,
                        "count": item.count,
                        "total": item.total(),
                        "mean": item.avg(),
                        "median": item.percentile(0.5),
                        "min": item.percentile(0.0),
                        "max": item.percentile(1.0),
                        "0.1 percentile": item.percentile(0.1),
                        "0.9 percentile": item.percentile(0.9),
                    }
                    row_values = [str(col_dict[col]) for col in metric_columns]
                    table.add_row(*row_values)

                console.print(table)

            # Export profiling timings
            if export_timings:
                TimingsT = TypeAdapter(dict[str, ProfilingItem])
                now = datetime.datetime.now()
                timings_file = Path(
                    output_dir / f"{doc_filename}-timings-{now:%Y-%m-%d_%H-%M-%S}.json"
                )
                with timings_file.open("wb") as fp:
                    r = TimingsT.dump_json(conv_res.timings, indent=2)
                    fp.write(r)

        else:
            _log.warning(f"Document {conv_res.input.file} failed to convert.")
            if _log.isEnabledFor(logging.INFO):
                for err in conv_res.errors:
                    _log.info(
                        f"  [Failure Detail] Component: {err.component_type}, "
                        f"Module: {err.module_name}, Message: {err.error_message}"
                    )
            failure_count += 1

    _log.info(
        f"Processed {success_count + failure_count} docs, of which {failure_count} failed"
    )


def _split_list(raw: Optional[str]) -> Optional[List[str]]:
    if raw is None:
        return None
    return re.split(r"[;,]", raw)


@app.command(no_args_is_help=True)
def convert(  # noqa: C901
    input_sources: Annotated[
        List[str],
        typer.Argument(
            ...,
            metavar="source",
            help="PDF files to convert. Can be local file / directory paths or URL.",
        ),
    ],
    from_formats: List[InputFormat] = typer.Option(
        None,
        "--from",
        help="Specify input formats to convert from. Defaults to all formats.",
    ),
    to_formats: List[OutputFormat] = typer.Option(
        None, "--to", help="Specify output formats. Defaults to Markdown."
    ),
    show_layout: Annotated[
        bool,
        typer.Option(
            ...,
            help="If enabled, the page images will show the bounding-boxes of the items.",
        ),
    ] = False,
    headers: str = typer.Option(
        None,
        "--headers",
        help="Specify http request headers used when fetching url input sources in the form of a JSON string",
    ),
    image_export_mode: Annotated[
        ImageRefMode,
        typer.Option(
            ...,
            help="Image export mode for the document (only in case of JSON, Markdown or HTML). With `placeholder`, only the position of the image is marked in the output. In `embedded` mode, the image is embedded as base64 encoded string. In `referenced` mode, the image is exported in PNG format and referenced from the main exported document.",
        ),
    ] = ImageRefMode.EMBEDDED,
    pipeline: Annotated[
        ProcessingPipeline,
        typer.Option(..., help="Choose the pipeline to process PDF or image files."),
    ] = ProcessingPipeline.STANDARD,
    vlm_model: Annotated[
        VlmModelType,
        typer.Option(..., help="Choose the VLM model to use with PDF or image files."),
    ] = VlmModelType.GRANITEDOCLING,
    asr_model: Annotated[
        AsrModelType,
        typer.Option(..., help="Choose the ASR model to use with audio/video files."),
    ] = AsrModelType.WHISPER_TINY,
    ocr: Annotated[
        bool,
        typer.Option(
            ..., help="If enabled, the bitmap content will be processed using OCR."
        ),
    ] = True,
    force_ocr: Annotated[
        bool,
        typer.Option(
            ...,
            help="Replace any existing text with OCR generated text over the full content.",
        ),
    ] = False,
    tables: Annotated[
        bool,
        typer.Option(
            ...,
            help="If enabled, the table structure model will be used to extract table information.",
        ),
    ] = True,
    ocr_engine: Annotated[
        str,
        typer.Option(
            ...,
            help=(
                f"The OCR engine to use. When --allow-external-plugins is *not* set, the available values are: "
                f"{', '.join(o.value for o in ocr_engines_enum_internal)}. "
                f"Use the option --show-external-plugins to see the options allowed with external plugins."
            ),
        ),
    ] = OcrAutoOptions.kind,
    ocr_lang: Annotated[
        Optional[str],
        typer.Option(
            ...,
            help="Provide a comma-separated list of languages used by the OCR engine. Note that each OCR engine has different values for the language names.",
        ),
    ] = None,
    psm: Annotated[
        Optional[int],
        typer.Option(
            ...,
            help="Page Segmentation Mode for the OCR engine (0-13).",
        ),
    ] = None,
    pdf_backend: Annotated[
        PdfBackend, typer.Option(..., help="The PDF backend to use.")
    ] = PdfBackend.DLPARSE_V4,
    pdf_password: Annotated[
        Optional[str], typer.Option(..., help="Password for protected PDF documents")
    ] = None,
    table_mode: Annotated[
        TableFormerMode,
        typer.Option(..., help="The mode to use in the table structure model."),
    ] = TableFormerMode.ACCURATE,
    enrich_code: Annotated[
        bool,
        typer.Option(..., help="Enable the code enrichment model in the pipeline."),
    ] = False,
    enrich_formula: Annotated[
        bool,
        typer.Option(..., help="Enable the formula enrichment model in the pipeline."),
    ] = False,
    enrich_picture_classes: Annotated[
        bool,
        typer.Option(
            ...,
            help="Enable the picture classification enrichment model in the pipeline.",
        ),
    ] = False,
    enrich_picture_description: Annotated[
        bool,
        typer.Option(..., help="Enable the picture description model in the pipeline."),
    ] = False,
    artifacts_path: Annotated[
        Optional[Path],
        typer.Option(..., help="If provided, the location of the model artifacts."),
    ] = None,
    enable_remote_services: Annotated[
        bool,
        typer.Option(
            ..., help="Must be enabled when using models connecting to remote services."
        ),
    ] = False,
    allow_external_plugins: Annotated[
        bool,
        typer.Option(
            ..., help="Must be enabled for loading modules from third-party plugins."
        ),
    ] = False,
    show_external_plugins: Annotated[
        bool,
        typer.Option(
            ...,
            help="List the third-party plugins which are available when the option --allow-external-plugins is set.",
            callback=show_external_plugins_callback,
            is_eager=True,
        ),
    ] = False,
    abort_on_error: Annotated[
        bool,
        typer.Option(
            ...,
            "--abort-on-error/--no-abort-on-error",
            help="If enabled, the processing will be aborted when the first error is encountered.",
        ),
    ] = False,
    output: Annotated[
        Path, typer.Option(..., help="Output directory where results are saved.")
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
    debug_visualize_cells: Annotated[
        bool,
        typer.Option(..., help="Enable debug output which visualizes the PDF cells"),
    ] = False,
    debug_visualize_ocr: Annotated[
        bool,
        typer.Option(..., help="Enable debug output which visualizes the OCR cells"),
    ] = False,
    debug_visualize_layout: Annotated[
        bool,
        typer.Option(
            ..., help="Enable debug output which visualizes the layour clusters"
        ),
    ] = False,
    debug_visualize_tables: Annotated[
        bool,
        typer.Option(..., help="Enable debug output which visualizes the table cells"),
    ] = False,
    version: Annotated[
        Optional[bool],
        typer.Option(
            "--version",
            callback=version_callback,
            is_eager=True,
            help="Show version information.",
        ),
    ] = None,
    document_timeout: Annotated[
        Optional[float],
        typer.Option(
            ...,
            help="The timeout for processing each document, in seconds.",
        ),
    ] = None,
    num_threads: Annotated[int, typer.Option(..., help="Number of threads")] = 4,
    device: Annotated[
        AcceleratorDevice, typer.Option(..., help="Accelerator device")
    ] = AcceleratorDevice.AUTO,
    docling_logo: Annotated[
        Optional[bool],
        typer.Option(
            "--logo", callback=logo_callback, is_eager=True, help="Docling logo"
        ),
    ] = None,
    page_batch_size: Annotated[
        int,
        typer.Option(
            "--page-batch-size",
            help=f"Number of pages processed in one batch. Default: {settings.perf.page_batch_size}",
        ),
    ] = settings.perf.page_batch_size,
    profiling: Annotated[
        bool,
        typer.Option(
            ...,
            help="If enabled, it summarizes profiling details for all conversion stages.",
        ),
    ] = False,
    save_profiling: Annotated[
        bool,
        typer.Option(
            ...,
            help="If enabled, it saves the profiling summaries to json.",
        ),
    ] = False,
):
    log_format = "%(asctime)s\t%(levelname)s\t%(name)s: %(message)s"

    if verbose == 0:
        logging.basicConfig(level=logging.WARNING, format=log_format)
    elif verbose == 1:
        logging.basicConfig(level=logging.INFO, format=log_format)
    else:
        logging.basicConfig(level=logging.DEBUG, format=log_format)

    settings.debug.visualize_cells = debug_visualize_cells
    settings.debug.visualize_layout = debug_visualize_layout
    settings.debug.visualize_tables = debug_visualize_tables
    settings.debug.visualize_ocr = debug_visualize_ocr
    settings.perf.page_batch_size = page_batch_size

    if from_formats is None:
        from_formats = list(InputFormat)

    parsed_headers: Optional[Dict[str, str]] = None
    if headers is not None:
        headers_t = TypeAdapter(Dict[str, str])
        parsed_headers = headers_t.validate_json(headers)

    if profiling or save_profiling:
        settings.debug.profile_pipeline_timings = True

    with tempfile.TemporaryDirectory() as tempdir:
        input_doc_paths: List[Path] = []
        for src in input_sources:
            try:
                # check if we can fetch some remote url
                source = resolve_source_to_path(
                    source=src, headers=parsed_headers, workdir=Path(tempdir)
                )
                input_doc_paths.append(source)
            except FileNotFoundError:
                err_console.print(
                    f"[red]Error: The input file {src} does not exist.[/red]"
                )
                raise typer.Abort()
            except IsADirectoryError:
                # if the input matches to a file or a folder
                try:
                    local_path = TypeAdapter(Path).validate_python(src)
                    if local_path.exists() and local_path.is_dir():
                        for fmt in from_formats:
                            for ext in FormatToExtensions[fmt]:
                                for path in local_path.glob(f"**/*.{ext}"):
                                    if path.name.startswith("~$") and ext == "docx":
                                        _log.info(
                                            f"Ignoring temporary Word file: {path}"
                                        )
                                        continue
                                    input_doc_paths.append(path)

                                for path in local_path.glob(f"**/*.{ext.upper()}"):
                                    if path.name.startswith("~$") and ext == "docx":
                                        _log.info(
                                            f"Ignoring temporary Word file: {path}"
                                        )
                                        continue
                                    input_doc_paths.append(path)
                    elif local_path.exists():
                        if not local_path.name.startswith("~$") and ext == "docx":
                            _log.info(f"Ignoring temporary Word file: {path}")
                            continue
                        input_doc_paths.append(local_path)
                    else:
                        err_console.print(
                            f"[red]Error: The input file {src} does not exist.[/red]"
                        )
                        raise typer.Abort()
                except Exception as err:
                    err_console.print(f"[red]Error: Cannot read the input {src}.[/red]")
                    _log.info(err)  # will print more details if verbose is activated
                    raise typer.Abort()

        if to_formats is None:
            to_formats = [OutputFormat.MARKDOWN]

        export_json = OutputFormat.JSON in to_formats
        export_yaml = OutputFormat.YAML in to_formats
        export_html = OutputFormat.HTML in to_formats
        export_html_split_page = OutputFormat.HTML_SPLIT_PAGE in to_formats
        export_md = OutputFormat.MARKDOWN in to_formats
        export_txt = OutputFormat.TEXT in to_formats
        export_doctags = OutputFormat.DOCTAGS in to_formats

        ocr_factory = get_ocr_factory(allow_external_plugins=allow_external_plugins)
        ocr_options: OcrOptions = ocr_factory.create_options(  # type: ignore
            kind=ocr_engine,
            force_full_page_ocr=force_ocr,
        )

        ocr_lang_list = _split_list(ocr_lang)
        if ocr_lang_list is not None:
            ocr_options.lang = ocr_lang_list
        if psm is not None and isinstance(
            ocr_options, TesseractOcrOptions | TesseractCliOcrOptions
        ):
            ocr_options.psm = psm

        accelerator_options = AcceleratorOptions(num_threads=num_threads, device=device)

        # pipeline_options: PaginatedPipelineOptions
        pipeline_options: PipelineOptions

        format_options: Dict[InputFormat, FormatOption] = {}
        pdf_backend_options: Optional[PdfBackendOptions] = PdfBackendOptions(
            password=pdf_password
        )

        if pipeline == ProcessingPipeline.STANDARD:
            pipeline_options = PdfPipelineOptions(
                allow_external_plugins=allow_external_plugins,
                enable_remote_services=enable_remote_services,
                accelerator_options=accelerator_options,
                do_ocr=ocr,
                ocr_options=ocr_options,
                do_table_structure=tables,
                do_code_enrichment=enrich_code,
                do_formula_enrichment=enrich_formula,
                do_picture_description=enrich_picture_description,
                do_picture_classification=enrich_picture_classes,
                document_timeout=document_timeout,
            )
            if isinstance(
                pipeline_options.table_structure_options, TableStructureOptions
            ):
                pipeline_options.table_structure_options.do_cell_matching = (
                    True  # do_cell_matching
                )
                pipeline_options.table_structure_options.mode = table_mode

            if image_export_mode != ImageRefMode.PLACEHOLDER:
                pipeline_options.generate_page_images = True
                pipeline_options.generate_picture_images = (
                    True  # FIXME: to be deprecated in version 3
                )
                pipeline_options.images_scale = 2

            backend: Type[PdfDocumentBackend]
            if pdf_backend == PdfBackend.DLPARSE_V1:
                backend = DoclingParseDocumentBackend
                pdf_backend_options = None
            elif pdf_backend == PdfBackend.DLPARSE_V2:
                backend = DoclingParseV2DocumentBackend
                pdf_backend_options = None
            elif pdf_backend == PdfBackend.DLPARSE_V4:
                backend = DoclingParseV4DocumentBackend  # type: ignore
            elif pdf_backend == PdfBackend.PYPDFIUM2:
                backend = PyPdfiumDocumentBackend  # type: ignore
            else:
                raise RuntimeError(f"Unexpected PDF backend type {pdf_backend}")

            pdf_format_option = PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=backend,  # pdf_backend
                backend_options=pdf_backend_options,
            )

            # METS GBS options
            mets_gbs_options = pipeline_options.model_copy()
            mets_gbs_options.do_ocr = False
            mets_gbs_format_option = PdfFormatOption(
                pipeline_options=mets_gbs_options,
                backend=MetsGbsDocumentBackend,
            )

            # SimplePipeline options
            simple_format_option = ConvertPipelineOptions(
                do_picture_description=enrich_picture_description,
                do_picture_classification=enrich_picture_classes,
            )
            if artifacts_path is not None:
                simple_format_option.artifacts_path = artifacts_path

            # Use image-native backend for IMAGE to avoid pypdfium2 locking
            image_format_option = PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=ImageDocumentBackend,
                backend_options=pdf_backend_options,
            )

            format_options = {
                InputFormat.PDF: pdf_format_option,
                InputFormat.IMAGE: image_format_option,
                InputFormat.METS_GBS: mets_gbs_format_option,
                InputFormat.DOCX: WordFormatOption(
                    pipeline_options=simple_format_option
                ),
                InputFormat.PPTX: PowerpointFormatOption(
                    pipeline_options=simple_format_option
                ),
                InputFormat.XLSX: ExcelFormatOption(
                    pipeline_options=simple_format_option
                ),
                InputFormat.HTML: HTMLFormatOption(
                    pipeline_options=simple_format_option
                ),
                InputFormat.MD: MarkdownFormatOption(
                    pipeline_options=simple_format_option
                ),
            }

        elif pipeline == ProcessingPipeline.VLM:
            pipeline_options = VlmPipelineOptions(
                enable_remote_services=enable_remote_services,
            )

            if vlm_model == VlmModelType.GRANITE_VISION:
                pipeline_options.vlm_options = (
                    vlm_model_specs.GRANITE_VISION_TRANSFORMERS
                )
            elif vlm_model == VlmModelType.GRANITE_VISION_OLLAMA:
                pipeline_options.vlm_options = vlm_model_specs.GRANITE_VISION_OLLAMA
            elif vlm_model == VlmModelType.GOT_OCR_2:
                pipeline_options.vlm_options = vlm_model_specs.GOT2_TRANSFORMERS
            elif vlm_model == VlmModelType.SMOLDOCLING:
                pipeline_options.vlm_options = vlm_model_specs.SMOLDOCLING_TRANSFORMERS
                if sys.platform == "darwin":
                    try:
                        import mlx_vlm

                        pipeline_options.vlm_options = vlm_model_specs.SMOLDOCLING_MLX
                    except ImportError:
                        _log.warning(
                            "To run SmolDocling faster, please install mlx-vlm:\n"
                            "pip install mlx-vlm"
                        )

            elif vlm_model == VlmModelType.GRANITEDOCLING:
                pipeline_options.vlm_options = (
                    vlm_model_specs.GRANITEDOCLING_TRANSFORMERS
                )
                if sys.platform == "darwin":
                    try:
                        import mlx_vlm

                        pipeline_options.vlm_options = (
                            vlm_model_specs.GRANITEDOCLING_MLX
                        )
                    except ImportError:
                        _log.warning(
                            "To run GraniteDocling faster, please install mlx-vlm:\n"
                            "pip install mlx-vlm"
                        )

            elif vlm_model == VlmModelType.SMOLDOCLING_VLLM:
                pipeline_options.vlm_options = vlm_model_specs.SMOLDOCLING_VLLM

            elif vlm_model == VlmModelType.GRANITEDOCLING_VLLM:
                pipeline_options.vlm_options = vlm_model_specs.GRANITEDOCLING_VLLM

            elif vlm_model == VlmModelType.DEEPSEEKOCR_OLLAMA:
                pipeline_options.vlm_options = vlm_model_specs.DEEPSEEKOCR_OLLAMA

            pdf_format_option = PdfFormatOption(
                pipeline_cls=VlmPipeline, pipeline_options=pipeline_options
            )

            format_options = {
                InputFormat.PDF: pdf_format_option,
                InputFormat.IMAGE: pdf_format_option,
            }

        # Set ASR options
        asr_pipeline_options = AsrPipelineOptions(
            accelerator_options=AcceleratorOptions(
                device=device,
                num_threads=num_threads,
            ),
            # enable_remote_services=enable_remote_services,
            # artifacts_path = artifacts_path
        )

        # Auto-selecting models (choose best implementation for hardware)
        if asr_model == AsrModelType.WHISPER_TINY:
            asr_pipeline_options.asr_options = WHISPER_TINY
        elif asr_model == AsrModelType.WHISPER_SMALL:
            asr_pipeline_options.asr_options = WHISPER_SMALL
        elif asr_model == AsrModelType.WHISPER_MEDIUM:
            asr_pipeline_options.asr_options = WHISPER_MEDIUM
        elif asr_model == AsrModelType.WHISPER_BASE:
            asr_pipeline_options.asr_options = WHISPER_BASE
        elif asr_model == AsrModelType.WHISPER_LARGE:
            asr_pipeline_options.asr_options = WHISPER_LARGE
        elif asr_model == AsrModelType.WHISPER_TURBO:
            asr_pipeline_options.asr_options = WHISPER_TURBO

        # Explicit MLX models (force MLX implementation)
        elif asr_model == AsrModelType.WHISPER_TINY_MLX:
            asr_pipeline_options.asr_options = WHISPER_TINY_MLX
        elif asr_model == AsrModelType.WHISPER_SMALL_MLX:
            asr_pipeline_options.asr_options = WHISPER_SMALL_MLX
        elif asr_model == AsrModelType.WHISPER_MEDIUM_MLX:
            asr_pipeline_options.asr_options = WHISPER_MEDIUM_MLX
        elif asr_model == AsrModelType.WHISPER_BASE_MLX:
            asr_pipeline_options.asr_options = WHISPER_BASE_MLX
        elif asr_model == AsrModelType.WHISPER_LARGE_MLX:
            asr_pipeline_options.asr_options = WHISPER_LARGE_MLX
        elif asr_model == AsrModelType.WHISPER_TURBO_MLX:
            asr_pipeline_options.asr_options = WHISPER_TURBO_MLX

        # Explicit Native models (force native implementation)
        elif asr_model == AsrModelType.WHISPER_TINY_NATIVE:
            asr_pipeline_options.asr_options = WHISPER_TINY_NATIVE
        elif asr_model == AsrModelType.WHISPER_SMALL_NATIVE:
            asr_pipeline_options.asr_options = WHISPER_SMALL_NATIVE
        elif asr_model == AsrModelType.WHISPER_MEDIUM_NATIVE:
            asr_pipeline_options.asr_options = WHISPER_MEDIUM_NATIVE
        elif asr_model == AsrModelType.WHISPER_BASE_NATIVE:
            asr_pipeline_options.asr_options = WHISPER_BASE_NATIVE
        elif asr_model == AsrModelType.WHISPER_LARGE_NATIVE:
            asr_pipeline_options.asr_options = WHISPER_LARGE_NATIVE
        elif asr_model == AsrModelType.WHISPER_TURBO_NATIVE:
            asr_pipeline_options.asr_options = WHISPER_TURBO_NATIVE

        else:
            _log.error(f"{asr_model} is not known")
            raise ValueError(f"{asr_model} is not known")

        _log.debug(f"ASR pipeline_options: {asr_pipeline_options}")

        audio_format_option = AudioFormatOption(
            pipeline_cls=AsrPipeline,
            pipeline_options=asr_pipeline_options,
        )
        format_options[InputFormat.AUDIO] = audio_format_option

        # Common options for all pipelines
        if artifacts_path is not None:
            pipeline_options.artifacts_path = artifacts_path
            asr_pipeline_options.artifacts_path = artifacts_path

        doc_converter = DocumentConverter(
            allowed_formats=from_formats,
            format_options=format_options,
        )

        start_time = time.time()

        _log.info(f"paths: {input_doc_paths}")
        conv_results = doc_converter.convert_all(
            input_doc_paths, headers=parsed_headers, raises_on_error=abort_on_error
        )

        output.mkdir(parents=True, exist_ok=True)
        export_documents(
            conv_results,
            output_dir=output,
            export_json=export_json,
            export_yaml=export_yaml,
            export_html=export_html,
            export_html_split_page=export_html_split_page,
            show_layout=show_layout,
            export_md=export_md,
            export_txt=export_txt,
            export_doctags=export_doctags,
            print_timings=profiling,
            export_timings=save_profiling,
            image_export_mode=image_export_mode,
        )

        end_time = time.time() - start_time

    _log.info(f"All documents were converted in {end_time:.2f} seconds.")


click_app = typer.main.get_command(app)

if __name__ == "__main__":
    app()
