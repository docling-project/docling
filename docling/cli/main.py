import datetime
import logging
import re
import sys
import tempfile
import time
import warnings
from collections.abc import Iterable
from enum import Enum
from pathlib import Path
from typing import Annotated, Type
from urllib.parse import urlparse
from docling.datamodel.service.responses import ChunkedDocumentResultItem

# Check for CLI dependencies
try:
    import rich.table
    import typer
except ImportError as e:
    missing_package = str(e).split("'")[1] if "'" in str(e) else "typer or rich"
    print(
        f"Error: Missing required CLI dependency '{missing_package}'", file=sys.stderr
    )
    print("\nThe docling CLI requires additional dependencies.", file=sys.stderr)
    print("Please install them using one of the following options:\n", file=sys.stderr)
    print("  1. Install the full docling package (recommended):", file=sys.stderr)
    print("     pip install docling\n", file=sys.stderr)
    print("  2. Install docling-slim with CLI support:", file=sys.stderr)
    print("     pip install docling-slim[cli]\n", file=sys.stderr)
    print("  3. Install just the missing dependencies:", file=sys.stderr)
    print("     pip install typer rich\n", file=sys.stderr)
    sys.exit(1)

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

from docling.cli.export_utils import (
    _export_flags_from_formats,
    _is_empty_output,
    _should_generate_export_images,
    _split_list,
)
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
from docling.datamodel.backend_options import (
    EpubBackendOptions,
    HTMLBackendOptions,
    LatexBackendOptions,
    PdfBackendOptions,
    ThreadedDoclingParseBackendOptions,
)
from docling.datamodel.base_models import (
    ConversionStatus,
    DoclingComponentType,
    ErrorItem,
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
    PdfBackend,
    PdfPipelineOptions,
    PipelineOptions,
    ProcessingPipeline,
    TableFormerMode,
    TableStructureOptions,
    TesseractCliOcrOptions,
    TesseractOcrOptions,
    VlmConvertOptions,
    VlmPipelineOptions,
    normalize_pdf_backend,
)
from docling.datamodel.settings import settings
from docling.document_converter import (
    AudioFormatOption,
    DocumentConverter,
    EpubFormatOption,
    ExcelFormatOption,
    FormatOption,
    HTMLFormatOption,
    LatexFormatOption,
    MarkdownFormatOption,
    OdpFormatOption,
    OdsFormatOption,
    OdtFormatOption,
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
from docling.utils.profiling import ProfilingItem

warnings.filterwarnings(action="ignore", category=UserWarning, module="pydantic|torch")
warnings.filterwarnings(action="ignore", category=FutureWarning, module="easyocr")

_log = logging.getLogger(__name__)

console = Console()
err_console = Console(stderr=True)


class HtmlImageFetchMode(str, Enum):
    NONE = "none"
    LOCAL = "local"
    REMOTE = "remote"
    ALL = "all"


class ChunkerType(str, Enum):
    HYBRID = "hybrid"
    HIERARCHICAL = "hierarchical"


def _is_http_url(source: str) -> bool:
    parsed = urlparse(source)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _name_matches_format(name: str, format: InputFormat) -> bool:
    name_lower = name.lower()
    return any(
        name_lower.endswith(f".{extension.lower()}")
        for extension in FormatToExtensions[format]
    )


def _is_html_source(source: str, from_formats: list[InputFormat]) -> bool:
    if InputFormat.HTML not in from_formats:
        return False
    if len(from_formats) == 1:
        return True

    source_name = urlparse(source).path if _is_http_url(source) else source
    return _name_matches_format(source_name, InputFormat.HTML)


def _is_temporary_word_file(path: Path) -> bool:
    return path.name.startswith("~$") and path.suffix.lower() == ".docx"


def _iter_input_paths_from_directory(
    local_path: Path, from_formats: list[InputFormat]
) -> Iterable[Path]:
    seen_paths: set[Path] = set()
    for path in sorted(local_path.rglob("*")):
        if not path.is_file() or not any(
            _name_matches_format(path.name, fmt) for fmt in from_formats
        ):
            continue
        if _is_temporary_word_file(path):
            _log.info(f"Ignoring temporary Word file: {path}")
            continue
        if path not in seen_paths:
            seen_paths.add(path)
            yield path


def _expand_from_formats(from_formats: list[str] | None) -> list[InputFormat]:
    if from_formats is None:
        return list(InputFormat)

    expanded_formats: list[InputFormat] = []
    for from_format in from_formats:
        normalized_format = from_format.lower()
        if normalized_format == "odf":
            expanded_formats.extend([InputFormat.ODT, InputFormat.ODS, InputFormat.ODP])
            continue
        try:
            expanded_formats.append(InputFormat(normalized_format))
        except ValueError:
            choices = ", ".join([format.value for format in InputFormat] + ["odf"])
            raise typer.BadParameter(
                f"{from_format!r} is not one of {choices}"
            ) from None

    return list(dict.fromkeys(expanded_formats))


ocr_factory_internal = get_ocr_factory(allow_external_plugins=False)
ocr_engines_enum_internal = ocr_factory_internal.get_enum()

# Get available VLM presets from the registry
vlm_preset_ids = VlmConvertOptions.list_preset_ids()

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


class _DefaultCommandGroup(typer.core.TyperGroup):
    """Route a bare ``docling <source>`` invocation to the ``convert`` command.

    Historically the CLI exposed a single command, so Typer let users run
    ``docling report.pdf`` without naming it. Adding a second command
    (``convert-remote``) would otherwise force ``docling convert report.pdf``
    on everyone. This group preserves the old behavior: when the first token is
    not a known subcommand (nor the top-level ``--help``), it is treated as
    arguments to ``convert``. ``docling --help`` still shows the command list.
    """

    default_command = "convert"

    def parse_args(self, ctx, args):
        if args and args[0] not in self.commands and args[0] not in ("--help", "-h"):
            args = [self.default_command, *args]
        return super().parse_args(ctx, args)


app = typer.Typer(
    name="Docling",
    cls=_DefaultCommandGroup,
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
    epilog=(
        "Remote conversion: when installed with the `service-client` extra, "
        "use `docling convert-remote` and read `docling convert-remote --help` "
        "for authentication (DOCLING_SERVICE_URL / DOCLING_SERVICE_API_KEY), "
        "supported options, and exit codes before invoking it."
    ),
)


def logo_callback(value: bool):
    if value:
        print(DOCLING_ASCII_ART)
        raise typer.Exit()


def version_callback(value: bool):
    if value:
        v = DoclingVersion()
        docling_version = (
            v.docling_version
            if v.docling_version != "unknown"
            else v.docling_slim_version
        )
        print(f"Docling version: {docling_version}")
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
    export_vtt: bool,
    export_doclang: bool,
    print_timings: bool,
    export_timings: bool,
    image_export_mode: ImageRefMode,
    export_chunks: bool = False,
    chunker_type: ChunkerType = ChunkerType.HYBRID,
    chunk_max_tokens: int | None = None,
    chunk_tokenizer: str = "sentence-transformers/all-MiniLM-L6-v2",
):
    success_count = 0
    failure_count = 0

    for conv_res in conv_results:
        doc_failed = conv_res.status != ConversionStatus.SUCCESS
        if not doc_failed:
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
                if _is_empty_output(fname):
                    error_message = (
                        "Markdown export produced empty output for "
                        f"{conv_res.input.file.name}"
                    )
                    _log.error(error_message)
                    conv_res.errors.append(
                        ErrorItem(
                            component_type=DoclingComponentType.DOC_ASSEMBLER,
                            module_name="export_documents",
                            error_message=error_message,
                        )
                    )
                    conv_res.status = ConversionStatus.FAILURE
                    doc_failed = True

            # Export Document Tags format:
            if export_doctags:
                fname = output_dir / f"{doc_filename}.doctags"
                _log.info(f"writing Doc Tags output to {fname}")
                conv_res.document.save_as_doctags(filename=fname)

            # Export WebVTT format:
            if export_vtt:
                fname = output_dir / f"{doc_filename}.vtt"
                _log.info(f"writing WebVTT output to {fname}")
                conv_res.document.save_as_vtt(filename=fname)

            # Export DocLang format:
            if export_doclang:
                fname = output_dir / f"{doc_filename}.dclg.xml"
                _log.info(f"writing DocLang output to {fname}")
                with fname.open("w", encoding="utf-8") as fp:
                    fp.write(conv_res.document.export_to_doclang())
            # Export Chunks format:
            if export_chunks:
                import json as _json

                from docling_core.transforms.chunker.hierarchical_chunker import (
                    HierarchicalChunker,
                )
                from docling_core.transforms.chunker.hybrid_chunker import (
                    HybridChunker,
                )
                from docling_core.transforms.chunker.tokenizer.huggingface import (
                    HuggingFaceTokenizer,
                )

                if chunker_type == ChunkerType.HIERARCHICAL:
                    chunker_obj = HierarchicalChunker()
                else:  # default: hybrid
                    hf_tok = HuggingFaceTokenizer.from_pretrained(
                        model_name=chunk_tokenizer,
                        max_tokens=chunk_max_tokens,  # None => tokenizer's own limit
                    )
                    chunker_obj = HybridChunker(tokenizer=hf_tok)

                from docling_core.transforms.chunker.hierarchical_chunker import (
                    DocChunk,
                )

                fname = output_dir / f"{doc_filename}.chunks.jsonl"
                _log.info(f"writing Chunks output to {fname}")
                with fname.open("w", encoding="utf-8") as fp:
                    for i, chunk in enumerate(
                        chunker_obj.chunk(dl_doc=conv_res.document)
                    ):
                        doc_chunk = DocChunk.model_validate(chunk)
                        page_numbers = sorted(
                            {
                                prov.page_no
                                for item in doc_chunk.meta.doc_items
                                for prov in item.prov
                            }
                        )
                        metadata = {}
                        if doc_chunk.meta.origin:
                            metadata["origin"] = doc_chunk.meta.origin.model_dump(
                                mode="json"
                            )
                        metadata["has_image"] = any(
                            item.self_ref.startswith("#/pictures/")
                            for item in doc_chunk.meta.doc_items
                        )
                        contextualized = chunker_obj.contextualize(doc_chunk)
                        num_tokens: int | None = None
                        if isinstance(chunker_obj, HybridChunker):
                            num_tokens = chunker_obj.tokenizer.count_tokens(
                                contextualized
                            )
                        chunk_record = ChunkedDocumentResultItem(
                            filename=doc_filename,
                            chunk_index=i,
                            text=contextualized,
                            raw_text=doc_chunk.text,
                            num_tokens=num_tokens,
                            headings=doc_chunk.meta.headings,
                            captions=doc_chunk.meta.captions,
                            doc_items=[
                                item.self_ref for item in doc_chunk.meta.doc_items
                            ],
                            page_numbers=page_numbers,
                            metadata=metadata,
                        )
                        fp.write(
                            _json.dumps(
                                chunk_record.model_dump(mode="json"),
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
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

        if doc_failed:
            _log.warning(f"Document {conv_res.input.file} failed to convert.")
            if _log.isEnabledFor(logging.INFO):
                for err in conv_res.errors:
                    _log.info(
                        f"  [Failure Detail] Component: {err.component_type}, "
                        f"Module: {err.module_name}, Message: {err.error_message}"
                    )
            failure_count += 1
        else:
            success_count += 1

    _log.info(
        f"Processed {success_count + failure_count} docs, of which {failure_count} failed"
    )


@app.command(no_args_is_help=True)
def convert(  # noqa: C901
    source: Annotated[
        list[str],
        typer.Argument(
            ...,
            metavar="source",
            help="PDF files to convert. Can be local file / directory paths or URL.",
        ),
    ],
    from_formats: list[str] = typer.Option(
        None,
        "--from",
        help="Input formats to accept. Use 'odf' for odt, ods, and odp. Defaults to all supported formats.",
    ),
    to_formats: list[OutputFormat] = typer.Option(
        None, "--to", help="Specify output formats. Defaults to Markdown."
    ),
    chunker_type: ChunkerType = typer.Option(
        ChunkerType.HYBRID,
        "--chunker",
        help="Chunker to use with '--to chunks'.",
    ),
    chunk_max_tokens: int | None = typer.Option(
        None,
        "--max-tokens",
        help="Max tokens per chunk. Defaults to the tokenizer's own limit.",
    ),
    chunk_tokenizer: str = typer.Option(
        "sentence-transformers/all-MiniLM-L6-v2",
        "--tokenizer",
        help="HuggingFace tokenizer model name/path. Determines token counting and default max-tokens.",
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
    html_image_headers: str = typer.Option(
        None,
        "--html-image-headers",
        help="Specify http request headers used when fetching HTML and EPUB image resources in the form of a JSON string",
    ),
    image_export_mode: Annotated[
        ImageRefMode,
        typer.Option(
            ...,
            help="Image export mode for image-capable document outputs (JSON, YAML, HTML, HTML split-page, and Markdown). Text, DocTags, and WebVTT outputs do not export images. With `placeholder`, only the position of the image is marked in the output. In `embedded` mode, the image is embedded as base64 encoded string. In `referenced` mode, the image is exported in PNG format and referenced from the main exported document.",
        ),
    ] = ImageRefMode.EMBEDDED,
    html_image_fetch: Annotated[
        HtmlImageFetchMode,
        typer.Option(
            ...,
            "--html-image-fetch",
            help="Fetch image resources referenced by HTML and EPUB inputs. Choose none, local, remote, or all.",
        ),
    ] = HtmlImageFetchMode.NONE,
    pipeline: Annotated[
        ProcessingPipeline,
        typer.Option(..., help="Choose the pipeline to process PDF or image files."),
    ] = ProcessingPipeline.STANDARD,
    vlm_model: Annotated[
        str,
        typer.Option(
            ...,
            help=f"Choose the VLM preset to use with PDF or image files. Available presets: {', '.join(vlm_preset_ids)}",
        ),
    ] = "granite_docling",
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
        str | None,
        typer.Option(
            ...,
            help="Provide a comma-separated list of languages used by the OCR engine. Note that each OCR engine has different values for the language names.",
        ),
    ] = None,
    psm: Annotated[
        int | None,
        typer.Option(
            ...,
            help="Page Segmentation Mode for the OCR engine (0-13).",
        ),
    ] = None,
    pdf_backend: Annotated[
        PdfBackend, typer.Option(..., help="The PDF backend to use.")
    ] = PdfBackend.DOCLING_PARSE,
    pdf_password: Annotated[
        str | None, typer.Option(..., help="Password for protected PDF documents")
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
    enrich_chart_extraction: Annotated[
        bool,
        typer.Option(
            ..., help="Enable chart data extraction from bar, pie, and line charts."
        ),
    ] = False,
    artifacts_path: Annotated[
        Path | None,
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
            ..., help="Enable debug output which visualizes the layout clusters"
        ),
    ] = False,
    debug_visualize_tables: Annotated[
        bool,
        typer.Option(..., help="Enable debug output which visualizes the table cells"),
    ] = False,
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            callback=version_callback,
            is_eager=True,
            help="Show version information.",
        ),
    ] = None,
    document_timeout: Annotated[
        float | None,
        typer.Option(
            ...,
            help="The timeout for processing each document, in seconds.",
        ),
    ] = None,
    num_threads: Annotated[int, typer.Option(..., help="Number of threads")] = 4,
    release_native_memory_every_n_pages: Annotated[
        int,
        typer.Option(
            ...,
            help=(
                "Release native parser memory after every N decoded pages when "
                "using the threaded docling-parse backend."
            ),
        ),
    ] = 128,
    device: Annotated[
        AcceleratorDevice, typer.Option(..., help="Accelerator device")
    ] = AcceleratorDevice.AUTO,
    docling_logo: Annotated[
        bool | None,
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
    # Heavy backend/converter/pipeline imports are deferred to here so the CLI
    # (and `convert-remote`) stay importable without the local PDF stack
    # (pypdfium2 / docling_parse). Only local `convert` needs them.
    from docling.backend.docling_parse_backend import (
        DoclingParseDocumentBackend,
        ThreadedDoclingParseDocumentBackend,
    )
    from docling.backend.image_backend import ImageDocumentBackend
    from docling.backend.mets_gbs_backend import MetsGbsDocumentBackend
    from docling.backend.pdf_backend import PdfDocumentBackend
    from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
    from docling.document_converter import (
        AudioFormatOption,
        DocumentConverter,
        EpubFormatOption,
        ExcelFormatOption,
        FormatOption,
        HTMLFormatOption,
        LatexFormatOption,
        MarkdownFormatOption,
        PdfFormatOption,
        PowerpointFormatOption,
        WordFormatOption,
    )
    from docling.pipeline.asr_pipeline import AsrPipeline
    from docling.pipeline.vlm_pipeline import VlmPipeline

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

    from_formats = _expand_from_formats(from_formats)

    parsed_headers: dict[str, str] | None = None
    if headers is not None:
        headers_t = TypeAdapter(dict[str, str])
        parsed_headers = headers_t.validate_json(headers)

    parsed_html_image_headers: dict[str, str] | None = None
    if html_image_headers is not None:
        headers_t = TypeAdapter(dict[str, str])
        parsed_html_image_headers = headers_t.validate_json(html_image_headers)

    html_fetch_images = html_image_fetch != HtmlImageFetchMode.NONE
    html_enable_local_fetch = html_image_fetch in {
        HtmlImageFetchMode.LOCAL,
        HtmlImageFetchMode.ALL,
    }
    html_enable_remote_fetch = html_image_fetch in {
        HtmlImageFetchMode.REMOTE,
        HtmlImageFetchMode.ALL,
    }
    if parsed_html_image_headers is not None and not html_enable_remote_fetch:
        err_console.print(
            "[red]Error: --html-image-headers requires --html-image-fetch remote or all.[/red]"
        )
        raise typer.Abort()

    if profiling or save_profiling:
        settings.debug.profile_pipeline_timings = True

    with tempfile.TemporaryDirectory() as tempdir:
        input_doc_paths: list[Path | str] = []
        for src in source:
            try:
                if _is_http_url(src) and _is_html_source(src, from_formats):
                    input_doc_paths.append(src)
                    continue

                local_path = TypeAdapter(Path).validate_python(src)
                if local_path.exists():
                    if local_path.is_dir():
                        input_doc_paths.extend(
                            _iter_input_paths_from_directory(local_path, from_formats)
                        )
                    elif _is_temporary_word_file(local_path):
                        _log.info(f"Ignoring temporary Word file: {local_path}")
                    elif _is_html_source(src, from_formats):
                        input_doc_paths.append(local_path)
                    else:
                        resolved_source = resolve_source_to_path(
                            source=src, headers=parsed_headers, workdir=Path(tempdir)
                        )
                        input_doc_paths.append(resolved_source)
                    continue

                # check if we can fetch some remote url
                resolved_source = resolve_source_to_path(
                    source=src, headers=parsed_headers, workdir=Path(tempdir)
                )
                input_doc_paths.append(resolved_source)
            except FileNotFoundError:
                err_console.print(
                    f"[red]Error: The input file {src} does not exist.[/red]"
                )
                raise typer.Abort()
            except (IsADirectoryError, PermissionError):
                # if the input matches to a file or a folder
                try:
                    local_path = TypeAdapter(Path).validate_python(src)
                    if local_path.exists() and local_path.is_dir():
                        input_doc_paths.extend(
                            _iter_input_paths_from_directory(local_path, from_formats)
                        )
                    elif local_path.exists():
                        if _is_temporary_word_file(local_path):
                            _log.info(f"Ignoring temporary Word file: {local_path}")
                        else:
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

        export_flags = _export_flags_from_formats(to_formats)

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
        pipeline_options: PipelineOptions
        format_options: dict[InputFormat, FormatOption] = {}
        pdf_backend_options: PdfBackendOptions | None = PdfBackendOptions(
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
                do_chart_extraction=enrich_chart_extraction,
                document_timeout=document_timeout,
            )
            if isinstance(
                pipeline_options.table_structure_options, TableStructureOptions
            ):
                pipeline_options.table_structure_options.do_cell_matching = True
                pipeline_options.table_structure_options.mode = table_mode

            if _should_generate_export_images(
                image_export_mode,
                to_formats,
            ):
                pipeline_options.generate_page_images = True
                pipeline_options.generate_picture_images = (
                    True  # FIXME: to be deprecated in version 3
                )
                pipeline_options.images_scale = 2
            pdf_backend = normalize_pdf_backend(pdf_backend)
            backend: Type[PdfDocumentBackend]
            if pdf_backend == PdfBackend.DOCLING_PARSE:
                backend = DoclingParseDocumentBackend  # type: ignore
            elif pdf_backend == PdfBackend.THREADED_DOCLING_PARSE:
                backend = ThreadedDoclingParseDocumentBackend  # type: ignore
                pdf_backend_options = ThreadedDoclingParseBackendOptions(
                    password=pdf_password,
                    parser_threads=num_threads,
                    release_native_memory_every_n_pages=(
                        release_native_memory_every_n_pages
                    ),
                )
            elif pdf_backend == PdfBackend.PYPDFIUM2:
                backend = PyPdfiumDocumentBackend  # type: ignore
            else:
                raise RuntimeError(f"Unexpected PDF backend type {pdf_backend}")

            pdf_format_option = PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=backend,  # pdf_backend
                backend_options=pdf_backend_options,
            )
            mets_gbs_options = pipeline_options.model_copy()
            mets_gbs_options.do_ocr = False
            mets_gbs_format_option = PdfFormatOption(
                pipeline_options=mets_gbs_options,
                backend=MetsGbsDocumentBackend,
            )
            simple_format_option = ConvertPipelineOptions(
                do_picture_description=enrich_picture_description,
                do_picture_classification=enrich_picture_classes,
                do_chart_extraction=enrich_chart_extraction,
            )
            if artifacts_path is not None:
                simple_format_option.artifacts_path = artifacts_path

            html_backend_options: HTMLBackendOptions | None = None
            if (
                html_fetch_images
                or html_enable_local_fetch
                or html_enable_remote_fetch
                or parsed_html_image_headers is not None
            ):
                html_backend_options = HTMLBackendOptions(
                    fetch_images=html_fetch_images,
                    enable_local_fetch=html_enable_local_fetch,
                    enable_remote_fetch=html_enable_remote_fetch,
                    headers=parsed_html_image_headers,
                )

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
                InputFormat.ODT: OdtFormatOption(pipeline_options=simple_format_option),
                InputFormat.ODP: OdpFormatOption(pipeline_options=simple_format_option),
                InputFormat.ODS: OdsFormatOption(pipeline_options=simple_format_option),
                InputFormat.HTML: HTMLFormatOption(
                    pipeline_options=simple_format_option,
                    backend_options=html_backend_options,
                ),
                InputFormat.EPUB: EpubFormatOption(
                    pipeline_options=simple_format_option,
                    backend_options=EpubBackendOptions(
                        fetch_images=html_fetch_images,
                        enable_local_fetch=html_enable_local_fetch,
                        enable_remote_fetch=html_enable_remote_fetch,
                    )
                    if (
                        html_fetch_images
                        or html_enable_local_fetch
                        or html_enable_remote_fetch
                    )
                    else None,
                ),
                InputFormat.MD: MarkdownFormatOption(
                    pipeline_options=simple_format_option
                ),
                InputFormat.LATEX: LatexFormatOption(
                    pipeline_options=simple_format_option,
                    backend_options=LatexBackendOptions(),
                ),
            }

        elif pipeline == ProcessingPipeline.VLM:
            pipeline_options = VlmPipelineOptions(
                accelerator_options=accelerator_options,
                enable_remote_services=enable_remote_services,
            )

            # Use the new preset system
            try:
                pipeline_options.vlm_options = VlmConvertOptions.from_preset(vlm_model)
                _log.info(f"Using VLM preset: {vlm_model}")
            except KeyError:
                err_console.print(
                    f"[red]Error: VLM preset '{vlm_model}' not found.[/red]"
                )
                err_console.print(
                    f"[yellow]Available presets: {', '.join(vlm_preset_ids)}[/yellow]"
                )
                raise typer.Abort()

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
            **export_flags,
            show_layout=show_layout,
            print_timings=profiling,
            export_timings=save_profiling,
            image_export_mode=image_export_mode,
            chunker_type=chunker_type,
            chunk_max_tokens=chunk_max_tokens,
            chunk_tokenizer=chunk_tokenizer,
        )

        end_time = time.time() - start_time

    _log.info(f"All documents were converted in {end_time:.2f} seconds.")


# Register `convert-remote` only when the service-client extra is installed.
# Imported here (after `app`, `export_documents`, and the source-collection
# helpers are defined) so the command is attached before the click app is built
# below.
try:
    from docling.cli.remote import register as _register_remote
except ImportError:
    _log.debug(
        "Skipping `convert-remote` registration because service-client dependencies are unavailable."
    )
else:
    _register_remote(app)

click_app = typer.main.get_command(app)

if __name__ == "__main__":
    app()
