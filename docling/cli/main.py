import importlib
import json
import logging
import re
import tempfile
import time
import warnings
from enum import Enum
from pathlib import Path
from typing import Annotated, Dict, Iterable, List, Optional, Type

import typer
from docling_core.types.doc import ImageRefMode
from docling_core.utils.file import resolve_source_to_path
from pydantic import TypeAdapter, ValidationError

from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
from docling.backend.pdf_backend import PdfDocumentBackend
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import (
    ConversionStatus,
    FormatToExtensions,
    InputFormat,
    OutputFormat,
)
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    EasyOcrOptions,
    OcrEngine,
    OcrMacOptions,
    OcrOptions,
    PdfBackend,
    PdfPipelineOptions,
    RapidOcrOptions,
    TableFormerMode,
    TesseractCliOcrOptions,
    TesseractOcrOptions,
)
from docling.datamodel.settings import settings
from docling.document_converter import DocumentConverter, FormatOption, PdfFormatOption

warnings.filterwarnings(action="ignore", category=UserWarning, module="pydantic|torch")
warnings.filterwarnings(action="ignore", category=FutureWarning, module="easyocr")

_log = logging.getLogger(__name__)
from rich.console import Console

err_console = Console(stderr=True)


app = typer.Typer(
    name="Docling",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)


def version_callback(value: bool):
    if value:
        docling_version = importlib.metadata.version("docling")
        docling_core_version = importlib.metadata.version("docling-core")
        docling_ibm_models_version = importlib.metadata.version("docling-ibm-models")
        docling_parse_version = importlib.metadata.version("docling-parse")
        print(f"Docling version: {docling_version}")
        print(f"Docling Core version: {docling_core_version}")
        print(f"Docling IBM Models version: {docling_ibm_models_version}")
        print(f"Docling Parse version: {docling_parse_version}")
        raise typer.Exit()


def export_documents(
    conv_results: Iterable[ConversionResult],
    output_dir: Path,
    export_json: bool,
    export_html: bool,
    export_md: bool,
    export_txt: bool,
    export_doctags: bool,
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

            # Export HTML format:
            if export_html:
                fname = output_dir / f"{doc_filename}.html"
                _log.info(f"writing HTML output to {fname}")
                conv_res.document.save_as_html(
                    filename=fname, image_mode=image_export_mode
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
                conv_res.document.save_as_document_tokens(filename=fname)

        else:
            _log.warning(f"Document {conv_res.input.file} failed to convert.")
            failure_count += 1

    _log.info(
        f"Processed {success_count + failure_count} docs, of which {failure_count} failed"
    )


def _split_list(raw: Optional[str]) -> Optional[List[str]]:
    if raw is None:
        return None
    return re.split(r"[;,]", raw)


@app.command(no_args_is_help=True)
def convert(
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
    ocr_engine: Annotated[
        OcrEngine, typer.Option(..., help="The OCR engine to use.")
    ] = OcrEngine.EASYOCR,
    ocr_lang: Annotated[
        Optional[str],
        typer.Option(
            ...,
            help="Provide a comma-separated list of languages used by the OCR engine. Note that each OCR engine has different values for the language names.",
        ),
    ] = None,
    pdf_backend: Annotated[
        PdfBackend, typer.Option(..., help="The PDF backend to use.")
    ] = PdfBackend.DLPARSE_V2,
    table_mode: Annotated[
        TableFormerMode,
        typer.Option(..., help="The mode to use in the table structure model."),
    ] = TableFormerMode.FAST,
    artifacts_path: Annotated[
        Optional[Path],
        typer.Option(..., help="If provided, the location of the model artifacts."),
    ] = None,
    abort_on_error: Annotated[
        bool,
        typer.Option(
            ...,
            "--abort-on-error/--no-abort-on-error",
            help="If enabled, the bitmap content will be processed using OCR.",
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
):
    if verbose == 0:
        logging.basicConfig(level=logging.WARNING)
    elif verbose == 1:
        logging.basicConfig(level=logging.INFO)
    elif verbose == 2:
        logging.basicConfig(level=logging.DEBUG)

    settings.debug.visualize_cells = debug_visualize_cells
    settings.debug.visualize_layout = debug_visualize_layout
    settings.debug.visualize_tables = debug_visualize_tables
    settings.debug.visualize_ocr = debug_visualize_ocr

    if from_formats is None:
        from_formats = [e for e in InputFormat]

    parsed_headers: Optional[Dict[str, str]] = None
    if headers is not None:
        headers_t = TypeAdapter(Dict[str, str])
        parsed_headers = headers_t.validate_json(headers)

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
                                input_doc_paths.extend(
                                    list(local_path.glob(f"**/*.{ext}"))
                                )
                                input_doc_paths.extend(
                                    list(local_path.glob(f"**/*.{ext.upper()}"))
                                )
                    elif local_path.exists():
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
        export_html = OutputFormat.HTML in to_formats
        export_md = OutputFormat.MARKDOWN in to_formats
        export_txt = OutputFormat.TEXT in to_formats
        export_doctags = OutputFormat.DOCTAGS in to_formats

        if ocr_engine == OcrEngine.EASYOCR:
            ocr_options: OcrOptions = EasyOcrOptions(force_full_page_ocr=force_ocr)
        elif ocr_engine == OcrEngine.TESSERACT_CLI:
            ocr_options = TesseractCliOcrOptions(force_full_page_ocr=force_ocr)
        elif ocr_engine == OcrEngine.TESSERACT:
            ocr_options = TesseractOcrOptions(force_full_page_ocr=force_ocr)
        elif ocr_engine == OcrEngine.OCRMAC:
            ocr_options = OcrMacOptions(force_full_page_ocr=force_ocr)
        elif ocr_engine == OcrEngine.RAPIDOCR:
            ocr_options = RapidOcrOptions(force_full_page_ocr=force_ocr)
        else:
            raise RuntimeError(f"Unexpected OCR engine type {ocr_engine}")

        ocr_lang_list = _split_list(ocr_lang)
        if ocr_lang_list is not None:
            ocr_options.lang = ocr_lang_list

        accelerator_options = AcceleratorOptions(num_threads=num_threads, device=device)
        pipeline_options = PdfPipelineOptions(
            accelerator_options=accelerator_options,
            do_ocr=ocr,
            ocr_options=ocr_options,
            do_table_structure=True,
            document_timeout=document_timeout,
        )
        pipeline_options.table_structure_options.do_cell_matching = (
            True  # do_cell_matching
        )
        pipeline_options.table_structure_options.mode = table_mode

        if image_export_mode != ImageRefMode.PLACEHOLDER:
            pipeline_options.generate_page_images = True
            pipeline_options.generate_picture_images = (
                True  # FIXME: to be deprecated in verson 3
            )
            pipeline_options.images_scale = 2

        if artifacts_path is not None:
            pipeline_options.artifacts_path = artifacts_path

        if pdf_backend == PdfBackend.DLPARSE_V1:
            backend: Type[PdfDocumentBackend] = DoclingParseDocumentBackend
        elif pdf_backend == PdfBackend.DLPARSE_V2:
            backend = DoclingParseV2DocumentBackend
        elif pdf_backend == PdfBackend.PYPDFIUM2:
            backend = PyPdfiumDocumentBackend
        else:
            raise RuntimeError(f"Unexpected PDF backend type {pdf_backend}")

        pdf_format_option = PdfFormatOption(
            pipeline_options=pipeline_options,
            backend=backend,  # pdf_backend
        )
        format_options: Dict[InputFormat, FormatOption] = {
            InputFormat.PDF: pdf_format_option,
            InputFormat.IMAGE: pdf_format_option,
        }
        doc_converter = DocumentConverter(
            allowed_formats=from_formats,
            format_options=format_options,
        )

        start_time = time.time()

        conv_results = doc_converter.convert_all(
            input_doc_paths, headers=parsed_headers, raises_on_error=abort_on_error
        )

        output.mkdir(parents=True, exist_ok=True)
        export_documents(
            conv_results,
            output_dir=output,
            export_json=export_json,
            export_html=export_html,
            export_md=export_md,
            export_txt=export_txt,
            export_doctags=export_doctags,
            image_export_mode=image_export_mode,
        )

        end_time = time.time() - start_time

    _log.info(f"All documents were converted in {end_time:.2f} seconds.")


click_app = typer.main.get_command(app)

if __name__ == "__main__":
    app()
