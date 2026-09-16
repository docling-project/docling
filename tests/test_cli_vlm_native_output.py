# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tests for exporting native VLM responses from the CLI."""

from pathlib import Path

import typer
from docling_core.types.doc import ImageRefMode

from docling.cli.main import app, export_documents
from docling.datamodel.base_models import (
    ConversionStatus,
    InputFormat,
    Page,
    VlmPrediction,
)
from docling.datamodel.document import (
    ConversionResult,
    InputDocument,
    _DummyBackend,
)


def test_cli_exposes_vlm_write_native_output() -> None:
    convert_command = typer.main.get_command(app).commands["convert"]

    assert any(
        "--vlm-write-native-output" in parameter.opts
        for parameter in convert_command.params
    )


def test_export_documents_writes_native_vlm_output_for_each_page(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.pdf"
    input_path.write_bytes(b"%PDF-1.4")
    input_doc = InputDocument(
        path_or_stream=input_path,
        format=InputFormat.PDF,
        backend=_DummyBackend,
    )
    first_page = Page(page_no=1)
    first_page.predictions.vlm_response = VlmPrediction(
        text="<class_Title><x_0.1>First page"
    )
    third_page = Page(page_no=3)
    third_page.predictions.vlm_response = VlmPrediction(text="Third page\n\n| A | B |")
    conv_res = ConversionResult(
        input=input_doc,
        pages=[first_page, third_page],
        status=ConversionStatus.PARTIAL_SUCCESS,
    )

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    export_documents(
        [conv_res],
        output_dir=output_dir,
        export_json=False,
        export_yaml=False,
        export_html=False,
        export_html_split_page=False,
        show_layout=False,
        export_md=False,
        export_txt=False,
        export_doctags=False,
        export_vtt=False,
        export_doclang=False,
        print_timings=False,
        export_timings=False,
        image_export_mode=ImageRefMode.PLACEHOLDER,
        vlm_write_native_output=True,
    )

    native_output_dir = output_dir / "input.vlm-native"
    assert (native_output_dir / "page_00001.txt").read_text(encoding="utf-8") == (
        "<class_Title><x_0.1>First page"
    )
    assert (native_output_dir / "page_00003.txt").read_text(encoding="utf-8") == (
        "Third page\n\n| A | B |"
    )
