# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""End-to-end tests for superscript and subscript recovery in the PDF pipeline."""

from pathlib import Path

import pytest
from docling_core.types.doc.document import InlineGroup, Script, TextItem
from PIL import Image

from docling.backend.docling_parse_backend import ThreadedDoclingParseDocumentBackend
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    ScriptPositionOptions,
)
from docling.document_converter import (
    DocumentConverter,
    ImageFormatOption,
    PdfFormatOption,
)

pytestmark = pytest.mark.ml_pdf_model

# Exported from Word-style formatting (LibreOffice), with an embedded font.
_PDF = Path("tests/data/pdf/script_position.pdf")


def _convert(backend, enabled: bool):
    options = PdfPipelineOptions(do_ocr=False, do_table_structure=False)
    options.script_position_options = ScriptPositionOptions(enabled=enabled)
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=options, backend=backend)
        }
    )
    return converter.convert(_PDF).document


def _runs(document) -> list[list[tuple[str, Script]]]:
    out = []
    for group in document.groups:
        if not isinstance(group, InlineGroup):
            continue
        runs = []
        for ref in group.children:
            item = ref.resolve(document)
            script = item.formatting.script if item.formatting else Script.BASELINE
            runs.append((item.text, script))
        out.append(runs)
    return out


def _scripts(runs) -> list[tuple[str, Script]]:
    return [
        (text.strip(), script)
        for group in runs
        for text, script in group
        if script != Script.BASELINE
    ]


def test_marks_superscript_and_subscript_runs():
    document = _convert(ThreadedDoclingParseDocumentBackend, enabled=True)
    runs = _runs(document)
    assert _scripts(runs) == [
        ("2", Script.SUPER),
        ("2", Script.SUB),
        ("2", Script.SUB),
        ("st", Script.SUPER),
        ("rd", Script.SUPER),
        ("[12]", Script.SUPER),
    ]
    # As in the Word backend, a run has no whitespace at its edges.
    assert runs[0][:2] == [
        ("Einstein wrote E = mc", Script.BASELINE),
        ("2", Script.SUPER),
    ]
    # A line without scripts stays a plain text item.
    assert any(
        isinstance(item, TextItem) and item.text.startswith("Plain line")
        for item in document.texts
    )


def test_marks_scripts_with_the_pypdfium2_backend():
    document = _convert(PyPdfiumDocumentBackend, enabled=True)
    # This backend puts the "2" of "mc2" in a text item of its own, so it has no line to
    # be measured against.
    assert _scripts(_runs(document)) == [
        ("2", Script.SUB),
        ("2", Script.SUB),
        ("st", Script.SUPER),
        ("rd", Script.SUPER),
        ("[12]", Script.SUPER),
    ]


def test_disabled_leaves_the_document_unchanged():
    document = _convert(ThreadedDoclingParseDocumentBackend, enabled=False)
    assert not _runs(document)
    assert "E = mc 2 in 1905." in document.export_to_markdown()


# One paragraph in Helvetica, as (text, size, rise) runs per line.
_LINES = [
    [
        ("Water is H", 11, 0),
        ("2", 7, -2),
        ("O and carbon dioxide is CO", 11, 0),
        ("2", 7, -2),
        (" in the air", 11, 0),
    ],
    [
        ("we can see that E = mc", 11, 0),
        ("2", 7, 4),
        (" holds on the 21", 11, 0),
        ("st", 7, 4),
        (" of June", 11, 0),
    ],
    [
        ("as reported before", 11, 0),
        ("12", 7, 4),
        (" and confirmed by the team today", 11, 0),
    ],
]


def _write_pdf(path: Path, page: str, render_mode: int = 0) -> None:
    """Write a one-page PDF of ``_LINES`` with the given page box entries."""
    ops = ["BT", f"{render_mode} Tr", "13 TL", "72 700 Td"]
    for number, line in enumerate(_LINES):
        if number:
            ops.append("T*")
        ops += [f"/F1 {size} Tf {rise} Ts ({text}) Tj" for text, size, rise in line]
    stream = "\n".join([*ops, "0 Ts", "ET"]).encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        f"<< /Type /Page /Parent 2 0 R {page} /Contents 4 0 R "
        "/Resources << /Font << /F1 5 0 R >> >> >>".encode(),
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    path.write_bytes(bytes(out))


@pytest.mark.parametrize(
    "backend", [ThreadedDoclingParseDocumentBackend, PyPdfiumDocumentBackend]
)
@pytest.mark.parametrize(
    "page",
    [
        pytest.param("/MediaBox [0 0 612 792] /Rotate 90", id="rotated"),
        # The visible page does not start at the origin of the PDF coordinates.
        pytest.param("/MediaBox [0 0 700 900] /CropBox [40 60 652 852]", id="cropped"),
    ],
)
def test_matches_characters_on_rotated_and_cropped_pages(tmp_path, backend, page):
    pdf = tmp_path / "page.pdf"
    _write_pdf(pdf, page)
    options = PdfPipelineOptions(do_ocr=False, do_table_structure=False)
    options.script_position_options = ScriptPositionOptions(enabled=True)
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=options, backend=backend)
        }
    )
    assert _scripts(_runs(converter.convert(pdf).document)) == [
        ("2", Script.SUB),
        ("2", Script.SUB),
        ("2", Script.SUPER),
        ("st", Script.SUPER),
        ("12", Script.SUPER),
    ]


@pytest.mark.parametrize(
    "backend", [ThreadedDoclingParseDocumentBackend, PyPdfiumDocumentBackend]
)
def test_leaves_invisible_text_unchanged(tmp_path, backend):
    """An invisible text layer, such as OCR over a scan, has no typeset baseline."""
    pdf = tmp_path / "page.pdf"
    _write_pdf(pdf, "/MediaBox [0 0 612 792]", render_mode=3)
    options = PdfPipelineOptions(do_ocr=False, do_table_structure=False)
    options.script_position_options = ScriptPositionOptions(enabled=True)
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=options, backend=backend)
        }
    )
    document = converter.convert(pdf).document

    assert "confirmed by the team today" in document.export_to_markdown()
    assert _scripts(_runs(document)) == []


def test_image_input_is_skipped_without_a_warning(tmp_path, caplog):
    """The PDF pipeline also converts images, which have no PDF characters."""
    image = tmp_path / "page.png"
    Image.new("RGB", (200, 100), "white").save(image)
    options = PdfPipelineOptions(do_ocr=False, do_table_structure=False)
    options.script_position_options = ScriptPositionOptions(enabled=True)
    converter = DocumentConverter(
        format_options={InputFormat.IMAGE: ImageFormatOption(pipeline_options=options)}
    )

    with caplog.at_level("WARNING", logger="docling.models.stages.script_position"):
        converter.convert(image)

    assert not [r for r in caplog.records if "recovery" in r.getMessage()]
