import logging
import os
from pathlib import Path

import pytest
from docling_core.types.doc import GroupItem

from docling.backend.docx.drawingml.utils import get_libreoffice_cmd
from docling.backend.msword_backend import MsWordDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import (
    ConversionResult,
    DoclingDocument,
    InputDocument,
    SectionHeaderItem,
    TextItem,
)
from docling.document_converter import DocumentConverter

from .test_data_gen_flag import GEN_TEST_DATA
from .verify_utils import verify_document, verify_export

_log = logging.getLogger(__name__)

GENERATE = GEN_TEST_DATA
IS_CI = bool(os.getenv("CI"))


@pytest.fixture(scope="module")
def docx_paths() -> list[Path]:
    # Define the directory you want to search
    directory = Path("./tests/data/docx/")

    # List all docx files in the directory and its subdirectories
    docx_files = sorted(directory.rglob("*.docx"))

    return docx_files


def get_converter():
    converter = DocumentConverter(allowed_formats=[InputFormat.DOCX])

    return converter


@pytest.fixture(scope="module")
def documents(docx_paths) -> list[tuple[Path, DoclingDocument]]:
    documents: list[dict[Path, DoclingDocument]] = []

    converter = get_converter()

    for docx_path in docx_paths:
        _log.debug(f"converting {docx_path}")

        gt_path = (
            docx_path.parent.parent / "groundtruth" / "docling_v2" / docx_path.name
        )

        conv_result: ConversionResult = converter.convert(docx_path)

        doc: DoclingDocument = conv_result.document

        assert doc, f"Failed to convert document from file {gt_path}"
        documents.append((gt_path, doc))

    return documents


def _test_e2e_docx_conversions_impl(docx_paths: list[tuple[Path, DoclingDocument]]):
    has_libreoffice = False
    try:
        cmd = get_libreoffice_cmd(raise_if_unavailable=True)
        if cmd is not None:
            has_libreoffice = True
    except Exception:
        pass

    for docx_path, doc in docx_paths:
        if not IS_CI and not has_libreoffice and docx_path.name == "drawingml.docx":
            print(f"Skipping {docx_path} because no Libreoffice is installed.")
            continue

        pred_md: str = doc.export_to_markdown()
        assert verify_export(pred_md, str(docx_path) + ".md", generate=GENERATE), (
            f"export to markdown failed on {docx_path}"
        )

        pred_itxt: str = doc._export_to_indented_text(
            max_text_len=70, explicit_tables=False
        )
        assert verify_export(pred_itxt, str(docx_path) + ".itxt", generate=GENERATE), (
            f"export to indented-text failed on {docx_path}"
        )

        assert verify_document(doc, str(docx_path) + ".json", generate=GENERATE), (
            f"DoclingDocument verification failed on {docx_path}"
        )

        if docx_path.name == "word_tables.docx":
            pred_html: str = doc.export_to_html()
            assert verify_export(
                pred_text=pred_html,
                gtfile=str(docx_path) + ".html",
                generate=GENERATE,
            ), f"export to html failed on {docx_path}"


flaky_file = "textbox.docx"


def test_e2e_docx_conversions(documents):
    target = [item for item in documents if item[0].name != flaky_file]
    _test_e2e_docx_conversions_impl(target)


@pytest.mark.xfail(strict=False)
def test_textbox_conversion(documents):
    target = [item for item in documents if item[0].name == flaky_file]
    _test_e2e_docx_conversions_impl(target)


@pytest.mark.xfail(strict=False)
def test_textbox_extraction(documents):
    name = "textbox.docx"
    doc = next(item[1] for item in documents if item[0].name == name)

    # Verify if a particular textbox content is extracted
    textbox_found = False
    for item, _ in doc.iterate_items():
        if item.text[:30] == """Suggested Reportable Symptoms:""":
            textbox_found = True
    assert textbox_found


def test_heading_levels(documents):
    name = "word_sample.docx"
    doc = next(item[1] for item in documents if item[0].name == name)

    found_lvl_1 = found_lvl_2 = False
    for item, _ in doc.iterate_items():
        if isinstance(item, SectionHeaderItem):
            if item.text == "Let\u2019s swim!":
                found_lvl_1 = True
                assert item.level == 1
            elif item.text == "Let\u2019s eat":
                found_lvl_2 = True
                assert item.level == 2
    assert found_lvl_1 and found_lvl_2


def test_text_after_image_anchors(documents):
    """Test to analyse whether text gets parsed after image anchors."""

    name = "word_image_anchors.docx"
    doc = next(item[1] for item in documents if item[0].name == name)

    found_text_after_anchor_1 = found_text_after_anchor_2 = (
        found_text_after_anchor_3
    ) = found_text_after_anchor_4 = False
    for item, _ in doc.iterate_items():
        if isinstance(item, TextItem):
            if item.text == "This is test 1":
                found_text_after_anchor_1 = True
            elif item.text == "0:08\nCorrect, he is not.":
                found_text_after_anchor_2 = True
            elif item.text == "This is test 2":
                found_text_after_anchor_3 = True
            elif item.text == "0:16\nYeah, exactly.":
                found_text_after_anchor_4 = True

    assert (
        found_text_after_anchor_1
        and found_text_after_anchor_2
        and found_text_after_anchor_3
        and found_text_after_anchor_4
    )


def test_is_rich_table_cell(docx_paths):
    """Test the function is_rich_table_cell."""

    name = "docx_rich_cells.docx"
    path = next(item for item in docx_paths if item.name == name)

    in_doc = InputDocument(
        path_or_stream=path,
        format=InputFormat.DOCX,
        backend=MsWordDocumentBackend,
        filename=name,
    )
    backend = MsWordDocumentBackend(
        in_doc=in_doc,
        path_or_stream=path,
    )

    gt_cells: list[bool] = []
    # table: Table with rich cells
    gt_cells.extend([False, False, True, True, True, True, True, False])
    # table: Table with nested table
    gt_cells.extend([False, False, False, True, True, True])
    # table: Table with pictures
    gt_cells.extend([False, False, False, True, True, False])
    gt_it = iter(gt_cells)

    for idx_t, table in enumerate(backend.docx_obj.tables):
        for idx_r, row in enumerate(table.rows):
            for idx_c, cell in enumerate(row.cells):
                assert next(gt_it) == backend._is_rich_table_cell(cell), (
                    f"Wrong cell type in table {idx_t}, row {idx_r}, col {idx_c} "
                    f"with text: {cell.text}"
                )


def test_add_header_footer(documents):
    """Test the funciton _add_header_footer."""

    name = "unit_test_formatting.docx"
    doc = next(item[1] for item in documents if item[0].name == name)

    headers: list[GroupItem] = []
    footers: list[GroupItem] = []
    for group in doc.groups:
        if not isinstance(group, GroupItem):
            continue
        if group.name == "page header":
            headers.append(group)
        elif group.name == "page footer":
            footers.append(group)

    assert len(headers) == 2, "Expected 2 different headers"
    assert len(footers) == 2, "Expected 2 different footers"

    assert len(headers[0].children) == 1, "First page header should have 1 paragraph"
    assert len(headers[1].children) == 2, "Second page header should have 2 paragraphs"

    assert len(footers[0].children) == 1, "First page footer should have 1 paragraph"
    assert len(footers[1].children) == 4, (
        "Second page footer should have 3 paragraphs and 1 picture"
    )


def test_handle_pictures(documents):
    """Test the function _handle_pictures."""

    name = "docx_grouped_images.docx"
    doc = next(item[1] for item in documents if item[0].name == name)

    assert len(doc.pictures) == 6
    assert isinstance(doc.pictures[0].parent.resolve(doc), GroupItem)
    assert doc.pictures[0].parent == doc.pictures[1].parent
    assert isinstance(doc.pictures[2].parent.resolve(doc), GroupItem)
    assert doc.pictures[2].parent == doc.pictures[3].parent
    assert isinstance(doc.pictures[4].parent.resolve(doc), SectionHeaderItem)
    assert doc.pictures[4].parent == doc.pictures[5].parent


@pytest.mark.parametrize(
    "style_label,expected_label,expected_level",
    [
        ("Heading 1", "Heading", 1),
        ("Heading 2", "Heading", 2),
        ("Heading 9", "Heading", 9),
        ("Heading 0", "Heading", 1),  # Custom style - level 0 should be clamped to 1
        ("1 Heading", "Heading", 1),  # Number before text
        ("0 Heading", "Heading", 1),  # Zero before text should be clamped to 1
    ],
)
def test_get_heading_and_level(docx_paths, style_label, expected_label, expected_level):
    """Test _get_heading_and_level handles edge cases like 'Heading 0' correctly."""
    # Create a backend instance using any existing docx file
    docx_path = docx_paths[0]
    in_doc = InputDocument(
        path_or_stream=docx_path,
        format=InputFormat.DOCX,
        backend=MsWordDocumentBackend,
    )
    backend = in_doc._backend

    label, level = backend._get_heading_and_level(style_label)
    assert label == expected_label, f"Expected label '{expected_label}' for '{style_label}', got '{label}'"
    assert level == expected_level, f"Expected level {expected_level} for '{style_label}', got {level}"


def test_get_outline_level_from_style(docx_paths):
    """Test that _get_outline_level_from_style correctly extracts outlineLvl.

    OOXML outlineLvl is 0-indexed, so outlineLvl=0 -> heading level 1.
    """
    from docx import Document

    docx_path = docx_paths[0]
    in_doc = InputDocument(
        path_or_stream=docx_path,
        format=InputFormat.DOCX,
        backend=MsWordDocumentBackend,
    )
    backend = in_doc._backend

    # Load the document to access paragraphs
    doc = Document(docx_path)

    # Find paragraphs with heading styles and verify outline level extraction
    for para in doc.paragraphs:
        if para.style and para.style.name:
            style_name = para.style.name.lower()
            if style_name.startswith("heading "):
                # Extract the expected level from the style name
                try:
                    expected_level = int(style_name.split()[-1])
                except ValueError:
                    continue

                # Get the outline level using our method
                outline_level = backend._get_outline_level_from_style(para)

                # outlineLvl is 0-indexed, so Heading 1 has outlineLvl=0
                # Our method should return the 1-indexed level
                if outline_level is not None:
                    assert outline_level == expected_level, (
                        f"Style '{para.style.name}' has outlineLvl that should map to "
                        f"level {expected_level}, but got {outline_level}"
                    )
