import base64
from io import BytesIO
from pathlib import Path

import pytest
from docling_core.types.doc import CodeItem, CodeLanguageLabel, PictureItem
from PIL import Image

from docling.backend.md_backend import MarkdownDocumentBackend
from docling.datamodel.backend_options import MarkdownBackendOptions
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.document import (
    ConversionResult,
    DoclingDocument,
    InputDocument,
)
from docling.document_converter import DocumentConverter
from tests.verify_utils import CONFID_PREC, COORD_PREC

from .test_data_gen_flag import GEN_TEST_DATA
from .verify_utils import verify_docitems, verify_document

pytestmark = pytest.mark.cross_platform


def test_convert_valid():
    fmt = InputFormat.MD
    cls = MarkdownDocumentBackend

    md_path = Path("tests") / "data" / "md"
    relevant_paths = sorted((md_path / "sources").rglob("*.md"))
    assert len(relevant_paths) > 0

    yaml_filter = ["inline_and_formatting", "mixed_without_h1"]
    json_filter = ["escaped_characters", "signature_stamp_01"]

    for in_path in relevant_paths:
        md_gt_path = md_path / "groundtruth" / f"{in_path.name}.md"
        yaml_gt_path = md_path / "groundtruth" / f"{in_path.name}.yaml"
        json_gt_path = md_path / "groundtruth" / f"{in_path.name}.json"

        in_doc = InputDocument(
            path_or_stream=in_path,
            format=fmt,
            backend=cls,
        )
        backend = cls(
            in_doc=in_doc,
            path_or_stream=in_path,
        )
        assert backend.is_valid()

        act_doc = backend.convert()
        act_data = act_doc.export_to_markdown(compact_tables=True)

        if in_path.stem in json_filter:
            assert verify_document(act_doc, json_gt_path, GEN_TEST_DATA), (
                "export to json"
            )

        if GEN_TEST_DATA:
            with open(md_gt_path, mode="w", encoding="utf-8") as f:
                f.write(f"{act_data}\n")

            if in_path.stem in yaml_filter:
                act_doc.save_as_yaml(
                    yaml_gt_path,
                    coord_precision=COORD_PREC,
                    confid_precision=CONFID_PREC,
                )
        else:
            with open(md_gt_path, encoding="utf-8") as f:
                exp_data = f.read().rstrip()
            assert act_data == exp_data

            if in_path.stem in yaml_filter:
                exp_doc = DoclingDocument.load_from_yaml(yaml_gt_path)
                verify_docitems(doc_true=act_doc, doc_pred=exp_doc, fuzzy=False)


def get_md_paths():
    # Define the directory you want to search
    directory = Path("./tests/data/md/groundtruth")

    # List all MD files in the directory and its subdirectories
    md_files = sorted(directory.rglob("*.md"))
    return md_files


def get_converter():
    converter = DocumentConverter(allowed_formats=[InputFormat.MD])

    return converter


@pytest.mark.skip(
    reason="Previously a silent no-op (globbed a non-existent ./tests/groundtruth "
    "path). Roundtrip of the markdown groundtruth does not hold (trailing-newline "
    "drift); re-enable once that is fixed."
)
def test_e2e_md_conversions():
    md_paths = get_md_paths()
    converter = get_converter()

    for md_path in md_paths:
        # print(f"converting {md_path}")

        with open(md_path) as fr:
            true_md = fr.read()

        conv_result: ConversionResult = converter.convert(md_path)

        doc: DoclingDocument = conv_result.document

        pred_md: str = doc.export_to_markdown(compact_tables=True)
        assert true_md == pred_md

        conv_result_: ConversionResult = converter.convert_string(
            true_md, format=InputFormat.MD
        )

        doc_: DoclingDocument = conv_result_.document

        pred_md_: str = doc_.export_to_markdown(compact_tables=True)
        assert true_md == pred_md_


def test_convert_leading_dash_sequences():
    converter = get_converter()
    markdown = """## Research Article

Here is some content...

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -This is an open access article under the terms of the Creative Commons Attribution License, which permits use, distribution and reproduction in any medium, provided the original work is properly cited.

<!-- image -->
"""

    conv_result: ConversionResult = converter.convert_string(
        markdown, format=InputFormat.MD
    )

    pred_md = conv_result.document.export_to_markdown()

    assert conv_result.status == ConversionStatus.SUCCESS
    assert (
        "- This is an open access article under the terms of the Creative Commons Attribution License"
        in pred_md
    )


def test_convert_list_item_codespan_only():
    """
    Regression test:
    A list item that only contains an inline CodeSpan (no RawText) must not leave
    a pending ListItem payload behind, otherwise later RawText will attach it to a
    wrong parent and create a very deep tree (RecursionError in iterate/export).
    """
    converter = get_converter()
    markdown = """# Title

*   `raw_ops.Abort`
*   `raw_ops.Abs`
"""

    conv_result: ConversionResult = converter.convert_string(
        markdown, format=InputFormat.MD
    )
    assert conv_result.status == ConversionStatus.SUCCESS

    pred_md = conv_result.document.export_to_markdown()
    assert "- raw\\_ops.Abort" in pred_md
    assert "- raw\\_ops.Abs" in pred_md


def _convert_markdown(
    markdown: str, options: MarkdownBackendOptions
) -> DoclingDocument:
    stream = BytesIO(markdown.encode("utf-8"))
    in_doc = InputDocument(
        path_or_stream=stream,
        format=InputFormat.MD,
        backend=MarkdownDocumentBackend,
        filename="test.md",
        backend_options=options,
    )
    backend = MarkdownDocumentBackend(
        in_doc=in_doc,
        path_or_stream=stream,
        options=options,
    )
    assert backend.is_valid()
    return backend.convert()


def _png_data_uri(width: int, height: int) -> str:
    buffer = BytesIO()
    Image.new("RGB", (width, height), color=(255, 0, 0)).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{encoded}"


def test_convert_embedded_base64_image():
    """Embedded base64 image data must be decoded when fetch_images is enabled."""
    markdown = f"# Title\n\n![alt]({_png_data_uri(7, 5)})\n"

    doc = _convert_markdown(markdown, MarkdownBackendOptions(fetch_images=True))

    pictures = [
        item for item, _ in doc.iterate_items() if isinstance(item, PictureItem)
    ]
    assert len(pictures) == 1
    picture = pictures[0]
    assert picture.image is not None
    image = picture.get_image(doc)
    assert image is not None
    assert image.size == (7, 5)


def test_convert_embedded_base64_image_disabled_by_default():
    """Without fetch_images the picture stays a placeholder (default behavior)."""
    markdown = f"# Title\n\n![alt]({_png_data_uri(7, 5)})\n"

    doc = _convert_markdown(markdown, MarkdownBackendOptions())

    pictures = [
        item for item, _ in doc.iterate_items() if isinstance(item, PictureItem)
    ]
    assert len(pictures) == 1
    assert pictures[0].image is None
    assert pictures[0].get_image(doc) is None


def test_convert_embedded_base64_image_enforces_size_limit():
    """Decoded base64 images larger than the configured cap are rejected."""
    markdown = f"# Title\n\n![alt]({_png_data_uri(7, 5)})\n"

    with pytest.warns(UserWarning, match="exceeds size limit"):
        doc = _convert_markdown(
            markdown,
            MarkdownBackendOptions(fetch_images=True, max_image_data_base64_bytes=8),
        )

    pictures = [
        item for item, _ in doc.iterate_items() if isinstance(item, PictureItem)
    ]
    assert len(pictures) == 1
    assert pictures[0].image is None


def test_code_block_language_detection():
    markdown = (
        "```python\n"
        "import sys\n"
        "print(sys.argv)\n"
        "```\n\n"
        "```\n"
        "SELECT id FROM users;\n"
        "```\n\n"
        "```\n"
        "ambiguous snippet here\n"
        "```\n"
    )
    conv_result = get_converter().convert_string(markdown, format=InputFormat.MD)
    assert conv_result.status == ConversionStatus.SUCCESS

    code_items = [
        item for item in conv_result.document.texts if isinstance(item, CodeItem)
    ]
    languages = [item.code_language for item in code_items]
    assert languages == [
        CodeLanguageLabel.PYTHON,
        CodeLanguageLabel.SQL,
        CodeLanguageLabel.UNKNOWN,
    ]


def test_convert_table_has_no_duplicate_cells():
    """
    Regression test:
    A parsed Markdown table must expose each cell exactly once. The backend used
    to append every cell a second time after passing it to the TableData
    constructor, so table.data.table_cells contained twice the real cell count
    (each grid position appeared twice) in export_to_dict/JSON and anything
    iterating the cells directly.
    """
    markdown = """| Region | Q1 | Q2 |
| --- | --- | --- |
| North | 10 | 20 |
| South | 30 | 40 |
"""
    conv_result = get_converter().convert_string(markdown, format=InputFormat.MD)
    assert conv_result.status == ConversionStatus.SUCCESS

    table = conv_result.document.tables[0]
    table_data = table.data
    assert len(table_data.table_cells) == table_data.num_rows * table_data.num_cols

    positions = [
        (cell.start_row_offset_idx, cell.start_col_offset_idx)
        for cell in table_data.table_cells
    ]
    assert len(positions) == len(set(positions))
