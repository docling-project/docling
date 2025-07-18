import io
from pathlib import Path
from textwrap import dedent
from typing import Annotated

import pytest
from _pytest.mark import ParameterSet
from docling_core.types.doc.document import DoclingDocument, GroupItem, RefItem

from docling.backend.md_backend import MarkdownDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import (
    InputDocument,
)
from tests.conftest import TEST_DATA_DIR
from tests.verify_utils import CONFID_PREC, COORD_PREC

from .test_data_gen_flag import GEN_TEST_DATA

GENERATE = True or GEN_TEST_DATA

ALSO_GENERATE_YAML = ["inline_and_formatting"]
"""A list of document names that should also be generated as yaml"""

# Test Input Directories
INPUT_DIR = TEST_DATA_DIR / "md"

# Test Output Directories
SNAPSHOT_DIR = TEST_DATA_DIR / "groundtruth" / "docling_v2"

TestCase = Annotated[tuple[str, Path, Path], "test_name, in_file, snapshot_file"]


def markdown_test_data() -> list[ParameterSet]:
    """Returns test cases for each of our input markdown files"""

    test_case_paths = sorted(INPUT_DIR.glob(pattern="*.md"), key=lambda x: x.name)

    test_cases: list[ParameterSet] = []

    for test_case_path in test_case_paths:
        name: str = test_case_path.stem

        markdown_document_path: Path = test_case_path.resolve()

        markdown_snapshot_path: Path = SNAPSHOT_DIR / f"{name}.md.md"
        yaml_snapshot_path: Path | None = (
            SNAPSHOT_DIR / f"{name}.md.yaml" if name in ALSO_GENERATE_YAML else None
        )

        test_cases.append(
            pytest.param(
                markdown_document_path,
                markdown_snapshot_path,
                yaml_snapshot_path,
                id=name,
            )
        )

    return test_cases


@pytest.mark.parametrize(
    ("markdown_document_path", "markdown_snapshot_path", "yaml_snapshot_path"),
    markdown_test_data(),
)
def test_convert_markdown(
    markdown_document_path: Path,
    markdown_snapshot_path: Path,
    yaml_snapshot_path: Path | None,
):
    """Test that the Markdown backend can:
     1) convert the input markdown file to a DoclingDocument
     2) export the markdown (and optionally, yaml) and verify it matches the committed snapshot
     """

    if not GENERATE and not markdown_snapshot_path.exists():
        pytest.skip(
            f"Test requires {markdown_snapshot_path} to exist, you may need to generate it with GENERATE=True"
        )

    document_backend = MarkdownDocumentBackend(
        in_doc=InputDocument(
            path_or_stream=markdown_document_path,
            format=InputFormat.MD,
            backend=MarkdownDocumentBackend,
        ),
        path_or_stream=markdown_document_path,
    )

    assert document_backend.is_valid()

    try:
        out_docling_document: DoclingDocument = document_backend.convert()
    except Exception as e:
        pytest.skip(f"Error converting {markdown_document_path}: {e}")

    # Validate the YAML/JSON Export
    if yaml_snapshot_path:
        if GENERATE:
            out_docling_document.save_as_yaml(
                yaml_snapshot_path,
                coord_precision=COORD_PREC,
                confid_precision=CONFID_PREC,
            )
        else:
            assert out_docling_document == DoclingDocument.load_from_yaml(
                yaml_snapshot_path
            )

    # Validate the Markdown Export
    out_markdown: str = out_docling_document.export_to_markdown()

    if GENERATE:
        _ = markdown_snapshot_path.write_text(out_markdown + "\n")
    else:
        assert (
            out_markdown == markdown_snapshot_path.read_text(encoding="utf-8")
        )


def test_convert_headers_to_groups():
    """Test that the Markdown backend can convert headers into hierarchical groups"""

    input_document = dedent("""
    # Header 1

    some content under the header 1

    ## Header 2a

    some content under the header 2

    ### Header 3

    some content under the header 3

    ## Header 2b
    """)

    in_doc = InputDocument(
        path_or_stream=io.BytesIO(input_document.encode("utf-8")),
        format=InputFormat.MD,
        filename="headers_to_groups.md",
        backend=MarkdownDocumentBackend,
    )
    backend = MarkdownDocumentBackend(
        in_doc=in_doc,
        path_or_stream=io.BytesIO(input_document.encode("utf-8")),
    )

    act_doc: DoclingDocument = backend.convert()

    assert len(act_doc.body.children) == 1
    body_first_child_ref: RefItem = act_doc.body.children[0]
    assert isinstance(body_first_child_ref, RefItem)

    assert body_first_child_ref.cref == "#/groups/0"

    body_first_child: GroupItem = body_first_child_ref.resolve(act_doc)

    # The first child should have the header, content and two subheaders
    assert len(body_first_child.children) == 4

    act_data = act_doc.export_to_markdown()

    expected_output = dedent("""
    # Header 1

    some content under the header 1

    ## Header 2a

    some content under the header 2

    ### Header 3

    some content under the header 3

    ## Header 2b
    """).strip()

    assert act_data == expected_output
