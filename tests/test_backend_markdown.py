from pathlib import Path

from docling_core.types.doc import BoundingBox, DocItem, DocItemLabel, TextItem

from docling.backend.md_backend import MarkdownDocumentBackend
from docling.datamodel.backend_options import MarkdownBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import (
    ConversionResult,
    DoclingDocument,
    InputDocument,
)
from docling.document_converter import DocumentConverter
from tests.verify_utils import CONFID_PREC, COORD_PREC

from .test_data_gen_flag import GEN_TEST_DATA
from .verify_utils import verify_document

GENERATE = GEN_TEST_DATA


def test_convert_valid():
    fmt = InputFormat.MD
    cls = MarkdownDocumentBackend

    root_path = Path("tests") / "data"
    relevant_paths = sorted((root_path / "md").rglob("*.md"))
    assert len(relevant_paths) > 0

    yaml_filter = ["inline_and_formatting", "mixed_without_h1"]
    json_filter = ["escaped_characters"]

    for in_path in relevant_paths:
        md_gt_path = root_path / "groundtruth" / "docling_v2" / f"{in_path.name}.md"
        yaml_gt_path = root_path / "groundtruth" / "docling_v2" / f"{in_path.name}.yaml"
        json_gt_path = root_path / "groundtruth" / "docling_v2" / f"{in_path.name}.json"

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
        act_data = act_doc.export_to_markdown()

        if in_path.stem in json_filter:
            assert verify_document(act_doc, json_gt_path, GENERATE), "export to json"

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
                assert act_doc == exp_doc, f"export to yaml failed on {in_path}"


def get_md_paths():
    # Define the directory you want to search
    directory = Path("./tests/groundtruth/docling_v2")

    # List all MD files in the directory and its subdirectories
    md_files = sorted(directory.rglob("*.md"))
    return md_files


def get_converter():
    converter = DocumentConverter(allowed_formats=[InputFormat.MD])

    return converter


def test_e2e_md_conversions():
    md_paths = get_md_paths()
    converter = get_converter()

    for md_path in md_paths:
        # print(f"converting {md_path}")

        with open(md_path) as fr:
            true_md = fr.read()

        conv_result: ConversionResult = converter.convert(md_path)

        doc: DoclingDocument = conv_result.document

        pred_md: str = doc.export_to_markdown()
        assert true_md == pred_md

        conv_result_: ConversionResult = converter.convert_string(
            true_md, format=InputFormat.MD
        )

        doc_: DoclingDocument = conv_result_.document

        pred_md_: str = doc_.export_to_markdown()
        assert true_md == pred_md_


def test_annotated_markdown():
    """Test parsing of annotated markdown with label and bbox information."""
    in_path = Path("tests") / "data" / "md" / "annotated_simple.md"

    # Test with parse_annotations=True
    options = MarkdownBackendOptions(parse_annotations=True)
    in_doc = InputDocument(
        path_or_stream=in_path,
        format=InputFormat.MD,
        backend=MarkdownDocumentBackend,
        backend_options=options,
    )
    backend = MarkdownDocumentBackend(
        in_doc=in_doc,
        path_or_stream=in_path,
        options=options,
    )
    assert backend.is_valid()

    doc = backend.convert()

    # Verify that items have proper labels and bounding boxes
    items_with_prov = []
    for item, _ in doc.iterate_items():
        if isinstance(item, DocItem) and len(item.prov) > 0:
            items_with_prov.append(item)

    # Should have multiple items with provenance
    assert len(items_with_prov) > 0, "Expected items with provenance information"

    # Check first text item has correct bbox
    text_items = [item for item in items_with_prov if isinstance(item, TextItem)]
    assert len(text_items) > 0, "Expected at least one text item"

    first_text = text_items[0]
    # prov is a list of ProvenanceItem objects
    prov_item = (
        first_text.prov[0]
        if not isinstance(first_text.prov[0], list)
        else first_text.prov[0][0]
    )
    assert prov_item.bbox.l == 217.0
    assert prov_item.bbox.t == 146.0
    assert prov_item.bbox.r == 785.0
    assert prov_item.bbox.b == 191.0

    # Check section header items
    section_headers = [
        item
        for item in items_with_prov
        if isinstance(item, TextItem) and item.label == DocItemLabel.SECTION_HEADER
    ]
    assert len(section_headers) >= 2, "Expected at least two section headers"

    # Test with parse_annotations=False (default behavior)
    options_no_parse = MarkdownBackendOptions(parse_annotations=False)
    in_doc_no_parse = InputDocument(
        path_or_stream=in_path,
        format=InputFormat.MD,
        backend=MarkdownDocumentBackend,
        backend_options=options_no_parse,
    )
    backend_no_parse = MarkdownDocumentBackend(
        in_doc=in_doc_no_parse,
        path_or_stream=in_path,
        options=options_no_parse,
    )

    doc_no_parse = backend_no_parse.convert()

    # When not parsing annotations, the annotation lines should appear as text
    all_text = doc_no_parse.export_to_markdown()
    assert "text[[217, 146, 785, 191]]" in all_text, (
        "Annotation should be preserved as text when parse_annotations=False"
    )
