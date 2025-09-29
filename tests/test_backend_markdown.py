from pathlib import Path

from docling.backend.md_backend import MarkdownDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import (
    ConversionResult,
    DoclingDocument,
    InputDocument,
    SectionHeaderItem,
)
from docling.document_converter import DocumentConverter
from tests.verify_utils import CONFID_PREC, COORD_PREC

from .test_data_gen_flag import GEN_TEST_DATA
from .verify_utils import verify_document, verify_export

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
