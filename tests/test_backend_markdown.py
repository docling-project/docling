from pathlib import Path

from docling.backend.md_backend import MarkdownDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import DoclingDocument, InputDocument

from .test_data_gen_flag import GEN_TEST_DATA


def test_convert_valid():
    fmt = InputFormat.MD
    cls = MarkdownDocumentBackend

    root_path = Path("tests") / "data"
    relevant_paths = sorted((root_path / "md").rglob("*.md"))
    assert len(relevant_paths) > 0

    yaml_filter = ["inline_and_formatting"]

    for in_path in relevant_paths:
        md_gt_path = root_path / "groundtruth" / "docling_v2" / f"{in_path.name}.md"
        yaml_gt_path = root_path / "groundtruth" / "docling_v2" / f"{in_path.name}.yaml"

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

        if GEN_TEST_DATA:
            with open(md_gt_path, mode="w", encoding="utf-8") as f:
                f.write(f"{act_data}\n")

            if in_path.stem in yaml_filter:
                with open(yaml_gt_path, mode="w", encoding="utf-8") as f:
                    act_doc.save_as_yaml(yaml_gt_path)
        else:
            with open(md_gt_path, encoding="utf-8") as f:
                exp_data = f.read().rstrip()
            assert act_data == exp_data

            if in_path.stem in yaml_filter:
                exp_doc = DoclingDocument.load_from_yaml(yaml_gt_path)
                assert act_doc == exp_doc
