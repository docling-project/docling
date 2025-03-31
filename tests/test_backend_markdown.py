from pathlib import Path

from docling.backend.md_backend import MarkdownDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument

from .test_data_gen_flag import GEN_TEST_DATA


def test_convert_valid():
    fmt = InputFormat.MD
    cls = MarkdownDocumentBackend

    test_data_path = Path("tests") / "data"
    relevant_paths = sorted((test_data_path / "md").rglob("*.md"))
    assert len(relevant_paths) > 0

    for in_path in relevant_paths:
        gt_path = test_data_path / "groundtruth" / "docling_v2" / f"{in_path.name}.md"

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
            with open(gt_path, mode="w", encoding="utf-8") as f:
                f.write(f"{act_data}\n")
        else:
            with open(gt_path, encoding="utf-8") as f:
                exp_data = f.read().rstrip()
            assert exp_data == act_data
