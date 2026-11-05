# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Native caption association survives the DCLX boundary used by evaluation."""

import json
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pytest
from docling_core.types.doc import DoclingDocument, FloatingItem, Size

from docling.utils.dots_utils import parse_dots_json
from docling.utils.mineru_utils import parse_mineru2, serialize_mineru2_transcript


def _parse(model: str, types: list[str]) -> DoclingDocument:
    size = Size(width=500, height=700)
    if model == "mineru":
        layout = "".join(
            f"<|box_start|>100 {100 + i * 100} 900 {150 + i * 100}"
            f"<|box_end|><|ref_start|>{kind}<|ref_end|>"
            for i, kind in enumerate(types)
        )
        recognition = [
            (
                i,
                "<fcel>Cell<nl>"
                if kind == "table"
                else "print(1)"
                if kind == "code"
                else "Caption text",
            )
            for i, kind in enumerate(types)
            if kind not in {"image", "image_block"}
        ]
        return parse_mineru2(
            serialize_mineru2_transcript(layout, recognition), size, page_no=1
        )
    blocks = [
        {
            "category": "Text" if kind == "Code" else kind,
            "text": (
                "<table><tr><td>Cell</td></tr></table>"
                if kind == "Table"
                else "```python\nprint(1)\n```"
                if kind == "Code"
                else "<b>Caption</b> <i>text</i>"
                if kind == "RichCaption"
                else "Caption text"
            ),
            "bbox": [10, 10 + i * 50, 400, 30 + i * 50],
        }
        for i, kind in enumerate(types)
    ]
    for block in blocks:
        if block["category"] == "Skipped":
            del block["bbox"]
        elif block["category"] == "RichCaption":
            block["category"] = "Caption"
    return parse_dots_json(json.dumps(blocks), size, page_no=1)


@pytest.mark.parametrize(
    "model,caption,owner,tag",
    [
        ("mineru", "table_caption", "table", "table"),
        ("mineru", "image_caption", "image", "picture"),
        ("dots", "Caption", "Table", "table"),
        ("dots", "Caption", "Picture", "picture"),
    ],
)
def test_caption_association_roundtrip(
    model: str, caption: str, owner: str, tag: str, tmp_path: Path
) -> None:
    for types in ([caption, owner], [owner, caption]):
        doc = _parse(model, types)
        floating = next(
            item for item, _ in doc.iterate_items() if isinstance(item, FloatingItem)
        )
        assert floating.caption_text(doc) == "Caption text"
        cap = floating.captions[0].resolve(doc)
        assert cap.prov[0].bbox != floating.prov[0].bbox
        assert doc.export_to_markdown().count("Caption text") == 1

        path = tmp_path / "caption.dclx"
        doc.save_as_doclang_archive(path)
        with zipfile.ZipFile(path) as archive:
            root = ET.fromstring(archive.read("document.xml"))
        serialized = root.find(f".//{tag}/caption")
        assert serialized is not None
        assert "Caption text" in "".join(serialized.itertext())
        imported = DoclingDocument.load_from_doclang_archive(path)
        imported_owner = next(
            item
            for item, _ in imported.iterate_items()
            if isinstance(item, FloatingItem)
        )
        assert imported_owner.caption_text(imported) == "Caption text"


@pytest.mark.parametrize(
    "model,types",
    [
        ("mineru", ["table", "table_caption", "table"]),
        ("mineru", ["image", "table_caption"]),
        ("mineru", ["table_caption", "text", "table"]),
        ("mineru", ["table_caption", "image_block", "table"]),
        ("mineru", ["code_caption", "code"]),
        ("mineru", ["code", "caption", "table"]),
        ("dots", ["Picture", "Caption", "Table"]),
        ("dots", ["Caption", "Text", "Table"]),
        ("dots", ["Caption", "Skipped", "Table"]),
        ("dots", ["RichCaption", "Table"]),
        ("dots", ["Caption", "Code"]),
        ("dots", ["Code", "Caption", "Table"]),
    ],
)
def test_unresolved_captions_preserve_text(model: str, types: list[str]) -> None:
    doc = _parse(model, types)
    assert all(
        not item.captions
        for item, _ in doc.iterate_items()
        if isinstance(item, FloatingItem)
    )
    assert "Caption" in doc.export_to_doclang()
    assert "<caption>" not in doc.export_to_doclang()


@pytest.mark.parametrize(
    "model,types",
    [
        ("mineru", ["table_caption", "table", "table_caption"]),
        ("dots", ["Caption", "Table", "Caption"]),
    ],
)
def test_additional_caption_remains_standalone(model: str, types: list[str]) -> None:
    doc = _parse(model, types)
    assert len(doc.tables[0].captions) == 1
    root = ET.fromstring(doc.export_to_doclang())
    assert len(root.findall(".//caption")) == 1
    assert len(root.findall("./text")) == 1
    assert doc.export_to_markdown().count("Caption text") == 2
