"""Contract tests for the dependency-free PaddleOCR-VL result adapter.

Most payloads in this module are deliberately synthetic mechanics fixtures.
The final regression uses an unedited ``prunedResult`` produced by the official
Baidu AI Studio PaddleOCR-VL 1.6 service; its provenance is recorded beside the
fixture.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from docling_core.types.doc import (
    CoordOrigin,
    DocItemLabel,
    DoclingDocument,
)

from docling.utils.paddleocr_vl_utils import parse_paddleocr_vl_result

_PADDLEOCR_VL_DATA_DIR = Path(__file__).parent / "data" / "json_paddleocr_vl"
_AISTUDIO_PRUNED_RESULT = (
    _PADDLEOCR_VL_DATA_DIR / "self_authored_page.paddleocr-vl-1.6.aistudio-pruned.json"
)


def _block(
    label: str,
    content: str,
    *,
    bbox: list[float] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Build one synthetic provider block for adapter mechanics tests."""
    return {
        "block_label": label,
        "block_content": content,
        "block_bbox": bbox or [10, 20, 300, 80],
        **extra,
    }


def _payload(
    blocks: list[Mapping[str, Any]] | None = None,
    *,
    page_index: int | None = 2,
    page_count: int = 3,
) -> dict[str, Any]:
    """Build a synthetic single-page PaddleOCR-VL serialization payload."""
    return {
        "input_path": "fixtures/测试文档.pdf",
        "page_index": page_index,
        "page_count": page_count,
        "width": 612,
        "height": 792,
        "model_settings": {"use_doc_preprocessor": True},
        "parsing_res_list": list(
            blocks if blocks is not None else [_block("text", "你好, PaddleOCR-VL")]
        ),
    }


@pytest.mark.parametrize(
    ("representation", "wrapped"),
    [
        ("mapping", False),
        ("mapping", True),
        ("string", False),
        ("string", True),
        ("bytes", False),
        ("bytes", True),
    ],
)
def test_accepts_bare_and_wrapped_mapping_string_and_utf8_bytes(
    representation: str,
    wrapped: bool,
) -> None:
    payload: Mapping[str, Any] = _payload()
    if wrapped:
        payload = {"res": payload}

    content: Mapping[str, Any] | str | bytes
    if representation == "mapping":
        content = payload
    else:
        serialized = json.dumps(payload, ensure_ascii=False)
        content = serialized if representation == "string" else serialized.encode()

    doc = parse_paddleocr_vl_result(content)

    assert doc.name == "测试文档"
    assert doc.origin is not None
    assert doc.origin.filename == "测试文档.pdf"
    assert doc.texts[0].text == "你好, PaddleOCR-VL"
    assert doc.texts[0].prov[0].page_no == 3


def test_preserves_native_array_order_and_ignores_order_and_group_metadata() -> None:
    payload = _payload(
        [
            _block("text", "first", block_order=30, group_id=8),
            _block("text", "second", block_order=10, group_id=7),
            _block("text", "third", block_order=20, group_id=8),
        ]
    )

    doc = parse_paddleocr_vl_result(payload)
    texts = [item.text for item, _ in doc.iterate_items()]

    assert texts == ["first", "second", "third"]


def test_maps_supported_textual_labels() -> None:
    expected = [
        ("abstract", DocItemLabel.TEXT),
        ("algorithm", DocItemLabel.TEXT),
        ("aside_text", DocItemLabel.TEXT),
        ("content", DocItemLabel.TEXT),
        ("doc_title", DocItemLabel.TITLE),
        ("figure_title", DocItemLabel.CAPTION),
        ("footer", DocItemLabel.PAGE_FOOTER),
        ("footnote", DocItemLabel.FOOTNOTE),
        ("formula_number", DocItemLabel.TEXT),
        ("header", DocItemLabel.PAGE_HEADER),
        ("number", DocItemLabel.TEXT),
        ("ocr", DocItemLabel.TEXT),
        ("paragraph_title", DocItemLabel.SECTION_HEADER),
        ("reference", DocItemLabel.TEXT),
        ("reference_content", DocItemLabel.REFERENCE),
        ("spotting", DocItemLabel.TEXT),
        ("text", DocItemLabel.TEXT),
        ("vertical_text", DocItemLabel.TEXT),
        ("vision_footnote", DocItemLabel.FOOTNOTE),
    ]
    payload = _payload(
        [
            _block(native_label, f"content-{index}")
            for index, (native_label, _) in enumerate(expected)
        ]
    )

    doc = parse_paddleocr_vl_result(payload)
    items = [item for item, _ in doc.iterate_items()]

    assert [item.label for item in items] == [label for _, label in expected]
    assert [item.text for item in items] == [
        f"content-{index}" for index in range(len(expected))
    ]


def test_normalizes_provider_markdown_wrappers_when_block_content_is_formatted() -> (
    None
):
    payload = _payload(
        [
            _block("doc_title", "# Main title"),
            _block("paragraph_title", "### Nested heading"),
            _block(
                "figure_title",
                '<div style="text-align: center;">Figure caption</div>\n',
            ),
        ]
    )
    payload["model_settings"]["format_block_content"] = True

    doc = parse_paddleocr_vl_result(payload)

    assert doc.texts[0].text == "Main title"
    assert doc.texts[1].text == "Nested heading"
    assert doc.texts[1].level == 2
    assert doc.texts[2].text == "Figure caption"
    assert doc.export_to_markdown() == (
        "# Main title\n\n### Nested heading\n\nFigure caption"
    )


def test_preserves_formatting_without_formatted_content_flag() -> None:
    payload = _payload(
        [
            _block("doc_title", "# Literal title"),
            _block("paragraph_title", "### Literal heading"),
            _block(
                "figure_title",
                '<div style="text-align: center;">Literal caption</div>',
            ),
        ]
    )

    doc = parse_paddleocr_vl_result(payload)

    assert doc.texts[0].text == "# Literal title"
    assert doc.texts[1].text == "### Literal heading"
    assert doc.texts[1].level == 1
    assert doc.texts[2].text == (
        '<div style="text-align: center;">Literal caption</div>'
    )


def test_parses_table_html_into_table_data() -> None:
    html = (
        "<table><tr><th>指标</th><th>值</th></tr>"
        "<tr><td>收入</td><td>42</td></tr></table>"
    )

    doc = parse_paddleocr_vl_result(_payload([_block("table", html)]))

    assert len(doc.tables) == 1
    table = doc.tables[0]
    assert table.data.num_rows == 2
    assert table.data.num_cols == 2
    assert [cell.text for cell in table.data.table_cells] == [
        "指标",
        "值",
        "收入",
        "42",
    ]


def test_preserves_formula_content_exactly() -> None:
    formulas = [r"E = mc^2", r"\frac{净利润}{营业收入}", r"x+y=z"]
    payload = _payload(
        [
            _block("display_formula", formulas[0]),
            _block("inline_formula", formulas[1]),
            _block("formula", formulas[2]),
        ]
    )

    doc = parse_paddleocr_vl_result(payload)
    items = [item for item, _ in doc.iterate_items()]

    assert [item.label for item in items] == [
        DocItemLabel.FORMULA,
        DocItemLabel.FORMULA,
        DocItemLabel.FORMULA,
    ]
    assert [item.text for item in items] == formulas


def test_preserves_picture_chart_and_seal_content_without_calling_it_description() -> (
    None
):
    contents = ["architecture diagram", "quarterly revenue chart", "company seal"]
    payload = _payload(
        [
            _block("image", contents[0]),
            _block("chart", contents[1]),
            _block("seal", contents[2]),
        ]
    )

    doc = parse_paddleocr_vl_result(payload)

    assert len(doc.pictures) == 3
    assert all(picture.label == DocItemLabel.PICTURE for picture in doc.pictures)
    assert all(picture.meta is None for picture in doc.pictures)
    assert [item.text for item in doc.texts] == contents


def test_unknown_label_warns_and_falls_back_to_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        doc = parse_paddleocr_vl_result(
            _payload([_block("future_provider_label", "keep this content")])
        )

    assert len(doc.texts) == 1
    assert doc.texts[0].label == DocItemLabel.TEXT
    assert doc.texts[0].text == "keep this content"
    assert "future_provider_label" in caplog.text


def test_uses_processed_top_left_canvas_page_index_and_block_bbox() -> None:
    payload = _payload(
        [
            _block(
                "text",
                "geometry",
                bbox=[11, 22, 333, 444],
                block_polygon_points=[
                    [500, 500],
                    [550, 500],
                    [550, 550],
                    [500, 550],
                ],
            )
        ],
        page_index=2,
        page_count=3,
    )

    doc = parse_paddleocr_vl_result(payload)
    prov = doc.texts[0].prov[0]

    assert doc.pages[3].size.width == 612
    assert doc.pages[3].size.height == 792
    assert prov.page_no == 3
    assert prov.bbox.coord_origin == CoordOrigin.TOPLEFT
    assert (prov.bbox.l, prov.bbox.t, prov.bbox.r, prov.bbox.b) == (
        11,
        22,
        333,
        444,
    )


def test_filename_and_page_number_overrides_take_precedence() -> None:
    doc = parse_paddleocr_vl_result(
        _payload(page_index=0, page_count=1),
        filename="override.json",
        page_no=9,
    )

    assert doc.name == "override"
    assert doc.origin is not None
    assert doc.origin.filename == "override.json"
    assert set(doc.pages) == {9}
    assert doc.texts[0].prov[0].page_no == 9


def test_derives_filename_from_foreign_platform_path() -> None:
    payload = _payload(page_index=0, page_count=1)
    payload["input_path"] = r"C:\provider\fixtures\cross-platform.pdf"

    doc = parse_paddleocr_vl_result(payload)

    assert doc.name == "cross-platform"
    assert doc.origin is not None
    assert doc.origin.filename == "cross-platform.pdf"


@pytest.mark.parametrize(
    "content",
    [
        '{"width": 612,',
        ["not", "an", "object"],
    ],
)
def test_rejects_malformed_json_or_non_object_schema(content: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        parse_paddleocr_vl_result(content)


def test_rejects_missing_required_page_schema() -> None:
    payload = _payload()
    payload.pop("width")

    with pytest.raises(ValueError):
        parse_paddleocr_vl_result(payload)


@pytest.mark.parametrize(
    "bbox",
    [
        [10, 20, 30],
        ["left", 20, 30, 40],
        [30, 20, 10, 40],
        [-1, 20, 30, 40],
        [10, 20, 700, 40],
    ],
)
def test_rejects_malformed_or_out_of_canvas_bbox(bbox: list[Any]) -> None:
    with pytest.raises(ValueError):
        parse_paddleocr_vl_result(_payload([_block("text", "bad", bbox=bbox)]))


def test_rejects_concatenated_multipage_result_without_page_identity() -> None:
    payload = _payload(page_index=None, page_count=2)

    with pytest.raises(ValueError, match="multi-page"):
        parse_paddleocr_vl_result(payload)

    with pytest.raises(ValueError, match="multi-page"):
        parse_paddleocr_vl_result(payload, page_no=4)


def test_rejects_page_index_outside_page_count() -> None:
    with pytest.raises(ValueError, match="outside page_count"):
        parse_paddleocr_vl_result(_payload(page_index=3, page_count=3))


@pytest.mark.parametrize(
    ("field", "value"), [("width", float("inf")), ("height", float("nan"))]
)
def test_rejects_non_finite_page_dimensions(field: str, value: float) -> None:
    payload = _payload()
    payload[field] = value

    with pytest.raises(ValueError):
        parse_paddleocr_vl_result(payload)


def test_rejects_provider_error_envelope() -> None:
    with pytest.raises(ValueError):
        parse_paddleocr_vl_result({"error": "invalid model settings"})


def test_preserves_empty_provider_content_without_inventing_text() -> None:
    doc = parse_paddleocr_vl_result(
        _payload(
            [
                _block("text", ""),
                _block("formula", ""),
                _block("image", ""),
            ]
        )
    )

    assert [item.text for item in doc.texts] == ["", ""]
    assert len(doc.pictures) == 1


def test_accepts_additive_provider_fields() -> None:
    payload = _payload(
        [
            _block(
                "text",
                "forward compatible",
                provider_block_extension={"future": True},
            )
        ]
    )
    payload["provider_page_extension"] = {"schema_revision": "future"}

    doc = parse_paddleocr_vl_result(payload)

    assert doc.texts[0].text == "forward compatible"


def test_docling_json_round_trip_preserves_adapter_output() -> None:
    payload = _payload(
        [
            _block("doc_title", "A title", bbox=[10, 10, 400, 50]),
            _block("text", "正文", bbox=[10, 60, 400, 100]),
            _block("image", "diagram description", bbox=[10, 110, 400, 300]),
        ]
    )
    doc = parse_paddleocr_vl_result(payload)

    restored = DoclingDocument.model_validate_json(doc.model_dump_json())

    assert restored.export_to_dict() == doc.export_to_dict()
    assert [item.label for item, _ in restored.iterate_items()] == [
        DocItemLabel.TITLE,
        DocItemLabel.TEXT,
        DocItemLabel.PICTURE,
        DocItemLabel.TEXT,
    ]


def test_imports_official_aistudio_paddleocr_vl_1_6_pruned_result() -> None:
    """Regress against a real hosted 1.6 result for the self-authored page."""
    doc = parse_paddleocr_vl_result(
        _AISTUDIO_PRUNED_RESULT.read_bytes(),
        filename="self_authored_page.png",
    )

    assert doc.pages[1].size.width == 960
    assert doc.pages[1].size.height == 1280

    assert doc.origin is not None
    assert doc.origin.filename == "self_authored_page.png"

    items = [item for item, _ in doc.iterate_items()]
    labels = [item.label for item in items]
    assert labels.count(DocItemLabel.PAGE_HEADER) == 1
    assert labels.count(DocItemLabel.SECTION_HEADER) == 2
    assert labels.count(DocItemLabel.TEXT) == 4
    assert labels.count(DocItemLabel.TABLE) == 1
    assert labels.count(DocItemLabel.CAPTION) == 1
    assert items[0].text == "PaddleOCR-VL Adapter Fixture"
    assert items[1].text == "Quarterly Metrics"
    assert items[3].text == "Profit = Revenue - Cost"
    assert items[5].text == "Table 1. Synthetic values for adapter testing."
    assert items[6].text == "Notes"

    for item in items:
        assert len(item.prov) == 1
        prov = item.prov[0]
        assert prov.page_no == 1
        assert prov.bbox.coord_origin == CoordOrigin.TOPLEFT
        assert 0 <= prov.bbox.l < prov.bbox.r <= 960
        assert 0 <= prov.bbox.t < prov.bbox.b <= 1280

    assert len(doc.tables) == 1
    table = doc.tables[0].data
    assert (table.num_rows, table.num_cols) == (3, 4)
    assert [cell.text for cell in table.table_cells[:8]] == [
        "Region",
        "Revenue",
        "Cost",
        "Profit",
        "North",
        "120",
        "80",
        "40",
    ]

    markdown = doc.export_to_markdown()
    assert "## Quarterly Metrics" in markdown
    assert "### Notes" in markdown
    assert "## ##" not in markdown
    assert "&lt;div" not in markdown
