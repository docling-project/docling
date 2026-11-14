# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Contract tests for the OXR parse-result adapter and its HTTP client.

``self_authored_pages.oxr.json`` follows the serialized OXR JSON output by
hand; ``real_code_and_formula.oxr.json`` is an unedited response of the
official free OXR API. See the ``.source.json`` note beside each fixture.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from docling_core.types.doc import CoordOrigin, DocItemLabel, DoclingDocument
from PIL import Image

from docling.utils.oxr_utils import parse_oxr_result, run_oxr_parse

_DATA_DIR = Path(__file__).parent / "data" / "json_oxr"
_FIXTURE = _DATA_DIR / "self_authored_pages.oxr.json"
_REAL_FIXTURE = _DATA_DIR / "real_code_and_formula.oxr.json"


def _block(
    kind: str,
    content: str,
    *,
    bbox: tuple[float, float, float, float] = (10, 20, 300, 80),
    fmt: str = "text",
) -> dict[str, Any]:
    return {"type": kind, "bbox": list(bbox), "content": content, "format": fmt}


def _page(
    blocks: list[dict[str, Any]],
    *,
    page_num: int = 0,
    width: float = 800,
    height: float = 1000,
) -> dict[str, Any]:
    return {
        "page_num": page_num,
        "page_size": {"w": width, "h": height},
        "blocks": blocks,
    }


def _result(
    pages: list[dict[str, Any]], *, file_name: str = "demo.pdf"
) -> dict[str, Any]:
    return {
        "file_name": file_name,
        "metadata": {},
        "outputs": {"markdown": "", "json": {"pages": pages}},
    }


def _response(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "code": 0,
        "message": "success",
        "warnings": [],
        "data": {"task_id": "t", "error": None, "result": result, "metadata": {}},
    }


def _parse_one(block: dict[str, Any]) -> DoclingDocument:
    return parse_oxr_result(_response(_result([_page([block])])))


@pytest.mark.parametrize(
    "shape", ["response", "result", "pages", "json_text", "json_bytes"]
)
def test_accepts_the_saved_oxr_response_shapes(shape: str) -> None:
    pages = [_page([_block("Text", "hello")])]
    result = _result(pages)
    content: Mapping[str, Any] | str | bytes
    if shape == "response":
        content = _response(result)
    elif shape == "result":
        content = result
    elif shape == "pages":
        content = {"pages": pages}
    elif shape == "json_text":
        content = json.dumps(_response(result))
    else:
        content = json.dumps(_response(result)).encode()

    doc = parse_oxr_result(content)

    assert [item.text for item in doc.texts] == ["hello"]
    expected_name = "oxr_result" if shape == "pages" else "demo"
    assert doc.name == expected_name


def test_preserves_page_order_block_order_and_canvas_geometry() -> None:
    first = _page(
        [
            _block("Text", "first", bbox=(10, 20, 110, 60)),
            _block("Text", "second", bbox=(10, 70, 110, 120)),
        ],
        page_num=0,
    )
    second = _page(
        [_block("Text", "third", bbox=(5, 5, 50, 50))],
        page_num=1,
        width=400,
        height=600,
    )

    # Pages arrive out of order on purpose; blocks stay in the given order.
    doc = parse_oxr_result(_response(_result([second, first])))

    assert [item.text for item in doc.texts] == ["first", "second", "third"]
    assert doc.pages[1].size.width == 800
    assert doc.pages[2].size.width == 400
    prov = doc.texts[0].prov[0]
    assert prov.page_no == 1
    assert prov.bbox.coord_origin == CoordOrigin.TOPLEFT
    assert (prov.bbox.l, prov.bbox.t, prov.bbox.r, prov.bbox.b) == (
        10,
        20,
        110,
        60,
    )


@pytest.mark.parametrize(
    ("oxr_type", "label"),
    [
        ("Text", DocItemLabel.TEXT),
        ("Caption", DocItemLabel.CAPTION),
        ("Footnote", DocItemLabel.FOOTNOTE),
        ("Page-header", DocItemLabel.PAGE_HEADER),
        ("Page-footer", DocItemLabel.PAGE_FOOTER),
    ],
)
def test_maps_each_textual_label(oxr_type: str, label: DocItemLabel) -> None:
    doc = _parse_one(_block(oxr_type, "value"))

    assert [item.label for item in doc.texts] == [label]


def test_first_title_is_the_document_title_and_later_titles_are_sections() -> None:
    doc = parse_oxr_result(
        _response(
            _result(
                [
                    _page(
                        [
                            _block("Title", "Doc title", bbox=(10, 10, 300, 40)),
                            _block("Title", "Section", bbox=(10, 50, 300, 80)),
                        ]
                    )
                ]
            )
        )
    )

    assert doc.texts[0].label == DocItemLabel.TITLE
    assert doc.texts[1].label == DocItemLabel.SECTION_HEADER


def test_parses_html_tables_into_table_items() -> None:
    html = (
        "<table><tr><th>name</th><th>value</th></tr>"
        "<tr><td>alpha</td><td>1</td></tr></table>"
    )

    doc = _parse_one(_block("Table", html, fmt="html"))

    assert len(doc.tables) == 1
    assert [cell.text for cell in doc.tables[0].data.table_cells] == [
        "name",
        "value",
        "alpha",
        "1",
    ]


def test_parses_markdown_tables_when_the_service_returns_them() -> None:
    markdown = "| name | value |\n| --- | --- |\n| alpha | 1 |"

    doc = _parse_one(_block("Table", markdown, fmt="markdown"))

    assert len(doc.tables) == 1
    assert doc.tables[0].data.num_cols == 2
    assert [cell.text for cell in doc.tables[0].data.table_cells] == [
        "name",
        "value",
        "alpha",
        "1",
    ]


def test_unparsable_table_content_is_kept_as_text() -> None:
    doc = _parse_one(_block("Table", "not a table", fmt="text"))

    assert not doc.tables
    assert [item.text for item in doc.texts] == ["not a table"]


def test_strips_math_delimiters_and_keeps_chemical_content_as_formula() -> None:
    doc = parse_oxr_result(
        _response(
            _result(
                [
                    _page(
                        [
                            _block("Formula", "$$E = mc^2$$", fmt="latex"),
                            _block(
                                "Chemical Equation",
                                r"\(2H_2 + O_2\)",
                                bbox=(10, 90, 300, 140),
                                fmt="latex",
                            ),
                            _block(
                                "Chemical Structure",
                                r"$\smiles{CCO}$",
                                bbox=(10, 150, 300, 200),
                                fmt="markdown",
                            ),
                        ]
                    )
                ]
            )
        )
    )

    assert [item.text for item in doc.texts] == [
        "E = mc^2",
        "2H_2 + O_2",
        r"\smiles{CCO}",
    ]
    assert [item.label for item in doc.texts] == [DocItemLabel.FORMULA] * 3


def test_removes_code_fences_from_code_blocks() -> None:
    doc = _parse_one(_block("Code Block", "```python\nprint(1)\n```", fmt="markdown"))

    assert doc.texts[0].label == DocItemLabel.CODE
    assert doc.texts[0].text == "print(1)"


def test_pictures_keep_no_image_unless_local_fetch_is_enabled(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "imgs" / "0_6.png"
    image_path.parent.mkdir()
    Image.new("RGB", (4, 4), color="white").save(image_path)
    page = _page([_block("Picture", "![](imgs/0_6.png)", fmt="markdown")])

    without_images = parse_oxr_result(_response(_result([page])))
    assert [picture.image for picture in without_images.pictures] == [None]

    with_images = parse_oxr_result(
        _response(_result([page])), load_local_images=True, base_dir=tmp_path
    )
    assert with_images.pictures[0].image is not None


def test_stamp_text_is_kept_as_text() -> None:
    doc = _parse_one(_block("Stamp", "APPROVED"))

    assert [item.label for item in doc.texts] == [DocItemLabel.TEXT]
    assert doc.texts[0].text == "APPROVED"


def test_unknown_region_type_falls_back_to_text_with_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        doc = _parse_one(_block("Mystery", "keep me"))

    assert doc.texts[0].text == "keep me"
    assert any(
        "Unsupported OXR block type" in record.message for record in caplog.records
    )


def test_clamps_regions_that_cross_the_canvas_edge() -> None:
    doc = _parse_one(_block("Text", "edge", bbox=(-5, 10, 900, 120)))

    bbox = doc.texts[0].prov[0].bbox
    assert (bbox.l, bbox.r) == (0, 800)


def test_region_without_page_area_is_rejected() -> None:
    with pytest.raises(ValueError, match="does not intersect"):
        _parse_one(_block("Text", "outside", bbox=(900, 10, 950, 120)))


def test_page_no_override_applies_to_a_single_page_payload() -> None:
    doc = parse_oxr_result(
        _response(_result([_page([_block("Text", "x")])])), page_no=7
    )

    assert list(doc.pages) == [7]


def test_page_no_override_is_rejected_for_multiple_pages() -> None:
    pages = [
        _page([_block("Text", "one")], page_num=0),
        _page([_block("Text", "two")], page_num=1),
    ]

    with pytest.raises(ValueError, match="single-page"):
        parse_oxr_result(_response(_result(pages)), page_no=1)


def test_page_no_override_rejects_a_non_positive_number() -> None:
    with pytest.raises(ValueError, match="1-based"):
        parse_oxr_result(_response(_result([_page([_block("Text", "x")])])), page_no=0)


def test_duplicate_page_numbers_are_rejected() -> None:
    pages = [_page([], page_num=0), _page([], page_num=0)]

    with pytest.raises(ValueError, match="unique"):
        parse_oxr_result(_response(_result(pages)))


def test_pending_async_result_is_reported() -> None:
    with pytest.raises(ValueError, match="may still be running"):
        parse_oxr_result({"code": 0, "data": {"task_id": "t", "result": None}})


def test_remote_picture_reference_keeps_a_picture_without_text() -> None:
    doc = _parse_one(
        _block("Picture", "![](https://example.com/a.png)", fmt="markdown")
    )

    assert len(doc.pictures) == 1
    assert doc.pictures[0].image is None
    assert not doc.texts


def test_picture_without_an_image_reference_keeps_its_text() -> None:
    doc = _parse_one(_block("Picture", "a red bar chart"))

    assert len(doc.pictures) == 1
    assert [item.text for item in doc.texts] == ["a red bar chart"]


def test_service_error_code_is_reported() -> None:
    with pytest.raises(ValueError, match="1133"):
        parse_oxr_result({"code": 1133, "message": "bad file", "data": None})


def test_result_without_a_json_output_is_reported() -> None:
    with pytest.raises(ValueError, match="output_formats=json"):
        parse_oxr_result({"file_name": "x.pdf", "outputs": {"markdown": ""}})


def test_non_json_content_is_rejected() -> None:
    with pytest.raises(ValueError, match="Invalid OXR JSON"):
        parse_oxr_result(b"not json")


def test_unsupported_content_type_is_rejected() -> None:
    with pytest.raises(TypeError):
        parse_oxr_result(42)  # type: ignore[arg-type]


def test_real_capture_builds_the_serialized_oxr_document() -> None:
    doc = parse_oxr_result(_REAL_FIXTURE.read_bytes())

    assert sorted(doc.pages) == [1, 2]
    assert (doc.pages[1].size.width, doc.pages[1].size.height) == (1275, 1651)
    assert (doc.pages[2].size.width, doc.pages[2].size.height) == (1241, 1754)

    labels = [item.label for item, _ in doc.iterate_items()]
    assert labels.count(DocItemLabel.CODE) == 1
    assert labels.count(DocItemLabel.FORMULA) == 1
    assert labels.count(DocItemLabel.CAPTION) == 1
    assert labels.count(DocItemLabel.TITLE) == 1

    markdown = doc.export_to_markdown()
    assert "function add(a, b)" in markdown
    assert "a^{2}+8=12" in markdown


def test_checked_in_fixture_builds_a_multi_page_document() -> None:
    doc = parse_oxr_result(_FIXTURE.read_bytes())

    assert doc.name == "self_authored"
    assert sorted(doc.pages) == [1, 2]
    assert doc.pages[1].size.width == 800
    assert len(doc.tables) == 1
    assert len(doc.pictures) == 1
    labels = [item.label for item in doc.texts]
    assert labels[0] == DocItemLabel.PAGE_HEADER
    for expected in (
        DocItemLabel.TITLE,
        DocItemLabel.CODE,
        DocItemLabel.FORMULA,
        DocItemLabel.PAGE_FOOTER,
        DocItemLabel.FOOTNOTE,
    ):
        assert expected in labels


@contextmanager
def _oxr_service(
    payload: Mapping[str, Any],
) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    received: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            received.append(
                {
                    "path": self.path,
                    "body": body,
                    "content_type": self.headers.get("Content-Type", ""),
                }
            )
            encoded = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format: str, *args: Any) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1", received
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_run_oxr_parse_posts_multipart_and_unwraps_the_result(
    tmp_path: Path,
) -> None:
    source = tmp_path / "doc.pdf"
    source.write_bytes(b"%PDF-1.4\n")
    payload = _response(_result([_page([_block("Text", "hello")])]))

    with _oxr_service(payload) as (base_url, received):
        result = run_oxr_parse(source, base_url=base_url, keep_header_footer=True)

    assert result["file_name"] == "demo.pdf"
    assert len(received) == 1
    request = received[0]
    assert request["path"] == "/v1/parse/sync"
    assert request["content_type"].startswith("multipart/form-data")
    body = request["body"]
    assert b'name="api_key"' in body
    assert b'name="output_formats"' in body
    assert b'filename="doc.pdf"' in body
    assert b'"keep_header_footer": true' in body
