import base64
import os
import threading
import time
from io import BytesIO
from pathlib import Path, PurePath
from unittest.mock import Mock, mock_open, patch

import pytest
import requests
from bs4 import BeautifulSoup
from docling_core.types.doc import PictureItem, RichTableCell
from docling_core.types.doc.document import ContentLayer
from pydantic import AnyUrl, ValidationError

from docling.backend.html_backend import (
    _BR_SENTINEL,
    HTMLDocumentBackend,
)
from docling.backend.image_resource_loader import (
    validate_url_safety as _validate_url_safety,
)
from docling.datamodel.backend_options import HTMLBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import (
    ConversionResult,
    DoclingDocument,
    InputDocument,
    SectionHeaderItem,
)
from docling.document_converter import DocumentConverter, HTMLFormatOption
from docling.exceptions import OperationNotAllowed

from .test_data_gen_flag import GEN_TEST_DATA
from .verify_utils import verify_document, verify_export

GENERATE = GEN_TEST_DATA


def _create_html_converter(backend_options):
    """Helper to create DocumentConverter with HTML format options."""
    return DocumentConverter(
        allowed_formats=[InputFormat.HTML],
        format_options={
            InputFormat.HTML: HTMLFormatOption(backend_options=backend_options)
        },
    )


def _create_mock_response(data=b"fake_image_data"):
    """Helper to create a mock HTTP response for image fetching."""
    mock_resp = Mock()
    mock_resp.headers = {}
    mock_resp.raise_for_status = Mock()
    mock_resp.iter_content = Mock(return_value=[data])
    mock_resp.is_redirect = False
    mock_resp.is_permanent_redirect = False
    return mock_resp


def test_html_backend_options():
    options = HTMLBackendOptions()
    assert options.kind == "html"
    assert not options.fetch_images
    assert options.source_uri is None

    url = "http://example.com"
    source_location = AnyUrl(url=url)
    options = HTMLBackendOptions(source_uri=source_location)
    assert options.source_uri == source_location

    source_location = PurePath("/local/path/to/file.html")
    options = HTMLBackendOptions(source_uri=source_location)
    assert options.source_uri == source_location

    with pytest.raises(ValidationError, match="Input is not a valid path"):
        HTMLBackendOptions(source_uri=12345)


def test_resolve_relative_path():
    html_path = Path("./tests/data/html/sources/example_01.html")
    in_doc = InputDocument(
        path_or_stream=html_path,
        format=InputFormat.HTML,
        backend=HTMLDocumentBackend,
        filename="test",
    )
    html_doc = HTMLDocumentBackend(path_or_stream=html_path, in_doc=in_doc)
    html_doc.base_path = "/local/path/to/file.html"

    relative_path = "subdir/another.html"
    expected_abs_loc = "/local/path/to/subdir/another.html"
    assert html_doc._resolve_relative_path(relative_path) == expected_abs_loc

    absolute_path = "/absolute/path/to/file.html"
    with pytest.raises(
        ValueError, match="Absolute paths are not allowed with local base_path"
    ):
        html_doc._resolve_relative_path(absolute_path)

    html_doc.base_path = "http://my_host.com"
    protocol_relative_url = "//example.com/file.html"
    expected_abs_loc = "https://example.com/file.html"
    assert html_doc._resolve_relative_path(protocol_relative_url) == expected_abs_loc

    html_doc.base_path = "http://example.com"
    remote_relative_path = "subdir/file.html"
    expected_abs_loc = "http://example.com/subdir/file.html"
    assert html_doc._resolve_relative_path(remote_relative_path) == expected_abs_loc

    html_doc.base_path = "http://example.com"
    remote_relative_path = "https://my_host.com/my_page.html"
    expected_abs_loc = "https://my_host.com/my_page.html"
    assert html_doc._resolve_relative_path(remote_relative_path) == expected_abs_loc

    html_doc.base_path = "http://example.com"
    remote_relative_path = "/static/images/my_image.png"
    expected_abs_loc = "http://example.com/static/images/my_image.png"
    assert html_doc._resolve_relative_path(remote_relative_path) == expected_abs_loc

    # when base_path is None, paths pass through unchanged
    # (validation happens in _load_image_data for actual file access)
    html_doc.base_path = None

    # Paths pass through _resolve_relative_path unchanged
    assert html_doc._resolve_relative_path("subdir/file.html") == "subdir/file.html"

    # Remote URLs also pass through
    remote_url = "https://example.com/file.html"
    assert html_doc._resolve_relative_path(remote_url) == remote_url

    # Fragment-only hrefs must pass through unchanged
    html_doc.base_path = "/local/path/to/file.html"
    assert html_doc._resolve_relative_path("#section1") == "#section1"
    assert html_doc._resolve_relative_path("#") == "#"

    html_doc.base_path = "http://example.com/page.html"
    assert html_doc._resolve_relative_path("#section1") == "#section1"

    html_doc.base_path = None
    assert html_doc._resolve_relative_path("#section1") == "#section1"


def test_heading_levels():
    in_path = Path("tests/data/html/sources/wiki_duck.html")
    in_doc = InputDocument(
        path_or_stream=in_path,
        format=InputFormat.HTML,
        backend=HTMLDocumentBackend,
    )
    backend = HTMLDocumentBackend(
        in_doc=in_doc,
        path_or_stream=in_path,
    )
    doc = backend.convert()

    found_lvl_1 = found_lvl_2 = False
    for item, _ in doc.iterate_items():
        if isinstance(item, SectionHeaderItem):
            if item.text == "Etymology":
                found_lvl_1 = True
                # h2 becomes level 1 because of h1 as title
                assert item.level == 1
            elif item.text == "Feeding":
                found_lvl_2 = True
                # h3 becomes level 2 because of h1 as title
                assert item.level == 2
    assert found_lvl_1 and found_lvl_2


def test_ordered_lists():
    test_set: list[tuple[bytes, str]] = []

    test_set.append(
        (
            b"<html><body><ol><li>1st item</li><li>2nd item</li></ol></body></html>",
            "1. 1st item\n2. 2nd item",
        )
    )
    test_set.append(
        (
            b'<html><body><ol start="1"><li>1st item</li><li>2nd item</li></ol></body></html>',
            "1. 1st item\n2. 2nd item",
        )
    )
    test_set.append(
        (
            b'<html><body><ol start="2"><li>1st item</li><li>2nd item</li></ol></body></html>',
            "2. 1st item\n3. 2nd item",
        )
    )
    test_set.append(
        (
            b'<html><body><ol start="0"><li>1st item</li><li>2nd item</li></ol></body></html>',
            "0. 1st item\n1. 2nd item",
        )
    )
    test_set.append(
        (
            b'<html><body><ol start="-5"><li>1st item</li><li>2nd item</li></ol></body></html>',
            "1. 1st item\n2. 2nd item",
        )
    )
    test_set.append(
        (
            b'<html><body><ol start="foo"><li>1st item</li><li>2nd item</li></ol></body></html>',
            "1. 1st item\n2. 2nd item",
        )
    )

    for idx, pair in enumerate(test_set):
        in_doc = InputDocument(
            path_or_stream=BytesIO(pair[0]),
            format=InputFormat.HTML,
            backend=HTMLDocumentBackend,
            filename="test",
        )
        backend = HTMLDocumentBackend(
            in_doc=in_doc,
            path_or_stream=BytesIO(pair[0]),
        )
        doc: DoclingDocument = backend.convert()
        assert doc
        assert doc.export_to_markdown() == pair[1], f"Error in case {idx}"


def test_description_lists():
    """Test that HTML description lists (<dl>, <dt>, <dd>) are properly parsed."""
    test_set: list[tuple[bytes, str]] = []

    # Simple description list
    test_set.append(
        (
            b"<html><body><dl><dt>Coffee</dt><dd>Black hot drink</dd><dt>Milk</dt><dd>White cold drink</dd></dl></body></html>",
            "- **Coffee**\n    - Black hot drink\n- **Milk**\n    - White cold drink",
        )
    )

    # Description list with multiple descriptions per term
    test_set.append(
        (
            b"<html><body><dl><dt>Python</dt><dd>A high-level programming language</dd><dd>Known for simplicity</dd></dl></body></html>",
            "- **Python**\n    - A high-level programming language\n    - Known for simplicity",
        )
    )

    # Description list with formatting in terms
    test_set.append(
        (
            b"<html><body><dl><dt><strong>HTML</strong></dt><dd>HyperText Markup Language</dd></dl></body></html>",
            "- **HTML**\n    - HyperText Markup Language",
        )
    )

    # Edge case: Empty description list
    test_set.append(
        (
            b"<html><body><dl></dl></body></html>",
            "",
        )
    )

    # Edge case: Description list with dd without dt (discouraged but valid HTML)
    test_set.append(
        (
            b"<html><body><dl><dd>Orphan description 1</dd><dd>Orphan description 2</dd></dl></body></html>",
            "- Orphan description 1\n- Orphan description 2",
        )
    )

    for idx, pair in enumerate(test_set):
        in_doc = InputDocument(
            path_or_stream=BytesIO(pair[0]),
            format=InputFormat.HTML,
            backend=HTMLDocumentBackend,
            filename="test",
        )
        backend = HTMLDocumentBackend(
            in_doc=in_doc,
            path_or_stream=BytesIO(pair[0]),
        )
        doc: DoclingDocument = backend.convert()
        assert doc
        markdown_output = doc.export_to_markdown()
        assert markdown_output == pair[1], (
            f"Error in case {idx}: expected '{pair[1]}', got '{markdown_output}'"
        )


def test_unicode_characters():
    raw_html = "<html><body><h1>Hello World!</h1></body></html>".encode()  # noqa: RUF001
    in_doc = InputDocument(
        path_or_stream=BytesIO(raw_html),
        format=InputFormat.HTML,
        backend=HTMLDocumentBackend,
        filename="test",
    )
    backend = HTMLDocumentBackend(
        in_doc=in_doc,
        path_or_stream=BytesIO(raw_html),
    )
    doc: DoclingDocument = backend.convert()
    assert doc.texts[0].text == "Hello World!"


def test_extract_parent_hyperlinks():
    html_path = Path("./tests/data/html/sources/hyperlink_04.html")
    in_doc = InputDocument(
        path_or_stream=html_path,
        format=InputFormat.HTML,
        backend=HTMLDocumentBackend,
        filename="test",
    )
    backend = HTMLDocumentBackend(
        in_doc=in_doc,
        path_or_stream=html_path,
    )
    div_tag = backend.soup.find("div")
    a_tag = backend.soup.find("a")
    annotated_text_list = backend._extract_text_and_hyperlink_recursively(
        div_tag, find_parent_annotation=True
    )
    assert str(annotated_text_list[0].hyperlink) == a_tag.get("href")


@pytest.fixture(scope="module")
def html_paths() -> list[Path]:
    # Define the directory you want to search
    directory = Path("./tests/data/html/sources/")

    # List all HTML files in the directory and its subdirectories
    html_files = sorted(directory.rglob("*.html"))

    return html_files


def get_converter():
    converter = DocumentConverter(allowed_formats=[InputFormat.HTML])

    return converter


def test_e2e_html_conversions(html_paths):
    converter = get_converter()

    for html_path in html_paths:
        gt_path = html_path.parent.parent / "groundtruth" / html_path.name

        conv_result: ConversionResult = converter.convert(html_path)

        doc: DoclingDocument = conv_result.document

        pred_md: str = doc.export_to_markdown(compact_tables=True)

        # Verify no sentinel characters leak into markdown output
        assert _BR_SENTINEL not in pred_md, (
            f"Sentinel character found in markdown output for {html_path.name}"
        )

        assert verify_export(pred_md, str(gt_path) + ".md", generate=GENERATE), (
            "export to md"
        )

        pred_itxt: str = doc._export_to_indented_text(
            max_text_len=70, explicit_tables=False
        )
        assert verify_export(pred_itxt, str(gt_path) + ".itxt", generate=GENERATE), (
            "export to indented-text"
        )

        assert verify_document(doc, str(gt_path) + ".json", GENERATE)


@patch("docling.backend.image_resource_loader.requests.get")
@patch("docling.backend.image_resource_loader.open", new_callable=mock_open)
def test_e2e_html_conversion_with_images(mock_local, mock_remote):
    source = "tests/data/html/sources/example_01.html"
    image_path = "tests/data/html/sources/example_image_01.png"
    with open(image_path, "rb") as f:
        img_bytes = f.read()

    # fetching image locally
    mock_local.return_value.__enter__.return_value = BytesIO(img_bytes)
    backend_options = HTMLBackendOptions(
        enable_local_fetch=True, fetch_images=True, source_uri=source
    )
    converter = DocumentConverter(
        allowed_formats=[InputFormat.HTML],
        format_options={
            InputFormat.HTML: HTMLFormatOption(backend_options=backend_options)
        },
    )
    res_local = converter.convert(source)
    mock_local.assert_called_once()
    assert res_local.document
    num_pic: int = 0
    for element, _ in res_local.document.iterate_items():
        if isinstance(element, PictureItem):
            assert element.image
            num_pic += 1
    assert num_pic == 1, "No embedded picture was found in the converted file"

    # fetching image remotely - need to mock Session.get instead of requests.get
    with patch(
        "docling.backend.image_resource_loader.requests.Session.get"
    ) as mocked_session_get:
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.headers = {}
        mock_resp.raise_for_status = Mock()
        mock_resp.iter_content = Mock(return_value=[img_bytes])
        mock_resp.is_redirect = False
        mock_resp.is_permanent_redirect = False
        mocked_session_get.return_value = mock_resp
        source_location = "https://example.com/example_01.html"

        backend_options = HTMLBackendOptions(
            enable_remote_fetch=True, fetch_images=True, source_uri=source_location
        )
        converter = DocumentConverter(
            allowed_formats=[InputFormat.HTML],
            format_options={
                InputFormat.HTML: HTMLFormatOption(backend_options=backend_options)
            },
        )
        res_remote = converter.convert(source)
        # Verify the session.get was called
        assert mocked_session_get.call_count == 1
        call_args = mocked_session_get.call_args
        assert call_args[0][0] == "https://example.com/example_image_01.png"
        assert call_args[1]["stream"] is True
        assert call_args[1]["headers"] == {"Range": "bytes=0-20971519"}
        assert call_args[1]["timeout"] == (5, 30)
    assert res_remote.document
    num_pic = 0
    for element, _ in res_remote.document.iterate_items():
        if isinstance(element, PictureItem):
            assert element.image
            assert element.image.mimetype == "image/png"
            num_pic += 1
    assert num_pic == 1, "No embedded picture was found in the converted file"

    # both methods should generate the same DoclingDocument
    assert res_remote.document == res_local.document

    # checking exported formats
    gt_path = "tests/data/html/groundtruth/" + str(Path(source).stem) + "_images.html"
    pred_md: str = res_local.document.export_to_markdown(compact_tables=True)
    assert verify_export(pred_md, gt_path + ".md", generate=GENERATE)
    assert verify_document(res_local.document, gt_path + ".json", GENERATE)


def test_html_furniture():
    raw_html = (
        b"<html><body><p>Initial content with some <strong>bold text</strong></p>"
        b"<h1>Main Heading</h1>"
        b"<p>Some Content</p>"
        b"<footer><p>Some Footer Content</p></footer></body></html"
    )

    in_doc = InputDocument(
        path_or_stream=BytesIO(raw_html),
        format=InputFormat.HTML,
        backend=HTMLDocumentBackend,
        filename="test",
    )
    backend = HTMLDocumentBackend(
        in_doc=in_doc,
        path_or_stream=BytesIO(raw_html),
    )
    doc: DoclingDocument = backend.convert()
    md_body = doc.export_to_markdown()
    assert md_body == "# Main Heading\n\nSome Content"
    md_all = doc.export_to_markdown(
        included_content_layers={ContentLayer.BODY, ContentLayer.FURNITURE}
    )
    assert md_all == (
        "Initial content with some **bold text**\n\n# Main Heading\n\nSome Content\n\n"
        "Some Footer Content"
    )


def test_fetch_remote_images(monkeypatch):
    source = "./tests/data/html/sources/example_01.html"

    # no image fetching: the image_fetch flag is False
    converter = _create_html_converter(
        HTMLBackendOptions(fetch_images=False, source_uri="http://example.com")
    )
    with patch("docling.backend.image_resource_loader.requests.get") as mocked_get:
        res = converter.convert(source)
        mocked_get.assert_not_called()
    assert res.document

    # no image fetching: the source location is False and enable_local_fetch is False
    converter = _create_html_converter(HTMLBackendOptions(fetch_images=True))
    with (
        patch("docling.backend.image_resource_loader.requests.get") as mocked_get,
        pytest.warns(
            match="Fetching local resources is only allowed when set explicitly"
        ),
    ):
        res = converter.convert(source)
        mocked_get.assert_not_called()
    assert res.document

    # no image fetching: the enable_remote_fetch is False
    converter = _create_html_converter(
        HTMLBackendOptions(fetch_images=True, source_uri="http://example.com")
    )
    with (
        patch("docling.backend.image_resource_loader.requests.get") as mocked_get,
        pytest.warns(
            match="Fetching remote resources is only allowed when set explicitly"
        ),
    ):
        res = converter.convert(source)
        mocked_get.assert_not_called()
    assert res.document

    # image fetching: all conditions apply, source location is remote
    converter = _create_html_converter(
        HTMLBackendOptions(
            enable_remote_fetch=True, fetch_images=True, source_uri="http://example.com"
        )
    )
    with patch(
        "docling.backend.image_resource_loader.requests.Session.get"
    ) as mocked_session_get:
        mocked_session_get.return_value = _create_mock_response()
        res = converter.convert(source)
        mocked_session_get.assert_called_once()
    assert res.document

    # image fetching: all conditions apply, local fetching allowed
    converter = _create_html_converter(
        HTMLBackendOptions(
            enable_local_fetch=True, fetch_images=True, source_uri=source
        )
    )
    with (
        patch("docling.backend.image_resource_loader.open") as mocked_open,
        pytest.warns(match="a bytes-like object is required"),
    ):
        res = converter.convert(source)
        expected_path = os.path.abspath("tests/data/html/sources/example_image_01.png")
        mocked_open.assert_called_once_with(expected_path, "rb")
        assert res.document


def test_fetch_remote_images_with_custom_headers():
    """Test that custom headers are passed when fetching remote images."""
    custom_headers = {"Authorization": "Bearer test-token", "X-API-Key": "test-api-key"}
    backend_options = HTMLBackendOptions(
        enable_remote_fetch=True,
        fetch_images=True,
        source_uri="http://example.com",
        headers=custom_headers,
    )
    # Verify sensitive headers are not exposed in string representation
    repr_str = repr(backend_options)
    assert (
        "test-token" not in repr_str
        and "test-api-key" not in repr_str
        and "headers=" not in repr_str
    )

    converter = _create_html_converter(backend_options)
    with patch(
        "docling.backend.image_resource_loader.requests.Session.get"
    ) as mocked_session_get:
        mocked_session_get.return_value = _create_mock_response()
        res = converter.convert("./tests/data/html/sources/example_01.html")
        headers_arg = mocked_session_get.call_args[1].get("headers", {})
        assert headers_arg["Authorization"] == "Bearer test-token"
        assert headers_arg["X-API-Key"] == "test-api-key" and "Range" in headers_arg
    assert res.document


def test_is_rich_table_cell(html_paths):
    """Test the function is_rich_table_cell."""

    name = "html_rich_table_cells.html"
    path = next(item for item in html_paths if item.name == name)

    in_doc = InputDocument(
        path_or_stream=path,
        format=InputFormat.HTML,
        backend=HTMLDocumentBackend,
        filename=name,
    )
    backend = HTMLDocumentBackend(
        in_doc=in_doc,
        path_or_stream=path,
    )

    gt_cells: dict[int, list[bool]] = {}
    # table: Basic duck facts
    gt_cells[0] = [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        True,
        True,
    ]
    # table: Duck family tree
    gt_cells[1] = [False, False, True, False, True, False, True, False]
    # table: Duck-related actions
    gt_cells[2] = [False, True, True, True, False, True, True]
    # table: nested table
    gt_cells[3] = [False, False, False, False, False, False]
    # table: Famous Ducks with Images
    gt_cells[4] = [
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        True,
        False,
        False,
        True,
        False,
        False,
        False,
    ]

    for idx_t, table in enumerate(backend.soup.find_all("table")):
        gt_it = iter(gt_cells[idx_t])
        num_cells = 0
        containers = table.find_all(["thead", "tbody"], recursive=False)
        for part in containers:
            for idx_r, row in enumerate(part.find_all("tr", recursive=False)):
                cells = row.find_all(["td", "th"], recursive=False)
                if not cells:
                    continue
                for idx_c, cell in enumerate(cells):
                    assert next(gt_it) == backend._is_rich_table_cell(cell), (
                        f"Wrong cell type in table {idx_t}, row {idx_r}, col {idx_c} "
                        f"with text: {cell.text}"
                    )
                    num_cells += 1
        assert num_cells == len(gt_cells[idx_t]), (
            f"Cell number does not match in table {idx_t}"
        )


def test_table_row_section_flag_from_tr_and_td_class():
    raw_html = b"""
    <html>
      <body>
        <table>
          <tr><th>Key</th><th>Value</th></tr>
          <tr class="row_section">
            <td>Section From TR</td>
            <td><a href="https://example.com">Rich Section From TR</a></td>
          </tr>
          <tr>
            <td class="row_section">Section From TD</td>
            <td>Normal Cell</td>
          </tr>
        </table>
      </body>
    </html>
    """

    in_doc = InputDocument(
        path_or_stream=BytesIO(raw_html),
        format=InputFormat.HTML,
        backend=HTMLDocumentBackend,
        filename="test_row_section.html",
    )
    backend = HTMLDocumentBackend(
        in_doc=in_doc,
        path_or_stream=BytesIO(raw_html),
    )
    doc: DoclingDocument = backend.convert()

    cells = doc.tables[0].data.table_cells
    cells_by_text = {cell.text: cell for cell in cells}

    assert cells_by_text["Section From TR"].row_section is True
    assert cells_by_text["Section From TD"].row_section is True
    assert cells_by_text["Normal Cell"].row_section is False

    rich_section_cell = cells_by_text["Rich Section From TR"]
    assert isinstance(rich_section_cell, RichTableCell)
    assert rich_section_cell.row_section is True


data_fix_par = [
    (
        "<p>Text<h2>Heading</h2>More text</p>",
        "<p>Text</p><h2>Heading</h2><p>More text</p>",
    ),
    (
        "<html><body><p>Some text<h2>A heading</h2>More text</p></body></html>",
        "<html><body><p>Some text</p><h2>A heading</h2><p>More text</p></body></html>",
    ),
    (
        "<p>Some text<h2>A heading</h2><i>Italics</i></p>",
        "<p>Some text</p><h2>A heading</h2><p><i>Italics</i></p>",
    ),
    (
        "<p>Some text<p>Another paragraph</p>More text</p>",
        "<p>Some text</p><p>Another paragraph</p><p>More text</p>",
    ),
    (
        "<p><table><tr><th>Name</th><th>Age</th></tr><tr><td>Alice</td><td>29</td></tr>"
        "<tr><td>Bob</td><td>34</td></tr></table></p>",
        "<table><tr><th>Name</th><th>Age</th></tr><tr><td>Alice</td><td>29</td></tr>"
        "<tr><td>Bob</td><td>34</td></tr></table>",
    ),
]


@pytest.mark.parametrize("html,expected", data_fix_par)
def test_fix_invalid_paragraph_structure(html, expected):
    """Test the function _fix_invalid_paragraph_structure."""

    soup = BeautifulSoup(html, "html.parser")
    HTMLDocumentBackend._fix_invalid_paragraph_structure(soup)
    assert str(soup) == expected


def test_e2e_inline_group_in_table_cell(html_paths):
    """Regression: InlineGroup in table cell must not cause content duplication."""
    name = "html_inline_group_in_table_cell.html"
    path = next(item for item in html_paths if item.name == name)

    converter = DocumentConverter()
    result = converter.convert(path)
    assert result.document is not None

    md = result.document.export_to_markdown()
    assert isinstance(md, str)
    assert len(md) > 0

    assert "Page A" in md
    assert "Page B" in md
    assert md.count("Page A") == 1
    assert md.count("Page B") == 1


def _build_large_rich_table_html(
    num_tables: int = 10, rows_per_table: int = 20
) -> bytes:
    """Build a synthetic HTML page with many tables whose cells have multiple hyperlinks."""
    parts = ["<html><body>"]
    for t in range(num_tables):
        parts.append(
            f"<h2>Table {t}</h2><table><thead><tr><th>Name</th><th>Links</th></tr></thead><tbody>"
        )
        for r in range(rows_per_table):
            cell_a = (
                f"<td><p>"
                f'<a href="https://example.com/{t}-{r}-0">Link {t}-{r}-0</a>, '
                f'<a href="https://example.com/{t}-{r}-1">Link {t}-{r}-1</a>, '
                f'<a href="https://example.com/{t}-{r}-2">Link {t}-{r}-2</a>'
                f"</p></td>"
            )
            cell_b = (
                f"<td><p>"
                f'<a href="https://example.com/b-{t}-{r}-0">B-Link {t}-{r}-0</a> and '
                f'<a href="https://example.com/b-{t}-{r}-1">B-Link {t}-{r}-1</a>'
                f"</p></td>"
            )
            parts.append(f"<tr>{cell_a}{cell_b}</tr>")
        parts.append("</tbody></table>")
    parts.append("</body></html>")
    return "\n".join(parts).encode()


def test_e2e_rich_table_oom_regression():
    """Regression: orphaned InlineGroups must not cause OOM on pages with many rich cells."""
    num_tables, rows_per_table = 30, 20
    html_bytes = _build_large_rich_table_html(
        num_tables=num_tables, rows_per_table=rows_per_table
    )

    in_doc = InputDocument(
        path_or_stream=BytesIO(html_bytes),
        format=InputFormat.HTML,
        backend=HTMLDocumentBackend,
        filename="rich_table_oom_test.html",
    )
    backend = HTMLDocumentBackend(
        in_doc=in_doc,
        path_or_stream=BytesIO(html_bytes),
    )
    doc: DoclingDocument = backend.convert()

    assert doc is not None, "Conversion returned None"

    result: list[str] = []

    def _run() -> None:
        result.append(doc.export_to_markdown())

    t = threading.Thread(target=_run, daemon=True)
    t0 = time.monotonic()
    t.start()
    t.join(timeout=15.0)
    elapsed = time.monotonic() - t0

    assert not t.is_alive(), (
        f"export_to_markdown() hung after {elapsed:.1f}s on rich table cells."
    )
    assert result, "export_to_markdown() produced no output"
    md = result[0]
    assert isinstance(md, str) and len(md) > 0

    max_expected_chars = num_tables * rows_per_table * 2 * 128 * 3
    assert len(md) <= max_expected_chars, (
        f"Markdown output is suspiciously large ({len(md):,} chars > {max_expected_chars:,})."
    )


def _build_nested_clade_html(depth: int) -> bytes:
    """Build nested-table HTML with one <img> per level, mirroring Wikipedia cladograms."""

    def _inner(lvl: int) -> str:
        img = f'<img src="level_{lvl}.png" width="16" height="16">'
        if lvl == depth - 1:
            return f"<table><tr><td>{img}</td></tr></table>"
        return f"<table><tr><td>{img}</td><td>{_inner(lvl + 1)}</td></tr></table>"

    return f"<html><body><h2>Cladogram</h2>{_inner(0)}</body></html>".encode()


def test_nested_table_images_no_quadratic_pictures():
    """Regression: nested tables must produce exactly one PictureItem per <img>."""
    DEPTH = 15

    html_bytes = _build_nested_clade_html(DEPTH)

    from bs4 import BeautifulSoup as _BS

    soup = _BS(html_bytes, "html.parser")
    num_img_tags = len(soup.find_all("img"))
    assert num_img_tags == DEPTH, "fixture sanity check"

    in_doc = InputDocument(
        path_or_stream=BytesIO(html_bytes),
        format=InputFormat.HTML,
        backend=HTMLDocumentBackend,
        filename="nested_clade_imgs.html",
    )
    backend = HTMLDocumentBackend(
        in_doc=in_doc,
        path_or_stream=BytesIO(html_bytes),
    )
    doc: DoclingDocument = backend.convert()

    num_pictures = sum(
        1 for item, _ in doc.iterate_items() if isinstance(item, PictureItem)
    )

    assert num_pictures == DEPTH, (
        f"Expected {DEPTH} PictureItems (one per <img>), got {num_pictures}."
    )

    t0 = time.time()
    md = doc.export_to_markdown()
    elapsed = time.time() - t0

    assert isinstance(md, str) and len(md) > 0
    assert elapsed < 5.0, f"export_to_markdown() took {elapsed:.2f}s; should be < 5s"


def test_validate_url_safety_rejects_private_ips():
    """Test that private and restricted IP addresses are rejected."""
    with pytest.raises(ValueError, match="Access to restricted IP address"):
        _validate_url_safety("http://127.0.0.1/file")

    with pytest.raises(ValueError, match="Access to restricted IP address"):
        _validate_url_safety("http://10.0.0.1/file")

    with pytest.raises(ValueError, match="Access to restricted IP address"):
        _validate_url_safety("http://192.168.1.1/file")

    with pytest.raises(ValueError, match="Access to restricted IP address"):
        _validate_url_safety("http://172.16.0.1/file")

    with pytest.raises(ValueError, match="Access to restricted IP address"):
        _validate_url_safety("http://169.254.169.254/metadata")


def test_load_image_data_enforces_size_limit(monkeypatch):
    """Test that image downloads are capped at the size limit."""

    class MockResponse:
        def __init__(self, content_size):
            self.status_code = 200
            self.headers = {"content-length": str(content_size)}
            self._content_size = content_size

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size=8192):
            remaining = self._content_size
            while remaining > 0:
                chunk_len = min(chunk_size, remaining)
                yield b"x" * chunk_len
                remaining -= chunk_len

    html_path = Path("./tests/data/html/sources/example_01.html")
    in_doc = InputDocument(
        path_or_stream=html_path,
        format=InputFormat.HTML,
        backend=HTMLDocumentBackend,
        filename="test",
    )
    backend = HTMLDocumentBackend(
        in_doc=in_doc,
        path_or_stream=html_path,
        options=HTMLBackendOptions(enable_remote_fetch=True),
    )

    oversized_response = MockResponse(25 * 1024 * 1024)  # 25 MB, exceeds 20 MB limit
    monkeypatch.setattr(
        requests.Session, "get", lambda *args, **kwargs: oversized_response
    )

    with pytest.raises(ValueError, match="Resource size exceeds limit"):
        backend._load_image_data("http://example.com/huge_image.png")


def test_load_image_data_enforces_data_uri_size_limit():
    """Test that base64 data URIs are capped at the size limit."""
    html_path = Path("./tests/data/html/sources/example_01.html")
    in_doc = InputDocument(
        path_or_stream=html_path,
        format=InputFormat.HTML,
        backend=HTMLDocumentBackend,
        filename="test",
    )
    backend = HTMLDocumentBackend(
        in_doc=in_doc,
        path_or_stream=html_path,
        options=HTMLBackendOptions(),
    )

    oversized_data = b"x" * (21 * 1024 * 1024)
    encoded = base64.b64encode(oversized_data).decode()
    data_uri = f"data:image/png;base64,{encoded}"

    with pytest.raises(ValueError, match="exceeds size limit"):
        backend._load_image_data(data_uri)


def test_anchor_fragment_links_with_source_uri():
    """Fragment-only hrefs must not be mangled when source_uri is set."""
    html_path = Path("tests/data/html/sources/hyperlink_06.html")
    in_doc = InputDocument(
        path_or_stream=html_path,
        format=InputFormat.HTML,
        backend=HTMLDocumentBackend,
        filename="test",
    )
    backend = HTMLDocumentBackend(
        in_doc=in_doc,
        path_or_stream=html_path,
        options=HTMLBackendOptions(source_uri=PurePath(str(html_path.resolve()))),
    )
    doc = backend.convert()
    md = doc.export_to_markdown()

    # Fragment links preserved
    assert "[Section 2](#section-2)" in md
    assert "[top link](#)" in md
    # External links still work (regression check)
    assert (
        "[Example](https://example.com)" in md
        or "[Example](https://example.com/)" in md
    )


def test_path_traversal_blocked_in_resolve_relative_path():
    """Test that path traversal attempts are blocked."""
    html_path = Path("./tests/data/html/sources/example_01.html")
    options = HTMLBackendOptions(enable_local_fetch=True, fetch_images=True)
    in_doc = InputDocument(
        path_or_stream=html_path,
        format=InputFormat.HTML,
        backend=HTMLDocumentBackend,
        filename="test",
    )
    html_doc = HTMLDocumentBackend(
        path_or_stream=html_path, in_doc=in_doc, options=options
    )
    html_doc.base_path = "/tmp/docs/report.html"

    # Path traversal with ../ blocked
    with pytest.raises(ValueError, match="Path traversal blocked"):
        html_doc._resolve_relative_path("../../../../../../../etc/something")

    with pytest.raises(ValueError, match="Path traversal blocked"):
        html_doc._resolve_relative_path("subdir/../../../../../../etc/something")

    # Valid relative paths work
    result = html_doc._resolve_relative_path("images/photo.png")
    assert "/tmp/docs/images/photo.png" in result
    assert "etc" not in result

    # Absolute paths blocked with local base_path
    with pytest.raises(
        ValueError, match="Absolute paths are not allowed with local base_path"
    ):
        html_doc._resolve_relative_path("/absolute/path/to/file.html")

    # file:// URIs blocked
    with pytest.raises(
        ValueError, match="Absolute paths are not allowed with local base_path"
    ):
        html_doc._resolve_relative_path("file:///etc/something")

    # Windows absolute paths blocked with local base_path (forward slashes)
    with pytest.raises(
        ValueError, match="Absolute paths are not allowed with local base_path"
    ):
        html_doc._resolve_relative_path("C:/Windows/System32/config/sam")

    with pytest.raises(
        ValueError, match="Absolute paths are not allowed with local base_path"
    ):
        html_doc._resolve_relative_path("D:/sensitive/data.txt")

    # Windows absolute paths with backslashes (native Windows separator)
    with pytest.raises(
        ValueError, match="Absolute paths are not allowed with local base_path"
    ):
        html_doc._resolve_relative_path(r"C:\Windows\System32\config\sam")

    with pytest.raises(
        ValueError, match="Absolute paths are not allowed with local base_path"
    ):
        html_doc._resolve_relative_path(r"D:\Users\Foo\Documents\something.txt")

    # Hypothetical single-letter URI schemes (c://, z://) should be rejected as URIs
    with pytest.raises(ValueError, match="Invalid base_path format"):
        html_doc.base_path = "c://example.com/path"
        html_doc._resolve_relative_path("image.png")

    # Reset base_path for remaining tests
    html_doc.base_path = "/tmp/docs/report.html"

    # Filesystem access blocked when base_path is None
    html_doc.base_path = None

    # Paths pass through unchanged for hyperlinks
    assert (
        html_doc._resolve_relative_path("../../../etc/something")
        == "../../../etc/something"
    )
    assert html_doc._resolve_relative_path("/etc/something") == "/etc/something"
    assert html_doc._resolve_relative_path("image.png") == "image.png"

    # But file access is blocked
    with pytest.raises(
        OperationNotAllowed, match="Local file access requires base_path"
    ):
        html_doc._load_image_data("../../../etc/something")

    with pytest.raises(
        OperationNotAllowed, match="Local file access requires base_path"
    ):
        html_doc._load_image_data("/etc/something")

    with pytest.raises(
        OperationNotAllowed, match="Local file access requires base_path"
    ):
        html_doc._load_image_data("image.png")


def test_valid_local_paths_still_work():
    """Test that valid paths within the base directory still work."""
    html_path = Path("./tests/data/html/sources/example_01.html").resolve()
    options = HTMLBackendOptions(enable_local_fetch=True, fetch_images=True)
    in_doc = InputDocument(
        path_or_stream=html_path,
        format=InputFormat.HTML,
        backend=HTMLDocumentBackend,
        filename="test",
    )
    html_doc = HTMLDocumentBackend(
        path_or_stream=html_path, in_doc=in_doc, options=options
    )
    html_doc.base_path = str(html_path)

    resolved = html_doc._resolve_relative_path("example_image_01.png")
    assert "tests/data/html" in resolved
    assert "example_image_01.png" in resolved


def test_html_newline_handling():
    """Test that HTML newlines are handled correctly per HTML spec.

    This test verifies:
    1. Newlines in HTML source within <p> tags are collapsed to spaces (HTML spec)
    2. Explicit <br> tags create line breaks
    3. <pre> blocks preserve newlines
    """
    converter = get_converter()

    # Paragraph newlines should be collapsed
    html_paragraph = """<!DOCTYPE html>
<html>
<body>
<p>
This document provides information about data processing that
can be performed using the application programming interface
(<a title="API">API</a>). This is a web-based service.
</p>
</body>
</html>"""

    result = converter.convert_string(html_paragraph, InputFormat.HTML)
    markdown = result.document.export_to_markdown()

    assert "data processing that can be performed" in markdown, (
        "Text should be continuous in markdown"
    )
    assert "\n\ncan be performed" not in markdown, (
        "Source newlines should not create paragraph breaks"
    )

    # Test 2: Single <br> tags should create line breaks within same paragraph
    html_single_br = """<!DOCTYPE html>
<html>
<body>
<p>foo<br>bar</p>
</body>
</html>"""

    result = converter.convert_string(html_single_br, InputFormat.HTML)
    markdown = result.document.export_to_markdown()

    # Single <br> should result in one paragraph with newline
    assert "foo\nbar" in markdown or "foo  \nbar" in markdown, (
        "Single <br> should create line break within same paragraph"
    )
    # Should NOT create separate paragraphs
    assert "\n\nbar" not in markdown, (
        "Single <br> should not create separate paragraphs"
    )

    # Test 3: Multiple consecutive <br> tags should create separate paragraphs
    html_double_br = """<!DOCTYPE html>
<html>
<body>
<p>foo<br><br>bar</p>
</body>
</html>"""

    result = converter.convert_string(html_double_br, InputFormat.HTML)
    markdown = result.document.export_to_markdown()

    # Double <br> should create separate paragraphs
    paragraphs = [p.strip() for p in markdown.split("\n\n") if p.strip()]
    assert len(paragraphs) >= 2, (
        f"Expected at least 2 paragraphs from double <br>, got {len(paragraphs)}"
    )
    assert any("foo" in p for p in paragraphs), "First paragraph should contain 'foo'"
    assert any("bar" in p for p in paragraphs), "Second paragraph should contain 'bar'"

    # <pre> blocks should preserve newlines
    html_pre = """<!DOCTYPE html>
<html>
<body>
<pre>
Line 1
Line 2
Line 3
</pre>
</body>
</html>"""

    result = converter.convert_string(html_pre, InputFormat.HTML)
    markdown = result.document.export_to_markdown()

    assert "Line 1" in markdown
    assert "Line 2" in markdown
    assert "Line 3" in markdown

    # Verify behavior applies to other tags (address, summary, td)
    html_other_tags = """<!DOCTYPE html>
<html>
<body>
<address>Street 1<br>City</address>
<details><summary>Title<br>Subtitle</summary></details>
<table>
<tr><td>Cell 1<br>Line 2</td></tr>
<tr><td>Cell A<br><br>Cell B</td></tr>
</table>
</body>
</html>"""

    result = converter.convert_string(html_other_tags, InputFormat.HTML)
    markdown = result.document.export_to_markdown()
    doc = result.document

    assert "Street 1\nCity" in markdown or "Street 1  \nCity" in markdown, (
        "Single <br> in <address> should create line break within same item"
    )

    assert "Title\nSubtitle" in markdown or "Title  \nSubtitle" in markdown, (
        "Single <br> in <summary> should create line break within same item"
    )

    table_found = False
    if doc.tables:
        cells = doc.tables[0].data.table_cells
        for cell in cells:
            if cell.text and "Cell 1" in cell.text:
                assert "\n" in cell.text, (
                    f"Single <br> in <td> should create newline in cell text. Got: {cell.text!r}"
                )
                table_found = True
                break

    assert table_found, "Should have found table with Cell 1"

    # Pre-existing sentinel characters should be cleaned up
    html_with_sentinel = f"""<!DOCTYPE html>
<html>
<body>
<p>
Text with pre-existing sentinel{_BR_SENTINEL}character should be cleaned.
</p>
</body>
</html>"""

    result = converter.convert_string(html_with_sentinel, InputFormat.HTML)
    markdown = result.document.export_to_markdown()

    assert _BR_SENTINEL not in markdown, (
        "Pre-existing sentinel characters should be cleaned up"
    )
    assert "sentinelcharacter" in markdown or "sentinel character" in markdown, (
        "Text should still be present after sentinel cleanup"
    )
