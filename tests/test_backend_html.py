from io import BytesIO
from pathlib import Path, PurePath
from unittest.mock import Mock, mock_open, patch

import pytest
from bs4 import BeautifulSoup
from docling_core.types.doc import PictureItem
from docling_core.types.doc.document import ContentLayer
from pydantic import AnyUrl, ValidationError

from docling.backend.html_backend import HTMLDocumentBackend
from docling.datamodel.backend_options import HTMLBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import (
    ConversionResult,
    DoclingDocument,
    InputDocument,
    SectionHeaderItem,
)
from docling.document_converter import DocumentConverter, HTMLFormatOption

from .test_data_gen_flag import GEN_TEST_DATA
from .verify_utils import verify_document, verify_export

GENERATE = GEN_TEST_DATA


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
    html_path = Path("./tests/data/html/example_01.html")
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
    assert html_doc._resolve_relative_path(absolute_path) == absolute_path

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

    html_doc.base_path = None
    relative_path = "subdir/file.html"
    assert html_doc._resolve_relative_path(relative_path) == relative_path


def test_heading_levels():
    in_path = Path("tests/data/html/wiki_duck.html")
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
    html_path = Path("./tests/data/html/hyperlink_04.html")
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
    directory = Path("./tests/data/html/")

    # List all HTML files in the directory and its subdirectories
    html_files = sorted(directory.rglob("*.html"))

    return html_files


def get_converter():
    converter = DocumentConverter(allowed_formats=[InputFormat.HTML])

    return converter


def test_e2e_html_conversions(html_paths):
    converter = get_converter()

    for html_path in html_paths:
        gt_path = (
            html_path.parent.parent / "groundtruth" / "docling_v2" / html_path.name
        )

        conv_result: ConversionResult = converter.convert(html_path)

        doc: DoclingDocument = conv_result.document

        pred_md: str = doc.export_to_markdown()
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


@patch("docling.backend.html_backend.requests.get")
@patch("docling.backend.html_backend.open", new_callable=mock_open)
def test_e2e_html_conversion_with_images(mock_local, mock_remote):
    source = "tests/data/html/example_01.html"
    image_path = "tests/data/html/example_image_01.png"
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

    # fetching image remotely
    mock_resp = Mock()
    mock_resp.status_code = 200
    mock_resp.content = img_bytes
    mock_remote.return_value = mock_resp
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
    mock_remote.assert_called_once_with(
        "https://example.com/example_image_01.png", stream=True
    )
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
    gt_path = (
        "tests/data/groundtruth/docling_v2/" + str(Path(source).stem) + "_images.html"
    )
    pred_md: str = res_local.document.export_to_markdown()
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
    source = "./tests/data/html/example_01.html"

    # no image fetching: the image_fetch flag is False
    backend_options = HTMLBackendOptions(
        fetch_images=False, source_uri="http://example.com"
    )
    converter = DocumentConverter(
        allowed_formats=[InputFormat.HTML],
        format_options={
            InputFormat.HTML: HTMLFormatOption(backend_options=backend_options)
        },
    )
    with patch("docling.backend.html_backend.requests.get") as mocked_get:
        res = converter.convert(source)
        mocked_get.assert_not_called()
    assert res.document

    # no image fetching: the source location is False and enable_local_fetch is False
    backend_options = HTMLBackendOptions(fetch_images=True)
    converter = DocumentConverter(
        allowed_formats=[InputFormat.HTML],
        format_options={
            InputFormat.HTML: HTMLFormatOption(backend_options=backend_options)
        },
    )
    with (
        patch("docling.backend.html_backend.requests.get") as mocked_get,
        pytest.warns(
            match="Fetching local resources is only allowed when set explicitly"
        ),
    ):
        res = converter.convert(source)
        mocked_get.assert_not_called()
    assert res.document

    # no image fetching: the enable_remote_fetch is False
    backend_options = HTMLBackendOptions(
        fetch_images=True, source_uri="http://example.com"
    )
    converter = DocumentConverter(
        allowed_formats=[InputFormat.HTML],
        format_options={
            InputFormat.HTML: HTMLFormatOption(backend_options=backend_options)
        },
    )
    with (
        patch("docling.backend.html_backend.requests.get") as mocked_get,
        pytest.warns(
            match="Fetching remote resources is only allowed when set explicitly"
        ),
    ):
        res = converter.convert(source)
        mocked_get.assert_not_called()
    assert res.document

    # image fetching: all conditions apply, source location is remote
    backend_options = HTMLBackendOptions(
        enable_remote_fetch=True, fetch_images=True, source_uri="http://example.com"
    )
    converter = DocumentConverter(
        allowed_formats=[InputFormat.HTML],
        format_options={
            InputFormat.HTML: HTMLFormatOption(backend_options=backend_options)
        },
    )
    with (
        patch("docling.backend.html_backend.requests.get") as mocked_get,
        pytest.warns(match="a bytes-like object is required"),
    ):
        res = converter.convert(source)
        mocked_get.assert_called_once()
    assert res.document

    # image fetching: all conditions apply, local fetching allowed
    backend_options = HTMLBackendOptions(
        enable_local_fetch=True, fetch_images=True, source_uri=source
    )
    converter = DocumentConverter(
        allowed_formats=[InputFormat.HTML],
        format_options={
            InputFormat.HTML: HTMLFormatOption(backend_options=backend_options)
        },
    )
    with (
        patch("docling.backend.html_backend.open") as mocked_open,
        pytest.warns(match="a bytes-like object is required"),
    ):
        res = converter.convert(source)
        mocked_open.assert_called_once_with(
            "tests/data/html/example_image_01.png", "rb"
        )
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
