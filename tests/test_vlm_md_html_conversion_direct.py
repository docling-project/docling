from unittest.mock import MagicMock

from docling.backend.html_backend import HTMLDocumentBackend
from docling.backend.md_backend import MarkdownDocumentBackend
from docling.datamodel.base_models import (
    InputFormat,
    Page,
    PagePredictions,
    VlmPrediction,
)
from docling.pipeline.vlm_pipeline import VlmPipeline


def _make_conv_res(texts):
    conv_res = MagicMock()
    conv_res.input.file.name = "vlm_output.txt"
    conv_res.pages = []

    for idx, text in enumerate(texts, start=1):
        page = Page(page_no=idx)
        page.predictions = PagePredictions(
            vlm_response=VlmPrediction(text=text)
        )
        conv_res.pages.append(page)

    return conv_res


def test_vlm_markdown_pages_convert_and_concatenate():
    pipeline = VlmPipeline.__new__(VlmPipeline)

    conv_res = _make_conv_res([
        "# Page One\n\n- item A\n- item B",
        "# Page Two\n\nThis is another page.",
    ])

    doc = pipeline._convert_text_with_backend(
        conv_res,
        InputFormat.MD,
        MarkdownDocumentBackend,
    )

    md = doc.export_to_markdown()
    assert "Page One" in md
    assert "Page Two" in md


def test_vlm_html_pages_convert_and_concatenate():
    pipeline = VlmPipeline.__new__(VlmPipeline)

    conv_res = _make_conv_res([
        "<html><body><h1>Page One</h1><p>Hello</p></body></html>",
        "<html><body><h1>Page Two</h1><p>World</p></body></html>",
    ])

    doc = pipeline._convert_text_with_backend(
        conv_res,
        InputFormat.HTML,
        HTMLDocumentBackend,
    )

    md = doc.export_to_markdown()
    assert "Page One" in md
    assert "Page Two" in md


def test_vlm_html_table_pages_convert_and_concatenate():
    pipeline = VlmPipeline.__new__(VlmPipeline)

    conv_res = _make_conv_res([
        """
        <html><body>
        <h1>Page One</h1>
        <table>
          <tr><th>Name</th><th>Link</th></tr>
          <tr><td>Page A</td><td><a href="https://example.com/a">A Link</a></td></tr>
        </table>
        </body></html>
        """,
        """
        <html><body>
        <h1>Page Two</h1>
        <table>
          <tr><th>Name</th><th>Link</th></tr>
          <tr><td>Page B</td><td><a href="https://example.com/b">B Link</a></td></tr>
        </table>
        </body></html>
        """,
    ])

    doc = pipeline._convert_text_with_backend(
        conv_res,
        InputFormat.HTML,
        HTMLDocumentBackend,
    )

    md = doc.export_to_markdown()
    assert "Page One" in md
    assert "Page Two" in md
    assert "Page A" in md
    assert "Page B" in md


def test_vlm_html_nested_table_pages_convert_and_concatenate():
    pipeline = VlmPipeline.__new__(VlmPipeline)

    conv_res = _make_conv_res([
        """
        <html><body>
        <h1>Page One</h1>
        <table>
          <tr>
            <td>Outer A</td>
            <td>
              <table>
                <tr><td>Nested A1</td></tr>
              </table>
            </td>
          </tr>
        </table>
        </body></html>
        """,
        """
        <html><body>
        <h1>Page Two</h1>
        <table>
          <tr>
            <td>Outer B</td>
            <td>
              <table>
                <tr><td>Nested B1</td></tr>
              </table>
            </td>
          </tr>
        </table>
        </body></html>
        """,
    ])

    doc = pipeline._convert_text_with_backend(
        conv_res,
        InputFormat.HTML,
        HTMLDocumentBackend,
    )

    md = doc.export_to_markdown()
    assert "Page One" in md
    assert "Page Two" in md
    assert "Nested A1" in md
    assert "Nested B1" in md


def test_vlm_html_rich_table_cells_pages_convert_and_concatenate():
    pipeline = VlmPipeline.__new__(VlmPipeline)

    conv_res = _make_conv_res([
        """
        <html><body>
        <h1>Page One</h1>
        <table>
          <tr><th>Name</th><th>Details</th></tr>
          <tr>
            <td>Item A</td>
            <td>
              <p><a href="https://example.com/a1">Link A1</a></p>
              <p><a href="https://example.com/a2">Link A2</a></p>
            </td>
          </tr>
        </table>
        </body></html>
        """,
        """
        <html><body>
        <h1>Page Two</h1>
        <table>
          <tr><th>Name</th><th>Details</th></tr>
          <tr>
            <td>Item B</td>
            <td>
              <p><a href="https://example.com/b1">Link B1</a></p>
              <p><a href="https://example.com/b2">Link B2</a></p>
            </td>
          </tr>
        </table>
        </body></html>
        """,
    ])

    doc = pipeline._convert_text_with_backend(
        conv_res,
        InputFormat.HTML,
        HTMLDocumentBackend,
    )

    md = doc.export_to_markdown()
    assert "Page One" in md
    assert "Page Two" in md
    assert "Link A1" in md
    assert "Link B1" in md


def test_vlm_markdown_nested_list_pages_convert_and_concatenate():
    pipeline = VlmPipeline.__new__(VlmPipeline)

    conv_res = _make_conv_res([
        "- item 1\n- item 2\n  - sub item 1\n  - sub item 2",
        "- item 3\n- item 4\n  - sub item 3\n  - sub item 4",
    ])

    doc = pipeline._convert_text_with_backend(
        conv_res,
        InputFormat.MD,
        MarkdownDocumentBackend,
    )

    md = doc.export_to_markdown()
    assert "item 1" in md
    assert "sub item 1" in md
    assert "item 3" in md
    assert "sub item 3" in md
