# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Test chandra-ocr-2 HTML parsing in VLM pipeline."""

from pathlib import Path

import pytest
from docling_core.types.doc import (
    CodeItem,
    ContentLayer,
    DocItemLabel,
    DoclingDocument,
    FormulaItem,
    GroupLabel,
    RichTableCell,
    Script,
    Size,
    TextItem,
)
from PIL import Image

from docling.utils.chandra_utils import parse_chandra_html


def get_chandra_test_paths():
    """Get all chandra HTML test files."""
    directory = Path("./tests/data/html_chandra/sources/")
    return sorted(directory.glob("*.html"))


def test_chandra_simple_parsing():
    """Test chandra HTML parsing produces expected document structure."""
    path = Path("./tests/data/html_chandra/sources/chandra_simple.html")
    content = path.read_text()
    source = path.with_suffix(".source.txt").read_text()

    doc = parse_chandra_html(
        content=content,
        original_page_size=Size(width=612, height=792),
        page_no=1,
        filename="chandra_simple.html",
    )

    assert isinstance(doc, DoclingDocument)
    assert len(doc.texts) > 0, "Should have text elements"

    labels = [
        t.label.value if hasattr(t.label, "value") else str(t.label) for t in doc.texts
    ]
    assert "section_header" in labels, "Should have section headers"
    assert "caption" in labels, "Should have caption"
    assert "page_header" in labels, "Should have page header"

    assert "tests/data/pdf/2305.03393v1-pg9.pdf, page 1" in source
    assert len(doc.tables) > 0, "Should have table elements"

    located = [*doc.texts, *doc.tables, *doc.pictures, *doc.field_regions]
    assert any(item.prov for item in located)
    for item in located:
        for prov in item.prov:
            assert prov.bbox.l >= 0 and prov.bbox.t >= 0


def test_chandra_br_spacing():
    """Test that br tags preserve spacing between text."""
    content = '<div data-bbox="0 0 1000 1000" data-label="Text">Hello<br/>World</div>'

    doc = parse_chandra_html(
        content=content,
        original_page_size=Size(width=1000, height=1000),
        page_no=1,
        filename="br.html",
    )

    assert len(doc.texts) == 1
    assert doc.texts[0].text == "Hello\nWorld"


def test_chandra_multiblock_parsing():
    """Test chandra parsing with a saved figure prediction."""
    path = Path("./tests/data/html_chandra/sources/chandra_multiblock.html")
    content = path.read_text()
    source = path.with_suffix(".source.txt").read_text()

    doc = parse_chandra_html(
        content=content,
        original_page_size=Size(width=612, height=792),
        page_no=1,
        filename="chandra_multiblock.html",
    )

    labels = [
        t.label.value if hasattr(t.label, "value") else str(t.label) for t in doc.texts
    ]
    assert "section_header" in labels, "Should have section header"
    assert "caption" in labels, "Should have caption"
    assert "page_footer" in labels, "Should have page footer"

    assert "tests/data/pdf/picture_classification.pdf, page 1" in source
    assert len(doc.pictures) > 0, "Should have picture/image elements"


def test_chandra_bbox_normalization():
    """Test that chandra bboxes (normalized 0-1000) map to page coordinates."""
    content = '<div data-bbox="0 0 1000 1000" data-label="Text"><p>full page</p></div>'

    doc = parse_chandra_html(
        content=content,
        original_page_size=Size(width=612, height=792),
        page_no=1,
        filename="test.html",
    )

    assert len(doc.texts) == 1
    bbox = doc.texts[0].prov[0].bbox
    assert abs(bbox.r - 612) < 1, f"Right edge should map to page width, got {bbox.r}"
    assert abs(bbox.b - 792) < 1, f"Bottom edge should map to page height, got {bbox.b}"


def test_chandra_empty_content():
    """Test that empty/whitespace content returns empty doc."""
    for content in ["", "   ", "\n\t"]:
        doc = parse_chandra_html(
            content=content,
            original_page_size=Size(width=612, height=792),
            page_no=1,
            filename="empty.html",
        )
        assert isinstance(doc, DoclingDocument)
        assert len(doc.texts) == 0


def test_chandra_malformed_divs():
    """Test graceful handling of divs with missing or bad attributes."""
    content = (
        '<div data-label="Text"><p>no bbox</p></div>'
        '<div data-bbox="0 0 500 500"><p>no label</p></div>'
        '<div data-bbox="bad coords" data-label="Text"><p>bad</p></div>'
        '<div data-bbox="0 0 500" data-label="Text"><p>incomplete</p></div>'
    )
    doc = parse_chandra_html(
        content=content,
        original_page_size=Size(width=612, height=792),
        page_no=1,
        filename="malformed.html",
    )
    assert isinstance(doc, DoclingDocument)
    assert [item.text for item in doc.texts] == [
        "no bbox",
        "no label",
        "bad",
        "incomplete",
    ]
    assert [bool(item.prov) for item in doc.texts] == [False, True, False, False]


def test_chandra_unknown_label_fallback():
    """Test that unknown labels fall back to TEXT."""
    content = '<div data-bbox="100 100 200 200" data-label="UnknownType"><p>fallback</p></div>'
    doc = parse_chandra_html(
        content=content,
        original_page_size=Size(width=612, height=792),
        page_no=1,
        filename="unknown.html",
    )
    assert len(doc.texts) == 1
    labels = [
        t.label.value if hasattr(t.label, "value") else str(t.label) for t in doc.texts
    ]
    assert "text" in labels


def test_chandra_table_parsing():
    """Test that Table elements use HTML table parser."""
    content = (
        '<div data-bbox="50 50 500 300" data-label="Table">'
        "<table><tr><th>Header</th></tr><tr><td>Cell</td></tr></table>"
        "</div>"
    )
    doc = parse_chandra_html(
        content=content,
        original_page_size=Size(width=612, height=792),
        page_no=1,
        filename="table.html",
    )
    assert len(doc.tables) == 1


def test_chandra_form_table_parsing():
    """Test a saved Chandra prediction containing tables labeled as Form."""
    path = Path("./tests/data/html_chandra/sources/chandra_form_table.html")
    content = path.read_text()

    doc = parse_chandra_html(
        content=content,
        original_page_size=Size(width=612, height=792),
        page_no=1,
        filename=path.name,
    )

    assert len(doc.tables) == 4
    assert all(region.prov for region in doc.field_regions)
    assert all(not table.prov for table in doc.tables)


def test_chandra_list_group_prediction_sample():
    """Test a saved chandra prediction containing list groups."""
    path = Path("./tests/data/html_chandra/sources/chandra_list_group.html")
    content = path.read_text()
    source = path.with_suffix(".source.txt").read_text()

    doc = parse_chandra_html(
        content=content,
        original_page_size=Size(width=612, height=792),
        page_no=1,
        filename=path.name,
    )

    list_items = [item for item in doc.texts if item.label == DocItemLabel.LIST_ITEM]

    assert "tests/data/pdf/multi_page.pdf, page 1" in source
    assert len(list_items) == 4
    for item, expected in zip(
        list_items, ["IBM MT/ST", "Wang Laboratories", "WordStar", "Microsoft Word"]
    ):
        text = " ".join(
            child.text
            for child, _ in doc.iterate_items(root=item)
            if isinstance(child, TextItem)
        )
        assert expected in text


def test_chandra_all_files_parse():
    """Ensure all chandra test files parse without errors."""
    for path in get_chandra_test_paths():
        content = path.read_text()
        doc = parse_chandra_html(
            content=content,
            original_page_size=Size(width=612, height=792),
            page_no=1,
            filename=path.name,
        )
        assert isinstance(doc, DoclingDocument), f"Failed to parse {path.name}"
        assert len(doc.texts) + len(doc.tables) + len(doc.pictures) > 0, (
            f"No elements parsed from {path.name}"
        )


def _parse_fragment(fragment: str, label: str = "Text") -> DoclingDocument:
    return parse_chandra_html(
        f"<div data-label='{label}' data-bbox='10.5 20 900 800'>{fragment}</div>",
        Size(width=1000, height=1000),
        page_no=3,
    )


def test_chandra_mixed_structure_and_rich_table_cells():
    doc = _parse_fragment(
        "<p>Before &lt; table</p><div><table>"
        "<tr><th rowspan='2'>Group</th><th>Value</th></tr>"
        "<tr><td><b>Bold</b><br>next <input type='checkbox' checked></td><td>Extra</td></tr>"
        "</table></div><p>Between</p><table><tr><td>Second</td></tr></table><p>After</p>",
        "Table-Of-Contents",
    )
    assert len(doc.tables) == 2
    first = doc.tables[0]
    assert (first.data.num_rows, first.data.num_cols) == (2, 3)
    assert first.data.table_cells[-1].text == "Extra"
    assert first.data.table_cells[-1].start_col_offset_idx == 2
    rich = first.data.table_cells[2]
    assert isinstance(rich, RichTableCell)
    assert rich.ref.resolve(doc).parent == first.get_ref()
    assert any(t.label == DocItemLabel.CHECKBOX_SELECTED for t in doc.texts)
    assert any(t.text == "Bold" and t.formatting.bold for t in doc.texts)
    assert [r.resolve(doc).label for r in doc.body.children] == [
        DocItemLabel.TEXT,
        DocItemLabel.TABLE,
        DocItemLabel.TEXT,
        DocItemLabel.TABLE,
        DocItemLabel.TEXT,
    ]
    assert doc.texts[0].text == "Before < table"
    assert doc.texts[-1].text == "After"
    assert doc.texts[0].prov[0].bbox.l == 10.5
    assert not first.prov
    assert all(not item.prov for item in [*doc.texts[1:], *doc.tables])
    assert doc.export_to_doclang().count("<location") == 4
    restored = DoclingDocument.model_validate_json(doc.model_dump_json())
    assert restored.validate_tree(restored.body)
    assert "Extra" in restored.export_to_doclang()


def test_chandra_nested_lists_preserve_parent_order_and_markers():
    doc = _parse_fragment(
        "<ol start='3'><li>Parent<ul><li>Child</li></ul>Tail</li><li>Next</li></ol>",
        "List-Group",
    )
    items = [t for t in doc.texts if t.label == DocItemLabel.LIST_ITEM]
    assert [t.text for t in items] == ["Parent", "Child", "Next"]
    assert items[0].marker == "3." and items[2].marker == "4."
    assert items[1].parent.resolve(doc).parent == items[0].get_ref()
    assert items[0].children[-1].resolve(doc).text == "Tail"
    assert all(not item.prov for item in items)
    assert "<location" not in doc.export_to_doclang()
    assert doc.validate_tree(doc.body)


def test_chandra_paragraph_list_and_form_controls():
    doc = _parse_fragment(
        "<p><input type='checkbox' checked>Yes</p><p><input type='radio'>No</p>",
        "List-Group",
    )
    assert len([t for t in doc.texts if t.label == DocItemLabel.LIST_ITEM]) == 2
    assert [
        t.label
        for t in doc.texts
        if t.label in {DocItemLabel.CHECKBOX_SELECTED, DocItemLabel.CHECKBOX_UNSELECTED}
    ] == [DocItemLabel.CHECKBOX_SELECTED, DocItemLabel.CHECKBOX_UNSELECTED]
    form = _parse_fragment(
        "<p>Name: <input type='text' value='Alice &amp; Bob'></p><input type='text'>",
        "Form",
    )
    assert len(form.field_regions) == 1
    values = [t for t in form.texts if t.label == DocItemLabel.FIELD_VALUE]
    assert [(v.text, v.kind) for v in values] == [
        ("Alice & Bob", "fillable"),
        ("", "fillable"),
    ]
    assert form.field_regions[0].prov
    assert all(not value.prov for value in values)


@pytest.mark.parametrize("kind", ["checkbox", "radio"])
@pytest.mark.parametrize("checked", [False, True])
@pytest.mark.parametrize("in_table", [False, True])
def test_chandra_checkbox_export_has_one_marker(kind, checked, in_table):
    control = f"<input type='{kind}' {'checked' if checked else ''} value='on'>"
    content = f"<p>{control}Choice</p><p>{control}</p>"
    if in_table:
        content = f"<table><tr><td>{content}</td></tr></table>"
    doc = _parse_fragment(content)
    label = (
        DocItemLabel.CHECKBOX_SELECTED if checked else DocItemLabel.CHECKBOX_UNSELECTED
    )
    controls = [t for t in doc.texts if t.label == label]
    assert len(controls) == 2
    assert all(t.text == "" for t in controls)
    restored = DoclingDocument.model_validate_json(doc.model_dump_json())
    markdown = restored.export_to_markdown()
    assert markdown.count("[x]" if checked else "[ ]") == 2
    assert "Choice" in markdown
    assert "☑" not in markdown and "☐" not in markdown


def test_chandra_labeled_multirun_block_emits_location_once():
    # A labeled block whose inline content splits into several runs used to
    # write the block bbox onto both the wrapper item and each inner run, so
    # doclang serialized the coordinate twice (8 <location> values).
    doc = _parse_fragment("<p><sup>1</sup> Footnote body text.</p>", "Footnote")
    footnote = next(t for t in doc.texts if t.label == DocItemLabel.FOOTNOTE)
    # The source block owns the bbox; its derived inline runs do not inherit it.
    assert footnote.prov
    assert all(not t.prov for t in doc.texts if t is not footnote)
    assert doc.export_to_doclang().count("<location") == 4


def test_chandra_inline_math_formatting_and_code_whitespace():
    doc = _parse_fragment(
        "<p>A <b>bold <i>word</i></b> x<sup>2</sup> H<sub>2</sub> "
        "<math>x &gt; 0</math> <code>a &lt; b</code> <a href='https://example.org'>link</a></p>"
        "<pre>  if x &gt; 0:\n      print(x)\n\n  done()</pre>"
        "<math display='block'>y = x^2</math><h3>Section</h3>"
    )
    assert doc.groups[0].label == GroupLabel.INLINE
    assert any(
        t.text == "word" and t.formatting.bold and t.formatting.italic
        for t in doc.texts
    )
    assert [t.formatting.script for t in doc.texts if t.text == "2"] == [
        Script.SUPER,
        Script.SUB,
    ]
    assert [t.text for t in doc.texts if isinstance(t, FormulaItem)] == [
        "x > 0",
        "y = x^2",
    ]
    assert [t.text for t in doc.texts if isinstance(t, CodeItem)] == [
        "a < b",
        "  if x > 0:\n      print(x)\n\n  done()",
    ]
    assert (
        str(next(t.hyperlink for t in doc.texts if t.text == "link"))
        == "https://example.org/"
    )
    assert doc.texts[-1].level == 3


def test_chandra_nested_math_does_not_inherit_table_bbox():
    doc = _parse_fragment(
        "<div><math display='block'>R</math></div>"
        "<table><tr><td><math>x</math></td></tr></table>",
        "Table",
    )
    table = doc.tables[0]
    formulas = [item for item in doc.texts if isinstance(item, FormulaItem)]
    rich_cell = table.data.table_cells[0]

    assert isinstance(rich_cell, RichTableCell)
    assert formulas[1].parent == rich_cell.ref
    assert table.prov
    assert all(not formula.prov for formula in formulas)
    assert doc.export_to_doclang().count("<location") == 4

    own_bbox = _parse_fragment(
        "<table><tr><td><math data-bbox='20 30 40 50'>x</math></td></tr></table>",
        "Table",
    )
    formula = next(item for item in own_bbox.texts if isinstance(item, FormulaItem))
    assert own_bbox.tables[0].prov and formula.prov
    assert own_bbox.export_to_doclang().count("<location") == 8


def test_chandra_picture_descriptions_data_captions_and_crop():
    content = (
        "<div data-label='Figure' data-bbox='100 200 600 700'>"
        "<img alt='A &amp; B chart'><p>Chart details</p>"
        "<table><caption>Data</caption><tr><td>A</td><td>2</td></tr></table>"
        "<pre>graph LR; A --&gt; B</pre>"
        "<div data-label='Caption' data-bbox='100 650 600 700'>Explicit caption</div>"
        "</div><div data-label='Caption' data-bbox='100 710 600 740'>Unlinked caption</div>"
    )
    doc = parse_chandra_html(
        content,
        Size(width=100, height=100),
        1,
        page_image=Image.new("RGB", (200, 200), "red"),
    )
    picture = doc.pictures[0]
    assert len(doc.pictures) == 1
    assert picture.meta.description.text == "A & B chart"
    assert doc.tables[0].parent == picture.get_ref()
    assert [r.resolve(doc).text for r in picture.captions] == ["Explicit caption"]
    assert [r.resolve(doc).text for r in doc.tables[0].captions] == ["Data"]
    assert any(isinstance(r.resolve(doc), CodeItem) for r in picture.children)
    assert picture.prov and not doc.tables[0].prov
    assert all(
        not item.prov
        for item in doc.texts
        if item.text in {"Chart details", "Data", "graph LR; A --> B"}
    )
    crop = picture.get_image(doc)
    assert crop.size == (100, 100) and crop.getpixel((0, 0)) == (255, 0, 0)
    assert doc.texts[-1].parent == doc.body.get_ref()
    assert doc.validate_tree(doc.body)


def test_chandra_chemical_structure_and_page_furniture():
    doc = _parse_fragment("<chem>CC(=O)O</chem>", "Chemical-Block")
    assert len(doc.pictures) == 1
    assert doc.pictures[0].meta.molecule.smi == "CC(=O)O"
    header = _parse_fragment("<p>Page <b>3</b></p>", "Page-Header")
    assert all(t.content_layer == ContentLayer.FURNITURE for t in header.texts)


def test_chandra_bare_html_recovery_warns_without_inventing_bbox(caplog):
    doc = parse_chandra_html(
        "Reasoning preamble<p>Actual text</p><math display='block'>x</math>",
        Size(width=100, height=100),
        1,
    )
    assert [t.text for t in doc.texts] == ["Actual text", "x"]
    assert all(not t.prov for t in doc.texts)
    assert "recovering HTML without coordinates" in caplog.text


@pytest.mark.parametrize(
    "content",
    ["A description of the page.", '[{"label":"Table","bbox":[0,0,100,100]}]'],
)
def test_chandra_rejects_non_transcriptions(content):
    with pytest.raises(ValueError, match="no HTML transcription"):
        parse_chandra_html(content, Size(width=100, height=100), 1)


@pytest.mark.parametrize(
    "bbox",
    ["nan 0 100 100", "0 0 inf 100", "100 0 0 100", "-1 0 100 100", "0 0 1001 1000"],
)
def test_chandra_invalid_coordinates_preserve_text_without_provenance(bbox, caplog):
    doc = parse_chandra_html(
        f'<div data-label="Text" data-bbox="{bbox}">Kept</div>',
        Size(width=100, height=100),
        1,
    )
    assert doc.texts[0].text == "Kept" and not doc.texts[0].prov
    assert "Invalid Chandra bbox" in caplog.text


def test_chandra_table_row_groups_and_optional_end_tags():
    doc = _parse_fragment(
        "<table><thead><tr><th>Heading<tr><th>Subheading</thead>"
        "<tbody><tr><th rowspan='0' scope='row'>Group<td colspan='bad'>A"
        "<tr><td>B</tbody><tbody><tr><td>C<td>D</tbody></table>"
    )
    table = doc.tables[0].data
    assert (table.num_rows, table.num_cols) == (5, 2)
    assert [cell.text for cell in table.table_cells] == [
        "Heading",
        "Subheading",
        "Group",
        "A",
        "B",
        "C",
        "D",
    ]
    assert all(cell.column_header for cell in table.table_cells[:2])
    assert table.table_cells[2].row_span == 2
    assert table.table_cells[2].row_header
    assert table.table_cells[-2].start_col_offset_idx == 0


def test_chandra_rejects_runaway_table_grids():
    with pytest.raises(ValueError, match="exceeds 1000 grid cells"):
        _parse_fragment(
            "<table><tr><td>Header</td></tr><tr>" + "<td></td>" * 501 + "</tr></table>"
        )


def test_chandra_malformed_overlapping_spans_do_not_shift_the_grid():
    doc = _parse_fragment(
        "<table><tr><td>A</td><td rowspan='2'>B</td></tr>"
        "<tr><td colspan='2'>C</td></tr></table>"
    )
    table = doc.tables[0].data
    assert table.num_cols == 2
    assert table.table_cells[-1].start_col_offset_idx == 0


def test_chandra_nested_table_remains_a_rich_cell():
    doc = _parse_fragment(
        "<table><tr><td>Outer<table><tr><td>Inner</td></tr></table></td></tr></table>"
    )
    assert len(doc.tables) == 2
    outer, inner = doc.tables
    assert (outer.data.num_rows, outer.data.num_cols) == (1, 1)
    cell = outer.data.table_cells[0]
    assert isinstance(cell, RichTableCell)
    assert inner.parent == cell.ref
    assert inner.data.table_cells[0].text == "Inner"


def test_chandra_empty_html_is_not_a_successful_transcription():
    with pytest.raises(ValueError, match="no document content"):
        _parse_fragment("<hr>")
    blank = _parse_fragment("", "Blank-Page")
    assert not blank.texts
