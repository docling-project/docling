"""Tests for ordered/unordered list-item inference in the PDF/image pipeline.

The reading-order stage strips only *simple* leading markers; compound/hierarchical markers
(``9a.``, ``3.a.``, ``1.2.1``) are left fused in the text with an empty ``marker``. Inside an
enumerated list the Markdown serializer then prepends a position-based number, producing a
doubled marker (``7. 9a. ...``). :class:`ListNormalizationModel` recovers those markers.
"""

from pathlib import Path
from types import SimpleNamespace

from docling_core.types.doc import DoclingDocument, GroupLabel
from docling_core.types.doc.document import ListItem

from docling.datamodel.pipeline_options import ListNormalizationOptions
from docling.models.stages.list_normalization.list_normalization_model import (
    ListNormalizationModel,
)

_GT_2203 = Path(__file__).parent / "data" / "pdf" / "groundtruth" / "2203.01017v2.json"


def _list_doc(entries: list[tuple[str, str, bool]]) -> DoclingDocument:
    """Build a single-list document from (marker, text, enumerated) tuples.

    This mirrors what the reading-order stage produces: items whose marker the marker processor
    could not parse arrive with ``marker == ""`` and the marker still fused into the text.
    """
    doc = DoclingDocument(name="t")
    group = doc.add_group(label=GroupLabel.LIST, name="list")
    for marker, text, enumerated in entries:
        doc.add_list_item(text=text, enumerated=enumerated, marker=marker, parent=group)
    return doc


def _normalize(doc: DoclingDocument, **opts) -> DoclingDocument:
    model = ListNormalizationModel(ListNormalizationOptions(enabled=True, **opts))
    return model.normalize(doc)


def _items(doc: DoclingDocument) -> list[ListItem]:
    return [it for it in doc.texts if isinstance(it, ListItem)]


def test_compound_marker_recovered_in_numbered_list():
    # The headline bug: a compound "3.a." fused into an item of an otherwise-numbered list
    # renders as a doubled "4. 3.a." because the empty marker triggers position numbering.
    doc = _list_doc(
        [
            ("1.", "Get the minimal grid dimensions.", True),
            ("2.", "Generate pair-wise matches.", True),
            ("", "3.a. If all IOU scores are below the threshold, discard.", False),
            ("3.", "Use a carefully selected IOU threshold.", True),
        ]
    )
    _normalize(doc)

    recovered = _items(doc)[2]
    assert recovered.marker == "3.a."
    assert recovered.text == "If all IOU scores are below the threshold, discard."
    assert recovered.enumerated is True

    md = doc.export_to_markdown()
    # The marker is preserved once, not doubled with a spurious position number.
    assert "3.a. If all IOU scores are below the threshold, discard." in md
    assert "4. 3.a." not in md


def test_digit_letter_marker_recovered():
    doc = _list_doc(
        [
            ("9.", "Pick up the remaining orphan cells.", True),
            ("", "9a. Compute the top and bottom boundary.", False),
            ("", "9b. Intersect the orphan's bounding box.", False),
        ]
    )
    _normalize(doc)

    items = _items(doc)
    assert [it.marker for it in items[1:]] == ["9a.", "9b."]
    assert all(it.enumerated for it in items[1:])
    assert items[1].text == "Compute the top and bottom boundary."


def test_bullet_list_left_unordered():
    doc = _list_doc(
        [
            ("·", "first bullet", False),
            ("·", "second bullet", False),
            ("·", "third bullet", False),
        ]
    )
    _normalize(doc)

    assert not any(it.enumerated for it in _items(doc))
    assert "1. first bullet" not in doc.export_to_markdown()


def test_dotted_value_in_prose_not_treated_as_marker():
    # A lone dotted-decimal *value* inside a bullet list must not be mistaken for a marker:
    # there is no numbered context and only one candidate.
    doc = _list_doc(
        [
            ("·", "revenue grew", False),
            ("", "1.1 million units were shipped", False),
            ("·", "profit rose", False),
        ]
    )
    _normalize(doc)

    prose = _items(doc)[1]
    assert prose.marker == ""
    assert prose.text == "1.1 million units were shipped"
    assert prose.enumerated is False


def test_compound_only_group_recovered_by_repetition():
    # No simple numbered item, but two or more compound-ordered items establish the context.
    doc = _list_doc(
        [
            ("", "9a. Compute the top and bottom boundary.", False),
            ("", "9b. Intersect the orphan's bounding box.", False),
            ("", "9c. Compute the left and right boundary.", False),
        ]
    )
    _normalize(doc)

    items = _items(doc)
    assert [it.marker for it in items] == ["9a.", "9b.", "9c."]
    assert all(it.enumerated for it in items)


def test_disabled_is_noop():
    entries = [
        ("1.", "First.", True),
        ("", "1.1 nested item here.", False),
    ]
    doc = _list_doc(entries)
    model = ListNormalizationModel(
        ListNormalizationOptions()
    )  # enabled=False (default)
    model(SimpleNamespace(document=doc))

    fused = _items(doc)[1]
    assert fused.marker == ""
    assert fused.text == "1.1 nested item here."


def test_real_document_recovers_missed_markers():
    # Regression on real reading-order output: the arXiv table-transformer paper has an
    # algorithm list with fused "3.a."/"9a.".."9d." sub-items that render as doubled markers.
    doc = DoclingDocument.load_from_json(_GT_2203)
    before = doc.export_to_markdown()
    assert (
        "4. 3.a." in before
    )  # the bug is present in the stored (feature-off) ground truth

    _normalize(doc)
    after = doc.export_to_markdown()

    for doubled in ("4. 3.a.", "7. 9a.", "8. 9b.", "9. 9c.", "10. 9d."):
        assert doubled not in after
    for recovered in ("9a. Compute", "9b. Intersect", "9c. Compute", "9d. Intersect"):
        assert recovered in after
