# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Recording a tagged PDF's MathML on the formulas the layout model detected."""

from pathlib import Path

import pypdfium2 as pdfium
import pytest
from docling_core.types.doc import (
    BoundingBox,
    CoordOrigin,
    DoclingDocument,
    ProvenanceItem,
    Size,
)
from docling_core.types.doc.common.meta import FormulaMeta, FormulaMetaField

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.pipeline_options import (
    CodeFormulaVlmOptions,
    PdfPipelineOptions,
)
from docling.models.stages.code_formula.code_formula_model import (
    CodeFormulaModel,
    CodeFormulaModelOptions,
)
from docling.models.stages.code_formula.code_formula_vlm_model import (
    CodeFormulaVlmModel,
)
from docling.models.stages.native_formula.native_formula_model import (
    NATIVE_FORMULA_SOURCE,
    NativeFormulaModel,
)
from docling.utils.formula_meta import has_native_mathml
from docling.utils.pdf_struct_tree import (
    _PdfFormulaStruct,
    extract_formula_structs_from_pdfium,
)

TAGGED = Path("tests/data/pdf/struct_tree/formula_mathml_tagged.pdf")

PAGE_HEIGHT = 200.0
MATHML = "<math><mfrac><mi>x</mi><mi>y</mi></mfrac></math>"


def _doc_with_formula(bbox: BoundingBox, text: str = "", orig: str = "x/y"):
    """A one-page document holding a single formula at *bbox* (bottom-left origin)."""
    doc = DoclingDocument(name="test")
    doc.add_page(page_no=1, size=Size(width=200.0, height=PAGE_HEIGHT))
    item = doc.add_formula(
        text=text,
        orig=orig,
        prov=ProvenanceItem(page_no=1, charspan=(0, 0), bbox=bbox),
    )
    return doc, item


def _struct(bbox_top_left: BoundingBox, **kwargs) -> _PdfFormulaStruct:
    payload = {"mathml": MATHML, "page_no": 1, "bbox": bbox_top_left}
    payload.update(kwargs)
    return _PdfFormulaStruct(**payload)


def _bl(left, bottom, right, top) -> BoundingBox:
    return BoundingBox(
        l=left, b=bottom, r=right, t=top, coord_origin=CoordOrigin.BOTTOMLEFT
    )


def _tl(left, top, right, bottom) -> BoundingBox:
    return BoundingBox(
        l=left, t=top, r=right, b=bottom, coord_origin=CoordOrigin.TOPLEFT
    )


def _apply(doc, structs) -> DoclingDocument:
    return NativeFormulaModel(enabled=True).apply_native_formulas(doc, structs)


def test_matching_formula_gets_the_mathml():
    doc, item = _doc_with_formula(_bl(15, 140, 85, 170))

    _apply(doc, [_struct(_tl(15, 30, 85, 60))])

    assert item.meta is not None and item.meta.formula is not None
    assert item.meta.formula.mathml == MATHML
    assert item.meta.formula.created_by == NATIVE_FORMULA_SOURCE


def test_padded_structure_bbox_still_matches():
    """A /BBox attribute is often looser than the detected box; IoU alone would miss it."""
    doc, item = _doc_with_formula(_bl(40, 145, 60, 160))

    _apply(doc, [_struct(_tl(10, 20, 90, 70))])

    assert has_native_mathml(item)


def test_unrelated_region_does_not_match():
    doc, item = _doc_with_formula(_bl(15, 140, 85, 170))

    _apply(doc, [_struct(_tl(120, 150, 190, 190))])

    assert item.meta is None


def test_each_structure_element_is_used_at_most_once():
    """Two formulas, two elements: the pairing must be one-to-one, best match first."""
    doc = DoclingDocument(name="test")
    doc.add_page(page_no=1, size=Size(width=200.0, height=PAGE_HEIGHT))
    top = doc.add_formula(
        text="",
        orig="x/y",
        prov=ProvenanceItem(page_no=1, charspan=(0, 0), bbox=_bl(15, 140, 85, 170)),
    )
    bottom = doc.add_formula(
        text="",
        orig="a+b",
        prov=ProvenanceItem(page_no=1, charspan=(0, 0), bbox=_bl(15, 90, 85, 120)),
    )

    _apply(
        doc,
        [
            _struct(_tl(15, 30, 85, 60), mathml="<math><mi>top</mi></math>"),
            _struct(_tl(15, 80, 85, 110), mathml="<math><mi>bottom</mi></math>"),
        ],
    )

    assert top.meta.formula.mathml == "<math><mi>top</mi></math>"
    assert bottom.meta.formula.mathml == "<math><mi>bottom</mi></math>"


def test_records_without_mathml_are_ignored():
    doc, item = _doc_with_formula(_bl(15, 140, 85, 170))

    _apply(doc, [_struct(_tl(15, 30, 85, 60), mathml=None, alt_text="x over y")])

    assert item.meta is None
    assert item.text == ""


def test_records_without_a_bbox_are_ignored():
    doc, item = _doc_with_formula(_bl(15, 140, 85, 170))

    _apply(doc, [_struct(None)])

    assert item.meta is None


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"actual_text": "a plus b", "alt_text": "alt"}, "a plus b"),
        ({"alt_text": "x over y"}, "x over y"),
        ({}, "x/y"),  # falls back to the item's own orig
    ],
)
def test_empty_text_is_filled_from_the_structure_element(kwargs, expected):
    """The enrichment model is skipped, so text has to come from somewhere."""
    doc, item = _doc_with_formula(_bl(15, 140, 85, 170))

    _apply(doc, [_struct(_tl(15, 30, 85, 60), **kwargs)])

    assert item.text == expected


def test_existing_text_is_not_overwritten():
    doc, item = _doc_with_formula(_bl(15, 140, 85, 170), text="\\frac{x}{y}")

    _apply(doc, [_struct(_tl(15, 30, 85, 60), alt_text="x over y")])

    assert item.text == "\\frac{x}{y}"


def test_disabled_stage_changes_nothing():
    doc, item = _doc_with_formula(_bl(15, 140, 85, 170))

    class _Res:
        document = doc
        _pdf_formula_structs = [_struct(_tl(15, 30, 85, 60))]

    NativeFormulaModel(enabled=False)(_Res())

    assert item.meta is None


def test_stage_releases_the_records_after_consuming_them():
    doc, _item = _doc_with_formula(_bl(15, 140, 85, 170))

    class _Res:
        document = doc
        _pdf_formula_structs = [_struct(_tl(15, 30, 85, 60))]

    res = _Res()
    NativeFormulaModel(enabled=True)(res)

    assert res._pdf_formula_structs is None


# --- the guard that keeps the enrichment model off a resolved formula -------------------


def test_has_native_mathml_only_fires_on_a_recorded_mathml():
    doc, item = _doc_with_formula(_bl(15, 140, 85, 170))
    assert not has_native_mathml(item)

    item.meta = FormulaMeta(formula=FormulaMetaField(latex="\\frac{x}{y}"))
    assert not has_native_mathml(item)

    item.meta = FormulaMeta(formula=FormulaMetaField(mathml=MATHML))
    assert has_native_mathml(item)


def test_legacy_enrichment_model_skips_a_resolved_formula():
    doc, item = _doc_with_formula(_bl(15, 140, 85, 170))
    model = CodeFormulaModel(
        enabled=False,
        artifacts_path=None,
        options=CodeFormulaModelOptions(),
        accelerator_options=AcceleratorOptions(),
    )
    assert not model.is_processable(doc, item)  # disabled anyway

    model.enabled = True
    assert model.is_processable(doc, item)

    item.meta = FormulaMeta(formula=FormulaMetaField(mathml=MATHML))
    assert not model.is_processable(doc, item)


def test_vlm_enrichment_model_skips_a_resolved_formula():
    doc, item = _doc_with_formula(_bl(15, 140, 85, 170))
    model = CodeFormulaVlmModel(
        enabled=False,
        enable_remote_services=False,
        artifacts_path=None,
        options=CodeFormulaVlmOptions.from_preset("codeformulav2"),
        accelerator_options=AcceleratorOptions(),
    )
    model.enabled = True
    model.options = model.options.model_copy(update={"extract_formulas": True})
    assert model.is_processable(doc, item)

    item.meta = FormulaMeta(formula=FormulaMetaField(mathml=MATHML))
    assert not model.is_processable(doc, item)


# --- the whole data path, minus the layout model -----------------------------------------


def test_mathml_read_from_a_real_pdf_reaches_the_html_export():
    """Extractor -> stage -> serializer, using the boxes the fixture actually declares."""
    pdoc = pdfium.PdfDocument(TAGGED)
    try:
        records = extract_formula_structs_from_pdfium(pdoc, [1])
    finally:
        pdoc.close()
    assert len(records) == 2

    doc = DoclingDocument(name="test")
    doc.add_page(page_no=1, size=Size(width=200.0, height=PAGE_HEIGHT))
    items = [
        doc.add_formula(
            text="",
            orig="",
            prov=ProvenanceItem(
                page_no=1,
                charspan=(0, 0),
                bbox=record.bbox.to_bottom_left_origin(page_height=PAGE_HEIGHT),
            ),
        )
        for record in records
    ]

    _apply(doc, records)

    assert all(has_native_mathml(item) for item in items)
    assert items[0].text == "x over y"  # from /Alt
    assert items[1].text == "a plus b"  # from /ActualText

    html = doc.export_to_html()
    assert "<mfrac><mi>x</mi><mi>y</mi></mfrac>" in html
    # latex2mathml would have added a TeX annotation; the authored MathML went out as-is.
    assert 'encoding="TeX"' not in html


def test_option_is_off_by_default():
    assert PdfPipelineOptions().do_native_formula_extraction is False
