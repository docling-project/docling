# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import json
from pathlib import PurePath
from types import SimpleNamespace

import pytest
from docling_core.types.doc import CodeItem, FormulaItem, Size

from docling.datamodel.base_models import Page, PagePredictions, VlmPrediction
from docling.datamodel.pipeline_options import VlmConvertOptions, VlmPipelineOptions
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling.utils.dots_utils import parse_dots_json
from docling.utils.nemotron_parse_utils import parse_nemotron_parse_v2


def test_markdown_vlm_preserves_math() -> None:
    pipeline = VlmPipeline.__new__(VlmPipeline)
    pipeline.pipeline_options = VlmPipelineOptions(
        vlm_options=VlmConvertOptions.from_preset("glm_ocr")
    )
    pipeline.force_backend_text = False
    source = "Before $x_i$ after.\n\n\\[y_j = x_i^2\\]"
    page = Page(
        page_no=5,
        size=Size(width=600, height=800),
        predictions=PagePredictions(vlm_response=VlmPrediction(text=source)),
    )
    result = SimpleNamespace(
        input=SimpleNamespace(file=PurePath("formulas.pdf")), errors=[]
    )
    page_document = pipeline._finalize_page_document(result, page)
    document = pipeline._concatenate_page_documents([(5, page_document)])
    formulas = [item for item in document.texts if isinstance(item, FormulaItem)]
    assert [item.text for item in formulas] == ["x_i", "y_j = x_i^2"]
    assert all(item.prov[0].page_no == 5 for item in formulas)
    assert sorted(document.pages) == [5]
    assert document.export_to_markdown() == "Before $x_i$ after.\n\n$$y_j = x_i^2$$"
    assert not result.errors


@pytest.mark.parametrize("parser", ["dots", "nemotron"])
@pytest.mark.parametrize(
    "category", ["Text", "Title", "Section-header", "List-item", "Caption"]
)
def test_inline_math_survives_export_without_changing_block_ownership(
    parser: str, category: str
) -> None:
    latex = r"x_1 * y_2 * z_3 < 2"
    text = f"Value ${latex}$ and `literal $x_1$`; cost: \\$5."
    if category == "Section-header":
        text = "### " + text
    size = Size(width=600, height=800)
    if parser == "dots":
        source = json.dumps(
            [{"category": category, "bbox": [100, 100, 500, 200], "text": text}]
        )
        doc = parse_dots_json(source, size, page_no=2)
    else:
        source = f"<x_0.2><y_0.2>{text}<x_0.8><y_0.4><class_{category}>"
        doc = parse_nemotron_parse_v2(source, size, size, page_no=2)
    formulas = [item for item in doc.texts if isinstance(item, FormulaItem)]
    assert [formula.text for formula in formulas] == [latex]
    assert all(not formula.prov for formula in formulas)
    assert len([item for item in doc.texts if item.prov]) == 1
    assert next(item for item in doc.texts if item.prov).prov[0].page_no == 2
    assert [item.text for item in doc.texts if isinstance(item, CodeItem)] == [
        "literal $x_1$"
    ]
    exported = doc.export_to_markdown()
    assert f"${latex}$" in exported
    assert "`literal $x_1$`" in exported
    assert "$5" in exported


@pytest.mark.parametrize("parser", ["dots", "nemotron"])
def test_inline_math_preserves_approximation_marks(parser: str) -> None:
    text = "about ~5 m and ~10 m, where $x$ is the length"
    size = Size(width=600, height=800)
    if parser == "dots":
        source = json.dumps(
            [{"category": "Text", "bbox": [100, 100, 500, 200], "text": text}]
        )
        doc = parse_dots_json(source, size, page_no=2)
    else:
        source = f"<x_0.2><y_0.2>{text}<x_0.8><y_0.4><class_Text>"
        doc = parse_nemotron_parse_v2(source, size, size, page_no=2)
    assert "about ~5 m and ~10 m" in doc.export_to_markdown()
    assert not any(
        item.formatting.strikethrough for item in doc.texts if item.formatting
    )
    assert [item.text for item in doc.texts if isinstance(item, FormulaItem)] == ["x"]


@pytest.mark.parametrize("parser", ["dots", "nemotron"])
@pytest.mark.parametrize(
    "price", ["$5-$10", "$5\u2013$10", "$5.00 to $10.00", "$1,000-$2,000"]
)
def test_currency_range_remains_text_next_to_math(parser: str, price: str) -> None:
    text = f"Price {price} and $x$."
    size = Size(width=600, height=800)
    if parser == "dots":
        source = json.dumps(
            [{"category": "Text", "bbox": [100, 100, 500, 200], "text": text}]
        )
        doc = parse_dots_json(source, size, page_no=2)
    else:
        source = f"<x_0.2><y_0.2>{text}<x_0.8><y_0.4><class_Text>"
        doc = parse_nemotron_parse_v2(source, size, size, page_no=2)
    assert [item.text for item in doc.texts if isinstance(item, FormulaItem)] == ["x"]
    assert f"Price {price}" in doc.export_to_markdown()
    assert len([item for item in doc.texts if item.prov]) == 1
