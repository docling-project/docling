# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""LightOnOCR-3 preset and grounded response tests, without model downloads."""

from pathlib import PurePath
from types import SimpleNamespace

import pytest
from docling_core.types.doc import ContentLayer, DocItemLabel, Size
from PIL import Image

from docling.datamodel.base_models import Page, PagePredictions, VlmPrediction
from docling.datamodel.pipeline_options import VlmConvertOptions, VlmPipelineOptions
from docling.datamodel.pipeline_options_vlm_model import (
    ResponseFormat,
    TransformersModelType,
    TransformersPromptStyle,
)
from docling.models.inference_engines.vlm.base import VlmEngineType
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling.utils.lightonocr_utils import parse_lightonocr_markdown

PAGE_SIZE = Size(width=600, height=800)


@pytest.mark.parametrize("size, suffix", [("0.8B", "08b"), ("1B", "1b"), ("4B", "4b")])
def test_preset_runtime_configuration(size: str, suffix: str):
    options = VlmConvertOptions.from_preset(f"lightonocr3_{suffix}")
    spec = options.model_spec
    config = spec.get_engine_config(VlmEngineType.TRANSFORMERS)
    runtime = spec.get_runtime_input_extra_config(VlmEngineType.TRANSFORMERS)
    assert config.repo_id == f"lightonai/LightOnOCR-3-{size}"
    assert config.min_engine_version == "5.5.4"
    assert config.torch_dtype == "bfloat16"
    assert (
        config.extra_config["transformers_model_type"]
        == TransformersModelType.AUTOMODEL_IMAGETEXTTOTEXT
    )
    assert runtime["transformers_prompt_style"] == TransformersPromptStyle.CHAT
    assert runtime["skip_special_tokens"] is True
    assert "extra_generation_config" not in runtime
    assert spec.prompt == "grounding"
    assert spec.response_format == ResponseFormat.LIGHTONOCR_MARKDOWN
    assert options.max_size == 1540
    for engine in (VlmEngineType.API, VlmEngineType.API_OPENAI):
        assert spec.get_api_params(engine) == {
            "model": config.repo_id,
            "max_tokens": 4096,
        }
    assert not spec.has_explicit_engine_export(VlmEngineType.MLX)


def test_grounded_text_and_continuation():
    content = """![title](100,50,900,100)
# Contract

![text](100,150,900,400)
First line
and its continuation.

A second paragraph.

![text+](100,450,900,500)
Continued block.
"""
    doc = parse_lightonocr_markdown(
        content, PAGE_SIZE, page_no=3, filename="contract.pdf"
    )
    assert doc.name == "contract"
    assert [t.text for t in doc.texts] == [
        "Contract",
        "First line and its continuation.",
        "A second paragraph.",
        "Continued block.",
    ]
    assert doc.texts[0].label == DocItemLabel.TITLE
    assert doc.texts[0].prov[0].page_no == 3
    bbox = doc.texts[0].prov[0].bbox
    assert (bbox.l, bbox.t, bbox.r, bbox.b) == (60, 40, 540, 80)
    assert doc.texts[1].prov
    assert not doc.texts[2].prov
    assert doc.texts[3].prov


def test_inline_markers_from_model_prediction():
    content = (
        "![title](51,110,448,180) # LEASE AGREEMENT\n\n"
        "![text](51,297,527,542) Tenant: Example AB Monthly rent: SEK 1000 "
        "Start date: 2026-10-01"
    )
    doc = parse_lightonocr_markdown(content, PAGE_SIZE, page_no=1)
    assert not doc.pictures
    assert doc.texts[0].text == "LEASE AGREEMENT"
    assert doc.texts[0].label == DocItemLabel.TITLE
    assert doc.texts[0].prov[0].bbox.l == pytest.approx(30.6)
    assert (
        doc.texts[1].text
        == "Tenant: Example AB Monthly rent: SEK 1000 Start date: 2026-10-01"
    )
    assert doc.texts[1].prov


def test_tables_images_and_charts():
    content = """![table](0,0,1000,250)
<table><tr><th>Rent</th></tr><tr><td>1000</td></tr></table>

![image](0,300,400,600)
A building with a sign saying "Office".

![chart](500,300,1000,600)
<table><tr><th>Year</th><th>Rent</th></tr><tr><td>2026</td><td>1000</td></tr></table>

![caption](500,650,1000,700)
Annual rent.
"""
    doc = parse_lightonocr_markdown(content, PAGE_SIZE, page_no=1)
    assert len(doc.tables) == 2
    assert doc.tables[0].data.grid[1][0].text == "1000"
    assert len(doc.pictures) == 2
    description = doc.pictures[0].meta.description
    assert description.text == 'A building with a sign saying "Office".'
    assert description.created_by == "lightonocr"
    assert doc.tables[1].parent == doc.pictures[1].get_ref()
    assert doc.pictures[1].prov
    assert not doc.tables[1].prov
    assert doc.texts[-1].label == DocItemLabel.CAPTION
    assert not doc.pictures[1].captions


def test_furniture_and_invalid_coordinates():
    content = """![header](0,0,1000,100)
Lease
![page_number](400,900,600,1000)
3
![header_image](0,0,100,100)
Company logo.
![text](900,200,100,300)
Retain this despite the reversed box.
"""
    doc = parse_lightonocr_markdown(content, PAGE_SIZE, page_no=1)
    assert doc.texts[0].label == DocItemLabel.PAGE_HEADER
    assert doc.texts[1].label == DocItemLabel.PAGE_FOOTER
    assert all(t.content_layer == ContentLayer.FURNITURE for t in doc.texts[:2])
    assert doc.pictures[0].content_layer == ContentLayer.FURNITURE
    assert doc.texts[-1].text == "Retain this despite the reversed box."
    assert not doc.texts[-1].prov


def test_plain_transcription_and_code_markers():
    content = """# Heading

| Field | Value |
| --- | --- |
| Rent | 1000 |

![code](0,200,1000,500)
```markdown
![text](0,0,1000,1000)
```
"""
    doc = parse_lightonocr_markdown(content, PAGE_SIZE, page_no=1)
    assert doc.texts[0].text == "Heading"
    assert not doc.texts[0].prov
    assert len(doc.tables) == 1
    assert doc.texts[-1].label == DocItemLabel.CODE
    assert doc.texts[-1].text == "![text](0,0,1000,1000)"
    assert doc.texts[-1].prov


def test_empty_page_preserves_image():
    image = Image.new("RGB", (600, 800), "white")
    doc = parse_lightonocr_markdown("", PAGE_SIZE, page_no=2, page_image=image)
    assert doc.pages[2].image.pil_image.size == (600, 800)
    assert not (doc.texts or doc.tables or doc.pictures)


def test_lists_and_empty_image_blocks():
    content = """![list](100,100,900,300)
1. First condition
2. Second condition

![image](100,400,900,700)
"""
    doc = parse_lightonocr_markdown(content, PAGE_SIZE, page_no=1)
    assert [t.text for t in doc.texts] == ["First condition", "Second condition"]
    assert all(t.label == DocItemLabel.LIST_ITEM for t in doc.texts)
    assert all(not t.prov for t in doc.texts)
    assert len(doc.pictures) == 1
    assert doc.pictures[0].prov


def test_pipeline_dispatch_and_page_concatenation():
    pipeline = VlmPipeline.__new__(VlmPipeline)
    pipeline.pipeline_options = VlmPipelineOptions(
        vlm_options=VlmConvertOptions.from_preset("lightonocr3_08b")
    )
    pipeline.force_backend_text = False
    page = Page(
        page_no=5,
        size=PAGE_SIZE,
        predictions=PagePredictions(
            vlm_response=VlmPrediction(text="![text](100,200,900,300)\nRent: 1000")
        ),
    )
    result = SimpleNamespace(
        input=SimpleNamespace(file=PurePath("contract.pdf")), errors=[]
    )
    page_document = pipeline._finalize_page_document(result, page)
    document = pipeline._concatenate_page_documents([(5, page_document)])
    assert sorted(document.pages) == [5]
    assert document.texts[0].text == "Rent: 1000"
    assert document.texts[0].prov[0].page_no == 5
    bbox = document.texts[0].prov[0].bbox
    assert (bbox.l, bbox.t, bbox.r, bbox.b) == (60, 160, 540, 240)
    assert not result.errors
