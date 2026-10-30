# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""LightOnOCR-3 preset and grounded response tests, without model downloads.

The model replies below are written by hand from the upstream model card, README
and grounding code. They are not captured model output.
"""

from pathlib import Path, PurePath
from types import SimpleNamespace

import pytest
from docling_core.types.doc import (
    CodeItem,
    ContentLayer,
    DocItemLabel,
    FormulaItem,
    PictureClassificationLabel,
    SectionHeaderItem,
    Size,
    TitleItem,
)
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


@pytest.mark.parametrize(
    "size, suffix, revision",
    [
        ("0.8B", "08b", "36d8636925327c671bd8e479c517a004f7566bd9"),
        ("4B", "4b", "b71010f095dc3735de7ee6b9969bff8f8f50b39a"),
    ],
)
def test_preset_runtime_configuration(size: str, suffix: str, revision: str):
    options = VlmConvertOptions.from_preset(f"lightonocr3_{suffix}")
    spec = options.model_spec
    config = spec.get_engine_config(VlmEngineType.TRANSFORMERS)
    runtime = spec.get_runtime_input_extra_config(VlmEngineType.TRANSFORMERS)
    assert config.repo_id == f"lightonai/LightOnOCR-3-{size}"
    assert config.revision == revision
    assert config.min_engine_version == "5.5.4"
    assert not spec.trust_remote_code
    assert spec.temperature == 0.1
    assert spec.max_new_tokens == 12288
    vllm_config = spec.get_engine_config(VlmEngineType.VLLM)
    assert vllm_config.revision == revision
    assert vllm_config.min_engine_version == "0.30.0"
    assert config.torch_dtype == "bfloat16"
    assert (
        config.extra_config["transformers_model_type"]
        == TransformersModelType.AUTOMODEL_IMAGETEXTTOTEXT
    )
    assert runtime["transformers_prompt_style"] == TransformersPromptStyle.CHAT
    assert runtime["skip_special_tokens"] is True
    assert runtime["top_k"] == 0
    assert "extra_generation_config" not in runtime
    assert spec.prompt == "grounding"
    assert spec.response_format == ResponseFormat.LIGHTONOCR_MARKDOWN
    assert options.scale == 2.0
    assert options.max_size == 2048
    for engine in (VlmEngineType.API, VlmEngineType.API_OPENAI):
        assert spec.get_api_params(engine) == {
            "model": config.repo_id,
            "max_tokens": 12288,
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


def test_inline_markers():
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
    assert len(doc.tables) == 1
    assert doc.tables[0].data.grid[1][0].text == "1000"
    assert len(doc.pictures) == 2
    description = doc.pictures[0].meta.description
    assert description.text == 'A building with a sign saying "Office".'
    assert description.created_by == "lightonocr"
    chart = doc.pictures[1]
    assert chart.label == DocItemLabel.PICTURE
    assert chart.prov
    assert chart.meta.description is None
    assert [p.class_name for p in chart.meta.classification.predictions] == [
        PictureClassificationLabel.OTHER_CHART.value
    ]
    assert chart.meta.tabular_chart.chart_data.grid[1][1].text == "1000"
    assert doc.texts[-1].label == DocItemLabel.CAPTION
    assert chart.captions == [doc.texts[-1].get_ref()]
    assert not doc.pictures[0].captions


def test_chart_without_table_keeps_its_description():
    doc = parse_lightonocr_markdown(
        "![chart](0,0,1000,500)\nRevenue grows each year.", PAGE_SIZE, page_no=1
    )
    chart = doc.pictures[0]
    assert chart.meta.description.text == "Revenue grows each year."
    assert chart.meta.tabular_chart is None
    assert [p.class_name for p in chart.meta.classification.predictions] == [
        PictureClassificationLabel.OTHER_CHART.value
    ]


def test_documented_block_sequence():
    content = (
        "![image](309,90,691,516) Diagram of the model.\n"
        "![caption](400,555,591,566) Figure 1: The model.\n"
        "![title](113,608,263,623) ## 2.2 Overview\n"
        "![aside_text](10,640,50,698) Draft\n"
        "![text](112,640,882,698) The first version set the design.\n"
        "![page_number](491,935,505,950) 3\n"
    )
    doc = parse_lightonocr_markdown(content, PAGE_SIZE, page_no=1)
    assert [(item.label, item.text) for item in doc.texts] == [
        (DocItemLabel.CAPTION, "Figure 1: The model."),
        (DocItemLabel.SECTION_HEADER, "2.2 Overview"),
        (DocItemLabel.TEXT, "Draft"),
        (DocItemLabel.TEXT, "The first version set the design."),
        (DocItemLabel.PAGE_FOOTER, "3"),
    ]
    assert all(item.prov for item in doc.texts)
    assert doc.texts[-1].content_layer == ContentLayer.FURNITURE
    assert doc.pictures[0].captions == [doc.texts[0].get_ref()]


def test_caption_between_two_owners_stays_unlinked():
    content = """![table](0,0,1000,250)
<table><tr><td>1</td></tr></table>
![caption](0,260,1000,300)
Which one?
![image](0,310,1000,600)
A photo.
"""
    doc = parse_lightonocr_markdown(content, PAGE_SIZE, page_no=1)
    assert not doc.tables[0].captions
    assert not doc.pictures[0].captions


@pytest.mark.parametrize("prefix", ["</think>\n\n", "<think>\n\n</think>\n\n"])
def test_leaked_think_tag_is_removed(prefix: str):
    doc = parse_lightonocr_markdown(
        f"{prefix}![text](100,200,900,300) Rent", PAGE_SIZE, page_no=1
    )
    assert [item.text for item in doc.texts] == ["Rent"]
    assert doc.texts[0].prov


def test_two_markers_on_one_line():
    doc = parse_lightonocr_markdown(
        "![text](0,0,500,100) Left ![text+](500,0,1000,100) Right",
        PAGE_SIZE,
        page_no=1,
    )
    assert [item.text for item in doc.texts] == ["Left", "Right"]
    assert doc.texts[1].prov[0].bbox.l == 300


@pytest.mark.parametrize("prefix", ["", "![text](0,0,1000,100) "])
@pytest.mark.parametrize("quote", ["`", "``"])
def test_marker_in_inline_code_stays_code(prefix: str, quote: str):
    code = "![image](0,0,1000,1000)"
    doc = parse_lightonocr_markdown(
        f"{prefix}The syntax is {quote}{code}{quote}.", PAGE_SIZE, page_no=1
    )
    assert [item.text for item in doc.texts] == ["The syntax is", code, "."]
    assert isinstance(doc.texts[1], CodeItem)
    assert not doc.pictures


def test_marker_after_inline_code_starts_a_block():
    doc = parse_lightonocr_markdown(
        "![text](0,0,500,100) Use `![image](0,0,1,1)` ![text](500,0,1000,100) Next",
        PAGE_SIZE,
        page_no=1,
    )
    assert [item.text for item in doc.texts] == ["Use", "![image](0,0,1,1)", "Next"]
    assert doc.texts[-1].prov[0].bbox.l == 300
    assert not doc.pictures


def test_cut_off_marker_stays_text():
    text = "Cut off here ![text](100,200"
    doc = parse_lightonocr_markdown(
        f"![text](0,0,1000,100) {text}", PAGE_SIZE, page_no=1
    )
    assert [item.text for item in doc.texts] == [text]
    assert not doc.pictures


@pytest.mark.parametrize("image", ["![logo](https://example.com/a.png)", "![a](1,2,3)"])
def test_markdown_image_is_not_a_marker(image: str):
    doc = parse_lightonocr_markdown(
        f"![text](0,0,1000,100) See {image} here", PAGE_SIZE, page_no=1
    )
    assert doc.texts[0].text == "See"
    assert [item for item in doc.texts if item.prov] == [doc.texts[0]]


def test_unknown_label_keeps_its_text():
    doc = parse_lightonocr_markdown(
        "![stamp](0,0,1000,100) Approved", PAGE_SIZE, page_no=1
    )
    assert [(item.label, item.text) for item in doc.texts] == [
        (DocItemLabel.TEXT, "Approved")
    ]
    assert doc.texts[0].prov


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


@pytest.mark.parametrize("bbox", ["24,1000,238,1000", "0,0,0,0", "5,5,5,50"])
def test_zero_area_picture_has_no_box_and_can_be_saved(bbox: str, tmp_path: Path):
    image = Image.new("RGB", (600, 800), "white")
    doc = parse_lightonocr_markdown(
        f"![image]({bbox})\nA small logo.", PAGE_SIZE, page_no=1, page_image=image
    )
    assert doc.pictures[0].meta.description.text == "A small logo."
    assert not doc.pictures[0].prov
    doc.save_as_doclang_archive(tmp_path / "doc.dclx")


@pytest.mark.parametrize("bbox", ["0,900,500,1200", "100,300,50,400", "900,10,100,20"])
def test_out_of_grid_and_inverted_boxes_keep_text(bbox: str):
    doc = parse_lightonocr_markdown(f"![text]({bbox}) Kept text", PAGE_SIZE, 1)
    assert [item.text for item in doc.texts] == ["Kept text"]
    assert not doc.texts[0].prov


def test_invalid_boxes_warn_once_per_page(caplog: pytest.LogCaptureFixture):
    content = """![header](1,2,24,1)
Top

![text](0,900,500,1200)
One

![text](0,0,1000,100)
Valid
"""
    doc = parse_lightonocr_markdown(content, PAGE_SIZE, page_no=3)
    assert [t.label for t in doc.texts] == [
        DocItemLabel.PAGE_HEADER,
        DocItemLabel.TEXT,
        DocItemLabel.TEXT,
    ]
    assert [bool(t.prov) for t in doc.texts] == [False, False, True]
    messages = [r.getMessage() for r in caplog.records]
    assert len(messages) == 1
    assert "page 3 has 2 block(s)" in messages[0]
    assert "Chandra" not in messages[0]


def test_table_in_image_block_becomes_a_table():
    content = """![image](100,200,900,500)
Skills
<table><tr><th>Skill</th><th>Level</th></tr><tr><td>Layout</td><td>90</td></tr></table>
Source: self-rated
"""
    doc = parse_lightonocr_markdown(content, PAGE_SIZE, page_no=1)
    assert not doc.pictures
    assert len(doc.tables) == 1
    assert doc.tables[0].data.grid[1][0].text == "Layout"
    assert doc.tables[0].prov[0].bbox.l == 60
    assert "Skills" in [t.text for t in doc.texts]
    assert "Source: self-rated" in [t.text for t in doc.texts]
    assert not any("<table" in t.text for t in doc.texts)


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


@pytest.mark.parametrize(
    "heading, level",
    [("## 2.2 LightOnOCR-1", 2), ("### Subsection", 3), ("Section\n-------", 2)],
)
def test_grounded_section_heading_preserves_level(heading: str, level: int):
    doc = parse_lightonocr_markdown(
        f"![title](100,100,900,200) {heading}", PAGE_SIZE, page_no=3
    )
    item = doc.texts[0]
    assert isinstance(item, SectionHeaderItem)
    assert item.level == level - 1
    assert item.prov[0].page_no == 3
    assert item.prov[0].bbox.t == 80
    assert doc.export_to_markdown() == f"{'#' * level} {item.text}"


@pytest.mark.parametrize(
    "body, latex",
    [
        ("$$\nx^2 + y^2 = z^2\n$$", "x^2 + y^2 = z^2"),
        (r"\[\left\{x : x > 0\right\}\]", r"\left\{x : x > 0\right\}"),
        (r"\(x_1 * y_2 * z_3\)", r"x_1 * y_2 * z_3"),
        (r"$x_1$", r"x_1"),
        (r"\frac{a}{b}", r"\frac{a}{b}"),
        (
            "$$\n\\begin{aligned}\na &= b \\\\\nc &= d\n\\end{aligned}\n$$",
            "\\begin{aligned}\na &= b \\\\\nc &= d\n\\end{aligned}",
        ),
    ],
)
def test_grounded_formula_preserves_latex(body: str, latex: str):
    doc = parse_lightonocr_markdown(
        f"![formula](100,100,900,300)\n{body}", PAGE_SIZE, page_no=2
    )
    assert len(doc.texts) == 1
    formula = doc.texts[0]
    assert isinstance(formula, FormulaItem)
    assert formula.text == latex
    assert formula.prov[0].page_no == 2
    assert formula.prov[0].bbox.l == 60
    assert doc.export_to_markdown() == f"$${latex}$$"


@pytest.mark.parametrize("marker", ["", "![text](100,100,900,300)\n"])
def test_inline_math_preserves_latex_and_code(marker: str):
    latex = r"\left\{x_1 * y_2 * z_3 < 2\right\}"
    source = marker + f"**Value** ${latex}$ and `literal $x_1$`. Cost: \\$5."
    doc = parse_lightonocr_markdown(source, PAGE_SIZE, page_no=1)
    formulas = [item for item in doc.texts if isinstance(item, FormulaItem)]
    assert [formula.text for formula in formulas] == [latex]
    assert all(not formula.prov for formula in formulas)
    assert [item.text for item in doc.texts if isinstance(item, CodeItem)] == [
        "literal $x_1$"
    ]
    exported = doc.export_to_markdown()
    assert f"${latex}$" in exported
    assert "`literal $x_1$`" in exported
    assert "**Value**" in exported
    assert "$5" in exported


@pytest.mark.parametrize("fence", ["```", "~~~~"])
@pytest.mark.parametrize("code", ["print(1)", "![text](0,0,1000,1000)"])
def test_inline_code_fence_preserves_following_block(fence: str, code: str):
    source = (
        f"![code](0,0,1000,500) {fence}python\n{code}\n{fence}\n"
        "![text](0,500,1000,600) After code.\n"
    )
    doc = parse_lightonocr_markdown(source, PAGE_SIZE, page_no=1)
    assert [(item.label, item.text) for item in doc.texts] == [
        (DocItemLabel.CODE, code),
        (DocItemLabel.TEXT, "After code."),
    ]
    assert not doc.pictures
    assert doc.texts[0].prov[0].bbox.b == 400
    assert doc.texts[1].prov[0].bbox.t == 400
    assert doc.export_to_markdown() == f"```\n{code}\n```\n\nAfter code."


@pytest.mark.parametrize("heading", ["Contract", "# Contract"])
def test_title_without_section_markup_is_the_title(heading: str):
    doc = parse_lightonocr_markdown(
        f"![title](100,100,900,200) {heading}", PAGE_SIZE, page_no=1
    )
    assert isinstance(doc.texts[0], TitleItem)
    assert doc.texts[0].text == "Contract"
    assert doc.export_to_markdown() == "# Contract"


@pytest.mark.parametrize(
    "heading, item_type, export",
    [
        ("## Theorem $x_1$", SectionHeaderItem, "## Theorem $x_1$"),
        ("# Theorem $x_1$", TitleItem, "# Theorem $x_1$"),
    ],
)
def test_heading_with_math_keeps_its_text(heading: str, item_type: type, export: str):
    doc = parse_lightonocr_markdown(
        f"![title](100,100,900,200) {heading}", PAGE_SIZE, page_no=1
    )
    # Like the other mappers, the heading holds rich text in child items.
    assert isinstance(doc.texts[0], item_type)
    assert doc.texts[0].prov
    assert [item.text for item in doc.texts[1:]] == ["Theorem", "x_1"]
    assert isinstance(doc.texts[-1], FormulaItem)
    assert doc.export_to_markdown() == export


def test_formatted_section_heading_owns_its_coordinates_once():
    doc = parse_lightonocr_markdown(
        "![title](100,100,900,200) ## **Rent** and terms", PAGE_SIZE, page_no=1
    )
    assert isinstance(doc.texts[0], SectionHeaderItem)
    assert doc.texts[0].level == 1
    assert doc.texts[0].prov
    assert all(not item.prov for item in doc.texts[1:])
    assert doc.export_to_markdown() == "## **Rent** and terms"
