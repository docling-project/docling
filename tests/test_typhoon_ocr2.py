# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT
# Typhoon OCR V2 by the Typhoon team: https://huggingface.co/typhoon-ai/typhoon-ocr2-2b

"""Typhoon OCR V2 presets and layout parsing, without model downloads.

Every reply below is written by hand from the output format in the model card and
the typhoon-ocr2 reference package. They are not model output. The API tests send
their requests to a fake OpenAI-compatible server on a local socket.
"""

import base64
import hashlib
import io
import json
import logging
import re
import unicodedata
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from docling_core.types.doc import (
    ContentLayer,
    CoordOrigin,
    DocItem,
    DocItemLabel,
    DoclingDocument,
    NodeItem,
    PictureItem,
    SectionHeaderItem,
    Size,
    TextItem,
)
from PIL import Image
from pydantic import AnyUrl

from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import VlmConvertOptions, VlmPipelineOptions
from docling.datamodel.pipeline_options_vlm_model import (
    ResponseFormat,
    TransformersModelType,
    TransformersPromptStyle,
)
from docling.datamodel.vlm_engine_options import ApiVlmEngineOptions
from docling.datamodel.vlm_prompts import TYPHOON_OCR2_PARSE_PROMPT
from docling.document_converter import (
    DocumentConverter,
    ImageFormatOption,
    PdfFormatOption,
)
from docling.models.inference_engines.vlm.base import VlmEngineType
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling.utils.typhoon_ocr2_utils import parse_typhoon_ocr2_json
from tests.fakes.http_service import FakeService
from tests.fakes.openai_compatible import FakeOpenAiApi

_SIZE = Size(width=612, height=792)
_THAI = "ทำงาน หลาย ๆ ครั้ง กำไร ๑๒๓"
# The prompt block of the 2B and 0.8B model cards.
_PROMPT_SHA256 = "b55c0349b52deac3958a9f356966c99acf50fbb5be25f717f7d2c90cf4dac27a"
_PDF = Path(__file__).parent / "data" / "pdf" / "sources" / "2305.03393v1-pg9.pdf"


def _element(
    category: str, text: str, bbox: Sequence[int] = (100, 100, 900, 140)
) -> dict[str, Any]:
    return {"bbox": list(bbox), "category": category, "text": text}


def _reply(*elements: dict[str, Any]) -> str:
    """Write the elements as the model does: a JSON list in a ``json`` fence."""
    return f"```json\n{json.dumps(list(elements), ensure_ascii=False)}\n```"


def _parse(content: str, **kwargs: Any) -> DoclingDocument:
    return parse_typhoon_ocr2_json(
        content, _SIZE, page_no=3, filename="page.pdf", **kwargs
    )


def _blocks(doc: DoclingDocument) -> list[NodeItem]:
    """Return the top-level items, which the parser adds once per element."""
    return [ref.resolve(doc) for ref in doc.body.children]


def _box(item: DocItem) -> tuple[float, float, float, float]:
    bbox = item.prov[0].bbox
    return bbox.l, bbox.t, bbox.r, bbox.b


def _summary(doc: DoclingDocument) -> list[tuple[DocItemLabel, str, tuple]]:
    return [
        (item.label, item.text if isinstance(item, TextItem) else "", _box(item))
        for item in _blocks(doc)
        if isinstance(item, DocItem)
    ]


@pytest.mark.parametrize(
    ("category", "text", "label", "layer"),
    [
        ("header", "Annual Report", DocItemLabel.PAGE_HEADER, ContentLayer.FURNITURE),
        ("footer", "12", DocItemLabel.PAGE_FOOTER, ContentLayer.FURNITURE),
        ("title", "Introduction", DocItemLabel.SECTION_HEADER, ContentLayer.BODY),
        ("text", "Body text.", DocItemLabel.TEXT, ContentLayer.BODY),
        ("figure", "", DocItemLabel.PICTURE, ContentLayer.BODY),
        (
            "table",
            "<table><tr><td>1</td></tr></table>",
            DocItemLabel.TABLE,
            ContentLayer.BODY,
        ),
        ("formula", "$$x^2$$", DocItemLabel.FORMULA, ContentLayer.BODY),
        ("figure_caption", "Figure 1.", DocItemLabel.CAPTION, ContentLayer.BODY),
        ("table_caption", "Table 1.", DocItemLabel.CAPTION, ContentLayer.BODY),
        ("formula_caption", "(1)", DocItemLabel.CAPTION, ContentLayer.BODY),
        (
            "figure_footnote",
            "Source: survey.",
            DocItemLabel.FOOTNOTE,
            ContentLayer.BODY,
        ),
        ("table_footnote", "Values in THB.", DocItemLabel.FOOTNOTE, ContentLayer.BODY),
        ("page_footnote", "1 See page 4.", DocItemLabel.FOOTNOTE, ContentLayer.BODY),
        # The category is read without case and surrounding spaces.
        (" Footer ", "12", DocItemLabel.PAGE_FOOTER, ContentLayer.FURNITURE),
        # The model has no list category; an unknown one stays text.
        ("list", "- one", DocItemLabel.TEXT, ContentLayer.BODY),
    ],
)
def test_categories_map_to_labels_and_layers(
    category: str,
    text: str,
    label: DocItemLabel,
    layer: ContentLayer,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        doc = _parse(_reply(_element(category, text)))

    assert [(item.label, item.content_layer) for item in _blocks(doc)] == [
        (label, layer)
    ]
    warnings = [record.getMessage() for record in caplog.records]
    if category == "list":
        assert len(warnings) == 1
        assert "unknown" in warnings[0]
        assert "'list'" in warnings[0]
    else:
        assert warnings == []


_ELEMENTS = [
    _element("title", "Results", [100, 50, 900, 80]),
    _element("text", "The first paragraph.", [100, 90, 900, 200]),
    _element("formula", "$$\\frac{a}{b}$$", [300, 210, 700, 260]),
]
_BARE = json.dumps(_ELEMENTS)
# With thinking on, a reply can hold a draft before the answer.
_DRAFT = _reply(_element("text", "A draft of the answer."))


@pytest.mark.parametrize(
    ("content", "count"),
    [
        (_BARE, 3),
        (f"```json\n{_BARE}", 3),
        (f"{_BARE}<|im_end|>", 3),
        (f"```json\n{_BARE}\n```<|im_end|>", 3),
        (f"Here is the layout.\n```json\n{_BARE}\n```", 3),
        (f"</think>\n\n```json\n{_BARE}\n```", 3),
        (f"<think>\n{_DRAFT}\n</think>\n\n```json\n{_BARE}\n```", 3),
        (json.dumps({"elements": _ELEMENTS}), 3),
        (json.dumps(_ELEMENTS[0]), 1),
    ],
    ids=[
        "bare_list",
        "no_closing_fence",
        "end_token",
        "end_token_after_fence",
        "prose_before_fence",
        "think_end_tag",
        "think_block_with_a_draft",
        "wrapper_object",
        "single_element_object",
    ],
)
def test_reply_shapes_give_the_same_items(content: str, count: int) -> None:
    expected = _summary(_parse(_reply(*_ELEMENTS)))
    assert [label for label, _, _ in expected] == [
        DocItemLabel.SECTION_HEADER,
        DocItemLabel.TEXT,
        DocItemLabel.FORMULA,
    ]

    assert _summary(_parse(content)) == expected[:count]


_FULL = _reply(*_ELEMENTS)


@pytest.mark.parametrize(
    "content",
    [
        # A cut at the last `}` would split this LaTeX and lose the whole page.
        _FULL[: _FULL.index("{a}{") + len("{a}{")],
        _FULL[: _FULL.index('{"bbox": [300')],
        _FULL[: _FULL.index("[300, 210") + len("[300,")],
    ],
    ids=["inside_latex", "after_a_comma", "inside_a_box"],
)
def test_cut_reply_keeps_complete_elements(
    content: str, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        doc = _parse(content)

    assert [item.text for item in doc.texts] == ["Results", "The first paragraph."]
    assert len(caplog.records) == 1
    assert "keeping 2 complete elements" in caplog.text


@pytest.mark.parametrize(
    "content",
    [
        "I cannot read this page.",
        '"x"',
        "[1, 2]",
        '{"a": 1}',
        '{"a": [1], "b": [2]}',
        '```json\n[{"bbox": [100, 200, 9',
    ],
    ids=[
        "prose",
        "json_string",
        "list_without_objects",
        "object_without_a_list",
        "object_with_two_lists",
        "cut_in_the_first_element",
    ],
)
def test_reply_without_elements_raises(content: str) -> None:
    with pytest.raises(ValueError):
        _parse(content)


@pytest.mark.parametrize("content", ["", " \n", "```json\n[]\n```", "</think>\n\n"])
def test_blank_reply_is_an_empty_page(content: str) -> None:
    doc = _parse(content)

    assert _blocks(doc) == []
    assert doc.pages[3].size == _SIZE


@pytest.mark.parametrize(
    ("bbox", "expected"),
    [
        ([100, 200, 900, 400], (61.2, 158.4, 550.8, 316.8)),
        # The grid is 0-1000, not 0-999: the far corner is the page corner.
        ([0, 0, 1000, 1000], (0.0, 0.0, 612.0, 792.0)),
        ([900, 400, 100, 200], (61.2, 158.4, 550.8, 316.8)),
        ([-5, 10, 1200, 30], (0.0, 7.92, 612.0, 23.76)),
    ],
    ids=["inside", "full_page", "swapped_corners", "clamped"],
)
def test_boxes_use_the_page_relative_0_1000_grid(
    bbox: list[int], expected: tuple[float, float, float, float]
) -> None:
    doc = _parse(_reply(_element("text", "Body", bbox)))

    prov = doc.texts[0].prov[0]
    assert _box(doc.texts[0]) == pytest.approx(expected)
    assert prov.bbox.coord_origin == CoordOrigin.TOPLEFT
    assert prov.page_no == 3


def test_elements_without_a_usable_box_keep_their_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # json reads NaN, so the parser must reject it itself.
    boxes = [
        "[1, 2, 3]",
        "[1, 2, 3, true]",
        '["1", "2", "3", "4"]',
        "[NaN, 2, 3, 4]",
        "[5, 5, 5, 9]",
        "null",
    ]
    elements = [
        f'{{"bbox": {box}, "category": "text", "text": "T{index}"}}'
        for index, box in enumerate(boxes)
    ]
    with caplog.at_level(logging.WARNING):
        doc = _parse(f"[{', '.join(elements)}]")

    assert [(item.text, item.prov) for item in doc.texts] == [
        (f"T{index}", []) for index in range(len(boxes))
    ]
    assert [record.getMessage() for record in caplog.records] == [
        "Keeping 6 Typhoon OCR V2 elements without a valid box"
    ]


def test_captions_link_to_one_adjacent_owner() -> None:
    doc = _parse(
        _reply(
            _element("figure", ""),
            _element("figure_caption", "Figure 1. Sales."),
            _element("table_caption", "Table 1. Costs."),
            _element("table", "<table><tr><td>1</td></tr></table>"),
            _element("formula", "$$x$$"),
            _element("formula_caption", "(1)"),
            _element("figure", ""),
            # An element without content still separates its neighbors.
            _element("text", ""),
            _element("figure_caption", "Figure 2. Not linked."),
        )
    )

    first, second = doc.pictures
    assert [ref.resolve(doc).text for ref in first.captions] == ["Figure 1. Sales."]
    assert [ref.resolve(doc).text for ref in doc.tables[0].captions] == [
        "Table 1. Costs."
    ]
    assert second.captions == []
    assert [item.label for item in _blocks(doc)] == [
        DocItemLabel.PICTURE,
        DocItemLabel.CAPTION,
        DocItemLabel.CAPTION,
        DocItemLabel.TABLE,
        DocItemLabel.FORMULA,
        DocItemLabel.CAPTION,
        DocItemLabel.PICTURE,
        DocItemLabel.CAPTION,
    ]


def test_title_is_always_a_section_header() -> None:
    reply = _reply(_element("title", "Intro"), _element("title", "## Method"))

    raw = _parse(reply, normalize_heading_levels=False)
    normalized = _parse(reply)

    assert [type(item) for item in raw.texts] == [SectionHeaderItem] * 2
    headings = [item for item in raw.texts if isinstance(item, SectionHeaderItem)]
    assert [(item.text, item.orig, item.level) for item in headings] == [
        ("Intro", "Intro", 2),
        ("Method", "## Method", 2),
    ]
    assert [
        item.level for item in normalized.texts if isinstance(item, SectionHeaderItem)
    ] == [1, 1]


@pytest.mark.parametrize("latex", ["$$a^2 + 8 = 12$$", "\\[a^2 + 8 = 12\\]"])
def test_formula_exports_its_latex_once(latex: str) -> None:
    doc = _parse(_reply(_element("formula", latex)))

    (formula,) = doc.texts
    assert (formula.label, formula.text, formula.orig) == (
        DocItemLabel.FORMULA,
        "a^2 + 8 = 12",
        latex,
    )
    assert doc.export_to_markdown() == "$$a^2 + 8 = 12$$"


def test_thai_text_is_kept_verbatim() -> None:
    # NFKC splits SARA AM, so this text shows any normalization.
    assert unicodedata.normalize("NFKC", _THAI) != _THAI

    doc = _parse(_reply(_element("text", _THAI), _element("title", _THAI)))

    assert [item.text for item in doc.texts] == [_THAI, _THAI]
    assert doc.export_to_markdown() == f"{_THAI}\n\n## {_THAI}"


def test_page_furniture_stays_out_of_markdown() -> None:
    doc = _parse(
        _reply(
            _element("header", "Annual **Report**"),
            _element("text", "Body"),
            _element("page_footnote", "1 See the appendix."),
            _element("footer", "Report\\(^{x}\\) 2025"),
        )
    )

    # Rich text gives a parent item, an inline group and runs; all are furniture.
    body = [item.text for item in doc.texts if item.content_layer == ContentLayer.BODY]
    assert body == ["Body", "1 See the appendix."]
    assert doc.groups
    assert all(group.content_layer == ContentLayer.FURNITURE for group in doc.groups)
    assert doc.export_to_markdown() == "Body\n\n1 See the appendix."
    assert doc.export_to_markdown(included_content_layers=set(ContentLayer)) == (
        "Annual **Report**\n\nBody\n\n1 See the appendix.\n\nReport$^{x}$ 2025"
    )


def test_table_fallbacks_keep_content(caplog: pytest.LogCaptureFixture) -> None:
    # 60 x 20 cells exceed the 1000-cell limit of the rich HTML table mapper.
    rows = "".join(
        "<tr>" + "".join(f"<td>{row}.{col}</td>" for col in range(20)) + "</tr>"
        for row in range(60)
    )
    pipe_table = "| a | b |\n|---|---|\n| 1 | 2 |"
    with caplog.at_level(logging.WARNING):
        doc = _parse(
            _reply(
                _element("table", f"<table>{rows}</table>", [100, 200, 900, 400]),
                _element("table", pipe_table),
                _element("table", "<table></table>"),
            )
        )

    (table,) = doc.tables
    assert (table.data.num_rows, table.data.num_cols) == (60, 20)
    assert len(table.data.table_cells) == 1200
    assert _box(table) == pytest.approx((61.2, 158.4, 550.8, 316.8))
    assert [(item.label, item.text) for item in doc.texts] == [
        (DocItemLabel.TEXT, pipe_table)
    ]
    assert _box(doc.texts[0]) == pytest.approx((61.2, 79.2, 550.8, 110.88))
    assert "without spans" in caplog.text
    assert "without cells as text" in caplog.text


def test_figure_keeps_box_and_unexpected_text() -> None:
    doc = _parse(
        _reply(
            _element("figure", "", [100, 200, 900, 400]),
            _element("figure", "A bar chart", [100, 500, 900, 700]),
        )
    )

    assert [type(item) for item in _blocks(doc)] == [
        PictureItem,
        PictureItem,
        TextItem,
    ]
    first, second = doc.pictures
    assert _box(first) == pytest.approx((61.2, 158.4, 550.8, 316.8))
    assert [item.text for item in doc.texts] == ["A bar chart"]
    assert _box(doc.texts[0]) == _box(second)


@pytest.mark.parametrize(
    ("preset_id", "repo_id"),
    [
        ("typhoon_ocr2_2b", "typhoon-ai/typhoon-ocr2-2b"),
        ("typhoon_ocr2_08b", "typhoon-ai/typhoon-ocr2-0.8b"),
    ],
)
def test_presets_pin_weights_and_engine_floors(preset_id: str, repo_id: str) -> None:
    preset = VlmConvertOptions.get_preset(preset_id)
    spec = preset.model_spec
    transformers = spec.get_engine_config(VlmEngineType.TRANSFORMERS)
    vllm = spec.get_engine_config(VlmEngineType.VLLM)
    mlx = spec.get_engine_config(VlmEngineType.MLX)
    engines = (transformers, vllm, mlx)

    # The models were trained with this prompt, so it must stay byte for byte.
    assert hashlib.sha256(spec.prompt.encode()).hexdigest() == _PROMPT_SHA256
    assert spec.response_format == ResponseFormat.TYPHOON_OCR2_JSON
    assert [config.repo_id for config in engines] == [
        repo_id,
        repo_id,
        f"{repo_id}-MLX-8bit",
    ]
    assert all(
        re.fullmatch(r"[0-9a-f]{40}", config.revision or "") for config in engines
    )
    assert vllm.revision == transformers.revision
    assert mlx.revision != transformers.revision
    assert [config.min_engine_version for config in engines] == [
        "5.15.1",
        "0.33.0",
        "0.7.6",
    ]
    assert transformers.torch_dtype == "bfloat16"
    assert (
        transformers.extra_config["transformers_model_type"]
        == TransformersModelType.AUTOMODEL_IMAGETEXTTOTEXT
    )
    assert spec.get_runtime_input_extra_config(VlmEngineType.TRANSFORMERS) == {
        "transformers_prompt_style": TransformersPromptStyle.CHAT,
        "skip_special_tokens": True,
    }
    assert vllm.extra_config == {"max_model_len": 40960}
    assert (spec.temperature, spec.max_new_tokens) == (0.0, 16384)
    assert spec.chat_template_kwargs == {"enable_thinking": False}
    assert not spec.trust_remote_code
    # Ollama ignores chat_template_kwargs, so the thinking block would stay open.
    assert not spec.is_engine_supported(VlmEngineType.API_OLLAMA)
    assert preset.scale * 72 == pytest.approx(2048)
    assert preset.max_size == 2048


_PAGE_REPLY = _reply(
    _element("header", "Annual Report 2025", [80, 30, 920, 50]),
    _element("title", "1. Overview", [80, 80, 600, 110]),
    _element("text", _THAI, [80, 120, 920, 200]),
    _element(
        "table",
        "<table><tr><th>Year</th><th>Total</th></tr>"
        "<tr><td>2025</td><td>12</td></tr></table>",
        [80, 220, 920, 400],
    ),
    _element("footer", "1", [490, 950, 510, 970]),
)
_PAGE_LAYOUT = [
    (DocItemLabel.PAGE_HEADER, ContentLayer.FURNITURE),
    (DocItemLabel.SECTION_HEADER, ContentLayer.BODY),
    (DocItemLabel.TEXT, ContentLayer.BODY),
    (DocItemLabel.TABLE, ContentLayer.BODY),
    (DocItemLabel.PAGE_FOOTER, ContentLayer.FURNITURE),
]


@pytest.fixture
def api() -> Iterator[FakeOpenAiApi]:
    service = FakeService()
    fake = FakeOpenAiApi(completion=_PAGE_REPLY)
    service.include(fake.router)
    service.start()
    fake.service = service
    try:
        yield fake
    finally:
        service.stop()


def _convert(
    api: FakeOpenAiApi,
    source: Path,
    engine_type: VlmEngineType = VlmEngineType.API_OPENAI,
) -> ConversionResult:
    options = VlmConvertOptions.from_preset(
        "typhoon_ocr2_2b",
        engine_options=ApiVlmEngineOptions(
            engine_type=engine_type,
            url=AnyUrl(f"{api.service.base_url}/v1/chat/completions"),
        ),
    )
    pipeline_options = VlmPipelineOptions(
        vlm_options=options, enable_remote_services=True
    )
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=VlmPipeline, pipeline_options=pipeline_options
            ),
            InputFormat.IMAGE: ImageFormatOption(
                pipeline_cls=VlmPipeline, pipeline_options=pipeline_options
            ),
        }
    )
    return converter.convert(source)


@pytest.mark.parametrize(
    ("engine_type", "source", "image_size"),
    [
        (VlmEngineType.API, "pdf", (1583, 2048)),
        (VlmEngineType.API_OPENAI, "pdf", (1583, 2048)),
        (VlmEngineType.API_OPENAI, "png", (2048, 1536)),
    ],
)
def test_api_request_matches_the_model_card(
    api: FakeOpenAiApi,
    tmp_path: Path,
    engine_type: VlmEngineType,
    source: str,
    image_size: tuple[int, int],
) -> None:
    path = _PDF
    if source == "png":
        path = tmp_path / "page.png"
        Image.new("RGB", (800, 600), "white").save(path)

    result = _convert(api, path, engine_type)

    (request,) = api.service.requests_for("POST", r"/v1/chat/completions")
    body = request.json()
    (message,) = body.pop("messages")
    # The keys and values that the client of the model card sends.
    assert body == {
        "model": "typhoon-ocr2",
        "max_tokens": 16384,
        "temperature": 0.0,
        "top_p": 1.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    assert message["role"] == "user"
    image_part, text_part = message["content"]
    assert text_part == {"type": "text", "text": TYPHOON_OCR2_PARSE_PROMPT}
    prefix = "data:image/png;base64,"
    assert image_part["image_url"]["url"].startswith(prefix)
    image_bytes = base64.b64decode(image_part["image_url"]["url"].removeprefix(prefix))
    # A page renders with a 2048-pixel long side. A PDF page can be 1 pixel off.
    assert Image.open(io.BytesIO(image_bytes)).size == pytest.approx(image_size, abs=1)
    assert result.status == ConversionStatus.SUCCESS
    assert [
        (item.label, item.content_layer) for item in _blocks(result.document)
    ] == _PAGE_LAYOUT
    assert result.document.texts[2].text == _THAI


def test_cut_reply_gives_partial_success(api: FakeOpenAiApi) -> None:
    api.completion = _PAGE_REPLY[: _PAGE_REPLY.index("<table>") + len("<table><tr>")]
    api.finish_reason = "length"

    result = _convert(api, _PDF)

    assert result.status == ConversionStatus.PARTIAL_SUCCESS
    assert [(error.error_message, error.page_no) for error in result.errors] == [
        ("VLM output incomplete (stop_reason=length).", 1)
    ]
    assert [
        (item.label, item.content_layer) for item in _blocks(result.document)
    ] == _PAGE_LAYOUT[:3]


def test_invalid_reply_is_a_page_error(api: FakeOpenAiApi) -> None:
    api.completion = "I cannot read this page."

    result = _convert(api, _PDF)

    assert result.status == ConversionStatus.PARTIAL_SUCCESS
    (error,) = result.errors
    assert error.page_no == 1
    assert error.error_message.startswith("Invalid Typhoon OCR V2 response: ")
    assert list(result.document.iterate_items()) == []
    assert list(result.document.pages) == [1]
