# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Model-free regression checks for Nemotron multi-pass inference and export."""

from io import BytesIO
from types import SimpleNamespace

import pytest
from docling_core.types.doc import DoclingDocument, Size
from PIL import Image

from docling.datamodel.base_models import ConversionStatus, Page, VlmStopReason
from docling.datamodel.pipeline_options import (
    NemotronParseOptions,
    VlmConvertOptions,
    VlmPipelineOptions,
)
from docling.datamodel.vlm_engine_options import (
    ApiVlmEngineOptions,
    TransformersVlmEngineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.models.inference_engines.vlm.base import VlmEngineOutput, VlmEngineType
from docling.models.stages.vlm_convert.vlm_convert_model import (
    VlmConvertModel,
    _prediction_from_engine_output,
)
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling.utils.nemotron_parse_multipass import (
    NEMOTRON_DETECT_PROMPT,
    merge_nemotron_crops,
    prepare_nemotron_crops,
    remap_nemotron_crop,
)
from docling.utils.nemotron_parse_v3_5_utils import (
    NemotronParseBlock,
    extract_nemotron_parse_v3_5_blocks,
    parse_nemotron_parse_v3_5,
)
from docling.utils.pipeline_cache import create_pipeline_options_hash


def _table(value="10", columns="cc") -> str:
    return (
        rf"\begin{{tabular}}{{{columns}}}"
        + "\n"
        + rf"Name & Value\\ A & {value}\\"
        + "\n"
        + r"\end{tabular}"
    )


def _block(text, bbox=(0.1, 0.1, 0.9, 0.4), label="Table") -> str:
    x1, y1, x2, y2 = bbox
    return f"<x_{x1}><y_{y1}>{text}<x_{x2}><y_{y2}><class_{label}>"


class _Engine:
    def __init__(self, page_outputs, detection="", crop_reason="end_of_sequence"):
        self.page_outputs = page_outputs
        self.detection = detection
        self.crop_reason = crop_reason
        self.batches = []
        self.crop_count = 0

    def predict_batch(self, batch):
        self.batches.append(batch)
        if len(self.batches) == 1:
            return [
                VlmEngineOutput(
                    text=text,
                    stop_reason="end_of_sequence",
                    metadata={"num_tokens": 10},
                )
                for text in self.page_outputs
            ]
        if batch[0].prompt == NEMOTRON_DETECT_PROMPT:
            return [
                VlmEngineOutput(text=self.detection, stop_reason="end_of_sequence")
                for _ in batch
            ]
        outputs = []
        for item in batch:
            self.crop_count += 1
            red = item.image.getpixel((0, 0))[0]
            outputs.append(
                VlmEngineOutput(
                    text=_block(_table(str(red)), (0.02, 0.02, 0.98, 0.98)),
                    stop_reason=self.crop_reason,
                    metadata={"num_tokens": 5, "usage": {"completion_tokens": 5}},
                )
            )
        return outputs

    def cleanup(self):
        pass


def _model(
    engine,
    mode="double",
    engine_type=VlmEngineType.TRANSFORMERS,
    regions="union",
    batch_size=8,
):
    model = VlmConvertModel.__new__(VlmConvertModel)
    model.enabled = True
    model.engine = engine
    runtime = (
        ApiVlmEngineOptions()
        if engine_type == VlmEngineType.API
        else TransformersVlmEngineOptions()
    )
    model.options = VlmConvertOptions.from_preset(
        "nemotron_parse_v3_5",
        engine_options=runtime,
        nemotron_parse=NemotronParseOptions(
            passes=mode, triple_regions=regions, crop_batch_size=batch_size
        ),
    )
    return model


def test_crop_geometry_jpeg_and_coordinate_mapping() -> None:
    image = Image.effect_noise((1000, 800), 50).convert("RGB")
    (crop,) = prepare_nemotron_crops(image, _block(_table(), (0.1, 0.2, 0.5, 0.6)))
    # 4% of region size + 12px on each side; floor left/top, ceil right/bottom.
    assert crop.bbox == (0.072, 0.16875, 0.528, 0.63125)
    expected = BytesIO()
    image.crop((72, 135, 528, 505)).save(expected, format="JPEG", quality=95)
    expected.seek(0)
    assert crop.image.tobytes() == Image.open(expected).convert("RGB").tobytes()
    (candidate,) = remap_nemotron_crop(
        _block(_table("11"), (0.05, 0.05, 0.95, 0.95)), crop
    )
    assert candidate.bbox == pytest.approx((0.0948, 0.191875, 0.5052, 0.608125))
    # A neighbour seen in the margin must not become the crop's target.
    assert not remap_nemotron_crop(_block(_table(), (0, 0, 0.1, 0.1)), crop)


def test_crop_edges_tiny_regions_invalid_boxes_and_cap() -> None:
    content = (
        _block("", (0, 0, 1, 1))
        + _block("", (0.4, 0.1, 0.2, 0.8))
        + _block("", (0.1, 0.1, 1.2, 1))
    )
    (crop,) = prepare_nemotron_crops(Image.new("RGBA", (100, 200)), content)
    assert crop.bbox == (0, 0, 1, 1)
    assert crop.image.mode == "RGB"
    assert prepare_nemotron_crops(Image.new("RGB", (7, 7)), content) == []
    regions = "".join(
        _block("", (i / 100, 0.1, (i + 0.5) / 100, 0.9)) for i in range(90)
    )
    assert len(prepare_nemotron_crops(Image.new("RGB", (1000, 800)), regions)) == 80


def test_triple_union_prefers_detector_and_keeps_parser_only_boxes() -> None:
    image = Image.new("RGB", (1000, 800))
    detector = _block("", (0.1, 0.1, 0.9, 0.4)) * 2
    parser = _block(_table(), (0.11, 0.1, 0.9, 0.4)) + _block(
        _table(), (0.1, 0.6, 0.9, 0.9)
    )
    assert [
        crop.region.bbox for crop in prepare_nemotron_crops(image, parser, detector)
    ] == [(0.1, 0.1, 0.9, 0.4), (0.1, 0.6, 0.9, 0.9)]
    assert len(prepare_nemotron_crops(image, parser, detector, union=False)) == 1
    assert prepare_nemotron_crops(image, parser, "", union=False) == []
    assert len(prepare_nemotron_crops(image, parser, "")) == 2


@pytest.mark.parametrize(
    "body",
    [
        _table(columns="ccc"),
        r"\begin{tabular}{cc}a} & b\\\end{tabular}",
        r"\begin{tabular}{cc}\end{tabular}",
        r"\begin{tabular}{cc}\multirow{2}{*} & b\\\end{tabular}",
        r"\begin{tabular}{cc}a & b & c\\\end{tabular}",
        r"\begin{tabular}{cc}\multirow{3}{*}{a} & b\\ & c\\\end{tabular}",
        r"\begin{tabular}{cc}\multirow{2}{*}{a} & b\\ x & c\\\end{tabular}",
        r"\begin{tabular}{cc}\multicolumn{0}{c}{a} & b\\\end{tabular}",
        r"\begin{tabular}{cc}a{ & b\\\end{tabular}",
        r"\begin{tabular}{cc}a & b\\",
        "",
    ],
)
def test_guard_keeps_page_table_when_crop_grid_is_bad(body) -> None:
    original = _block(_table())
    candidate = NemotronParseBlock("Table", (0.1, 0.1, 0.9, 0.4), body)
    assert merge_nemotron_crops(original, [candidate]) == original


def test_guard_accepts_spans_and_preserves_reading_order_without_alternatives() -> None:
    body = r"\begin{tabular}{cc}\multirow{2}{*}{a} & b\\ & c\\\end{tabular}"
    original = (
        _block("before", (0, 0, 1, 0.05), "Text")
        + _block(body)
        + _block("after", (0, 0.8, 1, 1), "Text")
    )
    refined = body.replace("{a}", "{A}")
    candidates = [
        NemotronParseBlock("Table", (0.11, 0.1, 0.89, 0.4), refined),
        NemotronParseBlock("Table", (0.1, 0.5, 0.9, 0.7), _table("12")),
    ]
    merged = merge_nemotron_crops(original, candidates * 2)
    blocks = extract_nemotron_parse_v3_5_blocks(merged)
    assert [b.text for b in blocks] == ["before", refined, _table("12"), "after"]
    assert blocks[1].bbox == (0.1, 0.1, 0.9, 0.4)
    document = parse_nemotron_parse_v3_5(merged, Size(width=600, height=800), 1)
    assert len(document.tables) == 2
    assert document.tables[0].data.table_cells[0].row_span == 2
    assert "A" in document.export_to_markdown()
    assert "<table>" in document.export_to_html()
    assert (
        len(DoclingDocument.model_validate_json(document.model_dump_json()).tables) == 2
    )


@pytest.mark.parametrize("engine_type", [VlmEngineType.TRANSFORMERS, VlmEngineType.API])
@pytest.mark.parametrize("mode,stages", [("single", 1), ("double", 2), ("triple", 3)])
def test_stages_batch_both_engines_and_keep_page_counts_separate(
    engine_type, mode, stages
) -> None:
    engine = _Engine([_block(_table()), _block(_table())], detection=_block(""))
    model = _model(engine, mode, engine_type, batch_size=1)
    pages = [Page(page_no=4), Page(page_no=7)]
    for page, red in zip(pages, [20, 80]):
        page._image_cache = {
            model.options.scale: Image.new("RGB", (1000, 800), (red, 0, 0))
        }
        page._default_image_scale = model.options.scale
    assert list(model(SimpleNamespace(timings={}), pages)) == pages
    assert len(engine.batches) == stages + (mode != "single")
    for page, value in zip(pages, [20, 80]):
        response = page.predictions.vlm_response
        assert response.nemotron_parse.requests == stages
        assert response.nemotron_parse.crop_requests == (mode != "single")
        assert response.num_tokens == (10 if mode == "single" else 15)
        assert (str(value) in response.text) == (mode != "single")
    assert engine.batches[0][0].temperature == (0 if mode == "single" else 0.2)
    if mode != "single":
        crop_input = engine.batches[-1][0]
        assert crop_input.temperature == 0.1
        config = crop_input.extra_generation_config
        if engine_type == VlmEngineType.API:
            config = config["api_params"]
        assert config["repetition_penalty"] == 1.0
        assert config["top_k"] == 50
    if mode == "triple":
        assert engine.batches[1][0].prompt == NEMOTRON_DETECT_PROMPT
        assert engine.batches[1][0].max_new_tokens == 4096


@pytest.mark.parametrize("reason", ["length", "inference_error", "content_filter"])
def test_bad_crop_keeps_page_and_counts_attempt(reason) -> None:
    original = _block(_table())
    engine = _Engine([original], crop_reason=reason)
    (response,) = _model(engine).process_images(
        [Image.new("RGB", (1000, 800))], "parse"
    )
    assert response.text == original
    assert response.stop_reason == VlmStopReason.END_OF_SEQUENCE
    assert response.nemotron_parse.requests == 2
    assert response.nemotron_parse.rejected_crop_requests == 1


@pytest.mark.parametrize("page_range", [(1, 2), (2, 2)])
def test_full_pipeline_export_keeps_metadata_and_options_change_cache(
    monkeypatch, tmp_path, page_range
) -> None:
    page_count = page_range[1] - page_range[0] + 1
    engine = _Engine([_block(_table())] * page_count)
    monkeypatch.setattr(
        "docling.models.stages.vlm_convert.vlm_convert_model.create_vlm_engine",
        lambda **kwargs: engine,
    )
    options = _model(engine).options
    pipeline_options = VlmPipelineOptions(vlm_options=options)
    converter = DocumentConverter(
        format_options={
            "pdf": PdfFormatOption(
                pipeline_cls=VlmPipeline, pipeline_options=pipeline_options
            )
        }
    )
    source = tmp_path / "tables.pdf"
    Image.new("RGB", (1000, 800), (20, 0, 0)).save(
        source, save_all=True, append_images=[Image.new("RGB", (1000, 800), (80, 0, 0))]
    )
    result = converter.convert(source, page_range=page_range)
    assert result.status == ConversionStatus.SUCCESS
    serialized = result.document.export_to_dict()
    metadata = serialized["body"]["meta"]["docling__nemotron_parse"]["pages"]
    assert set(metadata) == {str(i) for i in range(page_range[0], page_range[1] + 1)}
    assert all(
        page["passes"] == "double" and page["requests"] == 2
        for page in metadata.values()
    )
    assert len(serialized["tables"]) == page_count
    assert "80" in result.document.export_to_markdown()
    assert "80" in result.document.export_to_html()
    single = pipeline_options.model_copy(deep=True)
    single.vlm_options.nemotron_parse.passes = "single"
    assert create_pipeline_options_hash(single) != create_pipeline_options_hash(
        pipeline_options
    )


def test_chart_refinement_maps_data_and_rejects_unclosed_crop() -> None:
    chart = '<chart type="bar">\ntitle: Sales\n| Name | Value |\n| --- | --- |\n| A | 10 |\n</chart>'
    original = _block(chart, label="Chart")
    (crop,) = prepare_nemotron_crops(Image.new("RGB", (1000, 800)), original)
    refined = chart.replace("10", "12")
    candidates = remap_nemotron_crop(
        _block(refined, (0.02, 0.02, 0.98, 0.98), "Chart"), crop
    )
    merged = merge_nemotron_crops(original, candidates)
    document = parse_nemotron_parse_v3_5(merged, Size(width=600, height=800), 1)
    assert len(document.pictures) == 1
    assert (
        document.pictures[0].meta.tabular_chart.chart_data.table_cells[-1].text == "12"
    )
    assert document.pictures[0].prov[0].bbox.l == 60
    assert (
        remap_nemotron_crop(
            _block(refined.replace("</chart>", ""), (0.02, 0.02, 0.98, 0.98), "Chart"),
            crop,
        )
        == []
    )


@pytest.mark.parametrize("mode", ["double", "triple"])
def test_optional_engine_exception_preserves_page_parse(mode, monkeypatch) -> None:
    original = _block(_table())
    engine = _Engine([original])
    predict = engine.predict_batch

    def fail_optional(batch):
        if engine.batches:
            raise RuntimeError("service unavailable")
        return predict(batch)

    monkeypatch.setattr(engine, "predict_batch", fail_optional)
    (response,) = _model(engine, mode).process_images(
        [Image.new("RGB", (1000, 800))], "parse"
    )
    assert response.text == original
    assert response.nemotron_parse.requests == 2
    assert response.nemotron_parse.crop_requests == (mode == "double")


def test_empty_crop_and_partial_batch_failure_preserve_other_regions(
    monkeypatch,
) -> None:
    original = _block(_table()) + _block(_table(), (0.1, 0.6, 0.9, 0.9))
    engine = _Engine([original])
    predict = engine.predict_batch

    def one_failed_crop(batch):
        results = predict(batch)
        if len(engine.batches) > 1:
            results[1] = VlmEngineOutput(text="", stop_reason="end_of_sequence")
        return results

    monkeypatch.setattr(engine, "predict_batch", one_failed_crop)
    (response,) = _model(engine).process_images(
        [Image.new("RGB", (1000, 800), (20, 0, 0))], "parse"
    )
    blocks = extract_nemotron_parse_v3_5_blocks(response.text)
    assert [b.text for b in blocks] == [_table("20"), _table()]
    assert response.nemotron_parse.requests == 3
    assert response.nemotron_parse.rejected_crop_requests == 1


@pytest.mark.parametrize(
    "eos_token_ids,pad_token_id,final_eos",
    [(2, 0, 2), ([2, 7], 0, 7), (2, 2, 2), (None, 0, 2)],
)
def test_transformers_distinguishes_token_limit_from_final_eos_and_padding(
    eos_token_ids, pad_token_id, final_eos
) -> None:
    torch = pytest.importorskip("torch")
    from transformers import GenerationConfig

    from docling.models.inference_engines.vlm.base import VlmEngineInput
    from docling.models.inference_engines.vlm.transformers_engine import (
        TransformersVlmEngine,
    )

    generated = [
        [3, 4, 5, 6],
        [3, 4, 5, final_eos],
        [3, final_eos, pad_token_id, pad_token_id],
    ]

    class Processor:
        tokenizer = SimpleNamespace(
            pad_token_id=pad_token_id, eos_token_id=2, pad_token=None
        )

        def __call__(self, **kwargs):
            return {"input_ids": torch.tensor([[9, 9]] * 3)}

        def batch_decode(self, sequences, **kwargs):
            assert sequences.tolist() == generated
            return ["truncated", "complete at limit", "complete with padding"]

    engine = TransformersVlmEngine.__new__(TransformersVlmEngine)
    engine._initialized = True
    engine.options = TransformersVlmEngineOptions()
    engine.processor = Processor()
    engine.vlm_model = SimpleNamespace(
        generate=lambda **kwargs: torch.tensor([[9, 9, *row] for row in generated])
    )
    engine.device = "cpu"
    engine.generation_config = GenerationConfig(eos_token_id=eos_token_ids)
    engine.strip_stop_strings = False
    outputs = engine.predict_batch(
        [
            VlmEngineInput(
                image=Image.new("RGB", (8, 8)),
                prompt="parse",
                max_new_tokens=4,
                extra_generation_config={"transformers_prompt_style": "raw"},
            )
            for _ in range(3)
        ]
    )
    assert [out.stop_reason for out in outputs] == [
        "length",
        "unspecified",
        "unspecified",
    ]
    assert [out.metadata["num_tokens"] for out in outputs] == [
        sum(token != pad_token_id for token in row) for row in generated
    ]

    pipeline = VlmPipeline.__new__(VlmPipeline)
    for index, output in enumerate(outputs):
        page = Page(page_no=1)
        page.predictions.vlm_response = _prediction_from_engine_output(output)
        result = SimpleNamespace(
            pages=[page], status=ConversionStatus.STARTED, errors=[]
        )
        assert pipeline._determine_status(result) == (
            ConversionStatus.PARTIAL_SUCCESS if index == 0 else ConversionStatus.SUCCESS
        )
        assert len(result.errors) == (1 if index == 0 else 0)


@pytest.mark.parametrize(
    "body",
    [
        r"\begin{tabular}{ccc}Name & Value & Unit\\ A & 12 & kg\\\end{tabular}",
        _table("different " * 100),
    ],
)
def test_guard_rejects_valid_but_incompatible_or_unrelated_tables(body) -> None:
    original = _block(_table())
    candidate = NemotronParseBlock("Table", (0.1, 0.1, 0.9, 0.4), body)
    assert merge_nemotron_crops(original, [candidate]) == original


@pytest.mark.parametrize("stage", ["fullpage", "detection", "crop"])
def test_wrong_engine_output_count_never_attaches_a_result_to_the_wrong_page(
    stage, monkeypatch
) -> None:
    engine = _Engine([_block(_table())], detection=_block(""))
    predict = engine.predict_batch

    def omit_output(batch):
        current_stage = (
            "fullpage"
            if not engine.batches
            else "detection"
            if batch[0].prompt == NEMOTRON_DETECT_PROMPT
            else "crop"
        )
        if current_stage == stage:
            return []
        return predict(batch)

    monkeypatch.setattr(engine, "predict_batch", omit_output)
    model = _model(engine, "triple")
    page = Page(page_no=3)
    page._image_cache = {model.options.scale: Image.new("RGB", (1000, 800))}
    page._default_image_scale = model.options.scale
    with pytest.raises(RuntimeError, match="output count does not match"):
        list(model(SimpleNamespace(timings={}), [page]))
    assert page.predictions.vlm_response is None
