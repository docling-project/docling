# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Test the NVIDIA Nemotron 3.5 Parse preset, remote-code gate and output mapper."""

import sys
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest
from docling_core.types.doc import (
    ContentLayer,
    DocItemLabel,
    GroupItem,
    InlineGroup,
    ListItem,
    PictureClassificationLabel,
    RichTableCell,
    Script,
    SectionHeaderItem,
    Size,
    TextItem,
)

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.pipeline_options import VlmConvertOptions
from docling.datamodel.pipeline_options_vlm_model import (
    ResponseFormat,
    TransformersPromptStyle,
)
from docling.datamodel.vlm_engine_options import (
    AutoInlineVlmEngineOptions,
    MlxVlmEngineOptions,
    TransformersVlmEngineOptions,
)
from docling.models.inference_engines.vlm.base import VlmEngineType
from docling.utils.nemotron_parse_v3_5_utils import (
    extract_nemotron_parse_v3_5_blocks,
    parse_nemotron_parse_v3_5,
)

_REPO_ID = "nvidia/Nemotron-3.5-Parse"
_REVISION = "e7ea79242a87ea0ad3919b672bfedb429a683206"
_PAGE = Size(width=600, height=800)


def _block(label: str, body: str, box: str = "0.1 0.1 0.9 0.2") -> str:
    x1, y1, x2, y2 = box.split()
    return f"<x_{x1}><y_{y1}>{body}<x_{x2}><y_{y2}><class_{label}>\n\n"


def _parse(content: str):
    return parse_nemotron_parse_v3_5(content, _PAGE, page_no=1, filename="page.pdf")


def test_preset_pins_revision_and_requires_opt_in() -> None:
    preset = VlmConvertOptions.get_preset("nemotron_parse_v3_5")
    spec = preset.model_spec

    assert (preset.scale, preset.max_size) == (2.0, 2048)
    assert preset.default_engine_type == VlmEngineType.AUTO_INLINE
    assert spec.default_repo_id == _REPO_ID
    assert spec.revision == _REVISION
    assert spec.response_format == ResponseFormat.NEMOTRON_PARSE_V3_5
    assert spec.trust_remote_code is False
    assert spec.max_new_tokens == 8744
    assert spec.prompt == (
        "</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>"
    )
    assert spec.supported_engines == {
        VlmEngineType.TRANSFORMERS,
        VlmEngineType.MLX,
        VlmEngineType.API,
    }
    assert spec.has_explicit_engine_export(VlmEngineType.MLX)

    config = spec.get_engine_config(VlmEngineType.TRANSFORMERS)
    assert (config.repo_id, config.revision) == (_REPO_ID, _REVISION)
    assert config.extra_config["requires_trust_remote_code"] is True
    assert (
        config.extra_config["transformers_prompt_style"] == TransformersPromptStyle.RAW
    )
    assert spec.get_runtime_input_extra_config(VlmEngineType.TRANSFORMERS) == {
        "repetition_penalty": 1.1,
        "skip_special_tokens": True,
        "transformers_prompt_style": TransformersPromptStyle.RAW,
        "extra_processor_kwargs": {"add_special_tokens": False},
    }
    assert spec.get_api_params(VlmEngineType.API) == {
        "model": "nemotron-parse",
        "max_tokens": 8744,
        "top_k": 1,
        "repetition_penalty": 1.1,
        "skip_special_tokens": False,
    }


@pytest.mark.parametrize(
    ("system", "device", "mlx_version", "module_layout", "expected"),
    [
        ("Darwin", "mps", "0.7.9", "native", VlmEngineType.TRANSFORMERS),
        ("Darwin", "mps", "0.7.10", "native", VlmEngineType.TRANSFORMERS),
        ("Darwin", "mps", "0.7.10", "released", VlmEngineType.TRANSFORMERS),
        ("Darwin", "mps", "0.7.11", "native", VlmEngineType.MLX),
        ("Darwin", "mps", "0.7.11", "released", VlmEngineType.TRANSFORMERS),
        ("Darwin", "mps", "0.7.11", "no-models", VlmEngineType.TRANSFORMERS),
        ("Darwin", "cpu", "0.7.11", "native", VlmEngineType.TRANSFORMERS),
        ("Linux", "cuda:0", "0.7.11", "native", VlmEngineType.TRANSFORMERS),
    ],
)
def test_auto_inline_selects_nemotron_native_engine(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    system: str,
    device: str,
    mlx_version: str,
    module_layout: str,
    expected: VlmEngineType,
) -> None:
    from docling.models.inference_engines.vlm.auto_inline_engine import (
        AutoInlineVlmEngine,
    )

    options = VlmConvertOptions.from_preset("nemotron_parse_v3_5")
    engine = AutoInlineVlmEngine(
        options=AutoInlineVlmEngineOptions(prefer_vllm=True),
        accelerator_options=AcceleratorOptions(),
        artifacts_path=None,
    )
    engine.model_spec = options.model_spec
    monkeypatch.setattr("platform.system", lambda: system)
    monkeypatch.setattr(
        "docling.models.inference_engines.vlm.auto_inline_engine.decide_device",
        lambda *args, **kwargs: device,
    )
    # Use real module discovery to check the model independently of the version.
    # The minimal package needs no Apple Silicon runtime or model.
    mlx_module = ModuleType("mlx_vlm")
    mlx_module.__path__ = [str(tmp_path / "empty")]
    monkeypatch.setitem(sys.modules, "mlx_vlm", mlx_module)
    monkeypatch.delitem(sys.modules, "mlx_vlm.models", raising=False)
    monkeypatch.delitem(
        sys.modules, "mlx_vlm.models.nemotron_parse_v3_5", raising=False
    )
    if module_layout != "no-models":
        models_module = ModuleType("mlx_vlm.models")
        models_module.__path__ = [str(tmp_path)]
        monkeypatch.setitem(sys.modules, "mlx_vlm.models", models_module)
    if module_layout == "native":
        (tmp_path / "nemotron_parse_v3_5.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(
        "docling.models.inference_engines.vlm._utils._installed_engine_version",
        lambda engine_type: mlx_version,
    )

    assert engine._select_engine() == expected
    config = options.model_spec.get_engine_config(expected)
    if expected == VlmEngineType.MLX:
        assert config.repo_id == "geoHeil/Nemotron-3.5-Parse-MLX-bf16"
        assert config.revision != _REVISION
        assert config.extra_config.get("requires_trust_remote_code") is not True
        assert options.model_spec.get_runtime_input_extra_config(expected) == {
            "repetition_penalty": 1.1,
            "repetition_context_size": 0,
        }
    else:
        assert (config.repo_id, config.revision) == (_REPO_ID, _REVISION)


def test_explicit_native_mlx_reports_missing_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from docling.models.inference_engines.vlm.mlx_engine import MlxVlmEngine

    spec = VlmConvertOptions.get_preset("nemotron_parse_v3_5").model_spec
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    with pytest.raises(
        RuntimeError,
        match=r"mlx_vlm\.models\.nemotron_parse_v3_5.*Transformers engine",
    ):
        MlxVlmEngine(
            options=MlxVlmEngineOptions(),
            artifacts_path=None,
            model_config=spec.get_engine_config(VlmEngineType.MLX),
        )


@pytest.mark.parametrize(
    ("mlx_version", "supported"), [("0.7.10", False), ("0.7.11", True)]
)
def test_explicit_native_mlx_checks_version_before_loading_pinned_export(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mlx_version: str,
    supported: bool,
) -> None:
    from docling.models.inference_engines.vlm.mlx_engine import MlxVlmEngine

    model, processor, config = object(), object(), object()
    load = Mock(return_value=(model, processor))
    load_config = Mock(return_value=config)
    download = Mock(return_value=tmp_path)
    mlx_module = ModuleType("mlx_vlm")
    prompt_module = ModuleType("mlx_vlm.prompt_utils")
    utils_module = ModuleType("mlx_vlm.utils")
    native_module = ModuleType("mlx_vlm.models.nemotron_parse_v3_5")
    native_module.__spec__ = ModuleSpec(native_module.__name__, loader=None)
    # Stub only the optional vendor boundary; run Docling's module gate and loader.
    monkeypatch.setattr(mlx_module, "load", load, raising=False)
    monkeypatch.setattr(mlx_module, "stream_generate", Mock(), raising=False)
    monkeypatch.setattr(prompt_module, "apply_chat_template", Mock(), raising=False)
    monkeypatch.setattr(utils_module, "load_config", load_config, raising=False)
    for module in (mlx_module, prompt_module, utils_module, native_module):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(MlxVlmEngine, "download_models", staticmethod(download))
    monkeypatch.setattr(
        "docling.models.inference_engines.vlm._utils._installed_engine_version",
        lambda engine_type: mlx_version,
    )

    spec = VlmConvertOptions.get_preset("nemotron_parse_v3_5").model_spec
    if not supported:
        with pytest.raises(RuntimeError, match=r"requires mlx-vlm>=0\.7\.11"):
            MlxVlmEngine(
                options=MlxVlmEngineOptions(),
                artifacts_path=None,
                model_config=spec.get_engine_config(VlmEngineType.MLX),
            )
        download.assert_not_called()
        load.assert_not_called()
        load_config.assert_not_called()
        return

    engine = MlxVlmEngine(
        options=MlxVlmEngineOptions(),
        artifacts_path=None,
        model_config=spec.get_engine_config(VlmEngineType.MLX),
    )

    download.assert_called_once_with(
        "geoHeil/Nemotron-3.5-Parse-MLX-bf16",
        revision="d6a6b5386601e04706e36268e0c003fca9671ade",
        trust_remote_code=False,
    )
    load.assert_called_once_with(tmp_path, strict=True)
    load_config.assert_called_once_with(tmp_path)
    assert engine._initialized
    assert engine.vlm_model is model
    assert engine.processor is processor
    assert engine.config is config


def test_transformers_engine_refuses_remote_code_without_opt_in(
    tmp_path: Path,
) -> None:
    pytest.importorskip("transformers")
    from docling.models.inference_engines.vlm.transformers_engine import (
        TransformersVlmEngine,
    )

    spec = VlmConvertOptions.get_preset("nemotron_parse_v3_5").model_spec
    with pytest.raises(ValueError, match=f"{_REPO_ID} \\(revision {_REVISION}\\)"):
        TransformersVlmEngine(
            options=TransformersVlmEngineOptions(),
            accelerator_options=AcceleratorOptions(device="cpu"),
            artifacts_path=tmp_path,
            model_config=spec.get_engine_config(VlmEngineType.TRANSFORMERS),
        )


def test_auto_inline_passes_user_opt_in_to_transformers(tmp_path: Path) -> None:
    pytest.importorskip("transformers")
    from docling.models.inference_engines.vlm.auto_inline_engine import (
        AutoInlineVlmEngine,
    )

    spec = VlmConvertOptions.get_preset("nemotron_parse_v3_5").model_spec
    accelerator_options = AcceleratorOptions(device="cpu")
    with pytest.raises(ValueError, match="trust_remote_code=True"):
        AutoInlineVlmEngine(
            AutoInlineVlmEngineOptions(),
            accelerator_options=accelerator_options,
            artifacts_path=tmp_path,
            model_spec=spec,
        )
    # With the opt-in the gate passes; the empty artifacts path stops the load.
    with pytest.raises(FileNotFoundError, match=_REPO_ID):
        AutoInlineVlmEngine(
            AutoInlineVlmEngineOptions(trust_remote_code=True),
            accelerator_options=accelerator_options,
            artifacts_path=tmp_path,
            model_spec=spec,
        )


def test_blocks_drop_artifacts_duplicates_and_malformed_output() -> None:
    content = (
        "</s>"
        + _block("Text", "Hello<tbc> world")
        + _block("Text", "Hello<tbc> world")
        + "<x_0.5><y_0.5><class_Text>"
        + _block("Text", "next")
    )

    blocks = extract_nemotron_parse_v3_5_blocks(content)

    assert [block.text for block in blocks] == ["Hello world", "next"]
    assert blocks[0].bbox == (0.1, 0.1, 0.9, 0.2)


def test_bbox_scales_content_normalized_coordinates_to_page() -> None:
    doc = _parse(
        _block("Text", "inside", "0.1 0.25 0.5 0.5")
        + _block("Mystery", "outside", "0.0 0.9 1.2 1.1")
    )

    inside, outside = doc.texts
    assert inside.prov[0].bbox.as_tuple() == (60.0, 200.0, 300.0, 400.0)
    assert outside.prov[0].bbox.as_tuple() == (0.0, 720.0, 600.0, 800.0)
    assert outside.label == DocItemLabel.TEXT


def test_table_maps_multirow_multicolumn_and_thead() -> None:
    table = (
        r"\begin{tabular}{cccc}"
        "\n"
        r"\multirow{2}{*}{\thead{Solution}} & "
        r"\multicolumn{3}{c}{\thead{Intracellular solutions}}\\"
        "\n"
        r"\mbox{} & \thead{K<sup>+</sup>} & \thead{**Na**} & \thead{Cl}\\"
        "\n"
        r"Kin \$ & 145 & \mbox{} & 5 \% \textless{} 6\\"
        "\n"
        r"\end{tabular}"
    )

    data = _parse(_block("Table", table)).tables[0].data

    assert (data.num_rows, data.num_cols) == (3, 4)
    cells = {
        (c.start_row_offset_idx, c.start_col_offset_idx): c for c in data.table_cells
    }
    assert (cells[0, 0].text, cells[0, 0].row_span) == ("Solution", 2)
    assert (cells[0, 1].text, cells[0, 1].col_span) == ("Intracellular solutions", 3)
    assert [cells[1, col].text for col in (1, 2, 3)] == ["K+", "Na", "Cl"]
    assert all(cells[row, col].column_header for row, col in cells if row < 2)
    assert not cells[2, 0].column_header
    assert isinstance(cells[1, 1], RichTableCell)
    assert [cells[2, col].text for col in range(4)] == ["Kin $", "145", "", "5 % < 6"]


def test_table_with_oversized_span_is_rejected_without_building_it() -> None:
    # Building the span grid used memory linear in the predicted span value.
    table = r"\begin{tabular}{c}\multicolumn{1000000000}{c}{x}\\\end{tabular}"

    data = _parse(_block("Table", table)).tables[0].data

    assert (data.num_rows, data.num_cols) == (0, 0)


def test_table_cell_markup_stays_cell_text() -> None:
    table = (
        r"\begin{tabular}{cc}"
        r'a</td></tr><tr><td colspan="9">b & x<sup>2</sup> \& y\\'
        r"\end{tabular}"
    )

    data = _parse(_block("Table", table)).tables[0].data

    assert (data.num_rows, data.num_cols) == (1, 2)
    assert data.grid[0][0].text == 'a</td></tr><tr><td colspan="9">b'
    assert data.grid[0][1].text == "x2 & y"


def test_table_cell_latex_angle_brackets_stay_cell_text() -> None:
    table = (
        r"\begin{tabular}{cc}"
        r"\textless{}b\textgreater{}x\textless{}/b\textgreater{} & <code>a<b</code>\\"
        r"\end{tabular}"
    )

    data = _parse(_block("Table", table)).tables[0].data

    assert [cell.text for cell in data.grid[0]] == ["<b>x</b>", "a<b"]


def test_table_cell_under_multirow_with_text_ends_the_span() -> None:
    table = r"\begin{tabular}{cc}\multirow{3}{*}{A} & 1\\ \mbox{} & 2\\ B & 3\\\end{tabular}"

    data = _parse(_block("Table", table)).tables[0].data

    grid = [[cell.text for cell in row] for row in data.grid]
    assert grid == [["A", "1"], ["A", "2"], ["B", "3"]]


def test_chart_maps_classification_title_and_data() -> None:
    chart = (
        '<chart type="grouped_bar">\n'
        "title: Freight volume\n"
        "legend[color]: items=2010 ; 2005\n"
        "bind: 2010 ;  ; y\n\n"
        "| mode | 2010 | 2005 |\n"
        "| --- | --- | --- |\n"
        "| road | 229 | 138 |\n"
        "</chart>"
    )

    picture = _parse(_block("Chart", chart)).pictures[0]

    assert picture.meta is not None
    assert picture.meta.classification is not None
    assert picture.meta.classification.predictions[0].class_name == (
        PictureClassificationLabel.BAR_CHART.value
    )
    assert picture.meta.tabular_chart is not None
    assert picture.meta.tabular_chart.title == "Freight volume"
    chart_data = picture.meta.tabular_chart.chart_data
    assert [[cell.text for cell in row] for row in chart_data.grid] == [
        ["mode", "2010", "2005"],
        ["road", "229", "138"],
    ]


def test_multi_panel_chart_maps_one_picture_per_panel() -> None:
    chart = (
        '<chart type="faceted_bar">\n'
        "panel: p1 ; multi_line ; Early\n"
        "panel: p2 ; scatter\n\n"  # the model usually omits the label
        "table[p1]:\n| t | v |\n| --- | --- |\n| 1 | 2 |\n\n"
        "table[p2]:\n| t | v |\n| --- | --- |\n| 3 | 4 |\n| 5 | 6 |\n"
        "</chart>"
    )

    pictures = _parse(_block("Chart", chart)).pictures

    summary = [
        (
            picture.meta.classification.predictions[0].class_name,
            picture.meta.tabular_chart.title,
            picture.meta.tabular_chart.chart_data.num_rows,
        )
        for picture in pictures
        if picture.meta and picture.meta.classification and picture.meta.tabular_chart
    ]
    assert summary == [
        (PictureClassificationLabel.LINE_CHART.value, "Early", 2),
        (PictureClassificationLabel.SCATTER_CHART.value, None, 3),
    ]


def test_text_maps_inline_bold_and_superscript_runs() -> None:
    raw = "Mass is **heavy** in m<sup>2</sup>"

    doc = _parse(_block("Text", raw))

    parent = doc.texts[0]
    assert (parent.text, parent.orig) == ("", raw)
    group = parent.children[0].resolve(doc)
    assert isinstance(group, InlineGroup)
    runs = [child.resolve(doc) for child in group.children]
    assert [run.text for run in runs] == ["Mass is", "heavy", "in m", "2"]
    assert runs[1].formatting is not None and runs[1].formatting.bold
    assert runs[3].formatting is not None
    assert runs[3].formatting.script == Script.SUPER


def test_caption_links_to_adjacent_table() -> None:
    doc = _parse(
        _block("Caption", "Table 1: Results")
        + _block("Table", r"\begin{tabular}{c}a\\\end{tabular}")
    )

    assert doc.tables[0].captions[0].resolve(doc).text == "Table 1: Results"


def test_list_items_keep_markers_and_detect_ordered_lists() -> None:
    doc = _parse(
        _block("List-item", "1. First step")
        + _block("List-item", "b) Second step")
        + _block("List-item", "• Loose point")
        + _block("Text", "After the list")
    )

    group = doc.groups[0]
    items = [child.resolve(doc) for child in group.children]
    assert all(isinstance(item, ListItem) for item in items)
    assert [(item.marker, item.enumerated, item.text) for item in items] == [
        ("1.", True, "First step"),
        ("b)", True, "Second step"),
        ("•", False, "Loose point"),
    ]
    assert doc.texts[-1].parent is not None
    assert doc.texts[-1].parent.cref == "#/body"


def test_page_header_and_formatted_page_footer_are_furniture() -> None:
    doc = _parse(
        _block("Page-header", "Journal of Tests")
        + _block("Page-footer", "Page **4**")
        + _block("Text", "Body")
    )

    furniture = [
        item.text for item in doc.texts if item.content_layer == ContentLayer.FURNITURE
    ]
    assert furniture == ["Journal of Tests", "", "Page", "4"]
    assert "Journal of Tests" not in doc.export_to_markdown()
    assert doc.export_to_markdown() == "Body"


def test_headings_titles_formula_and_code() -> None:
    doc = _parse(
        _block("Title", "**A Study**")
        + _block("Section-header", r"### Data\_sets")
        + _block("Formula", r"E\_0 = mc^2")
        + _block("Code", "print('x')")
    )

    labels = [(item.label, item.text) for item in doc.texts]
    assert labels == [
        (DocItemLabel.TITLE, "A Study"),
        (DocItemLabel.SECTION_HEADER, "Data_sets"),
        (DocItemLabel.FORMULA, r"E\_0 = mc^2"),
        (DocItemLabel.CODE, "print('x')"),
    ]
    title = doc.texts[0]
    assert title.formatting is not None and title.formatting.bold
    heading = doc.texts[1]
    assert isinstance(heading, SectionHeaderItem) and heading.level == 2


@pytest.mark.parametrize(
    ("raw", "latex"),
    [
        (r"\(x^2\)", "x^2"),
        # Each segment has its own pair, with text between them.
        (r"(6) \(x_1=\frac{a}{b}\), \(x_2=y\)", r"(6) x_1=\frac{a}{b}, x_2=y"),
        (
            r"\(\[\begin{array}{c}1 \\ 2\end{array}\]\)",
            r"\begin{array}{c}1 \\ 2\end{array}",
        ),
        # Truncated output leaves an unpaired delimiter.
        (r"A = \([\begin{array}{c}1", r"A = [\begin{array}{c}1"),
    ],
)
def test_formula_text_has_no_math_delimiters(raw: str, latex: str) -> None:
    doc = _parse(_block("Formula", raw))

    formula = doc.texts[0]
    assert formula.label == DocItemLabel.FORMULA
    assert (formula.text, formula.orig) == (latex, raw)
    assert doc.export_to_markdown() == f"$${latex}$$"


def test_consecutive_toc_blocks_share_one_group() -> None:
    doc = _parse(
        _block("TOC", "1 Introduction ..... 1")
        + _block("TOC", "2 Methods ..... 4")
        + _block("Text", "Body")
    )

    toc = doc.body.children[0].resolve(doc)
    assert isinstance(toc, GroupItem) and toc.name == "toc"
    assert [child.resolve(doc).text for child in toc.children] == [
        "1 Introduction ..... 1",
        "2 Methods ..... 4",
    ]


def test_picture_text_is_added_as_picture_children() -> None:
    doc = _parse(_block("Picture", "") + _block("Picture", "axis label\nlegend"))

    empty, with_text = doc.pictures
    assert empty.children == []
    child = with_text.children[0].resolve(doc)
    assert isinstance(child, TextItem) and child.text == "axis label\nlegend"
