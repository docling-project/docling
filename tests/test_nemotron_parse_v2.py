# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Test NVIDIA Nemotron Parse 2.0 VLM preset configuration."""

import pytest
from docling_core.types.doc import DocItemLabel, Size

from docling.datamodel.pipeline_options import VlmConvertOptions
from docling.datamodel.pipeline_options_vlm_model import (
    ResponseFormat,
    TransformersModelType,
    TransformersPromptStyle,
)
from docling.models.inference_engines.vlm.base import VlmEngineType
from docling.utils.nemotron_parse_utils import (
    extract_nemotron_parse_v2_regions,
    parse_nemotron_parse_v2,
    transform_nemotron_bbox,
)


def test_nemotron_parse_v2_preset() -> None:
    preset = VlmConvertOptions.get_preset("nemotron_parse_v2")

    assert preset.name == "Nemotron Parse 2.0"
    assert preset.scale == 2.0
    assert preset.default_engine_type == VlmEngineType.AUTO_INLINE

    spec = preset.model_spec
    assert spec.default_repo_id == "nvidia/NVIDIA-Nemotron-Parse-2.0"
    assert spec.response_format == ResponseFormat.NEMOTRON_PARSE_V2
    assert spec.trust_remote_code is True
    assert spec.max_new_tokens == 9000
    assert spec.prompt == (
        "</s><s><predict_bbox><predict_classes><output_markdown>"
        "<predict_no_text_in_pic>"
    )
    assert spec.supported_engines == {
        VlmEngineType.TRANSFORMERS,
        VlmEngineType.MLX,
        VlmEngineType.VLLM,
    }


def test_nemotron_parse_v2_engine_configs() -> None:
    spec = VlmConvertOptions.get_preset("nemotron_parse_v2").model_spec

    transformers_config = spec.get_engine_config(VlmEngineType.TRANSFORMERS)
    assert transformers_config.repo_id == "nvidia/NVIDIA-Nemotron-Parse-2.0"
    assert transformers_config.torch_dtype == "bfloat16"
    assert transformers_config.min_engine_version == "5.6.1"
    assert (
        transformers_config.extra_config["transformers_model_type"]
        == TransformersModelType.AUTOMODEL
    )
    assert (
        transformers_config.extra_config["transformers_prompt_style"]
        == TransformersPromptStyle.RAW
    )
    assert transformers_config.extra_config["extra_processor_kwargs"] == {
        "add_special_tokens": False
    }
    assert transformers_config.extra_config["extra_generation_config"] == {
        "repetition_penalty": 1.1,
        "skip_special_tokens": True,
    }

    mlx_config = spec.get_engine_config(VlmEngineType.MLX)
    assert mlx_config.repo_id == "mlx-community/Nemotron-Parse-2.0-8bit"
    assert mlx_config.min_engine_version == "0.6.17"

    vllm_config = spec.get_engine_config(VlmEngineType.VLLM)
    assert vllm_config.repo_id == "nvidia/NVIDIA-Nemotron-Parse-2.0"
    assert vllm_config.min_engine_version == "0.20.0"
    assert vllm_config.extra_config["dtype"] == "bfloat16"
    assert (
        vllm_config.extra_config["transformers_prompt_style"]
        == TransformersPromptStyle.RAW
    )
    assert vllm_config.extra_config["extra_generation_config"] == {
        "repetition_penalty": 1.1,
        "top_k": 1,
        "skip_special_tokens": False,
    }


def test_nemotron_parse_v2_options_can_be_created() -> None:
    options = VlmConvertOptions.from_preset("nemotron_parse_v2")

    assert options.model_spec.default_repo_id == "nvidia/NVIDIA-Nemotron-Parse-2.0"
    assert options.engine_options.engine_type == VlmEngineType.AUTO_INLINE


def test_extract_nemotron_parse_v2_multiline_regions() -> None:
    content = (
        "<x_0.1><y_0.2># A title\non two lines"
        "<x_0.8><y_0.3><class_Title>\n\n"
        "<x_0.2><y_0.4><x_0.5><y_0.7><class_Picture>"
    )

    regions = extract_nemotron_parse_v2_regions(content)

    assert len(regions) == 2
    assert regions[0].label == "Title"
    assert regions[0].text == "# A title\non two lines"
    assert regions[0].bbox == (0.1, 0.2, 0.8, 0.3)
    assert regions[1].label == "Picture"
    assert regions[1].text == ""


def test_transform_nemotron_bbox_undoes_centered_padding() -> None:
    bbox = transform_nemotron_bbox(
        (220 / 1664, 232 / 2048, (220 + 1224) / 1664, (232 + 1584) / 2048),
        inference_image_size=Size(width=1224, height=1584),
        original_page_size=Size(width=612, height=792),
    )

    assert bbox.l == pytest.approx(0)
    assert bbox.t == pytest.approx(0)
    assert bbox.r == pytest.approx(612)
    assert bbox.b == pytest.approx(792)


def test_nemotron_latex_table_is_parsed_through_latex_backend() -> None:
    latex = r"""\begin{tabular}{ccc}
**Model** & **RMSE** & **PI**\\
ANN & 0.1337 & 1.2848\\
SVM & 0.1082 & 1.5087\\
\end{tabular}"""
    content = f"<x_0.1><y_0.2>{latex}<x_0.9><y_0.8><class_Table>"

    document = parse_nemotron_parse_v2(
        content=content,
        original_page_size=Size(width=612, height=792),
        inference_image_size=Size(width=1224, height=1584),
        page_no=1,
    )

    assert len(document.tables) == 1
    table = document.tables[0].data
    assert (table.num_rows, table.num_cols) == (3, 3)
    assert [cell.text for cell in table.table_cells[:3]] == ["Model", "RMSE", "PI"]
    assert all(cell.column_header for cell in table.table_cells[:3])


def test_parse_nemotron_markdown_and_semantic_classes() -> None:
    content = "\n\n".join(
        [
            "<x_0.2><y_0.2># Main **title**<x_0.8><y_0.25><class_Title>",
            "<x_0.2><y_0.3>### 1.2 Method<x_0.5><y_0.35><class_Section-header>",
            (
                "<x_0.2><y_0.4>_Article history:_<br>Received 12 June"
                "<x_0.8><y_0.5><class_Text>"
            ),
            "<x_0.2><y_0.55>First reference<x_0.8><y_0.6><class_Bibliography>",
            "<x_0.2><y_0.65>First item<x_0.8><y_0.7><class_List-item>",
            "<x_0.2><y_0.7>Second item<x_0.8><y_0.75><class_List-item>",
            r"<x_0.2><y_0.76>\(x^2\)<x_0.4><y_0.8><class_Formula>",
            "<x_0.2><y_0.82><x_0.4><y_0.9><class_Picture>",
            (
                "<x_0.45><y_0.82>| x | y |\n| --- | --- |\n| 1 | 2 |"
                "<x_0.8><y_0.9><class_Chart>"
            ),
        ]
    )

    document = parse_nemotron_parse_v2(
        content=content,
        original_page_size=Size(width=612, height=792),
        inference_image_size=Size(width=1224, height=1584),
        page_no=1,
    )

    assert document.texts[0].label == DocItemLabel.TITLE
    assert document.texts[0].text == "Main title"
    assert document.texts[1].label == DocItemLabel.SECTION_HEADER
    assert document.texts[1].level == 3
    assert document.texts[1].text == "1.2 Method"
    assert document.texts[2].text == "Article history:\nReceived 12 June"
    assert any(item.label == DocItemLabel.REFERENCE for item in document.texts)
    assert [
        item.text for item in document.texts if item.label == DocItemLabel.LIST_ITEM
    ] == ["First item", "Second item"]
    assert any(
        item.label == DocItemLabel.FORMULA and item.text == r"\(x^2\)"
        for item in document.texts
    )
    assert len(document.pictures) == 2
    assert document.pictures[1].meta is not None
    assert document.pictures[1].meta.tabular_chart is not None
    assert document.pictures[1].meta.tabular_chart.chart_data.num_cols == 2
    assert all(item.prov for item in document.texts)


def test_nemotron_parse_v2_requires_eager_attention() -> None:
    from docling.models.inference_engines.vlm.transformers_engine import (
        _EAGER_ATTN_REQUIRED_REPO_IDS,
    )

    assert "nvidia/NVIDIA-Nemotron-Parse-2.0" in _EAGER_ATTN_REQUIRED_REPO_IDS
