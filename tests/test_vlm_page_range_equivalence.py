# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Exercise range conversion with real PDF loading and deterministic inference."""

from pathlib import Path

import pytest
from docling_core.types.doc import ContentLayer, DoclingDocument

from docling.datamodel.base_models import ConversionStatus, InputFormat, VlmPrediction
from docling.datamodel.pipeline_options import VlmPipelineOptions
from docling.datamodel.pipeline_options_vlm_model import (
    InferenceFramework,
    InlineVlmOptions,
    ResponseFormat,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.models.vlm_pipeline_models.hf_transformers_model import (
    HuggingFaceTransformersVlmModel,
)
from docling.pipeline.vlm_pipeline import VlmPipeline


@pytest.mark.parametrize(
    "response_format", [ResponseFormat.MARKDOWN, ResponseFormat.DOCTAGS]
)
def test_complete_page_ranges_equal_whole_vlm_conversion(monkeypatch, response_format):
    seen_pages = []

    def predict(self, conv_res, pages):
        for page in pages:
            seen_pages.append(page.page_no)
            if response_format == ResponseFormat.MARKDOWN:
                text = f"# Page {page.page_no}\n\n- first\n- second\n\n| A | B |\n|---|---|\n| 1 | 2 |"
            else:
                text = (
                    f"<doctag><page_header><loc_0><loc_0><loc_100><loc_10>Header</page_header>"
                    f"<text><loc_10><loc_10><loc_90><loc_90>Page {page.page_no}</text>"
                    "<key_value_region><loc_10><loc_10><loc_90><loc_90>"
                    "<key_0><loc_10><loc_10><loc_40><loc_40>Page<link_1></key_0>"
                    f"<value_1><loc_50><loc_10><loc_90><loc_40>{page.page_no}</value_1>"
                    "</key_value_region>"
                    "<page_footer><loc_0><loc_90><loc_100><loc_100>Footer</page_footer></doctag>"
                )
            page.predictions.vlm_response = VlmPrediction(text=text)
            yield page

    monkeypatch.setattr(
        HuggingFaceTransformersVlmModel, "__init__", lambda self, **kwargs: None
    )
    monkeypatch.setattr(HuggingFaceTransformersVlmModel, "__call__", predict)
    options = VlmPipelineOptions(
        vlm_options=InlineVlmOptions(
            prompt="",
            repo_id="deterministic-test",
            response_format=response_format,
            inference_framework=InferenceFramework.TRANSFORMERS,
        ),
    )
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=VlmPipeline, pipeline_options=options
            ),
        }
    )
    source = Path(__file__).parent / "data/pdf/sources/2206.01062.pdf"
    whole = converter.convert(source)
    assert whole.status == ConversionStatus.SUCCESS
    page_count = whole.input.page_count
    assert page_count > 2
    seen_pages.clear()
    ranges = [(1, 2), (3, page_count)]
    parts = [converter.convert(source, page_range=span) for span in ranges]
    assert sorted(seen_pages) == list(range(1, page_count + 1))
    for result, (start, end) in zip(parts, ranges):
        assert result.status == ConversionStatus.SUCCESS
        assert sorted(result.document.pages) == list(range(start, end + 1))
        assert all(
            start <= prov.page_no <= end
            for item, _ in result.document.iterate_items(
                included_content_layers=set(ContentLayer)
            )
            for prov in item.prov
        )
        if response_format == ResponseFormat.DOCTAGS:
            assert len(result.document.key_value_items) == end - start + 1
            for item, page_no in zip(
                result.document.key_value_items, range(start, end + 1)
            ):
                assert len(item.graph.cells) == 2
                assert all(
                    cell.prov is not None and cell.prov.page_no == page_no
                    for cell in item.graph.cells
                )
    merged = DoclingDocument.concatenate([result.document for result in parts])
    assert merged.export_to_dict() == whole.document.export_to_dict()
