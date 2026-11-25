# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest
from docling_core.types.doc import CodeItem, TextItem
from docling_core.types.doc.labels import CodeLanguageLabel, DocItemLabel

from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline

pytestmark = pytest.mark.ml_pdf_model


def get_converter():
    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_page_images = True

    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = False
    pipeline_options.do_code_enrichment = True
    pipeline_options.do_formula_enrichment = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=StandardPdfPipeline,
                pipeline_options=pipeline_options,
            )
        }
    )

    return converter


def test_code_and_formula_conversion():
    pdf_path = Path("tests/data/pdf/sources/code_and_formula.pdf")
    converter = get_converter()

    print(f"converting {pdf_path}")

    doc_result: ConversionResult = converter.convert(pdf_path)

    results = doc_result.document.texts

    code_blocks = [el for el in results if isinstance(el, CodeItem)]
    assert len(code_blocks) == 1

    gt = "function add(a, b) {\n    return a + b;\n}\nconsole.log(add(3, 5));"

    predicted = code_blocks[0].text.strip()
    assert predicted == gt, f"mismatch in text {predicted=}, {gt=}"
    assert code_blocks[0].code_language == CodeLanguageLabel.JAVASCRIPT

    formula_blocks = [
        el
        for el in results
        if isinstance(el, TextItem) and el.label == DocItemLabel.FORMULA
    ]
    assert len(formula_blocks) == 1

    gt = "a ^ { 2 } + 8 = 1 2"
    predicted = formula_blocks[0].text
    assert predicted == gt, f"mismatch in text {predicted=}, {gt=}"


def test_formula_conversion_with_page_range():
    pdf_path = Path("tests/data/pdf/sources/code_and_formula.pdf")
    converter = get_converter()

    print(f"converting {pdf_path} with page range")

    doc_result: ConversionResult = converter.convert(pdf_path, page_range=(2, 2))

    results = doc_result.document.texts

    formula_blocks = [
        el
        for el in results
        if isinstance(el, TextItem) and el.label == DocItemLabel.FORMULA
    ]
    assert len(formula_blocks) == 1

    gt = "a ^ { 2 } + 8 = 1 2"
    predicted = formula_blocks[0].text
    assert predicted == gt, f"mismatch in text {predicted=}, {gt=}"


def test_code_formula_model_resolves_dtype_for_device(monkeypatch, tmp_path):
    """CodeFormulaV2 is not loaded in emulated bfloat16 on CPUs without it."""
    import torch
    import transformers

    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.models.stages.code_formula.code_formula_model import (
        CodeFormulaModel,
        CodeFormulaModelOptions,
    )
    from docling.utils.accelerator_utils import transformers_dtype_kwargs

    captured_kwargs = {}

    class FakeModel:
        def eval(self):
            return None

    def fake_model_from_pretrained(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return FakeModel()

    class FakeTokenizer:
        model_max_length = 8192

    class FakeProcessor:
        tokenizer = FakeTokenizer()

    monkeypatch.setattr(torch.backends.cpu, "get_cpu_capability", lambda: "AVX2")
    monkeypatch.setattr(
        transformers.AutoProcessor,
        "from_pretrained",
        lambda *args, **kwargs: FakeProcessor(),
    )
    monkeypatch.setattr(
        transformers.AutoModelForImageTextToText,
        "from_pretrained",
        fake_model_from_pretrained,
    )

    CodeFormulaModel(
        enabled=True,
        artifacts_path=tmp_path,
        options=CodeFormulaModelOptions(),
        accelerator_options=AcceleratorOptions(device="cpu"),
    )

    (dtype_arg_name,) = transformers_dtype_kwargs(None)
    assert captured_kwargs[dtype_arg_name] == "float32"
