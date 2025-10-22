from io import BytesIO
from pathlib import Path
from unittest.mock import Mock

import pytest

from docling.datamodel.accelerator_options import AcceleratorDevice
from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.pipeline_options_vlm_model import (
    InferenceFramework,
    InlineVlmOptions,
    ResponseFormat,
    TransformersPromptStyle,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.models.base_model import BaseVlmPageModel

from .test_data_gen_flag import GEN_TEST_DATA
from .verify_utils import verify_conversion_result_v2

GENERATE = GEN_TEST_DATA


def get_pdf_path():
    pdf_path = Path("./tests/data/pdf/2305.03393v1-pg9.pdf")
    return pdf_path


@pytest.fixture
def converter():
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True
    pipeline_options.accelerator_options.device = AcceleratorDevice.CPU
    pipeline_options.generate_parsed_pages = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=PdfFormatOption().backend,
            )
        }
    )

    return converter


def test_convert_path(converter: DocumentConverter):
    pdf_path = get_pdf_path()
    print(f"converting {pdf_path}")

    # Avoid heavy torch-dependent models by not instantiating layout models here in coverage run
    doc_result = converter.convert(pdf_path)
    verify_conversion_result_v2(
        input_path=pdf_path, doc_result=doc_result, generate=GENERATE
    )


def test_convert_stream(converter: DocumentConverter):
    pdf_path = get_pdf_path()
    print(f"converting {pdf_path}")

    buf = BytesIO(pdf_path.open("rb").read())
    stream = DocumentStream(name=pdf_path.name, stream=buf)

    doc_result = converter.convert(stream)
    verify_conversion_result_v2(
        input_path=pdf_path, doc_result=doc_result, generate=GENERATE
    )


class _DummyVlm(BaseVlmPageModel):
    def __init__(self, prompt_style: TransformersPromptStyle, repo_id: str = ""):  # type: ignore[no-untyped-def]
        self.vlm_options = InlineVlmOptions(
            repo_id=repo_id or "dummy/repo",
            prompt="test prompt",
            inference_framework=InferenceFramework.TRANSFORMERS,
            response_format=ResponseFormat.PLAINTEXT,
            transformers_prompt_style=prompt_style,
        )
        self.processor = Mock()

    def __call__(self, conv_res, page_batch):  # type: ignore[no-untyped-def]
        return []

    def process_images(self, image_batch, prompt):  # type: ignore[no-untyped-def]
        return []


def test_formulate_prompt_raw():
    model = _DummyVlm(TransformersPromptStyle.RAW)
    assert model.formulate_prompt("hello") == "hello"


def test_formulate_prompt_none():
    model = _DummyVlm(TransformersPromptStyle.NONE)
    assert model.formulate_prompt("ignored") == ""


def test_formulate_prompt_phi4_special_case():
    model = _DummyVlm(
        TransformersPromptStyle.RAW, repo_id="ibm-granite/granite-docling-258M"
    )
    # RAW style with granite-docling should still invoke the special path only when style not RAW;
    # ensure RAW returns the user text
    assert model.formulate_prompt("describe image") == "describe image"


def test_formulate_prompt_chat_uses_processor_template():
    model = _DummyVlm(TransformersPromptStyle.CHAT)
    model.processor.apply_chat_template.return_value = "templated"
    out = model.formulate_prompt("summarize")
    assert out == "templated"
    model.processor.apply_chat_template.assert_called()


def test_formulate_prompt_unknown_style_raises():
    # Create an InlineVlmOptions with an invalid enum by patching attribute directly
    model = _DummyVlm(TransformersPromptStyle.RAW)
    model.vlm_options.transformers_prompt_style = "__invalid__"  # type: ignore[assignment]
    with pytest.raises(RuntimeError):
        model.formulate_prompt("x")


def test_vlm_prompt_style_none_and_chat_variants():
    # NONE always empty
    m_none = _DummyVlm(TransformersPromptStyle.NONE)
    assert m_none.formulate_prompt("anything") == ""

    # CHAT path ensures processor used even with complex prompt
    m_chat = _DummyVlm(TransformersPromptStyle.CHAT)
    m_chat.processor.apply_chat_template.return_value = "ok"
    out = m_chat.formulate_prompt("details please")
    assert out == "ok"
