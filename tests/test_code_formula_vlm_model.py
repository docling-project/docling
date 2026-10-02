# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from docling_core.types.doc import DocItemLabel, DoclingDocument
from docling_core.types.doc.labels import CodeLanguageLabel
from PIL import Image

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import ItemAndImageEnrichmentElement
from docling.datamodel.pipeline_options import CodeFormulaVlmOptions
from docling.models.inference_engines.vlm import (
    BaseVlmEngine,
    VlmEngineInput,
    VlmEngineOutput,
)
from docling.models.stages.code_formula.code_formula_vlm_model import (
    CodeFormulaVlmModel,
)


class _FailingEngine(BaseVlmEngine):
    def initialize(self) -> None:
        self._initialized = True

    def predict_batch(self, input_batch: list[VlmEngineInput]) -> list[VlmEngineOutput]:
        raise RuntimeError("CUDA out of memory")


def test_failed_batch_keeps_the_extracted_text():
    options = CodeFormulaVlmOptions.from_preset("codeformulav2")
    model = CodeFormulaVlmModel(
        enabled=False,
        enable_remote_services=False,
        artifacts_path=None,
        options=options,
        accelerator_options=AcceleratorOptions(),
    )
    model.enabled = True
    model.engine = _FailingEngine(options=options.engine_options)

    doc = DoclingDocument(name="test")
    formula = doc.add_text(label=DocItemLabel.FORMULA, text="E = mc^2")
    code = doc.add_code(text="print('hi')", code_language=CodeLanguageLabel.PYTHON)
    image = Image.new("RGB", (32, 16))
    batch = [
        ItemAndImageEnrichmentElement(item=item, image=image)
        for item in (formula, code)
    ]

    enriched = list(model(doc, batch))

    assert enriched == [formula, code]
    assert formula.text == "E = mc^2"
    assert code.text == "print('hi')"
    assert code.code_language == CodeLanguageLabel.PYTHON
