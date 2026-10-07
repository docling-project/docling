# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import pytest
from docling_core.types.doc import (
    BoundingBox,
    CoordOrigin,
    DocItemLabel,
    DoclingDocument,
    ProvenanceItem,
)
from docling_core.types.doc.labels import CodeLanguageLabel
from PIL import Image

from docling.backend.image_backend import ImageDocumentBackend
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import (
    InputFormat,
    ItemAndImageEnrichmentElement,
    Page,
)
from docling.datamodel.document import ConversionResult, InputDocument
from docling.datamodel.pipeline_options import CodeFormulaVlmOptions
from docling.datamodel.vlm_engine_options import ApiVlmEngineOptions
from docling.models.inference_engines.vlm import (
    BaseVlmEngine,
    VlmEngineInput,
    VlmEngineOutput,
    VlmEngineType,
)
from docling.models.stages.code_formula.code_formula_vlm_model import (
    CodeFormulaVlmModel,
)
from docling.models.utils.generation_utils import TailRepetitionStopper


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


class _ScriptedEngine(BaseVlmEngine):
    """Answers each batch with the given texts and keeps the inputs it was sent."""

    def __init__(self, options, texts: list[str]):
        super().__init__(options)
        self.texts = texts
        self.inputs: list[VlmEngineInput] = []

    def initialize(self) -> None:
        self._initialized = True

    def predict_batch(self, input_batch: list[VlmEngineInput]) -> list[VlmEngineOutput]:
        self.inputs.extend(input_batch)
        return [VlmEngineOutput(text=text) for text in self.texts]


LOOPED = r"A y = \lambda x , \quad ( 9 4 ) " + r"\ " * 300
HEALTHY = r"E = m c ^ { 2 }"


def _enrich(options: CodeFormulaVlmOptions, texts: list[str]):
    """Run the stage on two formulas whose model output is `texts`."""
    model = CodeFormulaVlmModel(
        enabled=False,
        enable_remote_services=False,
        artifacts_path=None,
        options=options,
        accelerator_options=AcceleratorOptions(),
    )
    model.enabled = True
    engine = _ScriptedEngine(options.engine_options, texts)
    model.engine = engine

    doc = DoclingDocument(name="test")
    formulas = [doc.add_text(label=DocItemLabel.FORMULA, text="?") for _ in texts]
    image = Image.new("RGB", (32, 16))
    list(
        model(
            doc,
            [ItemAndImageEnrichmentElement(item=f, image=image) for f in formulas],
        )
    )
    return [f.text for f in formulas], engine.inputs


def _stoppers(engine_input: VlmEngineInput) -> list:
    return engine_input.extra_generation_config.get("custom_stopping_criteria", [])


def test_a_looping_formula_is_cut_back_and_its_neighbour_kept():
    options = CodeFormulaVlmOptions.from_preset("codeformulav2")

    texts, inputs = _enrich(options, [LOOPED, HEALTHY])

    assert texts == [r"A y = \lambda x , \quad ( 9 4 )", HEALTHY]
    assert all(
        any(isinstance(s, TailRepetitionStopper) for s in _stoppers(engine_input))
        for engine_input in inputs
    )


def test_api_engines_are_not_sent_a_stopper_but_their_output_is_cleaned():
    """A stopper would switch the API engine to a streaming request."""
    options = CodeFormulaVlmOptions.from_preset(
        "codeformulav2",
        engine_options=ApiVlmEngineOptions(engine_type=VlmEngineType.API),
    )

    texts, inputs = _enrich(options, [LOOPED, HEALTHY])

    assert texts == [r"A y = \lambda x , \quad ( 9 4 )", HEALTHY]
    assert all(_stoppers(engine_input) == [] for engine_input in inputs)


def test_repetition_handling_can_be_turned_off():
    options = CodeFormulaVlmOptions.from_preset(
        "codeformulav2", stop_on_repetition=False
    )

    texts, inputs = _enrich(options, [LOOPED, HEALTHY])

    assert texts == [LOOPED, HEALTHY]
    assert all(_stoppers(engine_input) == [] for engine_input in inputs)


@pytest.mark.parametrize("expansion_factor", [0.0, 0.08, 0.18])
def test_the_crop_margin_follows_the_option(tmp_path, expansion_factor):
    page_path = tmp_path / "page.png"
    Image.new("RGB", (600, 400), "white").save(page_path)
    in_doc = InputDocument(
        path_or_stream=page_path,
        format=InputFormat.IMAGE,
        backend=ImageDocumentBackend,
    )
    conv_res = ConversionResult(input=in_doc)
    page_backend = in_doc._backend.load_page(0)
    page = Page(page_no=1, size=page_backend.get_size())
    page._backend = page_backend
    conv_res.pages = [page]

    formula = conv_res.document.add_text(
        label=DocItemLabel.FORMULA,
        text="x",
        prov=ProvenanceItem(
            page_no=1,
            bbox=BoundingBox(
                l=100, t=300, r=500, b=250, coord_origin=CoordOrigin.BOTTOMLEFT
            ),
            charspan=(0, 1),
        ),
    )
    options = CodeFormulaVlmOptions.from_preset(
        "codeformulav2", expansion_factor=expansion_factor
    )
    model = CodeFormulaVlmModel(
        enabled=False,
        enable_remote_services=False,
        artifacts_path=None,
        options=options,
        accelerator_options=AcceleratorOptions(),
    )
    model.enabled = True

    element = model.prepare_element(conv_res, formula)

    assert element is not None
    width, height = element.image.size
    scale = model.images_scale
    assert width == pytest.approx(400 * (1 + 2 * expansion_factor) * scale, abs=2)
    assert height == pytest.approx(50 * (1 + 2 * expansion_factor) * scale, abs=2)
