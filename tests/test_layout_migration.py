"""Guards for the migration of layout inference onto the object-detection path.

Covers the contract the deprecated `LayoutOptions` / `LayoutModel` shim owes.
"""

import warnings
from pathlib import Path

import pytest
from docling_core.types.doc import DocItemLabel

from docling.datamodel.layout_model_specs import (
    DOCLING_LAYOUT_EGRET_LARGE,
    DOCLING_LAYOUT_EGRET_MEDIUM,
    DOCLING_LAYOUT_EGRET_XLARGE,
    DOCLING_LAYOUT_HERON,
    DOCLING_LAYOUT_HERON_101,
    DOCLING_LAYOUT_V2,
    LayoutModelConfig,
)
from docling.datamodel.pipeline_options import (
    LayoutObjectDetectionOptions,
    LayoutOptions,
    PdfPipelineOptions,
)
from docling.models.factories import get_layout_factory
from docling.models.stages.layout.layout_model import LayoutModel, _translate
from docling.models.stages.layout.layout_object_detection_model import (
    LayoutObjectDetectionModel,
)

SUPPORTED_SPECS = [
    (DOCLING_LAYOUT_HERON, "layout_heron_default"),
    (DOCLING_LAYOUT_HERON_101, "layout_heron_101"),
    (DOCLING_LAYOUT_EGRET_MEDIUM, "layout_egret_medium"),
    (DOCLING_LAYOUT_EGRET_LARGE, "layout_egret_large"),
    (DOCLING_LAYOUT_EGRET_XLARGE, "layout_egret_xlarge"),
]


def _layout_options(**kwargs) -> LayoutOptions:
    """Construct the deprecated options without tripping the warning filter."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return LayoutOptions(**kwargs)


# --------------------------------------------------------------------------
# The guarantee: LayoutOptions still constructs, resolves and selects models
# --------------------------------------------------------------------------


def test_layout_options_construction_warns():
    with pytest.warns(DeprecationWarning, match="LayoutOptions is deprecated"):
        LayoutOptions()


def test_default_pipeline_uses_object_detection_options():
    options = PdfPipelineOptions()
    assert isinstance(options.layout_options, LayoutObjectDetectionOptions)
    assert (
        options.layout_options.model_spec.repo_id
        == "docling-project/docling-layout-heron"
    )


@pytest.mark.parametrize(("spec", "preset_id"), SUPPORTED_SPECS)
def test_translate_matches_the_equivalent_preset(
    spec: LayoutModelConfig, preset_id: str
):
    """Every surviving model spec translates onto its object-detection preset."""
    translated = _translate(_layout_options(model_spec=spec))
    preset = LayoutObjectDetectionOptions.from_preset(preset_id)

    assert translated.model_spec.repo_id == preset.model_spec.repo_id
    assert translated.model_spec.revision == preset.model_spec.revision
    assert translated.engine_options == preset.engine_options


def test_translate_round_trips_postprocessing_flags():
    options = _layout_options(
        keep_empty_clusters=True,
        skip_cell_assignment=True,
        create_orphan_clusters=False,
    )
    translated = _translate(options)

    assert translated.keep_empty_clusters is True
    assert translated.skip_cell_assignment is True
    assert translated.create_orphan_clusters is False


def test_translate_falls_back_from_retired_v2_model():
    with pytest.warns(DeprecationWarning, match="DOCLING_LAYOUT_V2"):
        translated = _translate(_layout_options(model_spec=DOCLING_LAYOUT_V2))

    assert translated.model_spec.repo_id == DOCLING_LAYOUT_HERON.repo_id


def test_translate_preserves_user_defined_model_configs():
    """User-defined LayoutModelConfig instances keep selecting their own repo."""
    custom = LayoutModelConfig(name="custom", repo_id="acme/layout", revision="v1")
    translated = _translate(_layout_options(model_spec=custom))

    assert translated.model_spec.repo_id == "acme/layout"
    assert translated.model_spec.revision == "v1"


# --------------------------------------------------------------------------
# Factory dispatch: both option types must still resolve to a layout model
# --------------------------------------------------------------------------


def test_factory_dispatches_both_option_types():
    registered = get_layout_factory(allow_external_plugins=False).classes

    assert registered[LayoutOptions] is LayoutModel
    assert registered[LayoutObjectDetectionOptions] is LayoutObjectDetectionModel
    assert issubclass(LayoutModel, LayoutObjectDetectionModel)


# --------------------------------------------------------------------------
# Batching: `layout_batch_size` only means something if the engine sees a batch
# --------------------------------------------------------------------------


class _RecordingEngine:
    """Engine stub that records the batch shapes it is handed."""

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def get_label_mapping(self):
        return {0: "text"}

    def predict_batch(self, input_batch):
        from docling.models.inference_engines.object_detection import (
            ObjectDetectionEngineOutput,
        )

        self.batch_sizes.append(len(input_batch))
        return [ObjectDetectionEngineOutput() for _ in input_batch]


class _StubBackend:
    """Minimal PdfPageBackend stand-in: validity plus a rendered page image."""

    def __init__(self, valid: bool, image) -> None:
        self._valid = valid
        self._image = image

    def is_valid(self) -> bool:
        return self._valid

    def get_page_image(self, scale: float = 1.0, cropbox=None):
        return self._image


def _page(page_no: int, *, valid: bool = True, with_image: bool = True):
    from PIL import Image as PILImage

    from docling.datamodel.base_models import Page, Size

    page = Page(page_no=page_no)
    page.size = Size(width=612.0, height=792.0)
    image = PILImage.new("RGB", (612, 792)) if with_image else None
    page._backend = _StubBackend(valid, image)
    return page


def test_predict_layout_issues_a_single_batched_call():
    from types import SimpleNamespace

    from docling.datamodel.base_models import ConfidenceReport

    model = LayoutObjectDetectionModel.__new__(LayoutObjectDetectionModel)
    engine = _RecordingEngine()
    model.engine = engine
    model._label_map = model._build_label_map()
    model._unmapped_label_ids = set()

    pages = [_page(1), _page(2), _page(3)]
    conv_res = SimpleNamespace(confidence=ConfidenceReport(), timings={})

    predictions = model.predict_layout(conv_res, pages)

    assert engine.batch_sizes == [3]
    assert len(predictions) == 3
    assert all(page.predictions.layout is not None for page in pages)


def test_predict_layout_skips_unusable_pages_without_shifting_results():
    from types import SimpleNamespace

    from docling.datamodel.base_models import ConfidenceReport

    model = LayoutObjectDetectionModel.__new__(LayoutObjectDetectionModel)
    engine = _RecordingEngine()
    model.engine = engine
    model._label_map = model._build_label_map()
    model._unmapped_label_ids = set()

    pages = [_page(1), _page(2, valid=False), _page(3, with_image=False), _page(4)]
    conv_res = SimpleNamespace(confidence=ConfidenceReport(), timings={})

    predictions = model.predict_layout(conv_res, pages)

    assert engine.batch_sizes == [2]
    assert len(predictions) == len(pages)
    for page, prediction in zip(pages, predictions):
        assert page.predictions.layout is prediction


# --------------------------------------------------------------------------
# Picture-internal text: the regression that barely moves default markdown
# --------------------------------------------------------------------------


@pytest.mark.ml_pdf_model
def test_pictures_keep_their_non_caption_text_children():
    """Orphan clusters and unclamped boxes both show up as lost picture text.

    A `create_orphan_clusters` or bbox-clamping regression strips most text
    nodes out of pictures while moving the default markdown by ~0.2%, so the
    end-to-end markdown diff alone does not catch it.
    """
    from docling_core.types.doc import PictureItem, TextItem

    from docling.datamodel.accelerator_options import AcceleratorDevice
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions()
    options.do_ocr = False
    options.accelerator_options.device = AcceleratorDevice.CPU

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )
    source = Path(__file__).parent / "data/pdf/sources/amt_handbook_sample.pdf"
    doc = converter.convert(source).document

    pictures = [
        item for item, _ in doc.iterate_items() if isinstance(item, PictureItem)
    ]
    assert pictures, "expected the sample to contain pictures"

    text_children = [
        item
        for picture in pictures
        for item, _ in doc.iterate_items(root=picture, traverse_pictures=True)
        if isinstance(item, TextItem) and item.label != DocItemLabel.CAPTION
    ]
    assert text_children, "pictures lost all of their non-caption text descendants"


# --------------------------------------------------------------------------
# The point of the whole exercise
# --------------------------------------------------------------------------


def test_no_layout_model_imports_docling_ibm_models():
    package_root = Path(__file__).resolve().parents[1] / "docling"
    offenders = [
        str(path.relative_to(package_root.parent))
        for path in package_root.rglob("*.py")
        if "docling_ibm_models.layoutmodel" in path.read_text()
    ]
    assert offenders == []
