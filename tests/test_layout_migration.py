# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Guards for the migration of layout inference onto the object-detection path.

Covers the contract the deprecated `LayoutOptions` / `LayoutModel` shim owes.
"""

import warnings
from pathlib import Path

import pytest
from docling_core.types.doc import DocItemLabel
from PIL import Image

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.layout_model_specs import (
    DOCLING_LAYOUT_EGRET_LARGE,
    DOCLING_LAYOUT_EGRET_MEDIUM,
    DOCLING_LAYOUT_EGRET_XLARGE,
    DOCLING_LAYOUT_HERON,
    DOCLING_LAYOUT_HERON_101,
    DOCLING_LAYOUT_V2,
    LayoutModelConfig,
)
from docling.datamodel.object_detection_engine_options import (
    MlxObjectDetectionEngineOptions,
)
from docling.datamodel.pipeline_options import (
    LayoutObjectDetectionOptions,
    LayoutOptions,
    PdfPipelineOptions,
)
from docling.datamodel.settings import InferenceSettings, scoped
from docling.models.factories import get_layout_factory
from docling.models.inference_engines.object_detection.base import (
    BaseObjectDetectionEngineOptions,
    ObjectDetectionEngineOutput,
    ObjectDetectionEngineType,
)
from docling.models.inference_engines.object_detection.factory import (
    create_object_detection_engine,
)
from docling.models.inference_engines.object_detection.mlx_engine import (
    MlxObjectDetectionEngine,
)
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
    with scoped(inference=InferenceSettings(compile_torch_models=False)):
        options = PdfPipelineOptions()

    assert isinstance(options.layout_options, LayoutObjectDetectionOptions)
    assert options.layout_options.engine_options.compile_model is False
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


# --------------------------------------------------------------------------
# MLX preset, engine resolution, and label handling
# --------------------------------------------------------------------------


def test_mlx_preset_selects_the_mlx_engine():
    options = LayoutObjectDetectionOptions.from_preset("layout_heron_mlx")

    assert options.engine_options.engine_type == ObjectDetectionEngineType.MLX
    assert (
        options.model_spec.repo_id == "mlx-community/docling-layout-heron-101-mlx-bf16"
    )


def test_engine_options_resolve_from_dict_and_json():
    options = LayoutObjectDetectionOptions.model_validate(
        {"engine_options": {"engine_type": "mlx", "score_threshold": 0.5}}
    )
    assert isinstance(options.engine_options, MlxObjectDetectionEngineOptions)
    assert options.engine_options.score_threshold == 0.5

    restored = LayoutObjectDetectionOptions.model_validate_json(
        options.model_dump_json()
    )
    assert isinstance(restored.engine_options, MlxObjectDetectionEngineOptions)


def test_factory_returns_the_mlx_engine():
    options = LayoutObjectDetectionOptions.from_preset("layout_heron_mlx")

    engine = create_object_detection_engine(
        options=options.engine_options,
        model_spec=options.model_spec,
        accelerator_options=AcceleratorOptions(),
        artifacts_path=None,
    )

    assert isinstance(engine, MlxObjectDetectionEngine)


def test_factory_rejects_base_options_for_mlx():
    with pytest.raises(ValueError, match="MlxObjectDetectionEngineOptions"):
        create_object_detection_engine(
            options=BaseObjectDetectionEngineOptions(
                engine_type=ObjectDetectionEngineType.MLX
            ),
            model_spec=None,
            accelerator_options=AcceleratorOptions(),
            artifacts_path=None,
        )


class _LabelEngine:
    def __init__(self, mapping):
        self._mapping = mapping

    def get_label_mapping(self):
        return self._mapping


def test_build_label_map_normalizes_model_labels():
    model = LayoutObjectDetectionModel.__new__(LayoutObjectDetectionModel)
    model.engine = _LabelEngine({0: "List-item", 1: "Key-Value Region", 2: "text"})

    label_map = model._build_label_map()

    assert label_map[0] is DocItemLabel.LIST_ITEM
    assert label_map[1] is DocItemLabel.KEY_VALUE_REGION
    assert label_map[2] is DocItemLabel.TEXT


def test_build_label_map_rejects_unknown_labels():
    model = LayoutObjectDetectionModel.__new__(LayoutObjectDetectionModel)
    model.engine = _LabelEngine({0: "not-a-label"})

    with pytest.raises(RuntimeError, match="not-a-label"):
        model._build_label_map()


def test_predictions_to_clusters_drops_unknown_labels_and_clamps_boxes():
    from docling.datamodel.base_models import Page, Size

    model = LayoutObjectDetectionModel.__new__(LayoutObjectDetectionModel)
    model._label_map = {0: DocItemLabel.TEXT}
    model._unmapped_label_ids = set()

    page = Page(page_no=1)
    page.size = Size(width=100.0, height=200.0)
    image = Image.new("RGB", (50, 100))
    output = ObjectDetectionEngineOutput(
        label_ids=[0, 99],
        scores=[0.9, 0.8],
        bboxes=[[-10.0, -20.0, 120.0, 240.0], [1.0, 2.0, 3.0, 4.0]],
    )

    clusters = model._predictions_to_clusters(
        page=page, image=image, engine_output=output
    )

    assert len(clusters) == 1
    bbox = clusters[0].bbox
    assert (bbox.l, bbox.t, bbox.r, bbox.b) == (0.0, 0.0, 100.0, 200.0)
    assert 99 in model._unmapped_label_ids
