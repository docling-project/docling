# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from docling.backend.image_backend import ImageDocumentBackend
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import InputFormat, Page
from docling.datamodel.document import ConversionResult, InputDocument
from docling.datamodel.pipeline_options import OcrMode, RapidOcrOptions
from docling.models.stages.ocr.rapid_ocr_model import RapidOcrModel

pytestmark = pytest.mark.ml_ocr


def _capture_params(
    monkeypatch: pytest.MonkeyPatch,
    options: RapidOcrOptions,
    artifacts_path: Path,
    resolved_device: str = "cpu",
) -> dict[str, object]:
    """Build a RapidOcrModel with real rapidocr resolution but faked inference
    and downloading, returning the params dict handed to RapidOCR.

    artifacts_path is strictly offline, so the checkpoints are prefetched first.
    """
    import rapidocr

    captured: dict[str, object] = {}

    class FakeRapidOCR:
        def __init__(self, *, params):
            captured["params"] = params

    monkeypatch.setattr(rapidocr, "RapidOCR", FakeRapidOCR)
    monkeypatch.setattr(
        "docling.models.stages.ocr.rapid_ocr_model.decide_device",
        lambda device: resolved_device,
    )
    monkeypatch.setattr(
        "docling.models.stages.ocr.rapid_ocr_model.download_url_with_progress",
        lambda url, *, progress: BytesIO(b"dummy content"),
    )
    RapidOcrModel.download_models(
        backend=options.backend,
        lang=options.lang[0],
        local_dir=artifacts_path / RapidOcrModel._model_repo_folder,
    )

    RapidOcrModel(
        enabled=True,
        artifacts_path=artifacts_path,
        options=options,
        accelerator_options=AcceleratorOptions(device="cpu", num_threads=4),
    )
    return captured["params"]


@pytest.mark.parametrize(
    ("backend", "engine_key"),
    [
        ("onnxruntime", "EngineConfig.onnxruntime.intra_op_num_threads"),
        ("openvino", "EngineConfig.openvino.inference_num_threads"),
        ("paddle", "EngineConfig.paddle.cpu_math_library_num_threads"),
    ],
)
def test_rapidocr_num_threads_propagated_per_engine(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    backend: str,
    engine_key: str,
):
    params = _capture_params(monkeypatch, RapidOcrOptions(backend=backend), tmp_path)
    # num_threads must reach the engine actually in use, not only ONNXRuntime.
    assert params[engine_key] == 4


@pytest.mark.parametrize("backend", ["onnxruntime", "paddle", "torch"])
def test_rapidocr_gpu_device_uses_cuda_ep_cfg_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    backend: str,
):
    params = _capture_params(
        monkeypatch,
        RapidOcrOptions(backend=backend),
        tmp_path,
        resolved_device="cuda:2",
    )
    # The GPU device id must use the engine's real key; the legacy top-level
    # `gpu_id` key is not read by RapidOCR (see #3049 for the torch fix).
    assert f"EngineConfig.{backend}.cuda_ep_cfg.device_id" in params
    assert params[f"EngineConfig.{backend}.cuda_ep_cfg.device_id"] == 2
    assert params[f"EngineConfig.{backend}.use_cuda"] is True
    assert f"EngineConfig.{backend}.gpu_id" not in params


def test_rapidocr_pins_explicit_model_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    params = _capture_params(
        monkeypatch, RapidOcrOptions(backend="onnxruntime"), tmp_path
    )
    # Paths are always pinned now, so rapidocr never lazy-resolves models.
    assert params["Det.model_path"] is not None
    assert params["Rec.model_path"] is not None
    assert "Det.lang_type" not in params
    assert "Rec.lang_type" not in params


@pytest.mark.parametrize(
    "result_kind", ["complete", "detection", "no_text", "no_scores"]
)
def test_rapidocr_uses_only_complete_text_results(result_kind: str):
    from rapidocr.ch_ppocr_det.utils import TextDetOutput
    from rapidocr.utils.output import RapidOCROutput

    boxes = np.array([[[0, 0], [12, 0], [12, 6], [0, 6]]])
    if result_kind == "detection":
        result = TextDetOutput(boxes=boxes, scores=[0.9])
    else:
        result = RapidOCROutput(
            boxes=boxes,
            txts=None if result_kind == "no_text" else ("word",),
            scores=None if result_kind == "no_scores" else (0.9,),
        )

    source = BytesIO()
    Image.new("RGB", (20, 20), "white").save(source, format="PNG")
    source.seek(0)
    in_doc = InputDocument(
        path_or_stream=source,
        format=InputFormat.IMAGE,
        backend=ImageDocumentBackend,
        filename="ocr.png",
    )
    conv_res = ConversionResult(input=in_doc)
    page = Page(page_no=1)
    page._backend = in_doc._backend.load_page(0)
    page.size = page._backend.get_size()

    model = RapidOcrModel(
        enabled=False,
        artifacts_path=None,
        options=RapidOcrOptions(mode=OcrMode.FULL_PAGE, scale=2),
        accelerator_options=AcceleratorOptions(),
    )
    model.enabled = True
    model.reader = lambda *args, **kwargs: result

    assert list(model(conv_res, [page])) == [page]
    if result_kind == "complete":
        assert [cell.text for cell in page.cells] == ["word"]
        assert page.cells[0].confidence == 0.9
        assert page.cells[0].rect.to_bounding_box().as_tuple() == (0, 0, 6, 3)
    else:
        assert page.cells == []
