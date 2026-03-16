import logging
import platform
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Optional, Type, TypedDict, cast

import numpy
from docling_core.types.doc import BoundingBox, CoordOrigin
from docling_core.types.doc.page import BoundingRectangle, TextCell

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    NemotronOcrOptions,
    OcrOptions,
)
from docling.datamodel.settings import settings
from docling.models.base_ocr_model import BaseOcrModel
from docling.models.utils.hf_model_download import download_hf_model
from docling.utils.accelerator_utils import decide_device
from docling.utils.profiling import TimeRecorder

_log = logging.getLogger(__name__)


class _GridSamplerStorageWorkaround:
    # Temporary upstream workaround:
    # `nemotron_ocr_cpp.indirect_grid_sample_forward(...)` intermittently
    # raises "Tensor doesn't have storage" for tensors that appear valid from
    # Python. Force fresh owned contiguous storage at the wrapper boundary.
    def __init__(self, original_sampler: Any):
        self._original_sampler = original_sampler

    def __call__(self, input_tensor: Any, grid: Any, input_indices: Any) -> Any:
        # Workaround call site for the upstream custom-op storage bug.
        return self._original_sampler(
            input_tensor.contiguous().clone(),
            grid.contiguous().clone(),
            input_indices.contiguous().clone(),
        )


class NemotronOcrPrediction(TypedDict):
    """Exact prediction schema returned by `nemotron_ocr`."""

    text: str
    confidence: float
    left: float
    upper: float
    right: float
    lower: float


class NemotronOcrModel(BaseOcrModel):
    _repo_id = "nvidia/nemotron-ocr-v1"

    def __init__(
        self,
        enabled: bool,
        artifacts_path: Optional[Path],
        options: NemotronOcrOptions,
        accelerator_options: AcceleratorOptions,
    ):
        super().__init__(
            enabled=enabled,
            artifacts_path=artifacts_path,
            options=options,
            accelerator_options=accelerator_options,
        )
        self.options: NemotronOcrOptions
        self.scale = 3  # multiplier for 72 dpi == 216 dpi.

        if self.enabled:
            self._validate_runtime(accelerator_options=accelerator_options)

            try:
                from nemotron_ocr.inference.pipeline import NemotronOCR
            except ImportError as exc:
                raise ImportError(
                    "Nemotron OCR is not installed. Install the optional dependency "
                    'via `pip install "docling[nemotron-ocr]"` on Linux x86_64 with '
                    "Python 3.12 and CUDA 13.x."
                ) from exc

            model_dir = self._resolve_model_dir(artifacts_path=artifacts_path)
            self.reader = NemotronOCR(
                model_dir=None if model_dir is None else str(model_dir)
            )
            # Install the storage workaround only at the upstream grid-sampler
            # boundary, keeping the rest of the Nemotron integration unchanged.
            self.reader.grid_sampler = _GridSamplerStorageWorkaround(
                self.reader.grid_sampler
            )

    @staticmethod
    def _fail_runtime(message: str) -> None:
        _log.warning(message)
        raise RuntimeError(message)

    @classmethod
    def _validate_runtime(cls, accelerator_options: AcceleratorOptions) -> None:
        if sys.platform != "linux":
            cls._fail_runtime("Nemotron OCR is only supported on Linux.")

        if platform.machine() != "x86_64":
            cls._fail_runtime("Nemotron OCR is only supported on x86_64 machines.")

        if sys.version_info[:2] != (3, 12):
            cls._fail_runtime("Nemotron OCR requires Python 3.12.")

        requested_device = decide_device(accelerator_options.device)
        if not requested_device.startswith("cuda"):
            cls._fail_runtime(
                "Nemotron OCR requires a CUDA accelerator. Set "
                "`pipeline_options.accelerator_options.device` to CUDA or AUTO on a "
                "CUDA-enabled machine."
            )

        import torch

        if not torch.cuda.is_available():
            cls._fail_runtime(
                "Nemotron OCR requires CUDA at initialization time, but "
                "`torch.cuda.is_available()` is false."
            )

        cuda_version = torch.version.cuda
        if cuda_version is None or not cuda_version.startswith("13."):
            cls._fail_runtime(
                "Nemotron OCR requires CUDA 13.x, but the current PyTorch runtime "
                f"reports CUDA {cuda_version!r}."
            )

    @classmethod
    def _resolve_model_dir(cls, artifacts_path: Optional[Path]) -> Optional[Path]:
        if artifacts_path is None:
            return None

        repo_cache_folder = cls._repo_id.replace("/", "--")
        if (artifacts_path / repo_cache_folder).exists():
            return artifacts_path / repo_cache_folder / "checkpoints"

        available_dirs = []
        if artifacts_path.exists():
            available_dirs = sorted(
                path.name for path in artifacts_path.iterdir() if path.is_dir()
            )

        raise FileNotFoundError(
            "Nemotron OCR artifacts not found in artifacts_path.\n"
            f"Expected location: {artifacts_path / repo_cache_folder / 'checkpoints'}\n"
            f"Available directories in {artifacts_path}: {available_dirs}\n"
            "Use `docling-tools models download nemotron_ocr` to pre-download "
            "the checkpoints or unset artifacts_path to allow the upstream "
            "package to download them."
        )

    @staticmethod
    def download_models(
        local_dir: Optional[Path] = None,
        force: bool = False,
        progress: bool = False,
    ) -> Path:
        if local_dir is None:
            local_dir = (
                settings.cache_dir
                / "models"
                / NemotronOcrModel._repo_id.replace("/", "--")
            )

        local_dir.mkdir(parents=True, exist_ok=True)
        return download_hf_model(
            repo_id=NemotronOcrModel._repo_id,
            local_dir=local_dir,
            force=force,
            progress=progress,
        )

    @staticmethod
    def _prediction_to_cell(
        prediction: NemotronOcrPrediction,
        index: int,
        ocr_rect: BoundingBox,
        image_width: int,
        image_height: int,
        scale: int,
    ) -> TextCell:
        # `nemotron_ocr` returns normalized `left/right` and an inverted
        # pair `lower/upper`, where `lower` is the top Y and `upper` is the
        # bottom Y in image coordinates.
        left = (prediction["left"] * image_width) / scale + ocr_rect.l
        top = (prediction["lower"] * image_height) / scale + ocr_rect.t
        right = (prediction["right"] * image_width) / scale + ocr_rect.l
        bottom = (prediction["upper"] * image_height) / scale + ocr_rect.t
        text = prediction["text"]

        return TextCell(
            index=index,
            text=text,
            orig=text,
            from_ocr=True,
            confidence=float(prediction["confidence"]),
            rect=BoundingRectangle.from_bounding_box(
                BoundingBox(
                    l=left,
                    t=top,
                    r=right,
                    b=bottom,
                    coord_origin=CoordOrigin.TOPLEFT,
                )
            ),
        )

    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        if not self.enabled:
            yield from page_batch
            return

        for page in page_batch:
            assert page._backend is not None
            if not page._backend.is_valid():
                yield page
            else:
                with TimeRecorder(conv_res, "ocr"):
                    ocr_rects = self.get_ocr_rects(page)

                    all_ocr_cells = []
                    for ocr_rect in ocr_rects:
                        if ocr_rect.area() == 0:
                            continue

                        high_res_image = page._backend.get_page_image(
                            scale=self.scale, cropbox=ocr_rect
                        )
                        image_width, image_height = high_res_image.size
                        image_array = numpy.array(high_res_image)
                        raw_predictions = cast(
                            Sequence[NemotronOcrPrediction],
                            self.reader(
                                image_array,
                                merge_level=self.options.merge_level,
                                visualize=False,
                            ),
                        )

                        del high_res_image
                        del image_array

                        cells = [
                            self._prediction_to_cell(
                                prediction=prediction,
                                index=index,
                                ocr_rect=ocr_rect,
                                image_width=image_width,
                                image_height=image_height,
                                scale=self.scale,
                            )
                            for index, prediction in enumerate(raw_predictions)
                        ]
                        all_ocr_cells.extend(cells)

                    self.post_process_cells(all_ocr_cells, page)

                if settings.debug.visualize_ocr:
                    self.draw_ocr_rects_and_cells(conv_res, page, ocr_rects)

                yield page

    @classmethod
    def get_options_type(cls) -> Type[OcrOptions]:
        return NemotronOcrOptions
