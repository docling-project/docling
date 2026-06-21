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


_NEMOTRON_OCR_REPO_ID = "nvidia/nemotron-ocr-v2"
_NEMOTRON_OCR_COMMIT = "0e83e83f17943524b90afa6c0fd82ac2bc1a40ca"

_NEMOTRON_OCR_ENGLISH = "english"
_NEMOTRON_OCR_MULTILINGUAL = "multilingual"
_NEMOTRON_OCR_ENGLISH_GROUP = ["en", "eng", "english"]

# Mappings of nemotron language to the artifacts subdir
_NEMOTRON_OCR_LANG_TO_ARTIFACT_PATHS = {
    _NEMOTRON_OCR_ENGLISH: "v2_english",
    _NEMOTRON_OCR_MULTILINGUAL: "v2_multilingual",
}


def nemotron_ocr_model_dir() -> str:
    return _NEMOTRON_OCR_REPO_ID.replace("/", "--")


def _resolve_nemotronocr_language(req_languages: list[str] | None) -> str:
    r"""
    Map requested languages onto the nemotron-ocr language info
    """
    if not req_languages:
        # Use english by default
        return _NEMOTRON_OCR_ENGLISH

    # Map request language to nemotron language
    for language in req_languages:
        # "en-US" / "en_US" -> "en"
        normalized = language.strip().lower().replace("_", "-").split("-")[0]

        # Use the multilingual model to cover english and any non-english language
        if normalized not in _NEMOTRON_OCR_ENGLISH_GROUP:
            return _NEMOTRON_OCR_MULTILINGUAL
    return _NEMOTRON_OCR_ENGLISH


class NemotronOcrPrediction(TypedDict):
    """Exact prediction schema returned by `nemotron_ocr`."""

    text: str
    confidence: float
    left: float
    upper: float
    right: float
    lower: float


class NemotronOcrModel(BaseOcrModel):
    r"""Wrapper for Nvidia's nemotron-ocr-v2 model"""

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
            self.validate_runtime(accelerator_options=accelerator_options)

            try:
                from nemotron_ocr.inference.pipeline_v2 import NemotronOCRV2
            except ImportError as exc:
                raise ImportError(
                    "Nemotron OCR is not installed. Install the optional dependency "
                    'via `pip install "docling[feat-ocr-nemotron]"` on Linux x86_64 with '
                    "Python 3.12 and CUDA 13.x."
                ) from exc

            # Resolve the request language
            language = _resolve_nemotronocr_language(options.lang)

            # Initialize the model
            model_dir = NemotronOcrModel._resolve_model_dir(
                language, artifacts_path=artifacts_path
            )

            self.reader = NemotronOCRV2(
                model_dir=None if model_dir is None else str(model_dir),
                lang=language,
            )

    @staticmethod
    def _fail_runtime(message: str) -> None:
        _log.error(message)
        raise RuntimeError(message)

    @classmethod
    def validate_runtime(cls, accelerator_options: AcceleratorOptions) -> None:
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

    @staticmethod
    def _resolve_model_dir(
        language: str, artifacts_path: Optional[Path]
    ) -> Optional[Path]:
        if artifacts_path is None:
            return None

        nemotron_lang_dir = (
            artifacts_path
            / nemotron_ocr_model_dir()
            / _NEMOTRON_OCR_LANG_TO_ARTIFACT_PATHS[language]
        )
        if nemotron_lang_dir.exists():
            return nemotron_lang_dir

        available_dirs = []
        if artifacts_path.exists():
            available_dirs = sorted(
                path.name for path in artifacts_path.iterdir() if path.is_dir()
            )

        raise FileNotFoundError(
            "Nemotron OCR artifacts not found in artifacts_path.\n"
            f"Expected location: {nemotron_lang_dir}\n"
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
            local_dir = settings.cache_dir / "models" / nemotron_ocr_model_dir()

        local_dir.mkdir(parents=True, exist_ok=True)

        # The next command downloads the entire HF repo that contains artifacts for all languages
        return download_hf_model(
            repo_id=_NEMOTRON_OCR_REPO_ID,
            local_dir=local_dir,
            force=force,
            progress=progress,
            revision=_NEMOTRON_OCR_COMMIT,
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

                    # Process the OCR rectangles in batches of at most
                    # `batch_size` images. `self.reader` accepts a list of
                    # images and returns one prediction list per image.
                    all_ocr_cells = []
                    valid_rects = [
                        ocr_rect for ocr_rect in ocr_rects if ocr_rect.area() != 0
                    ]
                    batch_size = max(1, self.options.batch_size)

                    for batch_start in range(0, len(valid_rects), batch_size):
                        batch_rects = valid_rects[
                            batch_start : batch_start + batch_size
                        ]

                        image_arrays = []
                        # Image dimensions parallel to `batch_rects`, needed to
                        # map normalized predictions back to page coordinates.
                        image_sizes = []
                        for ocr_rect in batch_rects:
                            high_res_image = page._backend.get_page_image(
                                scale=self.scale, cropbox=ocr_rect
                            )
                            image_arrays.append(numpy.array(high_res_image))
                            image_sizes.append(high_res_image.size)

                        # Run the model to get the raw predictions
                        batch_predictions = cast(
                            Sequence[Sequence[NemotronOcrPrediction]],
                            self.reader(
                                image_arrays,
                                merge_level=self.options.merge_level,
                            ),
                        )

                        # Convert the raw predictions to docling's OCR cells
                        for ocr_rect, (
                            image_width,
                            image_height,
                        ), raw_predictions in zip(
                            batch_rects, image_sizes, batch_predictions
                        ):
                            cells = [
                                NemotronOcrModel._prediction_to_cell(
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
