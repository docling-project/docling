# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import logging
import platform
import sys
import time
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
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
from docling.exceptions import OcrLanguageNotSupportedError
from docling.models.base_ocr_model import BaseOcrModel
from docling.models.utils.hf_model_download import download_hf_model
from docling.utils.accelerator_utils import decide_device
from docling.utils.ocr_language import OcrLanguage, OcrLanguageSupport
from docling.utils.profiling import TimeIntervalRecorder

_log = logging.getLogger(__name__)


_NEMOTRON_OCR_REPO_ID = "nvidia/nemotron-ocr-v2"
_NEMOTRON_OCR_COMMIT = "0e83e83f17943524b90afa6c0fd82ac2bc1a40ca"

_NEMOTRON_OCR_ENGLISH = "english"
_NEMOTRON_OCR_MULTILINGUAL = "multilingual"

# Canonical tags the multilingual recognizer is trained on.
_NEMOTRON_OCR_MULTILINGUAL_TAGS = frozenset(
    {"zh-Hans", "zh-Hant", "ja-Jpan", "ko-Kore", "ru-Cyrl"}
)

# Mappings of nemotron language to the artifacts subdir
_NEMOTRON_CODE_TO_ARTIFACT = {
    _NEMOTRON_OCR_ENGLISH: "v2_english",
    _NEMOTRON_OCR_MULTILINGUAL: "v2_multilingual",
}


def nemotron_ocr_model_dir() -> str:
    return _NEMOTRON_OCR_REPO_ID.replace("/", "--")


class NemotronOcrPrediction(TypedDict):
    """Exact prediction schema returned by `nemotron_ocr`."""

    text: str
    confidence: float
    left: float
    upper: float
    right: float
    lower: float


@dataclass
class _PageOcrState:
    """Tracks the OCR progress of a single page while its rectangles are being
    processed in batches that may span several pages."""

    page: Page
    ocr_rects: list[BoundingBox]
    # Count-down counter of the number of valid OCR rectangles waiting for a prediction
    remaining: int
    # Whether this page should run through OCR post-processing on completion
    # Pages with an invalid backend are passed through untouched
    needs_ocr: bool
    cells: list[TextCell] = field(default_factory=list)
    # Per-page OCR stopwatch. Present but unused for invalid pass-through pages
    recorder: Optional[TimeIntervalRecorder] = None


@dataclass
class _BufferedRect:
    """A single OCR rectangle queued for inference, tied back to its page."""

    state: _PageOcrState
    ocr_rect: BoundingBox
    image: numpy.ndarray
    image_size: tuple[int, int]


class NemotronOcrModel(BaseOcrModel):
    r"""Wrapper for Nvidia's nemotron-ocr-v2 model"""

    language_support = OcrLanguageSupport(multiple_languages=False)

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
        # multiplier for 72 dpi; the default 3.0 == 216 dpi.
        self._scale = self.options.scale

        if self.enabled:
            self.validate_runtime(accelerator_options=accelerator_options)
            self._nemotron_checkpoint_files = []
            try:
                from nemotron_ocr.inference.pipeline import CHECKPOINT_FILES
                from nemotron_ocr.inference.pipeline_v2 import NemotronOCRV2

                self._nemotron_checkpoint_files = CHECKPOINT_FILES
            except ImportError as exc:
                raise ImportError(
                    "Nemotron OCR is not installed. Install the optional dependency "
                    'via `pip install "docling[feat-ocr-nemotron]"` on Linux x86_64 with '
                    "Python 3.12 and CUDA 13.x."
                ) from exc

            # Resolve the request language. An empty `lang` list means "the
            # engine's own default", which for nemotron-OCR is English.
            codes = self.resolve_ocr_languages()
            code = codes[0] if codes else _NEMOTRON_OCR_ENGLISH

            # Initialize the model
            model_dir = self._resolve_model_dir(code, artifacts_path=artifacts_path)

            self._reader = NemotronOCRV2(
                model_dir=None if model_dir is None else str(model_dir),
                lang=code,
            )

    def supported_ocr_languages(self) -> list[str]:
        return ["en-Latn", "mul", *sorted(_NEMOTRON_OCR_MULTILINGUAL_TAGS)]

    def map_ocr_language(self, language: OcrLanguage) -> str:
        if language.is_passthrough:
            # `english`, `multilingual`: nemotron's own recognizer names.
            if language.native in _NEMOTRON_CODE_TO_ARTIFACT:
                assert language.native is not None
                return language.native
        elif language.is_multilingual:
            return _NEMOTRON_OCR_MULTILINGUAL
        else:
            if language.bcp47 == "en-Latn":
                return _NEMOTRON_OCR_ENGLISH
            if language.bcp47 in _NEMOTRON_OCR_MULTILINGUAL_TAGS:
                return _NEMOTRON_OCR_MULTILINGUAL
        raise OcrLanguageNotSupportedError(
            self._engine_name,
            language.tag,
            supported=self.supported_ocr_languages(),
            detail=(
                "nemotron-OCR-v2 ships an English and a multilingual recognizer "
                "only; use 'mul' to run the multilingual one."
            ),
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

    def _resolve_model_dir(
        self, code: str, artifacts_path: Optional[Path]
    ) -> Optional[Path]:
        if artifacts_path is None:
            return None

        nemotron_lang_dir = (
            artifacts_path / nemotron_ocr_model_dir() / _NEMOTRON_CODE_TO_ARTIFACT[code]
        )
        if nemotron_lang_dir.is_dir() and all(
            (nemotron_lang_dir / f).is_file() for f in self._nemotron_checkpoint_files
        ):
            return nemotron_lang_dir

        available_dirs = []
        if artifacts_path.exists():
            available_dirs = sorted(
                path.name for path in artifacts_path.iterdir() if path.is_dir()
            )

        raise FileNotFoundError(
            "Nemotron OCR artifacts not found or incomplete in artifacts_path.\n"
            f"Expected location: {nemotron_lang_dir}\n"
            f"Required files: {self._nemotron_checkpoint_files}\n"
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
        scale: float,
    ) -> TextCell:
        # `nemotron_ocr` returns normalized `left/right` and an inverted pair `lower/upper`
        # where `lower` is the top Y and `upper` is the bottom Y in image coordinates.
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

    def _run_buffer(
        self, conv_res: ConversionResult, buffer: list[_BufferedRect]
    ) -> None:
        r"""Run the model over a buffer of OCR rectangles (possibly spanning
        multiple pages) and attach the resulting cells back to their pages."""
        if not buffer:
            return

        image_arrays = [entry.image for entry in buffer]

        # Measure the inference time and amortize it across the pages
        profile_inference = settings.debug.profile_pipeline_timings
        infer_start = time.monotonic() if profile_inference else 0.0
        batch_predictions = cast(
            Sequence[Sequence[NemotronOcrPrediction]],
            self._reader(
                image_arrays,
                merge_level=self.options.merge_level,
            ),
        )
        if profile_inference:
            per_rect_seconds = (time.monotonic() - infer_start) / len(buffer)
            for entry in buffer:
                if entry.state.recorder is not None:
                    entry.state.recorder.add(per_rect_seconds)

        # Convert the raw predictions to docling's OCR cells and route them to their page
        for entry, raw_predictions in zip(buffer, batch_predictions):
            image_width, image_height = entry.image_size
            cells = [
                NemotronOcrModel._prediction_to_cell(
                    prediction=prediction,
                    index=index,
                    ocr_rect=entry.ocr_rect,
                    image_width=image_width,
                    image_height=image_height,
                    scale=self._scale,
                )
                for index, prediction in enumerate(raw_predictions)
            ]
            entry.state.cells.extend(cells)
            entry.state.remaining -= 1

        buffer.clear()

    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        if not self.enabled:
            yield from page_batch
            return

        # Ensure the "ocr" timing entry always exists, even for documents that produce no rectangles
        TimeIntervalRecorder(conv_res, "ocr")

        # Pages currently in flight, kept in input order so they can be yielded in the same order
        pending: deque[_PageOcrState] = deque()

        def drain_completed() -> Iterable[Page]:
            # Yield pages from the front of the queue while they are fully processed
            while pending and pending[0].remaining == 0:
                state = pending.popleft()
                if state.needs_ocr:
                    assert state.recorder is not None
                    state.recorder.resume()
                    self.post_process_cells(state.cells, state.page, conv_res)
                    state.recorder.pause()
                    # One "ocr" sample per page (rects + image prep + this page's
                    # inference share + post-processing), so count == valid pages.
                    state.recorder.close()
                    if settings.debug.visualize_ocr:
                        self.draw_ocr_rects_and_cells(
                            conv_res, state.page, state.ocr_rects
                        )
                yield state.page

        # OCR rectangles are accumulated across pages so that the model is fed full batches even
        # when individual pages contribute only a few rectangles
        batch_size = max(1, self.options.batch_size)
        buffer: list[_BufferedRect] = []

        for page in page_batch:
            assert page._backend is not None
            if not page._backend.is_valid():
                # Add invalid pages in the queue so output order matches input order
                pending.append(
                    _PageOcrState(page=page, ocr_rects=[], remaining=0, needs_ocr=False)
                )
                yield from drain_completed()
                continue

            recorder = TimeIntervalRecorder(conv_res, "ocr")
            recorder.resume()
            ocr_rects = self.get_ocr_rects(page)
            recorder.pause()
            valid_rects = [ocr_rect for ocr_rect in ocr_rects if ocr_rect.area() != 0]
            state = _PageOcrState(
                page=page,
                ocr_rects=ocr_rects,
                remaining=len(valid_rects),
                needs_ocr=True,
                recorder=recorder,
            )
            pending.append(state)

            for ocr_rect in valid_rects:
                recorder.resume()
                high_res_image = page._backend.get_page_image(
                    scale=self._scale, cropbox=ocr_rect
                )
                buffered_rect = _BufferedRect(
                    state=state,
                    ocr_rect=ocr_rect,
                    image=numpy.array(high_res_image),
                    image_size=high_res_image.size,
                )
                recorder.pause()
                buffer.append(buffered_rect)

                if len(buffer) >= batch_size:
                    self._run_buffer(conv_res, buffer)
                    yield from drain_completed()

            # A page without any valid rectangles is complete right away.
            yield from drain_completed()

        # Flush any remaining rectangles that did not fill a full batch.
        self._run_buffer(conv_res, buffer)
        yield from drain_completed()

    @classmethod
    def get_options_type(cls) -> Type[OcrOptions]:
        return NemotronOcrOptions
