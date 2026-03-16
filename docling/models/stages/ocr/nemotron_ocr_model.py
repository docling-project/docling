import logging
import platform
import sys
import threading
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
from docling.utils.accelerator_utils import decide_device
from docling.utils.profiling import TimeRecorder

_log = logging.getLogger(__name__)


class _GridSamplerDebugWrapper:
    def __init__(self, original_sampler: Any):
        self._original_sampler = original_sampler
        self._call_seq = 0
        self._lock = threading.Lock()

    @staticmethod
    def _branch_name(grid: Any) -> str:
        shape = getattr(grid, "shape", None)
        if shape is None or len(shape) < 2:
            return "unknown"
        tail = tuple(int(dim) for dim in shape[-2:])
        if tail == (8, 32):
            return "recognizer"
        if tail == (2, 3):
            return "relational"
        return "unknown"

    @staticmethod
    def _describe_tensor(name: str, tensor: Any) -> str:
        shape = getattr(tensor, "shape", None)
        dtype = getattr(tensor, "dtype", None)
        device = getattr(tensor, "device", None)

        is_contiguous: Any
        try:
            is_contiguous = tensor.is_contiguous()
        except Exception as exc:  # pragma: no cover - debug path
            is_contiguous = f"err:{exc}"

        is_meta = getattr(tensor, "is_meta", "n/a")

        stride: Any
        try:
            stride = tuple(int(v) for v in tensor.stride())
        except Exception as exc:  # pragma: no cover - debug path
            stride = f"err:{exc}"

        storage_offset: Any
        try:
            storage_offset = tensor.storage_offset()
        except Exception as exc:  # pragma: no cover - debug path
            storage_offset = f"err:{exc}"

        data_ptr: Any
        try:
            data_ptr = tensor.data_ptr()
        except Exception as exc:  # pragma: no cover - debug path
            data_ptr = f"err:{exc}"

        return (
            f"{name}: type={type(tensor)} shape={shape} dtype={dtype} device={device} "
            f"contiguous={is_contiguous} is_meta={is_meta} stride={stride} "
            f"storage_offset={storage_offset} data_ptr={data_ptr}"
        )

    def __call__(self, input_tensor: Any, grid: Any, input_indices: Any) -> Any:
        with self._lock:
            self._call_seq += 1
            call_id = self._call_seq

        branch_name = self._branch_name(grid)
        print(
            f"[nemotron-debug] grid-sampler-enter call={call_id} branch={branch_name}"
        )
        print(f"[nemotron-debug] {self._describe_tensor('input', input_tensor)}")
        print(f"[nemotron-debug] {self._describe_tensor('grid', grid)}")
        print(
            f"[nemotron-debug] {self._describe_tensor('input_indices', input_indices)}"
        )

        try:
            result = self._original_sampler(input_tensor, grid, input_indices)
            print(
                f"[nemotron-debug] grid-sampler-ok call={call_id} branch={branch_name}"
            )
            return result
        except RuntimeError as exc:
            print(
                f"[nemotron-debug] grid-sampler-failed call={call_id} "
                f"branch={branch_name} error={exc}"
            )

            cloned_input = input_tensor.contiguous().clone()
            cloned_grid = grid.contiguous().clone()
            cloned_input_indices = input_indices.contiguous().clone()

            print(
                f"[nemotron-debug] grid-sampler-retry call={call_id} "
                f"branch={branch_name} mode=contiguous_clone"
            )
            print(
                f"[nemotron-debug] {self._describe_tensor('cloned_input', cloned_input)}"
            )
            print(
                f"[nemotron-debug] {self._describe_tensor('cloned_grid', cloned_grid)}"
            )
            print(
                f"[nemotron-debug] "
                f"{self._describe_tensor('cloned_input_indices', cloned_input_indices)}"
            )

            result = self._original_sampler(
                cloned_input, cloned_grid, cloned_input_indices
            )
            print(
                f"[nemotron-debug] grid-sampler-retry-ok call={call_id} "
                f"branch={branch_name}"
            )
            return result


class NemotronOcrPrediction(TypedDict):
    """Exact prediction schema returned by `nemotron_ocr`."""

    text: str
    confidence: float
    left: float
    upper: float
    right: float
    lower: float


class NemotronOcrModel(BaseOcrModel):
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

            model_dir = (
                str(self.options.model_dir)
                if self.options.model_dir is not None
                else None
            )
            self.reader = NemotronOCR(model_dir=model_dir)
            self.reader.grid_sampler = _GridSamplerDebugWrapper(
                self.reader.grid_sampler
            )
            self._reader_debug_lock = threading.Lock()
            self._active_reader_calls = 0
            self._reader_call_seq = 0

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

                    print(
                        "[nemotron-debug] "
                        f"page={page.page_no} rect_count={len(ocr_rects)} "
                        f"rects={[rect.as_tuple() for rect in ocr_rects]}"
                    )

                    all_ocr_cells = []
                    for crop_index, ocr_rect in enumerate(ocr_rects):
                        if ocr_rect.area() == 0:
                            continue

                        high_res_image = page._backend.get_page_image(
                            scale=self.scale, cropbox=ocr_rect
                        )
                        image_width, image_height = high_res_image.size
                        image_array = numpy.array(high_res_image)

                        print(
                            "[nemotron-debug] "
                            f"page={page.page_no} crop={crop_index} "
                            f"rect={ocr_rect.as_tuple()} "
                            f"size={image_width}x{image_height} "
                            f"shape={image_array.shape} "
                            f"dtype={image_array.dtype} "
                            f"contiguous={image_array.flags.c_contiguous} "
                            f"thread={threading.current_thread().name}:{threading.get_ident()} "
                            f"model_id={id(self)} reader_id={id(self.reader)}"
                        )

                        with self._reader_debug_lock:
                            self._reader_call_seq += 1
                            reader_call_id = self._reader_call_seq
                            self._active_reader_calls += 1
                            active_reader_calls = self._active_reader_calls

                        print(
                            "[nemotron-debug] "
                            f"reader-enter call={reader_call_id} "
                            f"active={active_reader_calls} "
                            f"page={page.page_no} crop={crop_index} "
                            f"thread={threading.current_thread().name}:{threading.get_ident()} "
                            f"model_id={id(self)} reader_id={id(self.reader)}"
                        )

                        try:
                            raw_predictions = cast(
                                Sequence[NemotronOcrPrediction],
                                self.reader(
                                    image_array,
                                    merge_level=self.options.merge_level,
                                    visualize=False,
                                ),
                            )
                        except Exception:
                            print(
                                "[nemotron-debug] "
                                f"FAILED page={page.page_no} crop={crop_index} "
                                f"rect={ocr_rect.as_tuple()} "
                                f"size={image_width}x{image_height} "
                                f"shape={image_array.shape} "
                                f"dtype={image_array.dtype} "
                                f"contiguous={image_array.flags.c_contiguous} "
                                f"reader_call={reader_call_id} "
                                f"thread={threading.current_thread().name}:{threading.get_ident()} "
                                f"model_id={id(self)} reader_id={id(self.reader)}"
                            )
                            raise
                        finally:
                            with self._reader_debug_lock:
                                self._active_reader_calls -= 1
                                active_reader_calls = self._active_reader_calls
                            print(
                                "[nemotron-debug] "
                                f"reader-exit call={reader_call_id} "
                                f"active={active_reader_calls} "
                                f"page={page.page_no} crop={crop_index} "
                                f"thread={threading.current_thread().name}:{threading.get_ident()} "
                                f"model_id={id(self)} reader_id={id(self.reader)}"
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
