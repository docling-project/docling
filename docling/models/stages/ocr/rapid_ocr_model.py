import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Literal, Optional, Type, TypedDict

import numpy
from docling_core.types.doc import BoundingBox, CoordOrigin
from docling_core.types.doc.page import BoundingRectangle, TextCell

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    OcrOptions,
    RapidOcrOptions,
)
from docling.datamodel.settings import settings
from docling.models.base_ocr_model import BaseOcrModel
from docling.utils.accelerator_utils import decide_device
from docling.utils.profiling import TimeRecorder
from docling.utils.utils import download_url_with_progress

_log = logging.getLogger(__name__)

_ModelPathEngines = Literal["onnxruntime", "torch"]
_ModelPathTypes = Literal[
    "det_model_path", "cls_model_path", "rec_model_path", "rec_keys_path", "font_path"
]


class _ModelPathDetail(TypedDict):
    url: str
    path: str


class RapidOcrModel(BaseOcrModel):
    _model_repo_folder = "RapidOcr"
    # from https://github.com/RapidAI/RapidOCR/blob/main/python/rapidocr/default_models.yaml
    # matching the default config in https://github.com/RapidAI/RapidOCR/blob/main/python/rapidocr/config.yaml
    # and naming f"{file_info.engine_type.value}.{file_info.ocr_version.value}.{file_info.task_type.value}"
    _default_models: dict[
        _ModelPathEngines, dict[_ModelPathTypes, _ModelPathDetail]
    ] = {
        "onnxruntime": {
            "det_model_path": {
                "url": "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.5.0/onnx/PP-OCRv4/det/ch_PP-OCRv4_det_infer.onnx",
                "path": "onnx/PP-OCRv4/det/ch_PP-OCRv4_det_infer.onnx",
            },
            "cls_model_path": {
                "url": "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.5.0/onnx/PP-OCRv4/cls/ch_ppocr_mobile_v2.0_cls_infer.onnx",
                "path": "onnx/PP-OCRv4/cls/ch_ppocr_mobile_v2.0_cls_infer.onnx",
            },
            "rec_model_path": {
                "url": "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.5.0/onnx/PP-OCRv4/rec/ch_PP-OCRv4_rec_infer.onnx",
                "path": "onnx/PP-OCRv4/rec/ch_PP-OCRv4_rec_infer.onnx",
            },
            "rec_keys_path": {
                "url": "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.5.0/paddle/PP-OCRv4/rec/ch_PP-OCRv4_rec_infer/ppocr_keys_v1.txt",
                "path": "paddle/PP-OCRv4/rec/ch_PP-OCRv4_rec_infer/ppocr_keys_v1.txt",
            },
            "font_path": {
                "url": "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.5.0/resources/fonts/FZYTK.TTF",
                "path": "fonts/FZYTK.TTF",
            },
        },
        "torch": {
            "det_model_path": {
                "url": "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.5.0/torch/PP-OCRv4/det/ch_PP-OCRv4_det_infer.pth",
                "path": "torch/PP-OCRv4/det/ch_PP-OCRv4_det_infer.pth",
            },
            "cls_model_path": {
                "url": "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.5.0/torch/PP-OCRv4/cls/ch_ptocr_mobile_v2.0_cls_infer.pth",
                "path": "torch/PP-OCRv4/cls/ch_ptocr_mobile_v2.0_cls_infer.pth",
            },
            "rec_model_path": {
                "url": "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.5.0/torch/PP-OCRv4/rec/ch_PP-OCRv4_rec_infer.pth",
                "path": "torch/PP-OCRv4/rec/ch_PP-OCRv4_rec_infer.pth",
            },
            "rec_keys_path": {
                "url": "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.5.0/paddle/PP-OCRv4/rec/ch_PP-OCRv4_rec_infer/ppocr_keys_v1.txt",
                "path": "paddle/PP-OCRv4/rec/ch_PP-OCRv4_rec_infer/ppocr_keys_v1.txt",
            },
            "font_path": {
                "url": "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.5.0/resources/fonts/FZYTK.TTF",
                "path": "fonts/FZYTK.TTF",
            },
        },
    }

    def __init__(
        self,
        enabled: bool,
        artifacts_path: Optional[Path],
        options: RapidOcrOptions,
        accelerator_options: AcceleratorOptions,
    ):
        super().__init__(
            enabled=enabled,
            artifacts_path=artifacts_path,
            options=options,
            accelerator_options=accelerator_options,
        )
        self.options: RapidOcrOptions

        self.scale = 3  # multiplier for 72 dpi == 216 dpi.

        if self.enabled:
            try:
                from rapidocr import EngineType, RapidOCR  # type: ignore
            except ImportError:
                raise ImportError(
                    "RapidOCR is not installed. Please install it via `pip install rapidocr onnxruntime` to use this OCR engine. "
                    "Alternatively, Docling has support for other OCR engines. See the documentation."
                )

            # Decide the accelerator devices
            device = decide_device(accelerator_options.device)
            use_cuda = str(AcceleratorDevice.CUDA.value).lower() in device
            use_dml = accelerator_options.device == AcceleratorDevice.AUTO
            intra_op_num_threads = accelerator_options.num_threads
            gpu_id = 0
            if use_cuda and ":" in device:
                gpu_id = int(device.split(":")[1])
            _ALIASES = {
                "onnxruntime": EngineType.ONNXRUNTIME,
                "openvino": EngineType.OPENVINO,
                "paddle": EngineType.PADDLE,
                "torch": EngineType.TORCH,
            }
            backend_enum = _ALIASES.get(self.options.backend, EngineType.ONNXRUNTIME)

            det_model_path = self.options.det_model_path
            cls_model_path = self.options.cls_model_path
            rec_model_path = self.options.rec_model_path
            rec_keys_path = self.options.rec_keys_path
            font_path = self.options.font_path
            if artifacts_path is not None:
                det_model_path = (
                    det_model_path
                    or artifacts_path
                    / self._model_repo_folder
                    / self._default_models[backend_enum.value]["det_model_path"]["path"]
                )
                cls_model_path = (
                    cls_model_path
                    or artifacts_path
                    / self._model_repo_folder
                    / self._default_models[backend_enum.value]["cls_model_path"]["path"]
                )
                rec_model_path = (
                    rec_model_path
                    or artifacts_path
                    / self._model_repo_folder
                    / self._default_models[backend_enum.value]["rec_model_path"]["path"]
                )
                rec_keys_path = (
                    rec_keys_path
                    or artifacts_path
                    / self._model_repo_folder
                    / self._default_models[backend_enum.value]["rec_keys_path"]["path"]
                )
                font_path = (
                    font_path
                    or artifacts_path
                    / self._model_repo_folder
                    / self._default_models[backend_enum.value]["font_path"]["path"]
                )

            for model_path in (
                rec_keys_path,
                cls_model_path,
                rec_model_path,
                rec_keys_path,
                font_path,
            ):
                if model_path is None:
                    continue
                if not Path(model_path).exists():
                    _log.warning(f"The provided model path {model_path} is not found.")

            params = {
                # Global settings (these are still correct)
                "Global.text_score": self.options.text_score,
                "Global.font_path": font_path,
                # "Global.verbose": self.options.print_verbose,
                # Detection model settings
                "Det.model_path": det_model_path,
                "Det.use_cuda": use_cuda,
                "Det.use_dml": use_dml,
                "Det.intra_op_num_threads": intra_op_num_threads,
                # Classification model settings
                "Cls.model_path": cls_model_path,
                "Cls.use_cuda": use_cuda,
                "Cls.use_dml": use_dml,
                "Cls.intra_op_num_threads": intra_op_num_threads,
                # Recognition model settings
                "Rec.model_path": rec_model_path,
                "Rec.font_path": font_path,
                "Rec.rec_keys_path": rec_keys_path,
                "Rec.use_cuda": use_cuda,
                "Rec.use_dml": use_dml,
                "Rec.intra_op_num_threads": intra_op_num_threads,
                "Det.engine_type": backend_enum,
                "Cls.engine_type": backend_enum,
                "Rec.engine_type": backend_enum,
                "EngineConfig.paddle.use_cuda": use_cuda,
                "EngineConfig.paddle.gpu_id": gpu_id,
                "EngineConfig.torch.use_cuda": use_cuda,
                "EngineConfig.torch.gpu_id": gpu_id,
            }

            if self.options.rec_font_path is not None:
                _log.warning(
                    "The 'rec_font_path' option for RapidOCR is deprecated. Please use 'font_path' instead."
                )
            user_params = self.options.rapidocr_params
            if user_params:
                _log.debug("Overwriting RapidOCR params with user-provided values.")
                params.update(user_params)

            self.reader = RapidOCR(
                params=params,
            )

    @staticmethod
    def download_models(
        backend: _ModelPathEngines,
        local_dir: Optional[Path] = None,
        force: bool = False,
        progress: bool = False,
    ) -> Path:
        if local_dir is None:
            local_dir = settings.cache_dir / "models" / RapidOcrModel._model_repo_folder

        local_dir.mkdir(parents=True, exist_ok=True)

        # Download models
        for model_type, model_details in RapidOcrModel._default_models[backend].items():
            output_path = local_dir / model_details["path"]
            if output_path.exists() and not force:
                continue
            output_path.parent.mkdir(exist_ok=True, parents=True)
            buf = download_url_with_progress(model_details["url"], progress=progress)
            with output_path.open("wb") as fw:
                fw.write(buf.read())

        return local_dir

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
                        # Skip zero area boxes
                        if ocr_rect.area() == 0:
                            continue
                        high_res_image = page._backend.get_page_image(
                            scale=self.scale, cropbox=ocr_rect
                        )
                        im = numpy.array(high_res_image)
                        result = self.reader(
                            im,
                            use_det=self.options.use_det,
                            use_cls=self.options.use_cls,
                            use_rec=self.options.use_rec,
                        )
                        if result is None or result.boxes is None:
                            _log.warning("RapidOCR returned empty result!")
                            continue
                        result = list(
                            zip(result.boxes.tolist(), result.txts, result.scores)
                        )

                        del high_res_image
                        del im

                        if result is not None:
                            cells = [
                                TextCell(
                                    index=ix,
                                    text=line[1],
                                    orig=line[1],
                                    confidence=line[2],
                                    from_ocr=True,
                                    rect=BoundingRectangle.from_bounding_box(
                                        BoundingBox.from_tuple(
                                            coord=(
                                                (line[0][0][0] / self.scale)
                                                + ocr_rect.l,
                                                (line[0][0][1] / self.scale)
                                                + ocr_rect.t,
                                                (line[0][2][0] / self.scale)
                                                + ocr_rect.l,
                                                (line[0][2][1] / self.scale)
                                                + ocr_rect.t,
                                            ),
                                            origin=CoordOrigin.TOPLEFT,
                                        )
                                    ),
                                )
                                for ix, line in enumerate(result)
                            ]
                            all_ocr_cells.extend(cells)

                    # Post-process the cells
                    self.post_process_cells(all_ocr_cells, page)

                # DEBUG code:
                if settings.debug.visualize_ocr:
                    self.draw_ocr_rects_and_cells(conv_res, page, ocr_rects)

                yield page

    @classmethod
    def get_options_type(cls) -> Type[OcrOptions]:
        return RapidOcrOptions
