import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Literal, Type, TypedDict

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
_RAPIDOCR_BACKENDS: tuple[_ModelPathEngines, ...] = ("onnxruntime", "torch")


class _ModelPathDetail(TypedDict):
    url: str
    path: str


_RAPIDOCR_MODELSCOPE_RELEASE = "v3.8.0"
_RAPIDOCR_MODELSCOPE_BASE_URL = (
    "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve"
)
_RAPIDOCR_DEFAULT_LANGUAGE = "chinese"
_RAPIDOCR_CHINESE_MODEL_PATHS: dict[_ModelPathEngines, dict[_ModelPathTypes, str]] = {
    "onnxruntime": {
        "det_model_path": "onnx/PP-OCRv4/det/ch_PP-OCRv4_det_mobile.onnx",
        "cls_model_path": "onnx/PP-OCRv4/cls/ch_ppocr_mobile_v2.0_cls_mobile.onnx",
        "rec_model_path": "onnx/PP-OCRv4/rec/ch_PP-OCRv4_rec_mobile.onnx",
        "rec_keys_path": "paddle/PP-OCRv4/rec/ch_PP-OCRv4_rec_mobile/ppocr_keys_v1.txt",
        "font_path": "resources/fonts/FZYTK.TTF",
    },
    "torch": {
        "det_model_path": "torch/PP-OCRv4/det/ch_PP-OCRv4_det_mobile.pth",
        "cls_model_path": "torch/PP-OCRv4/cls/ch_ptocr_mobile_v2.0_cls_mobile.pth",
        "rec_model_path": "torch/PP-OCRv4/rec/ch_PP-OCRv4_rec_mobile.pth",
        "rec_keys_path": "paddle/PP-OCRv4/rec/ch_PP-OCRv4_rec_mobile/ppocr_keys_v1.txt",
        "font_path": "resources/fonts/FZYTK.TTF",
    },
}
_RAPIDOCR_ENGLISH_MODEL_PATHS: dict[_ModelPathEngines, dict[_ModelPathTypes, str]] = {
    "onnxruntime": {
        "det_model_path": "onnx/PP-OCRv4/det/en_PP-OCRv3_det_mobile.onnx",
        "cls_model_path": "onnx/PP-OCRv4/cls/ch_ppocr_mobile_v2.0_cls_mobile.onnx",
        "rec_model_path": "onnx/PP-OCRv4/rec/en_PP-OCRv4_rec_mobile.onnx",
        "rec_keys_path": "paddle/PP-OCRv4/rec/en_PP-OCRv4_rec_mobile/en_dict.txt",
        "font_path": "resources/fonts/FZYTK.TTF",
    },
    "torch": {
        "det_model_path": "torch/PP-OCRv4/det/en_PP-OCRv3_det_mobile.pth",
        "cls_model_path": "torch/PP-OCRv4/cls/ch_ptocr_mobile_v2.0_cls_mobile.pth",
        "rec_model_path": "torch/PP-OCRv4/rec/en_PP-OCRv4_rec_mobile.pth",
        "rec_keys_path": "paddle/PP-OCRv4/rec/en_PP-OCRv4_rec_mobile/en_dict.txt",
        "font_path": "resources/fonts/FZYTK.TTF",
    },
}


_RAPIDOCR_LATIN_MODEL_PATHS: dict[_ModelPathEngines, dict[_ModelPathTypes, str]] = {
    # Latin-script European languages (German, French, Spanish, ...): the
    # latin rec model + dict; detector/classifier mirror the english set.
    "onnxruntime": {
        "det_model_path": "onnx/PP-OCRv4/det/en_PP-OCRv3_det_mobile.onnx",
        "cls_model_path": "onnx/PP-OCRv4/cls/ch_ppocr_mobile_v2.0_cls_mobile.onnx",
        "rec_model_path": "onnx/PP-OCRv4/rec/latin_PP-OCRv3_rec_mobile.onnx",
        "rec_keys_path": "paddle/PP-OCRv4/rec/latin_PP-OCRv3_rec_mobile/latin_dict.txt",
        "font_path": "resources/fonts/FZYTK.TTF",
    },
    "torch": {
        "det_model_path": "torch/PP-OCRv4/det/en_PP-OCRv3_det_mobile.pth",
        "cls_model_path": "torch/PP-OCRv4/cls/ch_ptocr_mobile_v2.0_cls_mobile.pth",
        "rec_model_path": "torch/PP-OCRv4/rec/latin_PP-OCRv3_rec_mobile.pth",
        "rec_keys_path": "paddle/PP-OCRv4/rec/latin_PP-OCRv3_rec_mobile/latin_dict.txt",
        "font_path": "resources/fonts/FZYTK.TTF",
    },
}


def _build_model_detail(path: str) -> _ModelPathDetail:
    return {
        "url": f"{_RAPIDOCR_MODELSCOPE_BASE_URL}/{_RAPIDOCR_MODELSCOPE_RELEASE}/{path}",
        "path": path,
    }


# Maps user-facing language names (ISO 639-1/639-2 codes and English names,
# tesseract-style values included) onto the bundled RapidOCR model sets.
_RAPIDOCR_LANGUAGE_GROUPS: dict[str, str] = {
    # english model set
    "en": "english",
    "eng": "english",
    "english": "english",
    # chinese model set
    "ch": "chinese",
    "chi": "chinese",
    "zh": "chinese",
    "zho": "chinese",
    "chinese": "chinese",
    # latin model set (latin_dict covers most Latin-script European languages)
    "latin": "latin",
    "de": "latin",
    "deu": "latin",
    "ger": "latin",
    "german": "latin",
    "fr": "latin",
    "fra": "latin",
    "fre": "latin",
    "french": "latin",
    "es": "latin",
    "spa": "latin",
    "spanish": "latin",
    "it": "latin",
    "ita": "latin",
    "italian": "latin",
    "pt": "latin",
    "por": "latin",
    "portuguese": "latin",
    "nl": "latin",
    "nld": "latin",
    "dut": "latin",
    "dutch": "latin",
    "fi": "latin",
    "fin": "latin",
    "finnish": "latin",
    "sv": "latin",
    "swe": "latin",
    "swedish": "latin",
    "da": "latin",
    "dan": "latin",
    "danish": "latin",
    "no": "latin",
    "nor": "latin",
    "norwegian": "latin",
    "pl": "latin",
    "pol": "latin",
    "polish": "latin",
    "cs": "latin",
    "ces": "latin",
    "cze": "latin",
    "czech": "latin",
    "ro": "latin",
    "ron": "latin",
    "rum": "latin",
    "romanian": "latin",
    "hu": "latin",
    "hun": "latin",
    "hungarian": "latin",
    "tr": "latin",
    "tur": "latin",
    "turkish": "latin",
    "hr": "latin",
    "hrv": "latin",
    "croatian": "latin",
    "sk": "latin",
    "slk": "latin",
    "slovak": "latin",
    "sl": "latin",
    "slv": "latin",
    "slovenian": "latin",
    "ca": "latin",
    "cat": "latin",
    "catalan": "latin",
    "id": "latin",
    "ind": "latin",
    "indonesian": "latin",
}


def _resolve_rapidocr_language(languages: list[str] | None) -> str:
    """Map requested languages onto a bundled RapidOCR model set.

    Falls back to the default set *loudly*: silently running the Chinese
    recognition model on Latin-script documents drops inter-word spaces
    (see docling issues #2887, #1635, #2927).
    """
    if not languages:
        return _RAPIDOCR_DEFAULT_LANGUAGE

    groups: list[str] = []
    unknown: list[str] = []
    for language in languages:
        # "en-US" / "en_US" -> "en"
        normalized = language.strip().lower().replace("_", "-").split("-")[0]
        group = _RAPIDOCR_LANGUAGE_GROUPS.get(normalized)
        if group is None:
            unknown.append(language)
        else:
            groups.append(group)

    if unknown:
        _log.warning(
            "RapidOCR has no bundled model set for language(s) %s; known values "
            "map onto the 'english', 'latin' or 'chinese' model sets.",
            unknown,
        )
    if not groups:
        _log.warning(
            "Falling back to the '%s' RapidOCR model set; note the Chinese "
            "recognition model drops inter-word spaces in Latin-script text.",
            _RAPIDOCR_DEFAULT_LANGUAGE,
        )
        return _RAPIDOCR_DEFAULT_LANGUAGE

    distinct = set(groups)
    if distinct == {"english"}:
        return "english"
    if distinct <= {"english", "latin"}:
        # the latin set covers English plus other Latin-script languages
        return "latin"
    if len(distinct) == 1:
        return groups[0]
    _log.warning(
        "Requested languages %s span multiple RapidOCR model sets %s; using "
        "'%s' (first requested). Run separate conversions for the others.",
        languages,
        sorted(distinct),
        groups[0],
    )
    return groups[0]


def _rapidocr_lang_type_params(ocr_lang: str) -> dict[str, object]:
    """Language params for the no-pinned-paths flow (no artifacts_path).

    Without explicit model paths RapidOCR resolves models itself — and its
    defaults are the Chinese set regardless of what was requested here, so the
    resolved language must be passed through. Older rapidocr versions without
    the typings module keep their defaults.
    """
    try:
        from rapidocr.utils.typings import LangDet, LangRec  # type: ignore
    except ImportError:
        return {}
    mapping: dict[str, dict[str, object]] = {
        "english": {"Det.lang_type": LangDet.EN, "Rec.lang_type": LangRec.EN},
        "latin": {"Det.lang_type": LangDet.EN, "Rec.lang_type": LangRec.LATIN},
    }
    return mapping.get(ocr_lang, {})


class RapidOcrModel(BaseOcrModel):
    _model_repo_folder = "RapidOcr"
    # from https://github.com/RapidAI/RapidOCR/blob/main/python/rapidocr/default_models.yaml
    # matching the default config in https://github.com/RapidAI/RapidOCR/blob/main/python/rapidocr/config.yaml
    # and naming f"{file_info.engine_type.value}.{file_info.ocr_version.value}.{file_info.task_type.value}"
    _models_by_language: dict[
        str, dict[_ModelPathEngines, dict[_ModelPathTypes, _ModelPathDetail]]
    ] = {
        "chinese": {
            backend: {
                key: _build_model_detail(path)
                for key, path in _RAPIDOCR_CHINESE_MODEL_PATHS[backend].items()
            }
            for backend in _RAPIDOCR_BACKENDS
        },
        "english": {
            backend: {
                key: _build_model_detail(path)
                for key, path in _RAPIDOCR_ENGLISH_MODEL_PATHS[backend].items()
            }
            for backend in _RAPIDOCR_BACKENDS
        },
        "latin": {
            backend: {
                key: _build_model_detail(path)
                for key, path in _RAPIDOCR_LATIN_MODEL_PATHS[backend].items()
            }
            for backend in _RAPIDOCR_BACKENDS
        },
    }
    _default_models: dict[
        _ModelPathEngines, dict[_ModelPathTypes, _ModelPathDetail]
    ] = _models_by_language[_RAPIDOCR_DEFAULT_LANGUAGE]

    def __init__(
        self,
        enabled: bool,
        artifacts_path: Path | None,
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
            backend_key: _ModelPathEngines = "onnxruntime"
            if backend_enum == EngineType.TORCH:
                backend_key = "torch"

            ocr_lang = _resolve_rapidocr_language(self.options.lang)
            model_set = self._models_by_language[ocr_lang][backend_key]

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
                    / model_set["det_model_path"]["path"]
                )
                cls_model_path = (
                    cls_model_path
                    or artifacts_path
                    / self._model_repo_folder
                    / model_set["cls_model_path"]["path"]
                )
                rec_model_path = (
                    rec_model_path
                    or artifacts_path
                    / self._model_repo_folder
                    / model_set["rec_model_path"]["path"]
                )
                rec_keys_path = (
                    rec_keys_path
                    or artifacts_path
                    / self._model_repo_folder
                    / model_set["rec_keys_path"]["path"]
                )
                font_path = (
                    font_path
                    or artifacts_path
                    / self._model_repo_folder
                    / model_set["font_path"]["path"]
                )

            for model_path in (
                det_model_path,
                rec_keys_path,
                cls_model_path,
                rec_model_path,
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
                # Engine-level ONNXRuntime settings
                "EngineConfig.onnxruntime.intra_op_num_threads": intra_op_num_threads,
                # Engine-level OpenVINO settings
                "EngineConfig.openvino.inference_num_threads": intra_op_num_threads,
                # "Global.verbose": self.options.print_verbose,
                # Detection model settings
                "Det.model_path": det_model_path,
                "Det.use_cuda": use_cuda,
                "Det.use_dml": use_dml,
                # Classification model settings
                "Cls.model_path": cls_model_path,
                "Cls.use_cuda": use_cuda,
                "Cls.use_dml": use_dml,
                # Recognition model settings
                "Rec.model_path": rec_model_path,
                "Rec.font_path": font_path,
                "Rec.rec_keys_path": rec_keys_path,
                "Rec.use_cuda": use_cuda,
                "Rec.use_dml": use_dml,
                "Det.engine_type": backend_enum,
                "Cls.engine_type": backend_enum,
                "Rec.engine_type": backend_enum,
                "EngineConfig.paddle.use_cuda": use_cuda,
                "EngineConfig.paddle.gpu_id": gpu_id,
                "EngineConfig.torch.use_cuda": use_cuda,
                "EngineConfig.torch.cuda_ep_cfg.device_id": gpu_id,
            }

            if self.options.rec_font_path is not None:
                _log.warning(
                    "The 'rec_font_path' option for RapidOCR is deprecated. Please use 'font_path' instead."
                )
            if det_model_path is None and rec_model_path is None:
                params.update(_rapidocr_lang_type_params(ocr_lang))

            user_params = self.options.rapidocr_params
            if user_params:
                _log.debug("Overwriting RapidOCR params with user-provided values.")
                params.update(user_params)

            self.reader = RapidOCR(
                params=params,
            )

    @classmethod
    def download_models(
        cls,
        backend: _ModelPathEngines,
        local_dir: Path | None = None,
        force: bool = False,
        progress: bool = False,
        lang: str = "chinese",
    ) -> Path:
        if local_dir is None:
            local_dir = settings.cache_dir / "models" / RapidOcrModel._model_repo_folder

        local_dir.mkdir(parents=True, exist_ok=True)

        # Download models
        resolved_lang = _resolve_rapidocr_language([lang])
        model_set = cls._models_by_language[resolved_lang][backend]
        for model_type, model_details in model_set.items():
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
