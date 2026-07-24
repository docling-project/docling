import logging
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Type

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

if TYPE_CHECKING:
    from rapidocr.inference_engine.base import FileInfo
    from rapidocr.utils.typings import EngineType, OCRVersion

_log = logging.getLogger(__name__)

# Recognition/detection model size for the PP-OCRv6 path; v4/v5 use "mobile".
_RAPIDOCR_DET_MODEL_LANG = "ch"
_RAPIDOCR_CLS_MODEL_LANG = "ch"
_RAPIDOCR_MODEL_TYPE = "small"
_RAPIDOCR_V4V5_MODEL_TYPE = "mobile"

# Docling's default language names -> rapidocr language codes.
_DOCLING_LANG_NORMALIZE: dict[str, str] = {"chinese": "ch", "english": "en"}

# Recognition languages served by the PP-OCRv4 backbone
_PPOCRV4_LANGS = frozenset(
    {"arabic", "cyrillic", "devanagari", "ka", "korean", "latin", "ta", "te"}
)
# Recognition languages served by the PP-OCRv5 backbone
_PPOCRV5_LANGS = frozenset(
    {
        "arabic",
        "ch",
        "cyrillic",
        "devanagari",
        "el",
        "en",
        "eslav",
        "korean",
        "latin",
        "ta",
        "te",
        "th",
    }
)


def _resolve_rapidocr(lang: list[str], backend: str) -> "tuple[OCRVersion, str]":
    """Map a requested language + backend onto a (PP-OCR version, rec language).

    - Prefer PP-OCRv6 (whose recognizer is multilingual and covers ~52 codes)
    - Otherwise fall back to PP-OCRv4 for the torch backend or PP-OCRv5 for the others.
    - Raises when the language cannot be served by the resolved backbone.
    """
    from rapidocr.utils.model_resolver import COMMON_LANG_ALIASES, PP_OCRV6_LANGS
    from rapidocr.utils.typings import OCRVersion

    langs = lang or ["ch"]
    if len(langs) > 1:
        _log.warning(
            "RapidOCR uses a single language; using %r and ignoring %r.",
            langs[0],
            langs[1:],
        )
    code = langs[0].strip().lower()
    code = _DOCLING_LANG_NORMALIZE.get(code, code)
    aliased = COMMON_LANG_ALIASES.get(code, code)

    if aliased in PP_OCRV6_LANGS:
        return OCRVersion.PPOCRV6, aliased

    if backend == "torch":
        if aliased in _PPOCRV4_LANGS:
            return OCRVersion.PPOCRV4, aliased
        raise ValueError(
            f"RapidOCR torch backend does not support language {langs[0]!r}. "
            f"Supported: {sorted(PP_OCRV6_LANGS | _PPOCRV4_LANGS)}."
        )

    if aliased in _PPOCRV5_LANGS:
        return OCRVersion.PPOCRV5, aliased
    raise ValueError(
        f"RapidOCR {backend} backend does not support language {langs[0]!r}. "
        f"Supported: {sorted(PP_OCRV6_LANGS | _PPOCRV5_LANGS)}."
    )


def _download_if_missing(url: str, dest: Path, *, force: bool, progress: bool) -> Path:
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    buf = download_url_with_progress(url, progress=progress)
    with dest.open("wb") as fw:
        fw.write(buf.read())
    return dest


def _download_rapidocr_model(
    target_dir: Path,
    file_info: "FileInfo",
    engine: "EngineType",
    *,
    force: bool,
    progress: bool,
) -> tuple[Path, Path | None]:
    """Resolve a checkpoint URL from rapidocr's registry and download it.

    Returns the local model path and the recognition-keys path when the entry ships a dict_url
    (v6/v5 onnx embed the charset, so this is None for them).
    """
    from rapidocr.inference_engine.base import InferSession
    from rapidocr.utils.typings import EngineType

    info = InferSession.get_model_url(file_info)
    model_url = info["model_dir"]

    if engine == EngineType.PADDLE:
        # paddle ships a directory bundle; the "model path" is that directory.
        model_url = model_url.rstrip("/")
        model_path = target_dir / Path(model_url).name
        for name, sha in info.items():
            if name in ("model_dir", "dict_url"):
                continue
            _download_if_missing(
                f"{model_url}/{name}",
                model_path / name,
                force=force,
                progress=progress,
            )
    else:
        model_path = target_dir / Path(model_url).name
        _download_if_missing(model_url, model_path, force=force, progress=progress)

    dict_path: Path | None = None
    dict_url = info.get("dict_url")
    if dict_url:
        dict_path = _download_if_missing(
            dict_url, target_dir / Path(dict_url).name, force=force, progress=progress
        )
    return model_path, dict_path


def _ensure_rapidocr_models(
    target_dir: Path,
    engine: "EngineType",
    version: "OCRVersion",
    rec_lang: str,
    *,
    force: bool = False,
    progress: bool = False,
) -> dict[str, Path | None]:
    """Ensure the det/cls/rec checkpoints exist locally, downloading if needed"""
    from rapidocr.inference_engine.base import FileInfo
    from rapidocr.utils.typings import ModelType, OCRVersion, TaskType

    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    size = ModelType(
        _RAPIDOCR_MODEL_TYPE
        if version == OCRVersion.PPOCRV6
        else _RAPIDOCR_V4V5_MODEL_TYPE
    )
    cls_size = ModelType(_RAPIDOCR_V4V5_MODEL_TYPE)

    det_path, _ = _download_rapidocr_model(
        target_dir,
        FileInfo(engine, version, TaskType.DET, _RAPIDOCR_DET_MODEL_LANG, size),
        engine,
        force=force,
        progress=progress,
    )
    cls_path, _ = _download_rapidocr_model(
        target_dir,
        FileInfo(
            engine, OCRVersion.PPOCRV4, TaskType.CLS, _RAPIDOCR_CLS_MODEL_LANG, cls_size
        ),
        engine,
        force=force,
        progress=progress,
    )
    rec_path, rec_keys_path = _download_rapidocr_model(
        target_dir,
        FileInfo(engine, version, TaskType.REC, rec_lang, size),
        engine,
        force=force,
        progress=progress,
    )
    return {
        "det_model_path": det_path,
        "cls_model_path": cls_path,
        "rec_model_path": rec_path,
        "rec_keys_path": rec_keys_path,
    }


class RapidOcrModel(BaseOcrModel):
    _model_repo_folder = "RapidOcr"

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

            ppocr_version, rec_lang = _resolve_rapidocr(
                self.options.lang, self.options.backend
            )

            det_model_path = self.options.det_model_path
            cls_model_path = self.options.cls_model_path
            rec_model_path = self.options.rec_model_path
            rec_keys_path = self.options.rec_keys_path
            font_path = self.options.font_path

            # Auto-resolve/download the model set unless the user pinned explicit
            # detection/recognition paths (mirrors the previous opt-out gate).
            if det_model_path is None and rec_model_path is None:
                if artifacts_path is not None:
                    target_dir = artifacts_path / self._model_repo_folder
                else:
                    target_dir = settings.cache_dir / "models" / self._model_repo_folder
                resolved = _ensure_rapidocr_models(
                    target_dir, backend_enum, ppocr_version, rec_lang
                )
                det_model_path = str(resolved["det_model_path"])
                rec_model_path = str(resolved["rec_model_path"])
                if cls_model_path is None and resolved["cls_model_path"] is not None:
                    cls_model_path = str(resolved["cls_model_path"])
                if rec_keys_path is None and resolved["rec_keys_path"] is not None:
                    rec_keys_path = str(resolved["rec_keys_path"])

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
                "EngineConfig.paddle.cpu_math_library_num_threads": intra_op_num_threads,
                "EngineConfig.paddle.use_cuda": use_cuda,
                "EngineConfig.paddle.cuda_ep_cfg.device_id": gpu_id,
                "EngineConfig.torch.use_cuda": use_cuda,
                "EngineConfig.torch.cuda_ep_cfg.device_id": gpu_id,
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

    @classmethod
    def download_models(
        cls,
        backend: str,
        local_dir: Path | None = None,
        force: bool = False,
        progress: bool = False,
        lang: str = "ch",
    ) -> Path:
        from rapidocr import EngineType  # type: ignore

        if local_dir is None:
            local_dir = settings.cache_dir / "models" / cls._model_repo_folder
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)

        version, rec_lang = _resolve_rapidocr([lang], backend)
        _ensure_rapidocr_models(
            local_dir,
            EngineType(backend),
            version,
            rec_lang,
            force=force,
            progress=progress,
        )
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
                    self.post_process_cells(all_ocr_cells, page, conv_res)

                # DEBUG code:
                if settings.debug.visualize_ocr:
                    self.draw_ocr_rects_and_cells(conv_res, page, ocr_rects)

                yield page

    @classmethod
    def get_options_type(cls) -> Type[OcrOptions]:
        return RapidOcrOptions
