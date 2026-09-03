# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import logging
from collections.abc import Iterable
from dataclasses import dataclass
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
    from rapidocr.utils.typings import EngineType, OCRVersion

_log = logging.getLogger(__name__)

_RAPIDOCR_DEFAULT_LANGUAGE = "ch"

# Recognition/detection model size for the PP-OCRv6 path; v4/v5 use "mobile".
_RAPIDOCR_DET_MODEL_LANG = "ch"
_RAPIDOCR_CLS_MODEL_LANG = "ch"
_RAPIDOCR_MODEL_TYPE = "small"
_RAPIDOCR_V4V5_MODEL_TYPE = "mobile"

# Docling's default language names -> rapidocr language codes.
_DOCLING_LANG_NORMALIZE: dict[str, str] = {"chinese": "ch", "english": "en"}

# Inference backends docling supports. Must stay in sync with the `backend` Literal of
# RapidOcrOptions in docling/datamodel/pipeline_options.py and with the mapping built in
# _backend_to_engine_type().
_RAPIDOCR_BACKENDS: tuple[str, ...] = ("onnxruntime", "openvino", "paddle", "torch")

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


@dataclass(frozen=True)
class _RapidOcrArtifact:
    """One resolved checkpoint: where it lives locally and where it comes from"""

    # What gets handed to RapidOCR: the checkpoint file, or its directory for paddle.
    model_path: Path

    # Every local file the checkpoint needs, mapped to the URL it is downloaded from.
    files: dict[Path, str]

    # Recognition-keys file, for the entries that ship one (v6/v5 onnx embed the charset).
    dict_path: Path | None


@dataclass(frozen=True)
class _RapidOcrModelSpec:
    """One RapidOCR checkpoint set, as requested or as resolved"""

    # Docling inference backend name, one of _RAPIDOCR_BACKENDS.
    backend: str

    # Language exactly as the user wrote it
    user_lang: str | None = None

    # Language code the rapidocr registry expects, after normalization and aliasing.
    rapidocr_lang_token: str | None = None

    # PP-OCR backbone that the (backend, language) pair resolves to.
    ppocr_version: "OCRVersion | None" = None


def _parse_rapidocr_model_spec(value: str) -> _RapidOcrModelSpec:
    """Parse a `<backend>:<lang>` prefetch spec into its requested form.

    The pair is routed through _resolve_rapidocr so the prefetcher can never accept a
    combination the runtime would reject, but only the user's own values are kept.
    """
    backend, separator, lang = value.partition(":")
    if not separator or not backend or not lang or ":" in lang:
        raise ValueError(
            f"Invalid RapidOCR model spec {value!r}. "
            "Expected '<backend>:<lang>', e.g. 'onnxruntime:th'."
        )
    if backend not in _RAPIDOCR_BACKENDS:
        raise ValueError(
            f"Unknown RapidOCR backend {backend!r} in {value!r}. "
            f"Supported: {list(_RAPIDOCR_BACKENDS)}."
        )
    try:
        _resolve_rapidocr(lang, backend)
    except ValueError as err:
        raise ValueError(f"Invalid RapidOCR model spec {value!r}: {err}") from err
    return _RapidOcrModelSpec(backend=backend, user_lang=lang)


def _backend_to_engine_type(backend: str) -> "EngineType":
    """Map a docling backend name onto the rapidocr EngineType it stands for."""
    from rapidocr.utils.typings import EngineType

    engine_types = {
        "onnxruntime": EngineType.ONNXRUNTIME,
        "openvino": EngineType.OPENVINO,
        "paddle": EngineType.PADDLE,
        "torch": EngineType.TORCH,
    }
    if backend not in engine_types:
        raise ValueError(
            f"Unknown RapidOCR backend {backend!r}. Supported: {list(_RAPIDOCR_BACKENDS)}."
        )
    return engine_types[backend]


def _resolve_rapidocr(lang: str, backend: str) -> _RapidOcrModelSpec:
    """Map one requested language + backend onto a fully populated _RapidOcrModelSpec.

    - Prefer PP-OCRv6 (whose recognizer is multilingual and covers ~52 codes)
    - Otherwise fall back to PP-OCRv4 for the torch backend or PP-OCRv5 for the others.
    - Raises when the language cannot be served by the resolved backbone.

    Callers pass a single language; reducing a multi-language request is up to them.
    """
    from rapidocr.utils.model_resolver import COMMON_LANG_ALIASES, PP_OCRV6_LANGS
    from rapidocr.utils.typings import OCRVersion

    code = lang.strip().lower()
    code = _DOCLING_LANG_NORMALIZE.get(code, code)
    aliased = COMMON_LANG_ALIASES.get(code, code)

    if aliased in PP_OCRV6_LANGS:
        version = OCRVersion.PPOCRV6
    elif backend == "torch":
        if aliased not in _PPOCRV4_LANGS:
            raise ValueError(
                f"RapidOCR torch backend does not support language {lang!r}. "
                f"Supported: {sorted(PP_OCRV6_LANGS | _PPOCRV4_LANGS)}."
            )
        version = OCRVersion.PPOCRV4
    elif aliased in _PPOCRV5_LANGS:
        version = OCRVersion.PPOCRV5
    else:
        raise ValueError(
            f"RapidOCR {backend} backend does not support language {lang!r}. "
            f"Supported: {sorted(PP_OCRV6_LANGS | _PPOCRV5_LANGS)}."
        )

    _log.debug(
        "RapidOCR resolved lang=%r backend=%r -> version=%s rec_lang=%r",
        lang,
        backend,
        version.value,
        aliased,
    )
    return _RapidOcrModelSpec(
        backend=backend,
        user_lang=lang,
        rapidocr_lang_token=aliased,
        ppocr_version=version,
    )


def _rapidocr_artifacts(
    target_dir: Path,
    engine: "EngineType",
    version: "OCRVersion",
    rec_lang: str,
    *,
    need_det: bool = True,
    need_cls: bool = True,
    need_rec: bool = True,
) -> dict[str, _RapidOcrArtifact]:
    """Resolve the det/cls/rec checkpoints for RapidOCR keyed by their task.

    This is a pure registry lookup: no network and no filesystem I/O
    """
    from rapidocr.inference_engine.base import FileInfo, InferSession
    from rapidocr.utils.typings import EngineType, ModelType, OCRVersion, TaskType

    size = ModelType(
        _RAPIDOCR_MODEL_TYPE
        if version == OCRVersion.PPOCRV6
        else _RAPIDOCR_V4V5_MODEL_TYPE
    )
    cls_size = ModelType(_RAPIDOCR_V4V5_MODEL_TYPE)

    file_infos: dict[str, FileInfo] = {}
    if need_det:
        file_infos["det"] = FileInfo(
            engine, version, TaskType.DET, _RAPIDOCR_DET_MODEL_LANG, size
        )
    if need_cls:
        file_infos["cls"] = FileInfo(
            engine, OCRVersion.PPOCRV4, TaskType.CLS, _RAPIDOCR_CLS_MODEL_LANG, cls_size
        )
    if need_rec:
        file_infos["rec"] = FileInfo(engine, version, TaskType.REC, rec_lang, size)

    artifacts: dict[str, _RapidOcrArtifact] = {}
    for task, file_info in file_infos.items():
        # Use RapidOCR's InferSession to get the URL for that FileInfo
        info = InferSession.get_model_url(file_info)
        model_url = info["model_dir"]

        files: dict[Path, str] = {}
        if engine == EngineType.PADDLE:
            # paddle ships a directory bundle; the "model path" is that directory.
            model_url = model_url.rstrip("/")
            model_path = target_dir / Path(model_url).name
            for name in info:
                if name in ("model_dir", "dict_url"):
                    continue
                files[model_path / name] = f"{model_url}/{name}"
        else:
            model_path = target_dir / Path(model_url).name
            files[model_path] = model_url

        dict_path: Path | None = None
        dict_url = info.get("dict_url")
        if dict_url:
            # v6/v5 onnx embed the charset, so only some entries ship a keys file.
            dict_path = target_dir / Path(dict_url).name
            files[dict_path] = dict_url

        artifacts[task] = _RapidOcrArtifact(
            model_path=model_path, files=files, dict_path=dict_path
        )
    return artifacts


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

        # multiplier for 72 dpi; the default 3.0 == 216 dpi.
        self.scale = self.options.scale

        if self.enabled:
            try:
                from rapidocr import ModelType, OCRVersion, RapidOCR  # type: ignore
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
            backend_enum = _backend_to_engine_type(self.options.backend)

            # Reduce the user provided language list to one language
            lang = (
                self.options.lang[0]
                if self.options.lang
                else _RAPIDOCR_DEFAULT_LANGUAGE
            )
            if len(self.options.lang) > 1:
                _log.warning(
                    "RapidOCR uses a single language; using %r and ignoring %r.",
                    lang,
                    self.options.lang[1:],
                )
            resolved: _RapidOcrModelSpec = _resolve_rapidocr(lang, self.options.backend)
            assert resolved.ppocr_version is not None
            assert resolved.rapidocr_lang_token is not None
            ppocr_version = resolved.ppocr_version
            rec_lang = resolved.rapidocr_lang_token

            det_model_path = self.options.det_model_path
            cls_model_path = self.options.cls_model_path
            rec_model_path = self.options.rec_model_path
            rec_keys_path = self.options.rec_keys_path
            font_path = self.options.font_path

            # A pinned path that does not exist is a configuration error
            missing_pinned = [
                model_path
                for model_path in (
                    det_model_path,
                    cls_model_path,
                    rec_model_path,
                    rec_keys_path,
                    font_path,
                )
                if model_path is not None and not Path(model_path).exists()
            ]
            if missing_pinned:
                listed = "\n".join(f"  - {path}" for path in missing_pinned)
                raise FileNotFoundError(
                    f"The following RapidOCR paths do not exist:\n{listed}"
                )

            # Params forwarded to RapidOCR only in the library-managed flow (no artifacts_path)
            lang_params: dict[str, object] = {}

            if artifacts_path is not None:
                # artifacts_path means fully-offline operation
                target_dir = artifacts_path / self._model_repo_folder
                artifacts: dict[str, _RapidOcrArtifact] = _rapidocr_artifacts(
                    target_dir,
                    backend_enum,
                    ppocr_version,
                    rec_lang,
                    need_det=det_model_path is None,
                    need_cls=cls_model_path is None,
                    need_rec=rec_model_path is None,
                )
                missing = [
                    dest
                    for artifact in artifacts.values()
                    for dest in artifact.files
                    if not dest.is_file()
                ]
                if missing:
                    listed = "\n".join(f"  - {path}" for path in missing)
                    # `lang` is the user's own token, so the hint mirrors their config.
                    raise FileNotFoundError(
                        "RapidOCR artifacts not found or incomplete in artifacts_path.\n"
                        f"Expected under: {target_dir}\n"
                        f"Resolved: backend={self.options.backend} "
                        f"ppocr_version={ppocr_version.value} rec_lang={rec_lang}\n"
                        f"Missing files:\n{listed}\n"
                        "Prefetch them with:\n"
                        f"  docling-tools models download rapidocr "
                        f"--rapidocr-backend-lang {self.options.backend}:{lang} "
                        f"-o {artifacts_path}\n"
                        "Or unset artifacts_path to let RapidOCR resolve and download "
                        "the checkpoints itself."
                    )

                if "det" in artifacts:
                    det_model_path = str(artifacts["det"].model_path)
                if "cls" in artifacts:
                    cls_model_path = str(artifacts["cls"].model_path)
                if "rec" in artifacts:
                    rec_model_path = str(artifacts["rec"].model_path)
                    if rec_keys_path is None and artifacts["rec"].dict_path is not None:
                        rec_keys_path = str(artifacts["rec"].dict_path)
            else:
                # Let RapidOCR resolve and cache the checkpoints itself
                size = ModelType(
                    _RAPIDOCR_MODEL_TYPE
                    if ppocr_version == OCRVersion.PPOCRV6
                    else _RAPIDOCR_V4V5_MODEL_TYPE
                )
                if det_model_path is None:
                    lang_params["Det.ocr_version"] = ppocr_version
                    lang_params["Det.lang_type"] = _RAPIDOCR_DET_MODEL_LANG
                    lang_params["Det.model_type"] = size
                if cls_model_path is None:
                    lang_params["Cls.ocr_version"] = OCRVersion.PPOCRV4
                    lang_params["Cls.lang_type"] = _RAPIDOCR_CLS_MODEL_LANG
                    lang_params["Cls.model_type"] = ModelType(_RAPIDOCR_V4V5_MODEL_TYPE)
                if rec_model_path is None:
                    lang_params["Rec.ocr_version"] = ppocr_version
                    lang_params["Rec.lang_type"] = rec_lang
                    lang_params["Rec.model_type"] = size

            params = {
                # Global settings (these are still correct)
                "Global.text_score": self.options.text_score,
                "Global.font_path": font_path,
                # Engine-level ONNXRuntime settings
                "EngineConfig.onnxruntime.intra_op_num_threads": intra_op_num_threads,
                "EngineConfig.onnxruntime.use_cuda": use_cuda,
                "EngineConfig.onnxruntime.cuda_ep_cfg.device_id": gpu_id,
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

            # Library-managed model-resolution params
            params.update(lang_params)

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
        lang: str = _RAPIDOCR_DEFAULT_LANGUAGE,
    ) -> Path:
        if local_dir is None:
            local_dir = settings.cache_dir / "models" / cls._model_repo_folder
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)

        resolved = _resolve_rapidocr(lang, backend)
        assert resolved.ppocr_version is not None
        assert resolved.rapidocr_lang_token is not None

        engine = _backend_to_engine_type(backend)
        for artifact in _rapidocr_artifacts(
            local_dir,
            engine,
            resolved.ppocr_version,
            resolved.rapidocr_lang_token,
        ).values():
            for dest, url in artifact.files.items():
                if dest.exists() and not force:
                    continue
                # paddle checkpoints are directory bundles, so dest may be nested.
                dest.parent.mkdir(parents=True, exist_ok=True)
                buf = download_url_with_progress(url, progress=progress)
                with dest.open("wb") as fw:
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
                    self.post_process_cells(all_ocr_cells, page, conv_res)

                # DEBUG code:
                if settings.debug.visualize_ocr:
                    self.draw_ocr_rects_and_cells(conv_res, page, ocr_rects)

                yield page

    @classmethod
    def get_options_type(cls) -> Type[OcrOptions]:
        return RapidOcrOptions
