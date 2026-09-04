# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Type, get_args

import numpy
from docling_core.types.doc import BoundingBox, CoordOrigin
from docling_core.types.doc.page import BoundingRectangle, TextCell

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    OcrOptions,
    RapidOcrBackend,
    RapidOcrOptions,
)
from docling.datamodel.settings import settings
from docling.exceptions import OcrLanguageNotSupportedError
from docling.models.base_ocr_model import BaseOcrModel
from docling.utils.accelerator_utils import decide_device
from docling.utils.ocr_language import (
    OcrLanguage,
    OcrLanguageResolver,
    OcrLanguageSupport,
)
from docling.utils.profiling import TimeRecorder
from docling.utils.utils import download_url_with_progress, safe_version

if TYPE_CHECKING:
    from rapidocr.utils.typings import EngineType, OCRVersion

_log = logging.getLogger(__name__)

# Default OCR language as a canonical tag, for the prefetch entry points.
_RAPIDOCR_DEFAULT_LANGUAGE = "ch"

# Docling's pre-canonicalization spellings for the two default recognizers, kept so that configs
# written before OCR languages were canonicalized keep working
_RAPIDOCR_LEGACY_ALIASES = {"chinese": "ch", "english": "en"}

# Recognition/detection model size for the PP-OCRv6 path; v4/v5 use "mobile".
_RAPIDOCR_DET_MODEL_LANG = "ch"
_RAPIDOCR_CLS_MODEL_LANG = "ch"
_RAPIDOCR_MODEL_TYPE = "small"
_RAPIDOCR_V4V5_MODEL_TYPE = "mobile"

# Inference backends docling supports, derived from the RapidOcrOptions.backend
# annotation so the two cannot drift.
_RAPIDOCR_BACKENDS: frozenset[str] = frozenset(get_args(RapidOcrBackend))

# Canonical BCP-47 to PP-OCR recognizer codes. The PP-OCRv6 set comes from the
# installed `rapidocr`; the v4/v5 sets below have no upstream equivalent to read,
# so they mirror the PP-OCR release notes summarised in `docs/concepts/OCR.md`.

# Recognition languages served by the PP-OCRv4 backbone (the torch fallback).
_PPOCRV4_CODES = frozenset(
    {"arabic", "cyrillic", "devanagari", "ka", "korean", "latin", "ta", "te"}
)

# Recognition languages served by the PP-OCRv5 backbone.
_PPOCRV5_CODES = frozenset(
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

# Canonical tag -> PP-OCR code, for the languages whose code is not simply the primary subtag.
# `None` marks a tag that must *not* fall through to the generic rules below,
# because the code that looks right means something else.
_PPOCR_CANONICAL_TO_CODE_DEVIATIONS: dict[str, str | None] = {
    "zh-Hans": "ch",
    "zh-Hant": "chinese_cht",
    "ja-Jpan": "japan",
    "ko-Kore": "korean",
    "sr-Latn": "rs_latin",
    # `tl` is PP-OCR's code; BCP-47 canonicalizes Tagalog to `fil`.
    "fil-Latn": "tl",
    # PP-OCR serves East Slavic with a narrower recognizer than `cyrillic`.
    "ru-Cyrl": "eslav",
    "uk-Cyrl": "eslav",
    "be-Cyrl": "eslav",
    # PP-OCR's `ka` is Kannada; BCP-47 `ka` is Georgian.
    "kn-Knda": "ka",
    "ka-Geor": None,
}

# For PPOCRv4 and PPOCRv5 some languages are described not individually but using their scripts
# This is a mapping from the canonical ISO 15924 script to PP-OCR script-family code
_PPOCR_SCRIPT_TO_CODE: dict[str, str] = {
    "Latn": "latin",
    "Cyrl": "cyrillic",
    "Arab": "arabic",
    "Deva": "devanagari",
}

# Reverse of the language table
_PPOCR_CODE_TO_CANONICAL_DEVIATIONS: dict[str, list[str]] = {}
for tag, token in _PPOCR_CANONICAL_TO_CODE_DEVIATIONS.items():
    if token is not None:
        _PPOCR_CODE_TO_CANONICAL_DEVIATIONS.setdefault(token, []).append(tag)

# PP-OCRv6 codes that duplicate a language already reachable by its subtag
_PPOCR_REDUNDANT_CODES = frozenset({"french", "german"})


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

    # PP-OCR code the rapidocr registry expects, after normalization and aliasing.
    rapidocr_code: str | None = None

    # PP-OCR backbone that the (backend, language) pair resolves to.
    ppocr_version: "OCRVersion | None" = None


@lru_cache(maxsize=len(_RAPIDOCR_BACKENDS))
def _ppocr_supported_languages(vocabulary: frozenset[str]) -> OcrLanguageSupport:
    """What a PP-OCR code vocabulary can serve, as tags and as native codes."""
    tags: set[str] = set()
    native: set[str] = set()
    for code in vocabulary:
        if code in _PPOCR_REDUNDANT_CODES:
            continue
        # A code several tags deviate onto -- `eslav` is `ru`, `uk` and `be` --
        # is advertised as every one of them.
        deviations = _PPOCR_CODE_TO_CANONICAL_DEVIATIONS.get(code, [code])
        for deviation in deviations:
            language = OcrLanguageResolver.canonicalize_bcp47(
                deviation, raise_exception=False
            )
            if language is not None and _ppocr_code(language, vocabulary) == code:
                tags.add(language.short_tag())
            else:
                # A recognizer no tag can name -- the `latin`, `cyrillic`,
                # `arabic` and `devanagari` script models -- is offered as the
                # code itself, which is the only spelling that reaches it.
                native.add(code)
    return OcrLanguageSupport(bcp47=sorted(tags), native=sorted(native))


@lru_cache(maxsize=1)
def _installed_ppocrv6_codes() -> frozenset[str]:
    """The PP-OCRv6 recognition languages, read from the installed rapidocr.

    `PP_OCRV6_LANGS` is not part of rapidocr's public API, so a version that
    moves it has to say so rather than leave docling guessing at the vocabulary.
    """
    try:
        from rapidocr.utils.model_resolver import PP_OCRV6_LANGS
    except ImportError as err:
        raise ImportError(
            f"The installed rapidocr ({safe_version('rapidocr')}) does not expose "
            "PP_OCRV6_LANGS, which docling needs to resolve PP-OCR recognizers. "
            "Install a version in the supported range: rapidocr>=3.9.1,<4.0.0."
        ) from err
    return frozenset(PP_OCRV6_LANGS)


@lru_cache(maxsize=len(_RAPIDOCR_BACKENDS))
def _rapidocr_vocabulary(backend: str) -> frozenset[str]:
    """PP-OCR codes a backend can serve: v6 plus its own v4/v5 fallback sets."""
    fallback = _PPOCRV4_CODES if backend == "torch" else _PPOCRV5_CODES | _PPOCRV4_CODES
    return _installed_ppocrv6_codes() | fallback


def _ppocr_code(language: OcrLanguage, vocabulary: frozenset[str]) -> str | None:
    """Map a canonical tag onto a PP-OCR code, or `None` if there is no model.

    `vocabulary` is the union of code sets the caller can actually reach, so
    the resolution never returns a code the backend cannot serve.
    """
    if language.is_passthrough():
        native = language.native
        alias = _RAPIDOCR_LEGACY_ALIASES.get(native or "")
        if alias is not None:
            _log.warning(
                "%r is a docling alias for the PP-OCR code %r. Write %r instead.",
                native,
                alias,
                alias,
            )
            native = alias
        return native if native in vocabulary else None

    if language.bcp47() in _PPOCR_CANONICAL_TO_CODE_DEVIATIONS:
        code = _PPOCR_CANONICAL_TO_CODE_DEVIATIONS[language.bcp47()]
        return code if code is not None and code in vocabulary else None

    # The bcp47 language matches the PPOCR code only when it uses its default script
    if language.has_default_script() and language.bcp47_language in vocabulary:
        return language.bcp47_language

    # Otherwise try to match the bcp47 script with the PPOCR script-based languages
    family = _PPOCR_SCRIPT_TO_CODE.get(language.bcp47_script or "")
    if family is not None and family in vocabulary:
        return family
    return None


def _parse_rapidocr_model_spec(value: str) -> _RapidOcrModelSpec:
    """Parse a `<backend>:<lang>` prefetch spec into its requested form.

    The split is on the first colon alone: a BCP-47 language carries one of its
    own, `onnxruntime:iso:th-Thai`, which is the spelling the model hands the user.
    """
    backend, separator, lang = value.partition(":")
    if not separator or not backend or not lang:
        raise ValueError(
            f"Invalid RapidOCR model spec {value!r}. "
            "Expected '<backend>:<lang>', e.g. 'onnxruntime:ch'."
        )
    if backend not in _RAPIDOCR_BACKENDS:
        raise ValueError(
            f"Unknown RapidOCR backend {backend!r} in {value!r}. "
            f"Supported: {sorted(_RAPIDOCR_BACKENDS)}."
        )
    try:
        _resolve_rapidocr(lang, backend)
    except (ValueError, OcrLanguageNotSupportedError) as err:
        raise ValueError(f"Invalid RapidOCR model spec {value!r}: {err}") from err
    return _RapidOcrModelSpec(backend=backend, user_lang=lang)


def _ppocr_version_for_code(code: str, backend: str) -> "OCRVersion":
    """Which PP-OCR backbone serves a code on this backend.

    Prefers PP-OCRv6 (whose recognizer covers ~52 codes). Torch then falls back
    to PP-OCRv4; the other backends try PP-OCRv5 first and PP-OCRv4 for the
    codes v5 lacks -- `ka`, PP-OCR's Kannada, is the only one.
    """
    from rapidocr.utils.typings import OCRVersion

    if code in _installed_ppocrv6_codes():
        return OCRVersion.PPOCRV6
    if backend == "torch":
        return OCRVersion.PPOCRV4
    return OCRVersion.PPOCRV5 if code in _PPOCRV5_CODES else OCRVersion.PPOCRV4


def _resolve_rapidocr(lang: str, backend: str) -> _RapidOcrModelSpec:
    """Map one language + backend onto a fully populated _RapidOcrModelSpec.

    `lang` may be one of PP-OCR's own codes or a BCP-47 tag behind `iso:`

    Raises:
        ValueError: `lang` is neither a PP-OCR code nor a valid BCP-47 tag.
        OcrLanguageNotSupportedError: No PP-OCR recognizer serves it on `backend`.

    Callers pass a single language; reducing a multi-language request is up to them.
    """
    language = OcrLanguageResolver.canonicalize_ocr_language(lang)
    code = _ppocr_code(language, _rapidocr_vocabulary(backend))
    if code is None:
        raise OcrLanguageNotSupportedError(
            f"RapidOCR (backend={backend})",
            language.tag(),
            supported=_ppocr_supported_languages(_rapidocr_vocabulary(backend)),
        )
    version = _ppocr_version_for_code(code, backend)

    _log.debug(
        "RapidOCR resolved lang=%r backend=%r -> version=%s rec_code=%r",
        lang,
        backend,
        version.value,
        code,
    )
    return _RapidOcrModelSpec(
        backend=backend,
        user_lang=lang,
        rapidocr_code=code,
        ppocr_version=version,
    )


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
            f"Unknown RapidOCR backend {backend!r}. Supported: {sorted(_RAPIDOCR_BACKENDS)}."
        )
    return engine_types[backend]


def _rapidocr_artifacts(
    target_dir: Path,
    engine: "EngineType",
    version: "OCRVersion",
    rec_code: str,
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
            engine,
            OCRVersion.PPOCRV4,
            TaskType.CLS,
            _RAPIDOCR_CLS_MODEL_LANG,
            cls_size,
        )
    if need_rec:
        file_infos["rec"] = FileInfo(engine, version, TaskType.REC, rec_code, size)

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

    multiple_languages = False

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
        self._native_codes: list[str] = []

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

            # One language, warn-and-truncate and coverage checks all happen here.
            self._native_codes = self.resolve_ocr_languages()
            rec_code = self._native_codes[0]
            lang = (
                self.languages[0].tag()
                if self.languages
                else _RAPIDOCR_DEFAULT_LANGUAGE
            )
            ppocr_version = _ppocr_version_for_code(rec_code, self.options.backend)

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
                    rec_code,
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
                    # `lang` is the canonical tag, which is what the prefetcher takes.
                    raise FileNotFoundError(
                        "RapidOCR artifacts not found or incomplete in artifacts_path.\n"
                        f"Expected under: {target_dir}\n"
                        f"Resolved: backend={self.options.backend} "
                        f"ppocr_version={ppocr_version.value} rec_code={rec_code}\n"
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
                    lang_params["Rec.lang_type"] = rec_code
                    lang_params["Rec.model_type"] = size

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

            # Library-managed model-resolution params
            params.update(lang_params)

            user_params = self.options.rapidocr_params
            if user_params:
                _log.debug("Overwriting RapidOCR params with user-provided values.")
                params.update(user_params)

            self.reader = RapidOCR(
                params=params,
            )

    def supported_ocr_languages(self) -> OcrLanguageSupport:
        return _ppocr_supported_languages(_rapidocr_vocabulary(self.options.backend))

    def resolve_ocr_languages(self) -> list[str]:
        # An empty `lang` list means "the engine's own default", which for PP-OCR
        # is the Simplified Chinese recognizer. Resolved the same way a request
        # is, so the recognizer this loads and the one the prefetch hint names
        # cannot drift apart.
        if not self.languages:
            default = OcrLanguageResolver.canonicalize_ocr_language(
                _RAPIDOCR_DEFAULT_LANGUAGE
            )
            code = _ppocr_code(default, _rapidocr_vocabulary(self.options.backend))
            assert code is not None, (
                f"the RapidOCR default {_RAPIDOCR_DEFAULT_LANGUAGE!r} has no "
                f"recognizer on backend {self.options.backend!r}"
            )
            return [code]
        return super().resolve_ocr_languages()

    def map_ocr_language(self, language: OcrLanguage) -> str:
        code = _ppocr_code(language, _rapidocr_vocabulary(self.options.backend))
        if code is None:
            raise OcrLanguageNotSupportedError(
                f"RapidOCR (backend={self.options.backend})",
                language.tag(),
                supported=self.supported_ocr_languages(),
            )
        return code

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
        assert resolved.rapidocr_code is not None

        engine = _backend_to_engine_type(backend)
        for artifact in _rapidocr_artifacts(
            local_dir,
            engine,
            resolved.ppocr_version,
            resolved.rapidocr_code,
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
