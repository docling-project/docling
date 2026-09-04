# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import logging
import os
import warnings
import zipfile
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Type

import numpy
from docling_core.types.doc import BoundingBox, CoordOrigin
from docling_core.types.doc.page import BoundingRectangle, TextCell

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    OcrOptions,
)
from docling.datamodel.settings import settings
from docling.exceptions import OcrLanguageNotSupportedError, SecurityError
from docling.models.base_ocr_model import BaseOcrModel
from docling.utils.accelerator_utils import decide_device
from docling.utils.ocr_language import (
    OcrLanguage,
    OcrLanguageResolver,
    OcrLanguageSupport,
)
from docling.utils.profiling import TimeRecorder
from docling.utils.utils import download_url_with_progress

_log = logging.getLogger(__name__)


# Canonical tag -> EasyOCR code, where EasyOCR deviates from ISO 639-1.
_EASYOCR_CANONICAL_TO_CODE_DEVIATIONS: dict[str, str] = {
    "zh-Hans": "ch_sim",
    "zh-Hant": "ch_tra",
    "sr-Cyrl": "rs_cyrillic",
    "sr-Latn": "rs_latin",
    "tg-Cyrl": "tjk",
    "fil-Latn": "tl",
    # EasyOCR names these two with their ISO 639-3 codes; canonicalization
    # reaches the 639-1 `av`/`ce`, which EasyOCR has no recognizer under.
    "av-Cyrl": "ava",
    "ce-Cyrl": "che",
    # EasyOCR's `ang` is Angika and its `mah` is Magahi, both Devanagari. CLDR
    # gives Angika a likely script of Latin and normalizes `mah` to Marshallese,
    # so neither is reachable without an explicit entry.
    "anp-Deva": "ang",
    "mag-Deva": "mah",
    # Tabasaran is written in Cyrillic; CLDR's likely script for it is Latin.
    "tab-Cyrl": "tab",
}

_EASYOCR_CODE_TO_CANONICAL_DEVIATIONS: dict[str, str] = {
    code: canonical for canonical, code in _EASYOCR_CANONICAL_TO_CODE_DEVIATIONS.items()
}

# EasyOCR has no "engine default": `lang_list` is a required positional argument, and
# an empty one leaves the reader with a symbols-only character set. Name one language.
_EASYOCR_DEFAULT_LANGUAGE = "en"


@lru_cache(maxsize=1)
def _easyocr_code_to_model() -> dict[str, str]:
    """EasyOCR code -> the recognition checkpoint that serves it.

    Doubles as EasyOCR's supported-language vocabulary: a code absent from this
    mapping has no recognizer.
    """
    from easyocr.config import (
        arabic_lang_list,
        bengali_lang_list,
        cyrillic_lang_list,
        devanagari_lang_list,
        latin_lang_list,
    )

    language_models: dict[str, str] = {}

    # First add the languages that come from big language groups
    for language_group, model_name in (
        (latin_lang_list, "latin_g2"),
        (arabic_lang_list, "arabic_g1"),
        (bengali_lang_list, "bengali_g1"),
        (cyrillic_lang_list, "cyrillic_g2"),
        (devanagari_lang_list, "devanagari_g1"),
    ):
        language_models.update(dict.fromkeys(language_group, model_name))

    # Add other supported languages, which are outside of the lang_lists. Overwrite "en".
    language_models.update(
        {
            "en": "english_g2",
            "th": "thai_g1",
            "ch_tra": "zh_tra_g1",
            "ch_sim": "zh_sim_g2",
            "ja": "japanese_g2",
            "ko": "korean_g2",
            "ta": "tamil_g1",
            "te": "telugu_g2",
            "kn": "kannada_g2",
        }
    )
    return language_models


def _easyocr_code(language: OcrLanguage) -> Optional[str]:
    """The EasyOCR code for a canonical language, or `None` when there is no model."""
    if language.is_passthrough():
        code = language.native
    else:
        code = _EASYOCR_CANONICAL_TO_CODE_DEVIATIONS.get(language.bcp47())
        if code is None:
            # Outside the deviations, an EasyOCR code is a bare language code with
            if not language.has_default_script():
                return None
            code = language.bcp47_language
    return code if code in _easyocr_code_to_model() else None


def resolve_easyocr_codes(tags: Iterable[str]) -> List[str]:
    """Canonicalize language tags into the EasyOCR codes they name.

    Accepts EasyOCR's own codes as well as BCP-47, matching what
    `EasyOcrOptions.lang` accepts
    """
    codes: List[str] = []
    for tag in tags:
        language = OcrLanguageResolver.canonicalize_ocr_language(tag)
        code = _easyocr_code(language)
        if code is None:
            raise ValueError(f"Unsupported EasyOCR language: {tag}")
        if code not in codes:
            codes.append(code)
    return codes


def _resolve_easyocr_recognition_models(codes: Iterable[str]) -> List[str]:
    """Map EasyOCR codes onto the checkpoints the prefetcher must fetch."""
    code_to_model = _easyocr_code_to_model()

    model_names: set[str] = set()
    for code in codes:
        try:
            model_names.add(code_to_model[code])
        except KeyError:
            raise ValueError(f"Unsupported EasyOCR language code: {code}") from None
    return sorted(model_names)


class EasyOcrModel(BaseOcrModel):
    _model_repo_folder = "EasyOcr"

    multiple_languages = True

    def __init__(
        self,
        enabled: bool,
        artifacts_path: Optional[Path],
        options: EasyOcrOptions,
        accelerator_options: AcceleratorOptions,
    ):
        super().__init__(
            enabled=enabled,
            artifacts_path=artifacts_path,
            options=options,
            accelerator_options=accelerator_options,
        )
        self.options: EasyOcrOptions

        # multiplier for 72 dpi; the default 3.0 == 216 dpi.
        self.scale = self.options.scale
        self._native_codes: List[str] = []

        if self.enabled:
            try:
                import easyocr
            except ImportError:
                raise ImportError(
                    "EasyOCR is not installed. Please install it via `pip install easyocr` to use this OCR engine. "
                    "Alternatively, Docling has support for other OCR engines. See the documentation."
                )

            self._native_codes = (
                self.resolve_ocr_languages()
                if self.languages
                else [_EASYOCR_DEFAULT_LANGUAGE]
            )

            if self.options.use_gpu is None:
                device = decide_device(accelerator_options.device)
                # Enable easyocr GPU if running on CUDA, MPS
                use_gpu = any(
                    device.startswith(x)
                    for x in [
                        AcceleratorDevice.CUDA.value,
                        AcceleratorDevice.MPS.value,
                    ]
                )
            else:
                warnings.warn(
                    "Deprecated field. Better to set the `accelerator_options.device` in `pipeline_options`. "
                    "When `use_gpu and accelerator_options.device == AcceleratorDevice.CUDA` the GPU is used "
                    "to run EasyOCR. Otherwise, EasyOCR runs in CPU."
                )
                use_gpu = self.options.use_gpu

            download_enabled = self.options.download_enabled
            model_storage_directory = self.options.model_storage_directory
            if artifacts_path is not None and model_storage_directory is None:
                download_enabled = False
                model_storage_directory = str(artifacts_path / self._model_repo_folder)

            with warnings.catch_warnings():
                if self.options.suppress_mps_warnings:
                    warnings.filterwarnings("ignore", message=".*pin_memory.*MPS.*")
                self.reader = easyocr.Reader(
                    lang_list=self._native_codes,
                    gpu=use_gpu,
                    model_storage_directory=model_storage_directory,
                    recog_network=self.options.recog_network,
                    download_enabled=download_enabled,
                    verbose=False,
                )

    def supported_ocr_languages(self) -> OcrLanguageSupport:
        r"""Report the native and BCP74 languages without script whenever it is not needed"""
        tags: set[str] = set()
        native: set[str] = set()
        for code in _easyocr_code_to_model():
            # First resolve against the deviational codes
            tag = _EASYOCR_CODE_TO_CANONICAL_DEVIATIONS.get(code, code)
            language = OcrLanguageResolver.canonicalize_bcp47(
                tag, raise_exception=False
            )
            if language is not None and _easyocr_code(language) == code:
                tags.add(language.short_tag())
            else:
                # A recognizer no tag can name is offered as the code itself
                native.add(code)
        return OcrLanguageSupport(bcp47=sorted(tags), native=sorted(native))

    def map_ocr_language(self, language: OcrLanguage) -> str | List[str]:
        code = _easyocr_code(language)
        if code is None:
            raise OcrLanguageNotSupportedError(
                self._engine_name,
                language.tag(),
                supported=self.supported_ocr_languages(),
            )
        return code

    @staticmethod
    def download_models(
        detection_models: List[str] = ["craft"],
        recognition_models: List[str] = ["english_g2", "latin_g2"],
        local_dir: Optional[Path] = None,
        force: bool = False,
        progress: bool = False,
    ) -> Path:
        # Models are located in https://github.com/JaidedAI/EasyOCR/blob/master/easyocr/config.py
        from easyocr.config import (
            detection_models as det_models_dict,
            recognition_models as rec_models_dict,
        )

        if local_dir is None:
            local_dir = settings.cache_dir / "models" / EasyOcrModel._model_repo_folder

        local_dir.mkdir(parents=True, exist_ok=True)

        download_list = []
        for model_name in detection_models:
            if model_name in det_models_dict:
                download_list.append(det_models_dict[model_name])

        recognition_models_by_name = {
            model_name: model_details
            for generation in rec_models_dict.values()
            for model_name, model_details in generation.items()
        }
        for model_name in recognition_models:
            if model_name in recognition_models_by_name:
                download_list.append(recognition_models_by_name[model_name])

        # Download models
        for model_details in download_list:
            buf = download_url_with_progress(model_details["url"], progress=progress)
            with zipfile.ZipFile(buf, "r") as zip_ref:
                for member in zip_ref.infolist():
                    member_path = os.path.realpath(
                        os.path.join(local_dir, member.filename)
                    )
                    if not member_path.startswith(os.path.realpath(local_dir) + os.sep):
                        raise SecurityError(f"ZIP slip attempt: {member.filename}")
                    zip_ref.extract(member, local_dir)

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

                        with warnings.catch_warnings():
                            if self.options.suppress_mps_warnings:
                                warnings.filterwarnings(
                                    "ignore", message=".*pin_memory.*MPS.*"
                                )

                            result = self.reader.readtext(im)

                        del high_res_image
                        del im

                        cells = [
                            TextCell(
                                index=ix,
                                text=line[1],
                                orig=line[1],
                                from_ocr=True,
                                confidence=line[2],
                                rect=BoundingRectangle.from_bounding_box(
                                    BoundingBox.from_tuple(
                                        coord=(
                                            (line[0][0][0] / self.scale) + ocr_rect.l,
                                            (line[0][0][1] / self.scale) + ocr_rect.t,
                                            (line[0][2][0] / self.scale) + ocr_rect.l,
                                            (line[0][2][1] / self.scale) + ocr_rect.t,
                                        ),
                                        origin=CoordOrigin.TOPLEFT,
                                    )
                                ),
                            )
                            for ix, line in enumerate(result)
                            if line[2] >= self.options.confidence_threshold
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
        return EasyOcrOptions
