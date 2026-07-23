# HWP/HWPX 변환 품질 복구 헬퍼
#
# intelligent_processor 가 HWP/HWPX → PDF 변환에서 내용을 잃었을 때(추출 텍스트 품질
# 점수가 낮을 때) 복구를 시도하기 위한 헬퍼 모듈. facade 가 아니라 converters 하위
# 모듈이며, intelligent_processor → 이 모듈의 단방향 의존만 갖는다(xlsx_processor 와 동일).
#
# 복구 전략(recover):
#   1) rhwp 백엔드만 강제해 재변환 → 재로딩 → 품질 점수가 개선되면 교체.
#   2) (.hwpx 한정) 여전히 저품질이면 HWPX XML 을 네이티브로 직접 파싱해 교체.
#
# docling 관련 import 는 각 메서드 내부에서만 수행한다(로컬 vendored docling import
# 충돌 회피 + 복구 미사용 시 docling 미로딩). convert_hwp_to_pdf 는 converters 내부
# 의존이라 top-level 상대 import 로 둔다(docling 미로딩).
from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

from .hwp_to_pdf import convert_hwp_to_pdf

_log = logging.getLogger(__name__)


class HwpQualityRecovery:
    """저품질 HWP/HWPX 추출을 rhwp 재변환 / 네이티브 HWPX 추출로 복구한다.

    문서 재로딩은 주입된 reload_fn(보통 DocumentProcessor._load_document)에 위임한다.
    표 셀 OCR 은 호출 측 파이프라인이 이후 일괄 수행하므로 여기서는 중복하지 않는다.
    """

    # 추출 텍스트 품질 점수가 이 값 미만이면 "의심(저품질)" 으로 판정한다.
    SUSPICIOUS_SCORE_THRESHOLD = 20

    def __init__(self, reload_fn: Callable[..., Any]):
        # reload_fn(file_path, **kwargs) -> DoclingDocument
        self._reload_fn = reload_fn

    @staticmethod
    def document_text_score(document) -> int:
        """의미있는 문자 수(표 셀 텍스트 포함)를 센다. HWP 변환 품질 판정용."""
        from docling_core.types.doc import TableItem

        parts: list[str] = []
        for item, _ in document.iterate_items():
            text = getattr(item, "text", None)
            if isinstance(text, str) and text:
                parts.append(text)

            if isinstance(item, TableItem) and item.data:
                for cell in item.data.table_cells:
                    cell_text = getattr(cell, "text", None)
                    if isinstance(cell_text, str) and cell_text:
                        parts.append(cell_text)

        return sum(1 for char in "".join(parts) if char.isalnum())

    @classmethod
    def is_suspicious(cls, document) -> bool:
        return cls.document_text_score(document) < cls.SUSPICIOUS_SCORE_THRESHOLD

    @staticmethod
    def load_hwpx_native(file_path: str):
        """PDF 변환이 내용을 잃을 때 HWPX XML 을 직접 파싱한다(품질 복구 폴백)."""
        from docling.backend.xml.hwpx_backend import HwpxDocumentBackend
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.document import ConversionResult
        from docling.datamodel.pipeline_options import PipelineOptions
        from docling.document_converter import DocumentConverter, HwpxFormatOption

        converter = DocumentConverter(
            format_options={
                InputFormat.XML_HWPX: HwpxFormatOption(
                    pipeline_options=PipelineOptions(),
                    backend=HwpxDocumentBackend,
                ),
            }
        )
        conv_result: ConversionResult = converter.convert(
            Path(file_path).resolve(),
            raises_on_error=True,
        )
        return conv_result.document

    def recover(
        self, document, source_file_path: str, file_path: str,
        converted_pdf_path: Optional[str], **kwargs: Any,
    ) -> "tuple[str, Optional[str], Any]":
        """저품질 HWP/HWPX 추출을 복구한다.

        재로딩은 주입된 reload_fn 을 사용한다. 갱신된
        (file_path, converted_pdf_path, document) 튜플을 반환한다.
        """
        source_suffix = Path(source_file_path).suffix.lower()
        if source_suffix not in {".hwp", ".hwpx"} or not self.is_suspicious(document):
            return file_path, converted_pdf_path, document

        # rhwp 재변환 재시도
        converted_score = self.document_text_score(document)
        original_pdf_backup: Optional[str] = None
        rhwp_recovered = False
        try:
            if converted_pdf_path and os.path.exists(converted_pdf_path):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as backup:
                    shutil.copy2(converted_pdf_path, backup.name)
                    original_pdf_backup = backup.name

            _log.warning(
                "[intelligent][hwp] Suspicious HWP/HWPX extraction (score=%s); retrying with rhwp",
                converted_score,
            )
            # rhwp 백엔드만 강제해 재변환한다(하위 변환기를 직접 호출).
            rhwp_pdf = convert_hwp_to_pdf(source_file_path, order=["rhwp"])
            if rhwp_pdf and os.path.exists(rhwp_pdf):
                rhwp_document = self._reload_fn(rhwp_pdf, **kwargs)
                rhwp_score = self.document_text_score(rhwp_document)
                if rhwp_score >= self.SUSPICIOUS_SCORE_THRESHOLD and rhwp_score > converted_score:
                    _log.info(
                        "[intelligent][hwp] rhwp recovered HWP/HWPX content (before=%s, after=%s)",
                        converted_score, rhwp_score,
                    )
                    document = rhwp_document
                    file_path = rhwp_pdf
                    converted_pdf_path = rhwp_pdf
                    rhwp_recovered = True
        except Exception as exc:
            _log.warning("[intelligent][hwp] rhwp quality fallback failed: %s", exc)
        finally:
            if not rhwp_recovered and original_pdf_backup and converted_pdf_path:
                shutil.copy2(original_pdf_backup, converted_pdf_path)
            if original_pdf_backup:
                try:
                    os.remove(original_pdf_backup)
                except OSError:
                    pass

        # 네이티브 HWPX 추출 폴백(여전히 의심스러우면)
        if source_suffix == ".hwpx" and self.is_suspicious(document):
            converted_score = self.document_text_score(document)
            try:
                native_document = self.load_hwpx_native(source_file_path)
                native_score = self.document_text_score(native_document)
                if native_score >= self.SUSPICIOUS_SCORE_THRESHOLD and native_score > converted_score:
                    _log.warning(
                        "[intelligent][hwp] HWPX conversion backends lost content "
                        "(converted_score=%s, native_score=%s); using native HWPX extraction",
                        converted_score, native_score,
                    )
                    document = native_document
            except Exception as exc:
                _log.warning("[intelligent][hwp] Native HWPX fallback failed: %s", exc)

        return file_path, converted_pdf_path, document
