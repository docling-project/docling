import subprocess
import logging
import os
from pathlib import Path
from typing import Union
from io import BytesIO
import random

import docling.backend.xml.hwpx_backend as hwpx_backend
from docling.backend.abstract_backend import DeclarativeDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument
from docling_core.types.doc import DoclingDocument
from docling.exceptions import HwpConversionError

_log = logging.getLogger(__name__)

class HwpDocumentBackend(DeclarativeDocumentBackend):
    def __init__(self, in_doc: InputDocument, path_or_stream: Union[Path, BytesIO]) -> None:
        """HWP 파일을 HWPX로 변환한 뒤 HwpxDocumentBackend에 위임하는 구버전 백엔드."""
        super().__init__(in_doc, path_or_stream)
        self.hwpx_backend = None
        self.valid = False
        temp_input_path = None

        if isinstance(path_or_stream, (Path, BytesIO)):
            try:
                if isinstance(path_or_stream, BytesIO):
                    # BytesIO를 임시 파일로 저장
                    random_int = random.randint(1000, 9999)
                    temp_hwp_path = Path(f'/tmp/temp_{random_int}.hwp')
                    with open(temp_hwp_path, 'wb') as temp_file:
                        temp_file.write(path_or_stream.getbuffer())
                    temp_input_path = temp_hwp_path
                    path_or_stream = temp_hwp_path

                hwpx_path = self._convert_hwp_to_hwpx(path_or_stream)
                self.hwpx_backend = hwpx_backend.HwpxDocumentBackend(in_doc, hwpx_path)
                self.valid = self.hwpx_backend.is_valid()
            except Exception as e:
                self.valid = False
                raise HwpConversionError(
                    f"HWP 파일을 변환하는 중에 오류가 발생하였습니다. "
                    f"HWPX로 직접 변환하신 후 다시 첨부해 주시기 바랍니다. "
                    f"번거롭게 해드려 죄송합니다.\n오류 내용: {e}"
                ) from e
            finally:
                # BytesIO에서 이 백엔드가 직접 생성한 임시 파일만 삭제
                if temp_input_path is not None:
                    try:
                        os.remove(temp_input_path)
                    except OSError:
                        pass
        else:
            raise RuntimeError("HwpDocumentBackend only supports .hwp files")

    def _convert_hwp_to_hwpx(self, hwp_path: Path) -> Path:
        """hwp2hwpx.sh 스크립트로 HWP → HWPX 변환."""
        input_hwp = str(hwp_path)
        output_hwpx = str(hwp_path.with_suffix('.hwpx'))

        try:
            result = subprocess.run(
                ["/app/hwp2hwpx/run_hwp2hwpx.sh", input_hwp, output_hwpx],
                capture_output=True,
                text=True,
                cwd=str(hwp_path.parent),
                timeout=600,
            )
        except subprocess.TimeoutExpired as e:
            _log.error(f"HWP→HWPX 변환 타임아웃: {input_hwp} → {output_hwpx}")
            raise RuntimeError("HWP to HWPX conversion timed out") from e

        if result.returncode != 0:
            raise RuntimeError(f"HWP to HWPX conversion failed: {result.stderr}")

        if not os.path.exists(output_hwpx):
            raise RuntimeError(f"HWPX file was not created: {output_hwpx}")

        return Path(output_hwpx)

    def is_valid(self) -> bool:
        return self.valid and self.hwpx_backend is not None

    @classmethod
    def supported_formats(cls) -> set:
        return set()  # HWP는 확장자로만 식별

    @classmethod
    def supports_pagination(cls) -> bool:
        return False

    def unload(self) -> None:
        if self.hwpx_backend:
            self.hwpx_backend.unload()
            self.hwpx_backend = None

    def convert(self) -> DoclingDocument:
        """실제 변환은 HwpxDocumentBackend에 위임."""
        if not self.is_valid():
            raise HwpConversionError("Invalid HWP document or conversion failed")
        return self.hwpx_backend.convert()
