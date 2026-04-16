import subprocess
import os
from pathlib import Path
from typing import Union
from io import BytesIO
import tempfile
import random

import docling.backend.xml.hwpx_backend as hwpx_backend
from docling.backend.abstract_backend import DeclarativeDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument
from docling_core.types.doc import DoclingDocument
from docling.exceptions import HwpConversionError


class HwpDocumentBackend(DeclarativeDocumentBackend):
    def __init__(self, in_doc: InputDocument, path_or_stream: Union[Path, BytesIO]) -> None:
        """Initialize the HWP backend by converting HWP to HWPX first."""
        super().__init__(in_doc, path_or_stream)
        self.hwpx_backend = None
        self.valid = False

        # HWP 파일인지 확인
        if isinstance(path_or_stream, (Path, BytesIO)):
            try:
                if isinstance(path_or_stream, BytesIO):
                    # BytesIO를 임시 파일로 저장
                    random_int = random.randint(1000, 9999)
                    temp_hwp_path = Path(f'/tmp/temp_{random_int}.hwp')
                    with open(temp_hwp_path, 'wb') as temp_file:
                        temp_file.write(path_or_stream.getbuffer())
                    path_or_stream = temp_hwp_path

                # HWP를 HWPX로 변환
                hwpx_path = self._convert_hwp_to_hwpx(path_or_stream)

                # HwpxDocumentBackend 인스턴스 생성
                self.hwpx_backend = hwpx_backend.HwpxDocumentBackend(in_doc, hwpx_path)
                self.valid = self.hwpx_backend.is_valid()
            except Exception as e:
                self.valid = False
                # HWP 변환 실패 시 구체적인 메시지와 함께 커스텀 예외 발생
                raise HwpConversionError(f"HWP 파일을 변환하는 중에 오류가 발생했습니다. HWPX로 직접 변환하신 후 다시 첨부해 주시기 바랍니다. 번거롭게 해드려 죄송합니다. \n오류 내용: {e}")
            finally:
                # 임시 파일 삭제
                if isinstance(path_or_stream, Path) and path_or_stream.name.startswith('temp_'):
                    os.remove(path_or_stream)
        else:
            raise RuntimeError("HwpDocumentBackend only supports .hwp files")

    def _convert_hwp_to_hwpx(self, hwp_path: Path) -> Path:
        """Convert HWP file to HWPX using hwp2hwpx.sh script."""
        try:
            # 입력 파일과 같은 디렉토리에 출력 파일 생성
            input_hwp = str(hwp_path)
            output_hwpx = str(hwp_path.with_suffix('.hwpx'))

            # hwp2hwpx 스크립트 실행
            result = subprocess.run([
                "/app/hwp2hwpx/run_hwp2hwpx.sh",
                input_hwp,
                output_hwpx
            ], capture_output=True, text=True, cwd=str(hwp_path.parent))

            if result.returncode != 0:
                raise RuntimeError(f"HWP to HWPX conversion failed: {result.stderr}")

            if not os.path.exists(output_hwpx):
                raise RuntimeError(f"HWPX file was not created: {output_hwpx}")

            return Path(output_hwpx)

        except Exception as e:
            raise RuntimeError(f"Failed to convert HWP to HWPX: {e}")

    def is_valid(self) -> bool:
        return self.valid and self.hwpx_backend is not None

    @classmethod
    def supported_formats(cls) -> set:
        return set()  # HWP는 확장자로만 식별

    @classmethod
    def supports_pagination(cls) -> bool:
        return False

    def unload(self) -> None:
        # HwpxDocumentBackend 정리
        if self.hwpx_backend:
            self.hwpx_backend.unload()
            self.hwpx_backend = None

    def convert(self) -> DoclingDocument:
        """Convert HWP file to DoclingDocument by delegating to HwpxDocumentBackend."""
        if not self.is_valid():
            raise RuntimeError("Invalid HWP document or conversion failed")

        # HwpxDocumentBackend에 실제 변환 작업 위임
        return self.hwpx_backend.convert()
