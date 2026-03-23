import json
import logging
import os
import subprocess
import tempfile
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from typing_extensions import override  # 상단 임포트에 추가
from io import BytesIO

from bs4 import BeautifulSoup
from docling_core.types.doc import (
    DocItemLabel, DoclingDocument, DocumentOrigin, GroupLabel,
    ImageRef, NodeItem, ProvenanceItem, TableCell, TableData,
    BoundingBox, Size, Formatting
)
from docling.backend.abstract_backend import DeclarativeDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument

_log = logging.getLogger(__name__)

# --- [1. HWP용 MIME 타입 패치] ---
_HWP_MIMETYPES = [
    "application/vnd.hancom.hwp", 
    "application/x-hwp",
    "application/vnd.hancom.hwpx", 
    "application/hwp+zip"
]

for mime in _HWP_MIMETYPES:
    # DocumentOrigin 클래스 내의 리스트에 값이 없으면 추가
    if mime not in DocumentOrigin._extra_mimetypes:
        DocumentOrigin._extra_mimetypes.append(mime)
# ------------------------------

# --- [2. SDK 경로 및 환경 변수 등록] ---
# 이 파일 위치 기준 hwp_sdk 폴더 계산
SDK_DIR = Path(__file__).resolve().parent.parent.parent / "hwp_sdk"
SDK_PATH_STR = str(SDK_DIR)

if SDK_DIR.exists():
    # PATH 등록
    if SDK_PATH_STR not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{SDK_PATH_STR}:{os.environ.get('PATH', '')}"
    
    # LD_LIBRARY_PATH 등록
    if SDK_PATH_STR not in os.environ.get("LD_LIBRARY_PATH", ""):
        os.environ["LD_LIBRARY_PATH"] = f"{SDK_PATH_STR}:{os.environ.get('LD_LIBRARY_PATH', '')}"
else:
    print(f"⚠️ 경고: HWP SDK 경로를 찾을 수 없습니다: {SDK_PATH_STR}")
# ------------------------------

class GenosHwpDocumentBackend(DeclarativeDocumentBackend):
    def __init__(self, in_doc: InputDocument, path_or_stream: Union[Path, BytesIO]) -> None:
        super().__init__(in_doc, path_or_stream)
        
        # 1. 환경 설정      
        self.valid = False
        
        # 2. 계층 및 상태 관리 (GenosMsWord 방식 이식)
        self.max_levels = 10
        self.parents: Dict[int, Optional[NodeItem]] = {i: None for i in range(-1, self.max_levels)}
        self.history: Dict[str, List[Any]] = {
            "names": [None], "levels": [None], "page_nos": [1]
        }
        
        # 3. 소스 파일 준비
        self.temp_input_path = None
        if isinstance(path_or_stream, BytesIO):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".hwp") as tmp:
                tmp.write(path_or_stream.getbuffer())
                self.temp_input_path = Path(tmp.name)
            self.source_path = self.temp_input_path
        else:
            self.source_path = path_or_stream

        if self.source_path and self.source_path.exists():
            self.valid = True

    @override
    def is_valid(self) -> bool:
        """추상 메서드 구현: 백엔드가 유효한지 반환"""
        return self.valid

    @classmethod
    @override
    def supported_formats(cls) -> set[InputFormat]:
        """추상 메서드 구현: 이 백엔드가 지원하는 포맷 정의 (클래스 메서드여야 함)"""
        return {InputFormat.HWP, InputFormat.XML_HWPX}

    @classmethod
    @override
    def supports_pagination(cls) -> bool:
        """추상 메서드 구현: 페이지 단위 처리를 지원하는지 여부 (HWP는 대개 False)"""
        return False

    # --- 기존에 생략되었던 유틸리티 함수들 ---
    def _update_history(self, name: str, level: Optional[int], page_no: int):
        self.history["names"].append(name)
        self.history["levels"].append(level)
        self.history["page_nos"].append(page_no)

    def _get_current_level(self) -> int:
        for k, v in self.parents.items():
            if k >= 0 and v is None:
                return k
        return 0

    def _map_font_to_formatting(self, font_info: Dict) -> Formatting:
        """SDK의 font 정보를 Docling의 Formatting 객체로 변환"""
        return Formatting(
            bold=font_info.get("bold", False),
            italic=font_info.get("italic", False),
            underline=font_info.get("underline", False), # SDK 지원 여부에 따라 조정
        )

    def _is_list_item(self, text: str) -> bool:
        """리스트 기호(□, o, -, 숫자.) 감지 로직"""
        list_patterns = [r'^□', r'^o\s', r'^-', r'^\d+\.', r'^[가-힣]\.']
        return any(re.match(p, text.strip()) for p in list_patterns)

    # --- 핵심 변환 로직 ---
    def convert(self) -> DoclingDocument:
        # 1. 유효성 검사
        if not self.is_valid():
            raise RuntimeError("Invalid HWP/HWPX document")

        # 🚀 [여기가 누락되었던 부분!] 
        # 문서의 '기원(Origin)' 정보를 생성합니다.
        origin = DocumentOrigin(
            filename=self.source_path.name or "file",
            mimetype="application/x-hwp",  # 패치한 MIME 타입 사용
            binary_hash=getattr(self, "document_hash", "unknown"), # 해시값이 없다면 unknown
        )
        
        # 실제 데이터를 담을 DoclingDocument 객체를 초기화합니다.
        # 이 객체가 바로 우리가 찾던 'doc'입니다.
        doc = DoclingDocument(name=self.source_path.stem or "file", origin=origin)

        # 2. 임시 디렉토리에서 SDK 작업 시작
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            json_out = temp_path / "output.json"
            info_out = temp_path / "output.info"
            img_dir = temp_path / "images"
            img_dir.mkdir()

            # 이미지 경로 끝에 '/' 보장
            img_path_str = str(img_dir)
            if not img_path_str.endswith("/"):
                img_path_str += "/"

            # 3. SDK 실행 명령어 구성
            cmd = [
                "convtext", 
                str(self.source_path.resolve()), 
                str(json_out.resolve()), 
                str(info_out.resolve()), 
                img_path_str
            ]
            
            print(f"DEBUG: Running SDK command: {' '.join(cmd)}")

            # 4. 실제 SDK 실행 (cwd 설정으로 11번 에러 방지)
            subprocess.run(
                cmd, 
                capture_output=True, 
                check=True, 
                text=True,
                cwd=str(SDK_DIR)
            )

            # 5. 결과 JSON 파일 읽기
            '''
            if not json_out.exists():
                raise RuntimeError("SDK failed to produce output JSON")

            with open(json_out, "r", encoding="utf-8") as f:
                try:
                    hwp_data = json.load(f)
                except json.JSONDecodeError:
                    f.seek(0)
                    hwp_data = [json.loads(line) for line in f if line.strip()]
            '''

            hwp_data = []
            with open(json_out, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    clean_line = line.strip()
                    if not clean_line:
                        continue
                    
                    try:
                        # 🚀 strict=False로 제어 문자 에러 방지
                        # 각 라인은 그 자체로 완벽한 리스트(batch)입니다.
                        batch = json.loads(clean_line, strict=False)
                        
                        if isinstance(batch, list):
                            # 리스트 안의 요소들을 하나씩 hwp_data에 통합
                            hwp_data.append(batch) 
                        else:
                            # 혹시 리스트가 아닌 단일 객체라면 그대로 추가
                            hwp_data.append([batch])
                            
                    except json.JSONDecodeError as e:
                        print(f"⚠️ {line_no}행 파싱 실패 (스킵): {e}")
                        continue
            
            # 6. JSON 데이터를 Docling 구조로 변환
            # 위에서 만든 'doc' 객체에 내용을 채워 넣습니다.
            self.current_img_dir = img_dir
            self._walk_hwp_data(hwp_data, doc)
            self.current_img_dir = None

        # 7. 완성된 문서 반환
        return doc

    def _walk_hwp_data(self, data: List[List[Dict]], doc: DoclingDocument):
        """전체 데이터를 순회하며 페이지 및 아이템 추가"""
        root_group = doc.add_group(label=GroupLabel.SECTION, name="root")
        self.parents[0] = root_group
        
        max_page = 1
        for paragraph_items in data:
            # 1. 문단 처리
            self._handle_paragraph(paragraph_items, doc)
            
            # 2. 최대 페이지 계산 (마지막에 Page 객체 생성을 위함)
            for item in paragraph_items:
                p_no = item.get("page", 1)
                max_page = max(max_page, p_no)

        # 3. 문서에 실제 페이지 규격 추가 (A4 기준)
        for p in range(1, max_page + 1):
            doc.add_page(page_no=p, size=Size(width=595, height=842))

    def _handle_paragraph(self, items: List[Dict], doc: DoclingDocument):
        """한 문단 내의 텍스트, 표, 이미지를 처리"""
        # 1. 아이템 추출
        text_item = next((i for i in items if i["item"] == "text"), None)
        table_item = next((i for i in items if i["item"] == "table"), None)
        image_item = next((i for i in items if i["item"] == "image"), None)
        
        page_no = items[0].get("page", 1)

        # 🚀 [수정] 각 아이템의 성격에 맞게 ProvenanceItem을 생성해야 합니다.
        if text_item:
            text_val = text_item.get("value", "").strip()
            # 텍스트 길이에 맞는 charspan 생성
            prov = ProvenanceItem(
                page_no=page_no, 
                bbox=BoundingBox(l=0, t=0, r=1, b=1),
                charspan=(0, len(text_val))# 👈 필수 추가
            )
            self._handle_text(text_item, doc, prov)

        elif table_item:
            # 표는 텍스트 길이가 없으므로 0으로 처리
            prov = ProvenanceItem(
                page_no=page_no, 
                bbox=BoundingBox(l=0, t=0, r=1, b=1),
                charspan=(0, 0) # 👈 필수 추가
            )
            table_data = self._parse_html_table(table_item["value"])
            doc.add_table(data=table_data, parent=self.parents[0], prov=prov)

        elif image_item:
            prov = ProvenanceItem(
                page_no=page_no, 
                bbox=BoundingBox(l=0, t=0, r=1, b=1),
                charspan=(0, 0) # 👈 필수 추가
            )
            self._handle_image(image_item, doc, prov)

    def _handle_text(self, item: Dict, doc: DoclingDocument, prov: ProvenanceItem):
        text_val = item.get("value", "").strip()
        if not text_val: return

        font_info = item.get("font", {})
        fmt = self._map_font_to_formatting(font_info)

        # A. 헤더 판별 (Word Backend 로직 이식)
        if font_info.get("size", 0) >= 20 or font_info.get("bold"):
            level = self._estimate_header_level(font_info)
            self._add_header(doc, level, text_val, prov.page_no)
        
        # B. 리스트 아이템 판별
        elif self._is_list_item(text_val):
            # 실제 제품에서는 여기서 ListGroup을 관리해야 함 (생략 가능하나 권장)
            doc.add_text(label=DocItemLabel.LIST_ITEM, text=text_val, parent=self.parents[0], prov=prov)
            
        # C. 일반 본문
        else:
            doc.add_text(
                label=DocItemLabel.PARAGRAPH,
                text=text_val,
                formatting=fmt,
                parent=self.parents[0],
                prov=prov
            )

    # --- 기존 msword_backend의 핵심 로직들 재구현 ---

    def _add_header(self, doc: DoclingDocument, level: int, text: str, page_no: int):
        # 하위 레벨 초기화
        for i in range(level, self.max_levels):
            self.parents[i] = None
        
        # 상위 그룹이 없으면 생성
        if level > 0 and self.parents[level-1] is None:
            self.parents[level-1] = doc.add_group(label=GroupLabel.SECTION, name=f"section-L{level-1}")

        # 🚀 [수정] 헤더 추가 시에도 ProvenanceItem에 charspan 추가
        self.parents[level] = doc.add_heading(
            text=text, 
            level=level, 
            parent=self.parents[level-1],
            prov=ProvenanceItem(
                page_no=page_no, 
                bbox=BoundingBox(l=0, t=0, r=1, b=1),
                charspan=(0, len(text))# 👈 필수 추가
            )
        )
        self._update_history(text, level, page_no)

    def _handle_image(self, item: Dict, doc: DoclingDocument, prov: ProvenanceItem):
        """SDK가 생성한 이미지 파일을 읽어 DoclingDocument에 추가"""
        img_filename = item.get("value", "") # 예: "picture1.png"
        if not img_filename or not self.current_img_dir:
            return

        # 🚀 이미지의 전체 경로 계산
        img_path = self.current_img_dir / img_filename
        
        if img_path.exists():
            from PIL import Image
            try:
                pil_img = Image.open(img_path)
                # Docling 표준 방식에 따라 사진 추가
                doc.add_picture(
                    image=ImageRef.from_pil(pil_img, dpi=72),
                    parent=self.parents[0],
                    prov=prov
                )
                print(f"✅ 이미지 삽입 성공: {img_filename}")
            except Exception as e:
                print(f"⚠️ 이미지 로드 에러 ({img_filename}): {e}")
        else:
            print(f"⚠️ 이미지를 찾을 수 없음: {img_path}")

    def _parse_html_table(self, html_str: str) -> TableData:
        """HTML 표를 파싱하여 Docling의 TableData 객체로 변환"""
        soup = BeautifulSoup(html_str, "html.parser")
        table_tag = soup.find("table")
        if not table_tag: 
            return TableData(num_rows=0, num_cols=0)

        rows = table_tag.find_all("tr")
        # 컬럼 수 계산
        num_cols = 0
        if rows:
            num_cols = sum(int(td.get("colspan", 1)) for td in rows[0].find_all(["td", "th"]))
        
        # TableData 초기화
        table_data = TableData(num_rows=len(rows), num_cols=num_cols)

        for r_idx, tr in enumerate(rows):
            c_idx_offset = 0
            for td in tr.find_all(["td", "th"]):
                rs, cs = int(td.get("rowspan", 1)), int(td.get("colspan", 1))
                
                # Docling TableCell 규격에 맞게 데이터 삽입
                table_data.table_cells.append(TableCell(
                    text=td.get_text(strip=True),
                    start_row_offset_idx=r_idx, 
                    end_row_offset_idx=r_idx + rs,
                    start_col_offset_idx=c_idx_offset, 
                    end_col_offset_idx=c_idx_offset + cs,
                    column_header=(tr.name == "th" or r_idx == 0)
                ))
                c_idx_offset += cs
        return table_data

    def _estimate_header_level(self, font_info: Dict) -> int:
        size = font_info.get("size", 10)
        if size >= 28: return 1
        if size >= 20: return 2
        return 3

    @override
    def unload(self):
        """추상 메서드 구현: 리소스 해제"""
        if self.temp_input_path and self.temp_input_path.exists():
            try:
                os.remove(self.temp_input_path)
            except Exception:
                pass
        self.temp_input_path = None