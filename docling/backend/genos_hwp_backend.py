import json
import logging
import os
import subprocess
import tempfile
import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from typing_extensions import override  # 상단 임포트에 추가
from io import BytesIO

try:
    from wand.image import Image as WandImage
    from wand.exceptions import WandException # 👈 예외 클래스 추가
    WAND_AVAILABLE = True
except ImportError:
    WAND_AVAILABLE = False

from bs4 import BeautifulSoup
from docling_core.types.doc import (
    DocItemLabel, DoclingDocument, DocumentOrigin, GroupLabel,
    ImageRef, NodeItem, ProvenanceItem, TableCell, TableData,
    BoundingBox, CoordOrigin, Size, Formatting
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
    def __init__(self, in_doc: InputDocument, path_or_stream: Union[Path, BytesIO], **kwargs) -> None:
        super().__init__(in_doc, path_or_stream)

        # 1. PipelineOptions 등에서 넘어오는 이미 저장 여부 설정 받기
        # kwargs에 없으면 기본적으로 True로 설정합니다.
        self.save_images = kwargs.get("save_images", True)
        
        # 만약 WMF 변환 포함 여부도 기존처럼 쓰고 싶다면 추가
        self.include_wmf = kwargs.get("include_wmf", True)

        self._processed_hashes = set()  # 중복 텍스트(머리말/꼬리말) 필터링용
        
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
        """추상 메서드 구현: 페이지 단위 처리를 지원하는지 여부 (HWP는 대개 False이나, 자유소프트 SDK는 true)"""
        return True
    
    def _get_active_parent(self):
        """현재 트리에서 가장 하위에 있는(None이 아닌) 부모 노드를 반환합니다."""
        for level in sorted(self.parents.keys(), reverse=True):
            if self.parents[level] is not None:
                return self.parents[level]
        return None # 정 없으면 Root

    def _get_label_and_level_hwp(self, text, size, is_bold):
        """폰트와 텍스트 패턴으로 p_style_id와 p_level을 결정합니다."""
        # 명시적 헤더 패턴 (1. , 가. , Ⅰ. 등)
        is_explicit_pattern = bool(re.match(r'^(?:\d+\.|\*|[-•]|[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\.)\s+', text))
        
        if is_explicit_pattern or size >= 18 or is_bold:
            # 폰트 크기에 따라 계층(Level) 세분화
            level = 1 if size >= 20 else 2
            return "Heading", level
            
        return "Normal", 0

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

        # 2. 문서의 '기원(Origin)' 정보 생성
        # a) 확장자에 따른 MIME 타입 동적 할당
        file_ext = self.source_path.suffix.lower()
        if file_ext == ".hwpx":
            mimetype = "application/vnd.hancom.hwpx"
        elif file_ext == ".hwp":
            mimetype = "application/x-hwp"
        else:
            mimetype = "application/octet-stream" # 알 수 없는 경우 기본값
        
        # b) binary_hash는 보통 정수형(int)을 기대하므로, 
        # 만약 문자열 해시라면 정수로 변환하거나 적절히 처리해야 합니다.
        try:
            bin_hash = int(self.document_hash, 16) & ((1 << 64) - 1) if hasattr(self, "document_hash") else 0
        except (ValueError, TypeError):
            bin_hash = 0

        # c) origin 정보 생성 부분
        origin = DocumentOrigin(
            filename=self.source_path.name or "file",
            mimetype=mimetype,
            binary_hash=bin_hash,
        )

        # 3. 실제 데이터를 담을 DoclingDocument 객체를 초기화합니다.
        doc = DoclingDocument(name=self.source_path.stem or "file", origin=origin)

        #print(f"self.source_path:{self.source_path}")

        # 4. 임시 디렉토리에서 SDK 작업 시작
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

            # 4-a) SDK 실행 명령어 구성
            cmd = [
                "convtext", 
                str(self.source_path.resolve()), 
                str(json_out.resolve()), 
                str(info_out.resolve()), 
                img_path_str
            ]
            
            print(f"DEBUG: Running SDK command: {' '.join(cmd)}")

            # 4-b) 실제 SDK 실행 (cwd 설정으로 11번 에러 방지)
            subprocess.run(
                cmd, 
                capture_output=True, 
                check=True, 
                text=True,
                cwd=str(SDK_DIR)
            )

            # 5. 자유소프트의 결과중, '.info'를 활용해서 페이지 길이 & 크기 정보 DoclingDocument에 설정 
            self._setup_pages(doc, info_out)

            # [페이지 설정 후 값 확인]
            print(f"\n--- 📄 페이지 설정 결과 확인 (총 {len(doc.pages)}개) ---")
            if not doc.pages:
                print("❌ [ERROR] 등록된 페이지 정보가 없습니다!")
            else:
                for page_no, page_item in sorted(doc.pages.items()):
                    # page_item.size에서 width와 height를 가져옵니다.
                    w = page_item.size.width
                    h = page_item.size.height
                    print(f"📍 [Page {page_no:02}] Size: {w} x {h} pt")
            print("------------------------------------------\n")

            # 6. 자유소프트 SDK의 결과를 hwp_data에 저장
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
            
            # [hwp_data 임시로 저장]
            # 스크립트를 실행한 현재 디렉토리(CWD)에 저장
            debug_json_name = f"{self.source_path.stem}_debug_raw.json"
            debug_json_path = Path.cwd() / debug_json_name
            
            try:
                with open(debug_json_path, "w", encoding="utf-8") as df:
                    json.dump(hwp_data, df, ensure_ascii=False, indent=2)
                
                # 절대 경로로 출력해줘야 나중에 터미널에서 찾기 편합니다.
                print(f"🔍 [DEBUG] SDK 생데이터가 현재 경로에 저장됨: {debug_json_path.resolve()}")
            except Exception as e:
                print(f"⚠️ [DEBUG] 실행 경로 파일 저장 실패: {e}")

            # 7. hwp_data를 Docling 구조로 변환
            # 3번에서 만든 DoclingDocument 객체에 내용을 채워 넣기.
            self.current_img_dir = img_dir
            self._walk_hwp_data(hwp_data, doc)
            self.current_img_dir = None

            # 7.5 [DEBUG] 최종 변환된 DoclingDocument 객체를 JSON으로 저장
            # _walk_hwp_data 처리가 완료된 최종 결과물을 확인합니다.
            debug_doc_name = f"{self.source_path.stem}_debug_docling.json"
            debug_doc_path = Path.cwd() / debug_doc_name
            
            try:
                # Pydantic의 model_dump를 이용해 딕셔너리로 변환 후 저장 (한글 깨짐 방지)
                with open(debug_doc_path, "w", encoding="utf-8") as f:
                    #import json
                    json.dump(doc.model_dump(), f, ensure_ascii=False, indent=2)
                
                print(f"✅ [DEBUG] 최종 DoclingDocument 저장 완료: {debug_doc_path.resolve()}")
            except Exception as e:
                print(f"⚠️ [DEBUG] 최종 문서 저장 실패: {e}")

        # 8. 완성된 DoclingDocument 형태 문서 반환
        return doc

    # --- [Step 1] 텍스트 처리에 집중한 구현 ---
    def _walk_hwp_data(self, data: List[List[Dict]], doc: DoclingDocument):
            """기존 페이지 그룹화 로직을 유지하며 표(Table)와 본문을 분리 처리합니다."""
            self._processed_hashes = set()
            last_page_no = -1
            
            # 1. Root 그룹 생성
            self.parents = {0: doc.add_group(label=GroupLabel.SECTION, name="root")}

            for paragraph_items in data:
                if not paragraph_items:
                    continue
                
                # 페이지 번호 추출 및 페이지 그룹 생성 로직
                page_no = 1
                for item in paragraph_items:
                    if "page" in item:
                        page_no = item.get("page", 1)
                        break
                
                # 페이지 번호 변경시 감지
                if page_no != last_page_no:
                    page_group = doc.add_group(
                        label=GroupLabel.SECTION, 
                        name=f"page_{page_no}", 
                        parent=self.parents[0]
                    )
                    self.parents[1] = page_group 
                    last_page_no = page_no

                # --- [여기서부터 수정 포인트: Table vs Paragraph 분기] ---
                
                texts_in_batch = []
                
                for item in paragraph_items:
                    i_type = str(item.get("item", "")).lower()
                    i_value = item.get("value", "")
                    
                    # 1. 표라면 즉시 처리
                    if i_type == "table" or "<table>" in i_value.lower():
                        # 그동안 쌓인 텍스트가 있다면 먼저 처리 (순서 유지)
                        if texts_in_batch:
                            self._handle_paragraph(texts_in_batch, doc, page_no)
                            texts_in_batch = []
                        
                        self._handle_table(i_value, doc, page_no)
                    
                    # 2. 텍스트라면 리스트에 수집 (나중에 한꺼번에 병합 처리)
                    elif i_type == "text":
                        texts_in_batch.append(item)
                
                # 3. 남은 텍스트들 처리
                if texts_in_batch:
                    self._handle_paragraph(texts_in_batch, doc, page_no)

    def _handle_table(self, html_content: str, doc: DoclingDocument, page_no: int):
        """HTML 테이블을 분석하여 구조화된 Table 객체로 doc에 추가합니다."""
        soup = BeautifulSoup(html_content, "html.parser")
        table_tag = soup.find("table")
        if not table_tag:
            return

        cells = []
        occupied = set() # (row, col) 점유 맵
        rows = table_tag.find_all("tr")

        for r_idx, tr in enumerate(rows):
            c_idx = 0
            for td in tr.find_all(["td", "th"]):
                # 점유된 칸 건너뛰기
                while (r_idx, c_idx) in occupied:
                    c_idx += 1
                
                row_span = int(td.get("rowspan", 1))
                col_span = int(td.get("colspan", 1))
                text = td.get_text(strip=True)
                
                # TableCell 생성
                cell = TableCell(
                    text=text,                             # content 대신 text 사용 권장
                    start_row_offset_idx=r_idx,            # 시작 행
                    end_row_offset_idx=r_idx + row_span,   # 끝 행 (시작 행 + span)
                    start_col_offset_idx=c_idx,            # 시작 열
                    end_col_offset_idx=c_idx + col_span,   # 끝 열 (시작 열 + span)
                    row_span=row_span,    # 추가!
                    col_span=col_span,    # 추가!
                    column_header=True if td.name == "th" else False  # label 대신 이거 사용 가능
                )
                cells.append(cell)
                
                # 점유 표시
                for r in range(r_idx, r_idx + row_span):
                    for c in range(c_idx, c_idx + col_span):
                        occupied.add((r, c))
                
                c_idx += col_span
                
        if cells:
            # 전체 크기 계산
            max_row = max(c.end_row_offset_idx for c in cells)
            max_col = max(c.end_col_offset_idx for c in cells)
            
            # 임시 위치 정보 (추후 Step 3에서 정교화)
            prov = ProvenanceItem(
                page_no=page_no,
                bbox=BoundingBox(l=0, t=0, r=1, b=1, coord_origin=CoordOrigin.BOTTOMLEFT),
                charspan=(0, len(html_content))
            )

            # Docling Document에 표 추가
            doc.add_table(
                data=TableData(num_rows=max_row, num_cols=max_col, table_cells=cells),
                parent=self.parents[1], # 기본 페이지 그룹에 추가
                prov=prov
            )

    def _handle_paragraph(self, paragraph_items: List[Dict], doc: DoclingDocument, page_no: int):
        full_text = "".join([item.get("value", "") for item in paragraph_items]).strip()
        if not full_text: return

        # 폰트 정보 추출
        max_font_size = max([i.get("font", {}).get("size", 10.0) for i in paragraph_items])
        is_bold = any([i.get("font", {}).get("bold", False) for i in paragraph_items])

        # 1. 가상 스타일 판정
        p_style_id, p_level = self._get_label_and_level_hwp(full_text, max_font_size, is_bold)

        prov = ProvenanceItem(
            page_no=page_no,
            bbox=BoundingBox(l=0, t=0, r=1, b=1, coord_origin=CoordOrigin.BOTTOMLEFT),
            charspan=(0, len(full_text))
        )

        # 2. [강등 로직 수정]: 너무 빡빡한 '60자' 제한을 풀고, 
        # 대신 제목 패턴이 명확하면(1. 가. 등) 본문이 좀 길어도 헤더로 인정해 줍니다.
        is_explicit_pattern = bool(re.match(r'^(?:\d+\.|\*|[-•]|[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\.)\s+', full_text))
        
        if p_style_id == "Heading":
            # 패턴이 있으면 150자까지는 헤더로 인정 (마디를 만들기 위함)
            if not is_explicit_pattern and len(full_text) > 80:
                p_style_id = "Normal"
                p_level = 0

        # 3. 추가 실행
        if p_style_id == "Heading":
            self._add_header(doc, p_level, full_text, prov)
        else:
            # 본문 추가 시, 부모를 최상단 부모(parents[1])가 아닌 
            # 현재 활성화된 가장 깊은 레벨의 헤더로 지정해야 합니다.
            doc.add_text(
                label=DocItemLabel.PARAGRAPH,
                text=full_text,
                parent=self._get_active_parent(), # <- 이 함수가 중요!
                prov=prov
            )

    def _handle_text(self, item: Dict, doc: DoclingDocument, prov: ProvenanceItem):
        """텍스트의 성격(중복, TOC, 헤더, 본문)을 판별하여 추가합니다."""
        text_val = item.get("value", "").strip()
        if not text_val: 
            return
        
        # 1. 중복 제거 (머리말/꼬리말 등 방어)
        '''To DO: 현재는 자유 소프트 SDK에서 머리말/꼬리말 자체를 생략하는 상황. 추후 추가 시 처리 로직 구현해야.'''

        # 2. 분석용 변수 추출
        font_info = item.get("font", {})
        font_size = font_info.get("size", 10.0)
        is_bold = font_info.get("bold", False)

        # 3. 패턴 감지 (이전 hwpx 로직 + 현재 정규식 최적화)
        # TOC: 점(2개 이상), 탭, 또는 긴 공백 뒤에 숫자로 끝나는 패턴
        is_toc = bool(re.search(r'(\.{2,}|…|\t|\s{4,})\s*\d+$', text_val))
        
        # Explicit Header: 숫자+점(1. ) 또는 로마자+점(Ⅱ. )으로 시작하는 패턴
        is_explicit_header = bool(re.match(r'^(?:\d+\.\s+|[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\.\s*)', text_val))

        # 4. 분류 및 추가 로직
        if is_toc:
            # --- [DEBUG LOG START] ---
            print(f"\n📑 [TOC DETECTED]")
            print(f"   - Text: {text_val[:30]}...")
            print(f"   - Label: {DocItemLabel.DOCUMENT_INDEX}")
            
            parent_info = self.parents[1]
            p_ref = getattr(parent_info, "self_ref", "N/A")
            p_label = getattr(parent_info, "label", "N/A")
            print(f"   - Parent: {p_label} (Ref: {p_ref})")
            print(f"   - Prov: Page {prov.page_no}, BBox {prov.bbox}")
            print(f"------------------------------------------")
            # --- [DEBUG LOG END] ---
            
            doc.add_text(
                label=DocItemLabel.DOCUMENT_INDEX, 
                text=text_val, 
                parent=self.parents[1], 
                prov=prov
            )

        # 5. 헤더 판별: 명시적 패턴이 있거나, 폰트가 크거나, 볼드체인 경우
        elif is_explicit_header or font_size >= 20 or is_bold:
            level = self._estimate_header_level(font_info)
            # 만약 명시적 패턴(1. 등)이 있는데 폰트가 작다면 레벨을 낮추는 등 미세조정 가능
            self._add_header(doc, level, text_val, prov)

        else:
            # 6. 일반 본문
            doc.add_text(
                label=DocItemLabel.PARAGRAPH, 
                text=text_val, 
                parent=self.parents[1], 
                prov=prov
            )

    def _add_header(self, doc: DoclingDocument, level: int, text: str, prov: ProvenanceItem):
        """헤더 계층을 관리하며 Heading 아이템을 추가합니다."""
        # 하위 레벨 부모들 초기화
        for i in range(level, 10): # max_levels 가정
            if i in self.parents and i > 1: # root와 page_group은 유지
                self.parents[i] = None

        # 상위 계층이 비어있으면 현재 페이지 그룹 아래에 배치
        parent_node = self.parents.get(level - 1) or self.parents[1]

        # Heading 추가 및 부모 등록
        header_item = doc.add_heading(
            text=text, 
            level=level, 
            parent=parent_node,
            prov=prov
        )
        self.parents[level] = header_item

    def _estimate_header_level(self, font_info: Dict) -> int:
        """폰트 크기에 따른 헤더 레벨 추정"""
        size = font_info.get("size", 10.0)
        if size >= 28: return 1
        if size >= 20: return 2
        return 3
    
    def _setup_pages(self, doc: DoclingDocument, info_path: Path):
        """ .info 파일을 읽어 DoclingDocument에 페이지 정보를 설정합니다. """
        
        # 단위 변환 상수: 1cm = 28.3465pt (72 DPI 기준)
        CM_TO_PT = 28.3465
        
        try:
            with open(info_path, "r", encoding="utf-8") as f:
                info_data = json.load(f)
            
            page_info_list = info_data.get("page_info", [])
            
            if page_info_list:
                # 1. 실제 페이지 정보가 있는 경우
                for p in page_info_list:
                    p_no = p.get("page")
                    # cm를 pt로 변환 (정수형으로 변환하는 것이 일반적입니다)
                    w_pt = round(float(p.get("width", 21.00)) * CM_TO_PT)
                    h_pt = round(float(p.get("height", 29.70)) * CM_TO_PT)
                    
                    doc.pages[p_no] = doc.add_page(
                        page_no=p_no, 
                        size=Size(width=w_pt, height=h_pt)
                    )
                print(f"✅ 총 {len(page_info_list)}페이지 정보 로드 완료.")
            else:
                raise ValueError("page_info is empty")

        except Exception as e:
            # 2. 오류가 나거나 정보가 없는 경우 (Fallback)
            print(f"⚠️ 페이지 정보 로드 실패({e}). 기본 1페이지로 설정합니다.")
            doc.pages[1] = doc.add_page(
                page_no=1, 
                size=Size(width=595, height=842) # 표준 A4 pt
            )

    @override
    def unload(self):
        """추상 메서드 구현: 리소스 해제"""
        if self.temp_input_path and self.temp_input_path.exists():
            try:
                os.remove(self.temp_input_path)
            except Exception:
                pass
        self.temp_input_path = None