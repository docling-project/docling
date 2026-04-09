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
from PIL import Image, UnidentifiedImageError
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
        # 자유소프트 SDK 사용 직후의 결과 저장 여부
        self.jayu_sdk_save = kwargs.get("jayu_sdk_save", False)

        print(f"(init)⚠️ self.jayu_sdk_save: {self.jayu_sdk_save}")

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
        self.original_path = Path(in_doc.file) if in_doc.file else None  # 원본 입력 경로 보존
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
        # 만약 문자열 해시라면 정수로 변환하거나 적절히 처리해야.
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

        # 4. 작업 디렉토리 결정 (임시 vs 영구)
        if self.jayu_sdk_save:
            # 영구 저장: 원본 파일의 부모 폴더 / jayu_sdk_result / {파일명} 구조로 생성
            base = self.original_path or self.source_path
            work_dir = base.parent / "jayu_sdk_result" / base.stem
            work_dir.mkdir(parents=True, exist_ok=True)
            temp_dir_context = None  # 삭제할 임시 컨텍스트 없음
            print(f"(if) ⚠️ work_dir: {work_dir}")
        else:
            # 임시 저장: 기존처럼 tempfile 사용
            temp_dir_context = tempfile.TemporaryDirectory()
            work_dir = Path(temp_dir_context.name)

        try:
            # 경로 설정 (work_dir 기준)
            json_out = work_dir / "output.json"
            info_out = work_dir / "output.info"
            img_dir = work_dir / "images"
            img_dir.mkdir(exist_ok=True)

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

            # 4-b) 실제 SDK 실행
            subprocess.run(
                cmd, 
                capture_output=True, 
                check=True, 
                text=True,
                cwd=str(SDK_DIR)
            )

            print(f"⚠️⚠️ json_out: {json_out}")
            print(f"⚠️⚠️ info_out: {info_out}")
            print(f"⚠️⚠️ img_dir: {img_dir}")

            # 5. '.info' 활용 설정
            self._setup_pages(doc, info_out)

            # 6. SDK 결과를 hwp_data에 저장 (읽기 로직 단순화)
            hwp_data = []
            with open(json_out, "r", encoding="utf-8") as f:
                for line in f:
                    clean_line = line.strip()
                    if not clean_line:
                        continue
                    try:
                        batch = json.loads(clean_line, strict=False)
                        hwp_data.append(batch) # SDK 결과는 항상 리스트이므로 바로 append
                    except json.JSONDecodeError as e:
                        print(f"⚠️ 파싱 실패 (스킵): {e}")
                        continue

            # 7. hwp_data를 Docling 구조로 변환
            self.current_img_dir = img_dir
            self._walk_hwp_data(hwp_data, doc)
            self.current_img_dir = None

        finally:
            # 8. 사후 정리: 임시 디렉토리인 경우에만 삭제 실행
            if temp_dir_context is not None:
                temp_dir_context.cleanup()
                print("DEBUG: Temporary directory cleaned up.")
            else:
                print(f"DEBUG: SDK outputs saved at: {work_dir}")

        return doc

    # --- DoclingDocument 객체에 자유소프트 SDK의 결과를 채워주는 함수 ---
    def _walk_hwp_data(self, data: List[List[Dict]], doc: DoclingDocument):
        """페이지 그룹화를 제거하고 모든 아이템을 body에 직접 나열하여 DOCX 스타일로 구성합니다."""
        self._processed_hashes = set()
        root_parent = doc.body 
        self.active_main_parent = root_parent # 🚀 초기값 설정

        for paragraph_items in data:
            if not paragraph_items:
                continue
            
            # 1. 페이지 번호 추출 (Provenance/메타데이터용으로만 사용)
            page_no = 1
            for item in paragraph_items:
                if "page" in item:
                    page_no = item.get("page", 1)
                    break

            # --- [아이템 분기 처리: Text & Table] ---
            texts_in_batch = []
            for item in paragraph_items:
                i_type = str(item.get("item", "")).lower()
                i_value = item.get("value", "")
                
                # 2-1. 표 처리
                if i_type == "table" or "<table>" in i_value.lower():
                    if texts_in_batch:
                        # 쌓인 텍스트 본문을 처리할 때 parent를 doc.body로 지정
                        self._handle_paragraph(texts_in_batch, doc, page_no, parent=self.active_main_parent)
                        texts_in_batch = []
                    # 표를 처리할 때 parent를 doc.body로 지정
                    self._handle_table(i_value, doc, page_no, parent=self.active_main_parent)
                
                # 🚀 2. [Step 3] 이미지(Picture) 처리 추가
                elif i_type == "image":
                    if texts_in_batch:
                        self._handle_paragraph(texts_in_batch, doc, page_no, parent=self.active_main_parent)
                        texts_in_batch = []
                    # 이미지 핸들러 호출
                    self._handle_image(item, doc, page_no, parent=self.active_main_parent)

                # 2-3. 텍스트 수집
                elif i_type == "text":
                    texts_in_batch.append(item)
            
            # 3. 루프 종료 후 남은 텍스트들 처리
            if texts_in_batch:
                self._handle_paragraph(texts_in_batch, doc, page_no, parent=self.active_main_parent)

    def _handle_table(self, html_content: str, doc: DoclingDocument, page_no: int, parent: Any):
        """HTML 테이블을 분석하여 구조화된 Table 객체를 지정된 parent(body)에 추가합니다."""
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
                
                # TableCell 생성 (기존 로직 유지)
                cell = TableCell(
                    text=text,
                    start_row_offset_idx=r_idx,
                    end_row_offset_idx=r_idx + row_span,
                    start_col_offset_idx=c_idx,
                    end_col_offset_idx=c_idx + col_span,
                    row_span=row_span,
                    col_span=col_span,
                    column_header=True if td.name == "th" else False
                )
                cells.append(cell)
                
                # 점유 표시 (기존 로직 유지)
                for r in range(r_idx, r_idx + row_span):
                    for c in range(c_idx, c_idx + col_span):
                        occupied.add((r, c))
                
                c_idx += col_span
                
        if cells:
            # 전체 크기 계산
            max_row = max(c.end_row_offset_idx for c in cells)
            max_col = max(c.end_col_offset_idx for c in cells)
            
            # 위치 정보 (Provenance) 생성
            prov = ProvenanceItem(
                page_no=page_no,
                bbox=BoundingBox(l=0, t=0, r=1, b=1, coord_origin=CoordOrigin.BOTTOMLEFT),
                charspan=(0, len(html_content))
            )

            # --- [수정 포인트]: 넘겨받은 parent(doc.body)를 사용합니다 ---
            doc.add_table(
                data=TableData(
                    num_rows=max_row, 
                    num_cols=max_col, 
                    table_cells=cells # 필드명 table_cells 확인!
                ),
                parent=parent, # <--- self.parents[1] 대신 인자로 받은 parent 사용
                prov=prov
            )

    def _handle_image(self, item: Dict, doc: DoclingDocument, page_no: int, parent: Any):
        # 1. JSON에 적힌 값(이미지 경로)을 가져옴 (예: "/tmp/old_path/images/image6.bmp")
        img_path = item.get("value", "")

        if not img_path or not os.path.exists(img_path):
            return

        # 🚀 [Salvaged 1] 매직 넘버 기반의 강력한 유효성 검사 (XML/가짜파일 방어)
        def is_really_image(file_path):
            signatures = [
                b'\x89PNG',           # PNG
                b'\xff\xd8\xff',      # JPEG
                b'GIF8',              # GIF
                b'BM',                # BMP
                b'RIFF',              # WebP (RIFF 컨테이너)
                b'\x00\x00\x01\x00', # ICO
                b'\xd7\xcd\xc6\x9a', # WMF
                b'\x01\x00\x00\x00', # EMF
                b'II*\x00',          # TIFF (little-endian)
                b'MM\x00*',          # TIFF (big-endian)
            ]
            try:
                with open(file_path, 'rb') as f:
                    header = f.read(4)
                return any(header.startswith(s) for s in signatures)
            except: return False

        if not is_really_image(img_path):
            print(f"⚠️ 유효하지 않은 이미지 데이터(XML 등 가능성): {os.path.basename(img_path)}")
            return

        pil_image = None
        
        # 🚀 [Salvaged 2] Pillow -> Wand 단계별 시도 (WMF/EMF 구제)
        try:
            # 1단계: 표준 포맷 시도
            pil_image = Image.open(img_path)
            pil_image.load() # 강제 로드하여 오류 조기 감지
        except (UnidentifiedImageError, OSError):
            # 2단계: Pillow 실패 시 Wand 가동
            if WAND_AVAILABLE:
                try:
                    with WandImage(filename=img_path) as wand_img:
                        wand_img.format = 'png'
                        pil_image = Image.open(BytesIO(wand_img.make_blob()))
                        print(f"🪄 Wand로 복구 성공: {os.path.basename(img_path)}")
                except Exception as e:
                    print(f"❌ Wand 변환 실패: {e}")
            else:
                print(f"⚠️ Pillow 실패 및 Wand 미설치로 복구 불가: {img_path}")

        # 🚀 [Salvaged 3] Docling 임베딩 (DPI 고정 및 BBox 설정)
        if pil_image:
            image_ref = ImageRef.from_pil(image=pil_image, dpi=72)
            
            doc.add_picture(
                parent=parent,
                image=image_ref,
                caption=item.get("image", {}).get("title", "그림"),
                prov=ProvenanceItem(
                    page_no=page_no,
                    # HWP 좌표계에 맞춰 TOPLEFT 명시
                    bbox=BoundingBox(
                        l=0.1, t=0.1, r=0.4, b=0.4, 
                        coord_origin=CoordOrigin.TOPLEFT
                    ),
                    charspan=(0, 0)
                )
            )
            #print(f"✅ 이미지 임베딩 완료: {os.path.basename(img_path)}")

    def _handle_paragraph(self, paragraph_items: List[Dict], doc: DoclingDocument, page_no: int, parent: Any):
        """TOC(목차) 감지 로직이 추가된 버전입니다. 넘겨받은 parent에 텍스트를 추가합니다."""
        
        # 1. 텍스트 병합 및 기본 검사
        full_text = "".join([item.get("value", "") for item in paragraph_items]).strip()
        if not full_text: 
            return

        # 2. 폰트/스타일 정보 추출
        max_font_size = max([i.get("font", {}).get("size", 10.0) for i in paragraph_items])
        is_bold = any([i.get("font", {}).get("bold", False) for i in paragraph_items])

        # 가상 스타일 판정
        p_style_id, p_level = self._get_label_and_level_hwp(full_text, max_font_size, is_bold)

        # 3. 패턴 감지 (TOC 및 헤더)
        # 🚀 [추가]: TOC 패턴 감지 (점 2개 이상, 탭, 또는 긴 공백 뒤에 숫자로 끝나는 경우)
        is_toc = bool(re.search(r'(\.{2,}|…|\t|\s{4,})\s*\d+$', full_text))
        
        # 명시적 헤더 패턴 (1., 가., * 등)
        is_explicit_pattern = bool(re.match(r'^(?:\d+\.|\*|[-•]|[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\.)\s+', full_text))
        
        # 4. 강등 로직
        if p_style_id == "Heading":
            if not is_explicit_pattern and len(full_text) > 80:
                p_style_id = "Normal"
                p_level = 0

        # 위치 정보(Provenance) 생성
        prov = ProvenanceItem(
            page_no=page_no,
            bbox=BoundingBox(l=0, t=0, r=1, b=1, coord_origin=CoordOrigin.BOTTOMLEFT),
            charspan=(0, len(full_text))
        )

        # 5. 최종 문서 추가 (TOC -> Heading -> Paragraph 순서로 우선순위 적용)
        if is_toc:
            # 목차로 판별된 경우
            doc.add_text(
                label=DocItemLabel.DOCUMENT_INDEX, 
                text=full_text, 
                parent=parent, 
                prov=prov
            )
        elif p_style_id == "Heading":
            # 헤더로 판별된 경우 (수정된 _add_header 호출)
            self._add_header(doc, p_level, full_text, prov, parent=parent)
        else:
            # 일반 본문으로 판별된 경우
            doc.add_text(
                label=DocItemLabel.PARAGRAPH,
                text=full_text,
                parent=parent,
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
            #print(f"\n📑 [TOC DETECTED]")
            #print(f"   - Text: {text_val[:30]}...")
            #print(f"   - Label: {DocItemLabel.DOCUMENT_INDEX}")
            
            parent_info = self.parents[1]
            p_ref = getattr(parent_info, "self_ref", "N/A")
            p_label = getattr(parent_info, "label", "N/A")
            #print(f"   - Parent: {p_label} (Ref: {p_ref})")
            #print(f"   - Prov: Page {prov.page_no}, BBox {prov.bbox}")
            #print(f"------------------------------------------")
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

    def _add_header(self, doc: DoclingDocument, level: int, text: str, prov: ProvenanceItem, parent: Any):
        """DOCX 표준 명칭(header-N)을 사용하여 논리적 섹션 마디를 만듭니다."""
        
        # 1. 하위 계층 부모 초기화 (기존 로직 유지)
        for i in range(level, 10):
            if i in self.parents:
                self.parents[i] = None

        # 🚀 2. [명칭 변경]: level 1 -> header-0, level 2 -> header-1 방식으로 명명
        header_name = f"header-{level - 1}"
        
        # 3. 새로운 섹션 그룹 생성
        # parent는 doc.body 또는 상위 헤더 그룹이 됩니다.
        new_section = doc.add_group(
            label=GroupLabel.SECTION, 
            name=header_name, # "header-0", "header-1" 등으로 박힘
            parent=parent
        )
        
        # 4. 헤더 텍스트 추가 (이 그룹의 자식으로 등록)
        header_item = doc.add_heading(
            text=text, 
            level=level, 
            parent=new_section,
            prov=prov
        )

        # 5. 현재 활성 그룹 상태 업데이트
        # 이후에 나오는 paragraph들이 이 'header-N' 그룹 안으로 들어오게 됩니다.
        self.parents[level] = new_section
        
        # _walk_hwp_data에서 문단들이 참고할 최신 부모 노드
        self.active_main_parent = new_section

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