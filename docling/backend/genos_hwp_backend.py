import base64
import binascii
import json
import logging
import os
import subprocess
import tempfile
import hashlib
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from typing_extensions import override  # 상단 임포트에 추가
from PIL import Image, UnidentifiedImageError
from io import BytesIO

try:
    from wand.image import Image as WandImage
    from wand.exceptions import WandException
    WAND_AVAILABLE = True
except ImportError:
    WAND_AVAILABLE = False

from bs4 import BeautifulSoup, NavigableString
from docling_core.types.doc import (
    DocItemLabel, DoclingDocument, DocumentOrigin, GroupLabel,
    ImageRef, NodeItem, ProvenanceItem, TableCell, TableData,
    BoundingBox, CoordOrigin, Size
)
from docling.backend.abstract_backend import DeclarativeDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument
from docling.exceptions import HwpConversionError

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
    _log.warning(f"HWP SDK 경로를 찾을 수 없습니다: {SDK_PATH_STR}")
# ------------------------------

# dump_sdk_output=True 시 HWP SDK 중간 산출물이 저장되는 디렉터리
_SDK_DEBUG_OUTPUT_DIR = Path("/tmp/docparser_debug")

# --- [헤더 판별용 마커 패턴] (Issue #206) ---
# 마커를 3계층으로 구분한다 (실측 10종 공문서 회귀로 도출):
#   1) divider  — 장/조/붙임/별표 등 "문서 구획 분절자". 폰트 무관 heading, 길이/문장 강등 면제
#                 (조항 헤더는 SDK가 본문과 한 단락으로 합쳐도 heading 유지해야 하므로 면제).
#   2) number   — 로마자./번호(N., N.N). 폰트 무관 heading이되, 긴 문장/서술형이면 본문으로 강등.
#   3) weak     — 글머리기호·①②·(1)·가. 등. 본문 글머리로도 흔해 "강조(bold/큰폰트) AND"일 때만 heading.
# (level 힌트: 1=장/구획, 2=절/번호, 3=세부)

# divider: 폰트·길이·문장 무관하게 heading (강등 면제)
_HEADING_DIVIDER_MARKERS = [
    (re.compile(r'^제\s*\d+\s*[편장]\b'), 1),                 # 제1편 / 제 2 장
    (re.compile(r'^제\s*\d+\s*[절관조]\b'), 2),               # 제1절 / 제2조
    # 붙임/별첨/별표/별지/참고/첨부/서식/양식/부록 — 괄호([<〈【) 안에 있어도 인식
    (re.compile(r'^[\[<〈【(]?\s*(?:붙임|별첨|별표|별지|참고|첨부물|첨부|서식|양식|부록)\b'), 1),
]

# number: 로마자./번호 — heading이되 긴 서술형이면 강등 대상
_HEADING_NUMBER_MARKERS = [
    (re.compile(r'^[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\.\s'), 1),               # Ⅰ. Ⅱ.
    (re.compile(r'^\d{1,2}\.\s'), 2),                        # 1.  (연도 등 3자리+ 숫자 제외)
    (re.compile(r'^\d{1,2}\.\d+(?:\.\d+)*\s'), 2),           # 1.1, 1.2.3
]

# 약한 마커(공문서 글머리 기호 + 불릿 + ①②·(1)·가.): 본문 글머리로도 흔히 쓰여
#   단독으로는 본문일 가능성이 커서 "강조(bold/큰폰트)"와 AND 결합 시에만 heading으로 본다.
#   Issue #206의 "마커 패턴 확장(❏ □ ■ ◆ ○ ● 등)"도 이 집합으로 흡수한다.
#   (실측: 조항 하위의 ①②, 입찰서의 가./나. 등은 baseline 폰트·비bold라 강한마커면 과다 검출됨)
_HEADING_WEAK_MARKER = re.compile(
    r'^(?:[❏□■◆◇○●▶▷▪▫※◦◈∙·•*\-①-⑳]\s|[가-힣]\.\s|\((?:[가-힣]|\d+)\)\s)'
)

# heading 후보가 "제목"이 아니라 서술형 본문 문장이면 강등한다 (divider 제외).
#   한국 공문서 본문은 종결형 '다./함/음/됨/임' 또는 의문형 '까?'로 끝나는 경우가 많다.
#   '요'는 명사(개요/요약 등) 오인이 많아 제외. 제목은 보통 명사로 끝난다.
_SENTENCE_FINAL = re.compile(r'(?:다|함|음|됨|임)[.)\]」』】〉]?\s*$|까[?]\s*$')
# 마커가 있어도 이 길이를 넘으면 제목이 아니라 본문으로 본다 (divider 제외).
_HEADING_MAX_LEN = 80

# 단락의 강조(bold/큰폰트) 비율 임계값 — 이 비율 이상일 때만 "강조 단락"으로 본다.
_EMPHASIS_RATIO_THRESHOLD = 0.7
# baseline(본문 최빈 폰트) 대비 이 배수 이상이면 "큰 폰트"로 본다.
_LARGE_FONT_FACTOR = 1.15
# heading run이 baseline 대비 얼마나 큰지로 L1/L2/L3 세분화.
_LEVEL1_FONT_FACTOR = 1.5
_LEVEL2_FONT_FACTOR = 1.25

class GenosHwpDocumentBackend(DeclarativeDocumentBackend):
    def __init__(self, in_doc: InputDocument, path_or_stream: Union[Path, BytesIO], **kwargs) -> None:
        super().__init__(in_doc, path_or_stream)

        self.include_wmf = kwargs.get("include_wmf", True)
        # include_wmf=True이면 save_images도 자동으로 True (hwpx_backend 동일 패턴)
        self.save_images = kwargs.get("save_images", True) or self.include_wmf
        # HWP SDK 중간 산출물(JSON, 이미지 등)을 디버깅 목적으로 보존할지 여부
        # 저장 기본 경로는 환경변수 DOCPARSER_OUTPUT_DIR로 서버에서 지정 (없으면 원본 파일 옆)
        self.dump_sdk_output = kwargs.get("dump_sdk_output", False)

        self._processed_hashes = set()  # 중복 텍스트(머리말/꼬리말) 필터링용

        # 문서 전체 최빈 폰트 크기(본문 baseline). 사전 패스에서 산출하며,
        # 헤더 판별의 "큰 폰트" 기준을 절대값 대신 이 값에 대한 상대값으로 잡는다. (Issue #206)
        self.body_font_size = 10.0

        # 1. 환경 설정
        self.valid = False
        
        # 2. 계층 및 상태 관리 (GenosMsWord 방식 이식)
        self.max_levels = 10
        self.parents: Dict[int, Optional[NodeItem]] = {i: None for i in range(-1, self.max_levels)}
        self.history: Dict[str, List[Any]] = {
            "names": [None], "levels": [None], "page_nos": [1]
        }
        
        # 3. 소스 파일 준비
        self.original_path = Path(in_doc.file) if in_doc.file else None
        self.temp_input_path = None
        if isinstance(path_or_stream, BytesIO):
            suffix = self._infer_suffix(path_or_stream, in_doc)
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(path_or_stream.getbuffer())
                self.temp_input_path = Path(tmp.name)
            self.source_path = self.temp_input_path
        else:
            self.source_path = path_or_stream

        if self.source_path and self.source_path.exists():
            self.valid = True

    @staticmethod
    def _infer_suffix(stream: BytesIO, in_doc: InputDocument) -> str:
        # 1순위: in_doc.file에 확장자가 있으면 그대로 사용
        if in_doc.file:
            suffix = Path(in_doc.file).suffix
            if suffix:
                return suffix

        # 2순위: 스트림 매직 바이트로 판별
        header = stream.read(4)
        stream.seek(0)  # 반드시 되감기

        if header[:2] == b"PK":               # ZIP 시그니처 → HWPX
            return ".hwpx"
        if header == b"\xd0\xcf\x11\xe0":     # OLE 시그니처 → HWP
            return ".hwp"

        # 3순위: InputFormat으로 폴백
        if getattr(in_doc, "format", None) == InputFormat.XML_HWPX:
            return ".hwpx"

        return ".hwp"

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
        """추상 메서드 구현: 페이지 단위 처리를 지원하는지 여부 (HWP는 대개 False이나, hwp_sdk는 true)"""
        return True
    
    @staticmethod
    def _match_structural_marker(text: str):
        """강한 마커(divider/number)에 매칭되면 (level, kind)를 반환, 아니면 (None, None).

        kind: 'divider'(장/조/붙임/별표 등 — 강등 면제) | 'number'(로마자./번호 — 강등 대상).
        약한 마커(글머리기호·①②·가. 등)는 여기서 매칭하지 않고 _HEADING_WEAK_MARKER로 따로 본다.
        """
        for pattern, level_hint in _HEADING_DIVIDER_MARKERS:
            if pattern.match(text):
                return level_hint, "divider"
        for pattern, level_hint in _HEADING_NUMBER_MARKERS:
            if pattern.match(text):
                return level_hint, "number"
        return None, None

    def _level_from_font(self, max_size: float, marker_level: Optional[int]) -> int:
        """baseline 대비 폰트 비율로 L1/L2/L3를 결정한다.
        폰트가 본문과 비슷해 분간이 안 되면(구조 마커만으로 heading인 경우)
        마커가 준 level 힌트를 사용한다. (Issue #206)
        """
        baseline = self.body_font_size or 10.0
        ratio = max_size / baseline if baseline else 1.0
        if ratio >= _LEVEL1_FONT_FACTOR:
            return 1
        if ratio >= _LEVEL2_FONT_FACTOR:
            return 2
        if ratio >= _LARGE_FONT_FACTOR:
            return 3
        # 폰트만으로는 본문과 구분이 안 됨 → 마커 힌트(없으면 가장 깊은 L3)로 폴백
        return marker_level if marker_level is not None else 3

    def _get_label_and_level_hwp(self, text, max_size, bold_ratio, large_ratio, marker=None):
        """텍스트 패턴 + 강조 비율로 p_style_id와 p_level을 결정한다. (Issue #206)

        판별 규칙:
          (A) 강한 마커(divider=장/조/붙임/별표, number=로마자./번호)는 폰트/강조 무관 heading.
          (B) 약한 마커(글머리기호·①②·(1)·가.)는 "강조(bold/큰폰트) AND"일 때만 heading.
              마커 없는 순수 bold/큰폰트 단락은 heading으로 보지 않는다 (Issue #183 가짜헤더 차단).
        서술형 본문 문장으로의 강등은 호출부(_handle_paragraph)에서 처리한다.
        level은 baseline 대비 폰트 비율로 L1/L2/L3 세분화한다.
        """
        if marker is None:
            marker = self._match_structural_marker(text)
        marker_level, _marker_kind = marker

        # 강조 단락 여부: 단락의 70% 이상이 bold이거나 큰 폰트일 때만.
        is_emphasized = (
            bold_ratio >= _EMPHASIS_RATIO_THRESHOLD
            or large_ratio >= _EMPHASIS_RATIO_THRESHOLD
        )

        # (A) 강한 마커(divider/number) → 단독으로 heading
        if marker_level is not None:
            return "Heading", self._level_from_font(max_size, marker_level)

        # (B) 약한 마커 AND 강조 동시 충족
        if bool(_HEADING_WEAK_MARKER.match(text)) and is_emphasized:
            return "Heading", self._level_from_font(max_size, None)

        return "Normal", 0

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
        if self.dump_sdk_output:
            # 영구 저장: _SDK_DEBUG_OUTPUT_DIR 하위에 파일명 서브폴더로 저장
            base = self.original_path or self.source_path
            # BytesIO 입력은 파일명이 고유하지 않을 수 있으므로 UUID 접미사로 충돌 방지
            if self.temp_input_path is not None:
                subdir_name = f"{base.stem}_{uuid.uuid4().hex[:8]}"
            else:
                subdir_name = base.stem
            work_dir = _SDK_DEBUG_OUTPUT_DIR / subdir_name / "hwp_sdk_result"
            work_dir.mkdir(parents=True, exist_ok=True)
            temp_dir_context = None  # 삭제할 임시 컨텍스트 없음
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
            
            # 4-b) 실제 SDK 실행
            try:
                subprocess.run(
                    cmd,
                    capture_output=True,
                    check=True,
                    text=True,
                    cwd=str(SDK_DIR),
                    timeout=600,
                )
            except subprocess.TimeoutExpired as e:
                raise HwpConversionError(
                    f"SDK 실행 타임아웃 (600초 초과): {self.source_path.name}"
                ) from e
            except subprocess.CalledProcessError as e:
                raise HwpConversionError(
                    f"SDK 실행 실패 (exit code={e.returncode}): {self.source_path.name}\n"
                    f"stderr: {e.stderr}"
                ) from e

            # 5. '.info' 활용 설정
            try:
                self._setup_pages(doc, info_out)
            except Exception as e:
                raise HwpConversionError(
                    f"페이지 정보 로드 실패: {self.source_path.name}"
                ) from e

            # 6. SDK 결과를 hwp_data에 저장
            # SDK 출력은 두 가지 비정상 케이스가 있어 일반 line-by-line json.loads로는 깨진다:
            #   (a) 긴 base64 value(latex 등)에 줄바꿈이 끼어 한 record가 여러 물리 줄에 걸침
            #   (b) 표 셀 HTML 안에 <latex value="..."/>를 임베드하면서 inner "를 escape 안 함
            #       → outer JSON 문자열이 그 지점에서 조기 종료되어 파싱 실패
            # 정규화로 (b)를 보정한 뒤, (a)는 JSONDecoder.raw_decode 기반 stream 파싱으로 처리.
            hwp_data = []
            try:
                with open(json_out, "r", encoding="utf-8") as f:
                    text = f.read()
                text = self._normalize_sdk_json_text(text)
                decoder = json.JSONDecoder(strict=False)
                pos = 0
                n = len(text)
                while pos < n:
                    while pos < n and text[pos] in " \r\n\t":
                        pos += 1
                    if pos >= n:
                        break
                    try:
                        batch, end = decoder.raw_decode(text, pos)
                    except json.JSONDecodeError as e:
                        _log.warning(f"파싱 실패 (스킵): pos={pos} err={e}")
                        next_nl = text.find("\n", pos)
                        if next_nl == -1:
                            break
                        pos = next_nl + 1
                        continue
                    hwp_data.append(batch)
                    pos = end
            except OSError as e:
                raise HwpConversionError(
                    f"SDK 결과 파일 읽기 실패: {json_out}"
                ) from e

            # 7. hwp_data를 Docling 구조로 변환
            self.current_img_dir = img_dir
            self._walk_hwp_data(hwp_data, doc)
            self.current_img_dir = None

        finally:
            # 8. 사후 정리: 임시 디렉토리인 경우에만 삭제 실행
            if temp_dir_context is not None:
                temp_dir_context.cleanup()

        return doc

    # --- DoclingDocument 객체에 HWP SDK의 결과를 채워주는 함수 ---
    @staticmethod
    def _compute_body_font_size(data: List[List[Dict]]) -> float:
        """문서 전체 text run의 폰트 크기를 글자수 가중으로 집계해 최빈값(본문 baseline)을 구한다.

        절대 임계(size>=18) 대신 이 baseline에 대한 상대값으로 "큰 폰트"를 판정하기 위함. (Issue #206)
        run이 하나도 없으면 기본 10.0을 반환한다.
        """
        from collections import Counter

        weighted: "Counter[float]" = Counter()
        for paragraph_items in data:
            if not paragraph_items:
                continue
            for item in paragraph_items:
                if str(item.get("item", "")).lower() != "text":
                    continue
                text = item.get("value", "") or ""
                weight = len(text.strip())
                if weight <= 0:
                    continue
                size = float(item.get("font", {}).get("size", 10.0) or 10.0)
                # 0.5pt 단위로 양자화하여 미세한 차이로 최빈값이 흩어지는 것을 방지
                weighted[round(size * 2) / 2] += weight

        if not weighted:
            return 10.0
        return weighted.most_common(1)[0][0]

    def _walk_hwp_data(self, data: List[List[Dict]], doc: DoclingDocument):
        """페이지 그룹화를 제거하고 모든 아이템을 body에 직접 나열하여 DOCX 스타일로 구성합니다."""
        self._processed_hashes = set()
        # 본문 baseline 폰트 사전 패스 (헤더 상대 판정용)
        self.body_font_size = self._compute_body_font_size(data)
        root_parent = doc.body
        self.active_main_parent = root_parent

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
                
                # 2. [Step 3] 이미지(Picture) 처리 추가
                elif i_type == "image":
                    if texts_in_batch:
                        self._handle_paragraph(texts_in_batch, doc, page_no, parent=self.active_main_parent)
                        texts_in_batch = []
                    # 이미지 핸들러 호출
                    self._handle_image(item, doc, page_no, parent=self.active_main_parent)

                # 2-2. 수식(LaTeX) 처리
                # SDK는 latex를 자체 batch에 단독으로 또는 텍스트 사이에 끼워서 emit할 수 있다.
                # 이미지와 동일한 패턴으로, 누적된 텍스트를 먼저 flush한 뒤 FORMULA 노드로 추가한다.
                elif i_type == "latex":
                    if texts_in_batch:
                        self._handle_paragraph(texts_in_batch, doc, page_no, parent=self.active_main_parent)
                        texts_in_batch = []
                    self._handle_latex(item, doc, page_no, parent=self.active_main_parent)

                # 2-3. 텍스트 수집
                elif i_type == "text":
                    texts_in_batch.append(item)
            
            # 3. 루프 종료 후 남은 텍스트들 처리
            if texts_in_batch:
                self._handle_paragraph(texts_in_batch, doc, page_no, parent=self.active_main_parent)

    def _handle_table(self, html_content: str, doc: DoclingDocument, page_no: int, parent: Any):
        """HTML 테이블을 분석하여 구조화된 Table 객체를 지정된 parent(body)에 추가합니다."""
        soup = BeautifulSoup(html_content, "html.parser")
        # SDK가 표 셀 HTML 안에 <latex value="<base64>"/> 형태로 임베드한 수식을
        # TableCell.text가 단순 문자열이라 별도 FORMULA 노드로 못 박는다.
        # 대신 셀 텍스트 추출 전에 <math>{decoded latex}</math> 텍스트 노드로 치환하여
        # chandra OCR prompt의 inline 수식 컨벤션(<math>...</math>, KaTeX-compatible
        # LaTeX 본문)과 정합을 맞춘다.
        for latex_tag in soup.find_all("latex"):
            decoded = self._decode_latex_b64(latex_tag.get("value", ""))
            replacement = f"<math>{decoded}</math>" if decoded else ""
            latex_tag.replace_with(NavigableString(replacement))
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
        if not self.save_images:
            return

        # 1. JSON에 적힌 값(이미지 경로)을 가져옴 (예: "/tmp/old_path/images/image6.bmp")
        img_path = item.get("value", "")

        if not img_path or not os.path.exists(img_path):
            return

        # [Salvaged 1] 매직 넘버 기반의 강력한 유효성 검사 (XML/가짜파일 방어)
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
            _log.warning(f"유효하지 않은 이미지 데이터(XML 등 가능성): {os.path.basename(img_path)}")
            return

        pil_image = None
        
        # [Salvaged 2] Pillow -> Wand 단계별 시도 (WMF/EMF 구제)
        try:
            # 1단계: 표준 포맷 시도
            pil_image = Image.open(img_path)
            pil_image.load() # 강제 로드하여 오류 조기 감지
        except (UnidentifiedImageError, OSError):
            # 2단계: Pillow 실패 시 Wand 가동 (include_wmf=True인 경우에만)
            if self.include_wmf and WAND_AVAILABLE:
                try:
                    with WandImage(filename=img_path) as wand_img:
                        wand_img.format = 'png'
                        pil_image = Image.open(BytesIO(wand_img.make_blob()))
                except Exception as e:
                    _log.error(f"Wand 변환 실패: {e}")
            else:
                _log.warning(f"Pillow 실패, WMF/EMF 복구 미시도 (include_wmf={self.include_wmf}): {img_path}")

        # [Salvaged 3] Docling 임베딩 (DPI 고정 및 BBox 설정)
        prov = ProvenanceItem(
            page_no=page_no,
            bbox=BoundingBox(l=0, t=0, r=1, b=1, coord_origin=CoordOrigin.BOTTOMLEFT),
            charspan=(0, 0)
        )
        if pil_image:
            doc.add_picture(
                parent=parent,
                image=ImageRef.from_pil(image=pil_image, dpi=72),
                caption=None,
                prov=prov,
            )
        else:
            # Pillow + Wand 모두 실패 시 빈 플레이스홀더라도 추가해 문서 구조 보존
            _log.error(f"이미지 로드 완전 실패, 플레이스홀더 추가: {os.path.basename(img_path)}")
            doc.add_picture(
                parent=parent,
                caption=None,
                prov=prov,
            )

    # HWP SDK의 비정상 출력 보정용. 외부에서도 단위 테스트하기 좋게 staticmethod.
    # 패턴: <latex value="<base64, 줄바꿈/공백 가능>"/>  (속성값 안의 "가 escape 없이 등장)
    # JSON 문자열 안에 들어오면 outer string이 그 시점에서 종료된 것으로 파싱돼 record 손실.
    # base64 charset(A-Z, a-z, 0-9, +, /, =)과 공백만 허용하여 다른 HTML 속성을 오인하지 않도록 한다.
    _LATEX_ATTR_PATTERN = re.compile(
        r'<latex\s+value="([A-Za-z0-9+/=\s]*?)"\s*/?>',
        flags=re.DOTALL,
    )

    @classmethod
    def _normalize_sdk_json_text(cls, text: str) -> str:
        """SDK 결과 JSON에서 임베드된 <latex value="..."/> 속성의 inner "를 escape하고
        base64 안의 공백/줄바꿈을 제거하여 JSON 파서가 outer 문자열을 끝까지 읽을 수 있도록 한다.
        """
        def _repl(m: "re.Match[str]") -> str:
            b64 = re.sub(r"\s+", "", m.group(1))
            return f'<latex value=\\"{b64}\\"/>'
        return cls._LATEX_ATTR_PATTERN.sub(_repl, text)

    @staticmethod
    def _decode_latex_b64(raw: str) -> Optional[str]:
        """SDK가 emit하는 base64 인코딩된 LaTeX 문자열을 디코드한다.
        긴 값은 SDK가 중간에 줄바꿈/공백을 끼울 수 있어 전처리 후 디코드한다.
        손상된 입력을 조용히 통과시키지 않도록 strict 모드로 검증하며,
        실패 시 경고 로그를 남기고 None 반환.
        """
        if not raw:
            return None
        # 공백/줄바꿈/HTML 잔여물 제거 (정상 SDK 출력의 line-wrap 흡수)
        cleaned = re.sub(r"\s+", "", raw)
        try:
            decoded = base64.b64decode(cleaned, validate=True).decode("utf-8").strip()
        except (binascii.Error, UnicodeDecodeError) as e:
            _log.warning(f"latex base64 디코드 실패: {e}; raw[:60]={raw[:60]!r}")
            return None
        return decoded or None

    def _handle_latex(self, item: Dict, doc: DoclingDocument, page_no: int, parent: Any):
        """SDK의 latex 아이템을 base64 디코드 후 FORMULA 노드로 추가한다."""
        latex = self._decode_latex_b64(item.get("value", ""))
        if not latex:
            return
        prov = ProvenanceItem(
            page_no=page_no,
            bbox=BoundingBox(l=0, t=0, r=1, b=1, coord_origin=CoordOrigin.BOTTOMLEFT),
            charspan=(0, len(latex))
        )
        doc.add_text(
            label=DocItemLabel.FORMULA,
            text=latex,
            parent=parent,
            prov=prov,
        )

    def _handle_paragraph(self, paragraph_items: List[Dict], doc: DoclingDocument, page_no: int, parent: Any):
        """TOC(목차) 감지 로직이 추가된 버전입니다. 넘겨받은 parent에 텍스트를 추가합니다."""
        
        # 1. 텍스트 병합 및 기본 검사
        full_text = "".join([item.get("value", "") for item in paragraph_items]).strip()
        if not full_text: 
            return

        # 2. 폰트/스타일 정보 추출 — run 단위 max()/any() 대신 글자수 가중 "비율"로 집계.
        #    단락의 일부 run만 강조돼도 전체가 강조로 잡히던 문제를 막는다. (Issue #206)
        baseline = self.body_font_size or 10.0
        total_chars = 0
        bold_chars = 0
        large_chars = 0
        max_font_size = 0.0
        for i in paragraph_items:
            weight = len(str(i.get("value", "")).strip())
            font = i.get("font", {}) or {}
            size = float(font.get("size", 10.0) or 10.0)
            max_font_size = max(max_font_size, size)
            if weight <= 0:
                continue
            total_chars += weight
            if font.get("bold", False):
                bold_chars += weight
            if size >= baseline * _LARGE_FONT_FACTOR:
                large_chars += weight
        if max_font_size <= 0.0:
            max_font_size = baseline
        bold_ratio = (bold_chars / total_chars) if total_chars else 0.0
        large_ratio = (large_chars / total_chars) if total_chars else 0.0

        # 가상 스타일 판정
        marker = self._match_structural_marker(full_text)
        p_style_id, p_level = self._get_label_and_level_hwp(
            full_text, max_font_size, bold_ratio, large_ratio, marker=marker
        )

        # 3. 패턴 감지 (TOC 및 헤더)
        # [추가]: TOC 패턴 감지 (점 2개 이상, 탭, 또는 긴 공백 뒤에 숫자로 끝나는 경우)
        is_toc = bool(re.search(r'(\.{2,}|…|\t|\s{4,})\s*\d+$', full_text))

        # 4. 강등 로직: heading으로 잡혔지만 실제로는 서술형 본문 문장/리스트 항목인 경우 강등.
        #    divider 마커(장/조/붙임/별표)는 SDK가 본문과 한 단락으로 합쳐도 heading 유지 → 면제.
        #    그 외(number/weak heading)는 종결형 문장으로 끝나거나 지나치게 길면 본문으로 강등.
        if p_style_id == "Heading" and marker[1] != "divider":
            if len(full_text) > _HEADING_MAX_LEN or _SENTENCE_FINAL.search(full_text):
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

    def _add_header(self, doc: DoclingDocument, level: int, text: str, prov: ProvenanceItem, parent: Any):
        """DOCX 표준 명칭(header-N)을 사용하여 논리적 섹션 마디를 만듭니다."""
        
        # 1. 하위 계층 부모 초기화 (기존 로직 유지)
        for i in range(level, 10):
            if i in self.parents:
                self.parents[i] = None

        # 2. [명칭 변경]: level 1 -> header-0, level 2 -> header-1 방식으로 명명
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
            else:
                raise ValueError("page_info is empty")

        except Exception as e:
            # 2. 오류가 나거나 정보가 없는 경우 (Fallback)
            _log.warning(f"페이지 정보 로드 실패({e}). 기본 1페이지로 설정합니다.")
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
        super().unload()