

# `attachment_processor.py` 코드 상세 설명서

---

## 📌 목차

1. [개요](#1-개요)
2. [전체 아키텍처](#2-전체-아키텍처)
3. [임포트 및 초기 설정](#3-임포트-및-초기-설정)
4. [유틸리티 함수](#4-유틸리티-함수)
   - 4.1 [`convert_to_pdf()`](#41-convert_to_pdf)
   - 4.2 [`_get_pdf_path()`](#42-_get_pdf_path)
   - 4.3 [`install_packages()`](#43-install_packages)
5. [데이터 모델](#5-데이터-모델)
   - 5.1 [`GenOSVectorMeta`](#51-genosvectormeta)
   - 5.2 [`GenOSVectorMetaBuilder`](#52-genosvectormetabuilder)
6. [파일 로더 (Loader) 클래스들](#6-파일-로더-loader-클래스들)
   - 6.1 [`HwpLoader`](#61-hwploader)
   - 6.2 [`TextLoader`](#62-textloader)
   - 6.3 [`TabularLoader`](#63-tabularloader)
   - 6.4 [`AudioLoader`](#64-audioloader)
7. [청킹(Chunking) 클래스들](#7-청킹chunking-클래스들)
   - 7.1 [`HierarchicalChunker`](#71-hierarchicalchunker)
   - 7.2 [`HybridChunker`](#72-hybridchunker)
8. [문서 프로세서 (Processor) 클래스들](#8-문서-프로세서-processor-클래스들)
   - 8.1 [`DocxProcessor`](#81-docxprocessor)
   - 8.2 [`HwpxProcessor`](#82-hwpxprocessor)
   - 8.3 [`DocumentProcessor` (메인 엔트리포인트)](#83-documentprocessor-메인-엔트리포인트)
9. [예외 클래스](#9-예외-클래스)
10. [실행 흐름 요약](#10-실행-흐름-요약)
11. [지원 파일 포맷 총정리](#11-지원-파일-포맷-총정리)

---

## 1. 개요

`attachment_processor.py`는 **첨부용 전처리기(Attachment Processor)** 입니다. 사용자가 채팅 중 첨부로 업로드하는 파일을 **실시간**으로 분석하기 위한 **경량화 전처리기**로, 복잡한 레이아웃 분석(Layout Detection) 과정을 생략하고 **텍스트 추출(Text Extraction)** 에 집중하여 즉각적인 응답 속도를 보장합니다.

### 핵심 설계 철학

```
"속도 중심: 다양한 포맷의 텍스트 즉시 추출"
```

| 특징 | 설명 |
|------|------|
| **Native 텍스트 추출** | HWP, HWPX, DOCX, XLSX 등 원본 파일의 텍스트를 직접 파싱 |
| **멀티미디어 지원** | MP3, WAV, M4A 등 오디오 파일의 음성→텍스트 변환(STT) |
| **데이터 변환** | CSV, Excel 등의 정형 데이터를 LLM이 이해하기 쉬운 형태로 변환 |

---

## 2. 전체 아키텍처

아래 다이어그램은 파일이 입력되었을 때 확장자에 따라 어떤 경로로 처리되는지를 보여줍니다.

```
사용자가 파일 업로드
        │
        ▼
 ┌──────────────────┐
 │ DocumentProcessor│  ◄── 메인 엔트리포인트 (__call__)
 │   (라우터 역할)   │
 └──────┬───────────┘
        │
        │  확장자(ext)에 따라 분기
        │
        ├── .wav/.mp3/.m4a ──────► AudioLoader ──► STT(Whisper) ──► GenOSVectorMeta
        │
        ├── .csv/.xlsx ──────────► TabularLoader ──► DataFrame 파싱 ──► GenOSVectorMeta
        │
        ├── .hwp ────────────────► HwpLoader ──► hwp5html ──► PDF ──► 텍스트 추출
        │                                                       │
        ├── .hwpx ───────────────► HwpxProcessor ──► Docling 파싱 ──► HybridChunker
        │                                                               │
        ├── .docx ───────────────► DocxProcessor ──► Docling 파싱 ──► HybridChunker
        │                                                               │
        └── 기타 (.pdf, .ppt,    ► get_loader() ──► LangChain Loader     │
            .doc, .jpg, .txt,                          │                │
            .json, .md 등)                             ▼                ▼
                                              RecursiveTextSplitter  HybridChunker
                                                       │                │
                                                       ▼                ▼
                                                compose_vectors()   compose_vectors()
                                                       │                │
                                                       ▼                ▼
                                              ┌─────────────────────────────┐
                                              │  List[GenOSVectorMeta]      │
                                              │  (최종 출력: 청크별 메타데이터)    │
                                              └─────────────────────────────┘
```

---

## 3. 임포트 및 초기 설정

```python
from collections import defaultdict
import asyncio
import fitz          # PyMuPDF: PDF 파일을 읽고 조작하는 라이브러리
import json
import math
import os
import pandas as pd
import pydub          # 오디오 파일 처리 (분할, 포맷 변환)
import requests       # HTTP 요청 (STT API 호출용)
import shutil         # 파일/디렉토리 복사 및 삭제
import subprocess     # 외부 프로세스 실행 (LibreOffice, hwp5html 등)
import sys
import threading      # 멀티스레드 (오디오 STT 병렬 요청)
import uuid           # 고유 임시 디렉토리명 생성
import warnings
from datetime import datetime
import logging
```

### 주요 외부 라이브러리 그룹

| 그룹 | 라이브러리 | 용도                                                     |
|------|-----------|--------------------------------------------------------|
| **문서 로딩** | `langchain_community.document_loaders` | PDF, DOCX, PPT, 이미지, 마크다운 등을 로드                        |
| **텍스트 분할** | `langchain_text_splitters.RecursiveCharacterTextSplitter` | 문서를 적절한 크기의 청크로 분할                                     |
| **Docling** | `docling`, `docling_core` | HWPX, DOCX 문서를 바로 읽어들여 구조적으로 파싱하여 `DoclingDocument` 생성 |
| **토크나이저** | `transformers.AutoTokenizer`, `semchunk` | 토큰 수 기반 청킹 및 시맨틱 분할                                    |
| **PDF 생성** | `weasyprint.HTML` | HTML → PDF 변환                                          |
| **인코딩 탐지** | `chardet` | 텍스트 파일의 인코딩 자동 감지                                      |

### 로깅 초기 설정

```python
for n in ("fontTools", "fontTools.ttLib", "fontTools.ttLib.ttFont"):
    lg = logging.getLogger(n)
    lg.setLevel(logging.CRITICAL)
    lg.propagate = False
    logging.getLogger().setLevel(logging.WARNING)
```
> **역할**: `fontTools` 라이브러리가 PDF 처리 시 과도하게 출력하는 디버그 로그를 억제합니다. 사용자에게 불필요한 로그 노이즈를 줄여 핵심 정보만 보이게 합니다.

### PDF 변환 대상 확장자 상수

```python
CONVERTIBLE_EXTENSIONS = ['.hwp', '.txt', '.json', '.md', '.ppt', '.pptx', '.docx']
```
> **역할**: `_get_pdf_path()` 함수에서 파일 경로의 확장자를 `.pdf`로 바꿀 때 사용되는 대상 확장자 목록입니다.

---

## 4. 유틸리티 함수

### 4.1 `convert_to_pdf()`

```python
def convert_to_pdf(file_path: str) -> str | None:
```

**목적**: [LibreOffice](https://www.libreoffice.org/)를 활용하여 다양한 문서 포맷을 PDF로 변환합니다.

**동작 흐름**:

```
입력 파일 (PPT, DOCX, DOC 등)
        │
        ▼
 확장자 판별 → 적절한 LibreOffice 필터 선택
        │
        │  .ppt/.pptx → "pdf:impress_pdf_Export"
        │  .doc/.docx → "pdf:writer_pdf_Export"
        │  .xls/.xlsx/.csv → "pdf:calc_pdf_Export"
        │  기타 → "pdf"
        │
        ▼
 비ASCII 파일명 체크
        │
        ├── ASCII 파일명 → 그대로 사용
        └── 비ASCII 파일명 → 임시 ASCII 파일명으로 복사본 생성
                              (LibreOffice의 비ASCII 파일명 이슈 우회)
        │
        ▼
 soffice --headless --convert-to ... 실행
        │
        ├── 성공 → PDF 경로 반환
        └── 실패 → None 반환 (예외를 던지지 않음!)
```

**핵심 포인트**:
- `--headless` 옵션으로 GUI 없이 변환 (서버 환경)
- 실패해도 **예외를 던지지 않고** `None`을 반환하는 **방어적 설계**
- `UTF-8` 로캘을 환경 변수로 보장하여 한글 파일명 문제 최소화

```python
    # 핵심 코드: LibreOffice 커맨드 실행
    cmd = [
        "soffice", "--headless",           # GUI 없이 실행
        "--convert-to", convert_arg,       # 출력 포맷 지정
        "--outdir", str(out_dir),          # 출력 디렉토리
        str(cand)                          # 입력 파일
    ]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
```

---

### 4.2 `_get_pdf_path()`

```python
def _get_pdf_path(file_path: str) -> str:
```

**목적**: 파일 경로에서 확장자를 `.pdf`로 단순 치환합니다.

```python
# 예시:
# "/data/report.hwp"  → "/data/report.pdf"
# "/data/notes.txt"   → "/data/notes.pdf"
# "/data/data.pptx"   → "/data/data.pdf"
```

> **주의**: 이 함수는 실제로 파일을 변환하지 않습니다. 단지 **경로 문자열만 변환**합니다. 실제 변환은 `convert_to_pdf()`가 담당합니다.

---

### 4.3 `install_packages()`

```python
def install_packages(packages):
```

**목적**: 런타임에 필요한 Python 패키지가 설치되어 있는지 확인하고, 없으면 자동으로 `pip install`을 실행합니다.

```python
# 사용 예시:
install_packages(['openpyxl', 'chardet'])
# → openpyxl이 없으면: pip install openpyxl 실행
# → chardet이 없으면: pip install chardet 실행
```

---

## 5. 데이터 모델

### 5.1 `GenOSVectorMeta`

```python
class GenOSVectorMeta(BaseModel):
    class Config:
        extra = 'allow'    # 정의되지 않은 추가 필드도 허용

    text: str | None = None                 # 청크의 실제 텍스트 내용
    n_char: int | None = None               # 텍스트의 문자 수
    n_word: int | None = None               # 텍스트의 단어 수
    n_line: int | None = None               # 텍스트의 줄 수
    i_page: int | None = None               # 청크가 시작하는 페이지 번호
    e_page: int | None = None               # 청크가 끝나는 페이지 번호
    i_chunk_on_page: int | None = None      # 해당 페이지 내에서의 청크 순서 (0부터)
    n_chunk_of_page: int | None = None      # 해당 페이지의 전체 청크 수
    i_chunk_on_doc: int | None = None       # 문서 전체에서의 청크 순서 (0부터)
    n_chunk_of_doc: int | None = None       # 문서 전체의 청크 수
    n_page: int | None = None               # 문서의 전체 페이지 수
    reg_date: str | None = None             # 처리 일시 (ISO 8601 형식)
    chunk_bboxes: str | None = None         # 청크의 위치 정보 (JSON 문자열)
    media_files: str | None = None          # 연관된 미디어 파일 정보 (JSON 문자열)
```

**역할**: 전처리 결과의 **최종 출력 단위(=벡터 메타데이터)** 입니다. 하나의 청크(chunk)가 하나의 `GenOSVectorMeta` 객체로 변환되며, 이 객체가 벡터 DB에 저장됩니다.

**필드 시각화**:

```
┌──────────────────────────────────────────────────┐
│ 문서 전체 (n_page=5, n_chunk_of_doc=12)            │
│                                                  │
│  ┌──────────── 페이지 1 ───────────────┐           │
│  │  ┌─────────────────────┐          │           │
│  │  │ 청크 0               │ i_page=1 │           │
│  │  │ i_chunk_on_page=0   │          │           │
│  │  │ i_chunk_on_doc=0    │          │           │
│  │  │ text="제1조 목적..."  │          │           │
│  │  │ n_char=245          │          │           │
│  │  └─────────────────────┘          │           │
│  │  ┌─────────────────────┐          │           │
│  │  │ 청크 1               │          │           │
│  │  │ i_chunk_on_page=1   │          │           │
│  │  │ i_chunk_on_doc=1    │ n_chunk_ │           │
│  │  │ text="제2조 범위..."  │ of_page=2│           │
│  │  └─────────────────────┘          │           │
│  └───────────────────────────────────┘           │
│                                                  │
│  ┌──────────── 페이지 2 ───────────────┐           │
│  │  ...                              │           │
│  └───────────────────────────────────┘           │
└──────────────────────────────────────────────────┘
```

---

### 5.2 `GenOSVectorMetaBuilder`

```python
class GenOSVectorMetaBuilder:
```

**역할**: `GenOSVectorMeta` 객체를 단계별로 조립하는 **빌더 패턴(Builder Pattern)** 구현체입니다. 복잡한 메타데이터를 실수 없이 조합할 수 있게 메서드 체이닝을 지원합니다.

**사용 예시 (코드에서 실제로 사용되는 패턴)**:

```python
vector = (GenOSVectorMetaBuilder()
          .set_text(content)                                    # ① 텍스트 설정
          .set_page_info(page, chunk_idx_on_page, total)        # ② 페이지 정보 설정
          .set_chunk_index(chunk_idx)                            # ③ 청크 인덱스 설정
          .set_global_metadata(**global_metadata)                # ④ 전역 메타데이터 병합
          .set_chunk_bboxes(doc_items, document)                 # ⑤ 위치 좌표 설정
          .set_media_files(doc_items)                            # ⑥ 미디어 파일 설정
          ).build()                                              # ⑦ 최종 객체 생성
```

**주요 메서드 설명**:

| 메서드 | 역할 |
|--------|------|
| `set_text(text)` | 텍스트를 설정하고, 동시에 `n_char`, `n_word`, `n_line`을 자동 계산 |
| `set_page_info(...)` | 페이지 번호, 페이지 내 청크 순서, 페이지 내 전체 청크 수 설정 |
| `set_chunk_index(idx)` | 문서 전체에서의 청크 순서 설정 |
| `set_global_metadata(**kw)` | `n_chunk_of_doc`, `n_page`, `reg_date` 등 문서 공통 정보 일괄 설정 |
| `set_chunk_bboxes(items, doc)` | 청크를 구성하는 문서 요소들의 **바운딩 박스(bbox)** 좌표를 정규화(0~1)하여 JSON으로 저장 |
| `set_media_files(items)` | 청크에 포함된 이미지 등 미디어 파일 정보를 JSON으로 저장 |
| `build()` | 모든 설정을 종합하여 `GenOSVectorMeta` 객체 반환 |

**`set_chunk_bboxes` 상세 설명**:

```python
def set_chunk_bboxes(self, doc_items: list, document: DoclingDocument):
    # 각 문서 요소(doc_item)의 위치 정보(prov)를 순회
    for item in doc_items:
        for prov in item.prov:
            size = document.pages.get(prov.page_no).size  # 페이지 크기 가져오기
            bbox = prov.bbox                               # 원본 좌표
            bbox_data = {
                'l': bbox.l / size.width,    # 왼쪽 (0~1로 정규화)
                't': bbox.t / size.height,   # 위쪽
                'r': bbox.r / size.width,    # 오른쪽
                'b': bbox.b / size.height,   # 아래쪽
                'coord_origin': bbox.coord_origin.value
            }
```

> 좌표를 0~1 범위로 정규화하는 이유는, 화면에 표시할 때 페이지 크기에 관계없이 동일한 비율로 하이라이트할 수 있게 하기 위함입니다.

---

## 6. 파일 로더 (Loader) 클래스들

각 파일 포맷별로 텍스트를 추출하는 전용 로더입니다.

### 6.1 `HwpLoader`

```python
class HwpLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.output_dir = os.path.join('/tmp', str(uuid.uuid4()))  # 고유한 임시 디렉토리
```

**목적**: HWP(한글) 파일에서 텍스트를 추출합니다.

**처리 흐름**:

```
.hwp 파일
    │
    ▼
hwp5html 명령어로 XHTML 변환
    │  (hwp5html은 pyhwp 패키지에 포함된 CLI 도구)
    │  결과: /tmp/{uuid}/index.xhtml
    │
    ▼
WeasyPrint로 XHTML → PDF 변환
    │  결과: 원본경로/파일명.pdf
    │
    ▼
PyMuPDFLoader로 PDF에서 텍스트 추출
    │
    ▼
List[Document] 반환
```

**주의**
> HWP는 한컴오피스 고유의 바이너리 포맷이므로 상용 소프트웨어가 아닌 이상 텍스트를 추출하기 어렵습니다. 
> `hwp5html`은 오픈소스로서 HTML 구조로 변환한 뒤 PDF를 거쳐 텍스트를 추출하지만, 오류를 자주 동반합니다.
> Site 별로 상황에 맞게 상용솔루션으로 이부분을 대체합니다.

```python
    def load(self):
        try:
            # 1단계: HWP → XHTML
            subprocess.run(
                ['hwp5html', self.file_path, '--output', self.output_dir],
                check=True,
                timeout=600          # 10분 타임아웃 (대용량 파일 대비)
            )
            # 2단계: XHTML → PDF
            converted_file_path = os.path.join(self.output_dir, 'index.xhtml')
            pdf_save_path = _get_pdf_path(self.file_path)
            HTML(converted_file_path).write_pdf(pdf_save_path)

            # 3단계: PDF → 텍스트 추출
            loader = PyMuPDFLoader(pdf_save_path)
            return loader.load()
        finally:
            # 임시 디렉토리 정리 (성공/실패 무관)
            if os.path.exists(self.output_dir):
                shutil.rmtree(self.output_dir)
```

---

### 6.2 `TextLoader`

```python
class TextLoader:
    def __init__(self, file_path: str):
```

**목적**: `.txt`, `.json`, `.md` 등 순수 텍스트 파일을 로드합니다. **인코딩 자동 감지**가 핵심 기능입니다.

**인코딩 감지 흐름**:

```
파일 바이너리 읽기
    │
    ▼
chardet로 인코딩 자동 감지
    │
    ▼
시도 순서로 디코딩:
    1. chardet이 감지한 인코딩
    2. UTF-8
    3. CP949 (한글 Windows)
    4. EUC-KR (한글 레거시)
    5. ISO-8859-1
    6. Latin-1
    │
    ├── 디코딩 성공 → 텍스트 획득
    └── 모두 실패 → UTF-8 + errors='replace' (깨진 문자는 ?로 대체)
```

**PDF 변환 분기**:

```python
    # WeasyPrint가 사용 가능한 경우
    if HTML:
        # 텍스트를 HTML로 감싸고 PDF로 변환
        html = f"<html><meta charset='utf-8'><body><pre>{content}</pre></body></html>"
        HTML(html_path).write_pdf(pdf_path)
        loader = PyMuPDFLoader(pdf_path)
        return loader.load()

    # WeasyPrint 불가 시 Document 직접 반환 (PDF 변환 없이)
    return [Document(page_content=content, metadata={'source': self.file_path, 'page': 0})]
```

> **설계 의도**: WeasyPrint가 설치되어 있으면 PDF를 통해 페이지 정보까지 추출하고, 없으면 텍스트만이라도 반환하는 **그레이스풀 폴백(graceful fallback)** 설계입니다.

---

### 6.3 `TabularLoader`

```python
class TabularLoader:
    def __init__(self, file_path: str, ext: str):
```

**목적**: CSV, XLSX 등 정형 데이터(테이블) 파일을 LLM이 이해할 수 있는 형태로 변환합니다.

**처리 흐름 (CSV 예시)**:

```
.csv 파일
    │
    ▼
chardet로 인코딩 감지 → pandas.read_csv()
    │
    ▼
check_sql_dtypes()로 각 컬럼의 데이터 타입 분석
    │  INT, FLOAT, BOOLEAN, DATE, VARCHAR 등
    │
    ▼
LangChain DataFrameLoader로 Document 변환
    │  (첫 번째 컬럼을 page_content로 사용)
    │
    ▼
process_data_rows()로 최종 데이터 구조 생성
    │
    ▼
data_dict = {
    "data": [{
        "sheet_name": "table_1",
        "data_rows": [...],        # 각 행의 데이터
        "data_types": [...]        # 각 컬럼의 SQL 데이터 타입
    }]
}
```

**`check_sql_dtypes()` 메서드 상세**:

이 메서드는 DataFrame의 각 컬럼 타입을 SQL 호환 타입으로 매핑합니다:

```python
# 타입 매핑 예시:
# pandas int64    → 'BIGINT'
# pandas int32    → 'INT'
# pandas float64  → 'FLOAT'
# pandas bool     → 'BOOLEAN'
# pandas object   → 'VARCHAR(최대길이+10)'
```

**`return_vectormeta_format()` 메서드**:

```python
def return_vectormeta_format(self):
    text = "[DA] " + str(self.data_dict)
    # "[DA]" 토큰은 이 데이터가 "Data Analysis"용임을 LLM에게 알리는 마커
```

> **`[DA]` 접두어의 의미**: LLM이 이 청크를 받았을 때, 일반 텍스트가 아니라 **정형 데이터**라는 것을 인식하고 적절한 분석 방식(표 분석, 통계 등)으로 처리하도록 유도합니다.

**XLSX 처리 시 다중 시트 지원**:

```python
def load_xlsx_documents(self, file_path: str):
    dfs = pd.read_excel(file_path, sheet_name=None)  # 모든 시트를 dict로 로드
    for sheet_name, df in dfs.items():
        # 각 시트를 독립적으로 처리하여 data_dict["data"]에 추가
```

---

### 6.4 `AudioLoader`

```python
class AudioLoader:
    def __init__(self,
                 file_path: str,       # 오디오 파일 경로
                 req_url: str,         # Whisper STT API URL
                 req_data: dict,       # API 요청 파라미터
                 chunk_sec: int = 29,  # 오디오 분할 단위 (초)
                 tmp_path: str = '.',  # 임시 파일 저장 경로
                 ):
```

**목적**: 오디오 파일(MP3, WAV, M4A)을 텍스트로 변환합니다 (Speech-to-Text).

**처리 흐름**:

```
오디오 파일 (예: 5분짜리 MP3)
    │
    ▼
split_file_as_chunks()
    │  29초 단위로 분할 (Whisper 모델의 입력 제한 대응)
    │  각 청크 앞에 0.3초 오버랩 추가 (문맥 연결 보장)
    │
    │  tmp_0.wav (0~29초)
    │  tmp_1.wav (28.7~58초)     ← 0.3초 오버랩
    │  tmp_2.wav (57.7~87초)
    │  ...
    │
    ▼
transcribe_audio()
    │  각 청크를 Whisper API에 병렬 전송 (멀티스레드)
    │
    │  Thread 0 → POST /v1/audio/transcriptions (tmp_0.wav)
    │  Thread 1 → POST /v1/audio/transcriptions (tmp_1.wav)
    │  Thread 2 → POST /v1/audio/transcriptions (tmp_2.wav)
    │
    ▼
파일명 기준 정렬 → 텍스트 병합
    │
    ▼
"[AUDIO] 안녕하세요 오늘 회의를 시작하겠습니다..."
```

**오버랩 설계의 이유**:

```python
overlap_start_ms = start_ms - 300 if start_ms > 0 else start_ms
```

> 오디오를 단순히 29초 단위로 자르면 단어가 중간에 잘릴 수 있습니다. 0.3초(300ms)의 오버랩을 두어 경계 부분의 단어가 완전히 인식되도록 합니다.

**병렬 처리 구현**:

```python
# 멀티스레드로 STT 요청을 동시에 보내 총 처리 시간 단축
threads = [threading.Thread(target=_send_request, args=(f,)) for f in file_path_lst]
for t in threads: t.start()   # 모든 스레드 시작
for t in threads: t.join()    # 모든 스레드 완료 대기
```

**`[AUDIO]` 접두어**: `[DA]`와 마찬가지로, LLM에게 이 텍스트가 음성 전사(transcription) 결과임을 알리는 마커입니다.

---

## 7. 청킹(Chunking) 클래스들

HWPX와 DOCX 파일 처리 시 사용되는 고급 청킹 로직입니다. `Genos 지능형 전처리기` 의 `DoclingDocument` 구조를 기반으로 동작합니다.

### 7.1 `HierarchicalChunker`

```python
class HierarchicalChunker(BaseChunker):
```

**목적**: 문서의 논리적 계층 구조(제목-본문-표-그림)를 유지하면서 청크를 생성합니다.

**핵심 개념 — `heading_by_level` 딕셔너리**:

```python
heading_by_level: dict[LevelNumber, str] = {}
```

이 딕셔너리는 현재 문서 탐색 위치의 **제목 컨텍스트**를 추적합니다:

```
문서 구조:                        heading_by_level 상태:
─────────                        ─────────────────────
제1장 총칙                        {0: "제1장 총칙"}
  제1절 목적                      {0: "제1장 총칙", 1: "제1절 목적"}
    제1조 이 법은...               {0: "제1장 총칙", 1: "제1절 목적"}
  제2절 범위                      {0: "제1장 총칙", 1: "제2절 범위"}
    ↑ level 1이 바뀌면 이전 level 1("제1절 목적")은 삭제됨
제2장 권리                        {0: "제2장 권리"}
    ↑ level 0이 바뀌면 하위 레벨 모두 삭제됨
```

**문서 요소별 처리 방식**:

```python
def chunk(self, dl_doc: DLDocument, **kwargs: Any) -> Iterator[BaseChunk]:
    for item, level in dl_doc.iterate_items():
        # ── 리스트 아이템 병합 ──
        if isinstance(item, ListItem):
            list_items.append(item)   # 연속된 리스트 아이템을 모아둠
            continue                  # 아직 yield하지 않음

        # ── 섹션 헤더 ──
        elif isinstance(item, SectionHeaderItem):
            heading_by_level[level] = item.text
            text = ''.join(heading_by_level.values())  # 누적된 제목 연결
            yield DocChunk(text=text, ...)

        # ── 일반 텍스트 ──
        elif isinstance(item, TextItem):
            text = item.text
            yield DocChunk(text=text, ...)

        # ── 표 ──
        elif isinstance(item, TableItem):
            text = item.export_to_markdown(dl_doc)  # 마크다운 표로 변환
            yield DocChunk(text=text, ...)

        # ── 그림 ──
        elif isinstance(item, PictureItem):
            text = ''.join(heading_by_level.values())  # 제목 컨텍스트만
            yield DocChunk(text=text, ...)
```

**리스트 아이템 병합 로직**:

```
원본 문서:                          병합 결과 (하나의 청크):
─────────                          ─────────────────────
• 사과                              "사과\n배\n포도"
• 배
• 포도
```

연속된 리스트 아이템을 하나의 청크로 병합하여, 불필요하게 많은 소규모 청크가 생성되는 것을 방지합니다.

---

### 7.2 `HybridChunker`

```python
class HybridChunker(BaseChunker):
```

**목적**: `HierarchicalChunker`의 결과를 받아, **토큰 수 기반**으로 청크 크기를 조절합니다. 레이아웃 구조 + 토큰 제한을 **하이브리드**로 결합합니다.

**처리 파이프라인**:

```
DoclingDocument
    │
    ▼
HierarchicalChunker.chunk()
    │  → 문서 구조 기반 초기 청크 생성
    │
    ▼
_split_by_doc_items()
    │  → max_tokens 초과 청크를 doc_item 단위로 분할
    │
    ▼
_split_using_plain_text()
    │  → 그래도 초과하면 semchunk로 텍스트 레벨 분할
    │
    ▼
_merge_chunks_with_matching_metadata() (merge_peers=True일 때)
    │  → 같은 제목/캡션을 가진 작은 청크들을 다시 병합
    │
    ▼
최종 청크 리스트
```

**토큰 카운팅**:

```python
@model_validator(mode="after")
def _patch_tokenizer_and_max_tokens(self) -> Self:
    self._tokenizer = (
        self.tokenizer
        if isinstance(self.tokenizer, PreTrainedTokenizerBase)
        else AutoTokenizer.from_pretrained(self.tokenizer)
        # 기본값: "/models/doc_parser_models/sentence-transformers-all-MiniLM-L6-v2"
    )
```

> 토크나이저를 사용하여 텍스트의 정확한 토큰 수를 계산합니다. 단순 문자 수가 아닌 토큰 수로 제한하므로, 벡터 모델의 입력 제한에 정확히 맞출 수 있습니다.

**`_split_by_doc_items()` — 슬라이딩 윈도우 분할**:

```
[아이템A, 아이템B, 아이템C, 아이템D, 아이템E]
  window_start=0                    max_tokens=500
  │
  ├─ window_end=0: A만 → 200토큰 → OK, 확장 시도
  ├─ window_end=1: A+B → 380토큰 → OK, 확장 시도
  ├─ window_end=2: A+B+C → 520토큰 → 초과!
  │  → A+B를 하나의 청크로 확정 (380토큰)
  │  → window_start=2 (C부터 새 윈도우)
  ├─ window_end=2: C만 → 140토큰 → OK, 확장 시도
  └─ ...
```

**`_split_using_plain_text()` — 시맨틱 텍스트 분할**:

```python
sem_chunker = semchunk.chunkerify(self._tokenizer, chunk_size=available_length)
segments = sem_chunker.chunk(text)
# semchunk는 문장/문단 경계를 존중하면서 텍스트를 분할합니다
```

> 단일 아이템이 `max_tokens`를 초과하는 경우(예: 매우 긴 본문 단락), `semchunk` 라이브러리를 사용하여 의미적 경계(문장, 문단)를 고려한 분할을 수행합니다.

**`_merge_chunks_with_matching_metadata()` — 동일 메타데이터 청크 병합**:

```
병합 전:                              병합 후:
─────────                            ──────────
청크1: heading="제1절", text="가"     청크1: heading="제1절", text="가\n나"
청크2: heading="제1절", text="나"     청크2: heading="제2절", text="다"
청크3: heading="제2절", text="다"
```

> 같은 제목(heading)과 캡션(caption)을 가진 작은 청크들을 토큰 제한 내에서 하나로 합칩니다. 이는 검색 시 문맥이 분절되는 것을 방지합니다.

**이 코드에서의 특수한 사용 방식**:

```python
chunker = HybridChunker(max_tokens=int(1e30), merge_peers=True)
```

> `max_tokens=int(1e30)` (약 10억)으로 설정되어 있어, 실질적으로 **토큰 제한이 없습니다**. 즉, 이 첨부용 전처리기에서는 HybridChunker를 사실상 **레이아웃 기반 병합 도구**로만 사용하고 있으며, 토큰 수에 의한 분할은 발생하지 않습니다. 이는 속도 우선의 설계 철학에 부합합니다.

---

## 8. 문서 프로세서 (Processor) 클래스들

### 8.1 `DocxProcessor`

```python
class DocxProcessor:
```

**목적**: `.docx` (Microsoft Word) 파일을 Docling 엔진으로 파싱하여 구조화된 청크를 생성합니다.

**Docling 변환기 설정**:

```python
self.converter = DocumentConverter(
    format_options={
        InputFormat.DOCX: WordFormatOption(
            pipeline_cls=SimplePipeline,                    # 경량 파이프라인 사용
            backend=GenosMsWordDocumentBackend              # Genos 커스텀 백엔드
        ),
    }
)
```

> `SimplePipeline`은 AI 기반 레이아웃 분석을 생략하는 경량 파이프라인입니다. 속도 우선의 첨부 전처리기 철학에 맞게, 단순 파싱만 수행합니다.

**처리 흐름**:

```python
async def __call__(self, request: Request, file_path: str, **kwargs: dict):
    # 1단계: Docling으로 문서 구조 파싱
    document: DoclingDocument = self.load_documents(file_path)

    # 2단계: 이미지 참조 경로 설정
    artifacts_dir, reference_path = self.get_paths(file_path)
    document = document._with_pictures_refs(...)

    # 3단계: HybridChunker로 청킹
    chunks: list[DocChunk] = self.split_documents(document)

    # 4단계: GenOSVectorMeta로 조립
    vectors: list[dict] = await self.compose_vectors(document, chunks, file_path, request)

    return vectors
```

**`compose_vectors()` 상세**:

```python
async def compose_vectors(self, document, chunks, file_path, request, **kwargs):
    # 전역 메타데이터 (모든 청크에 공통으로 적용)
    global_metadata = dict(
        n_chunk_of_doc=len(chunks),     # 총 청크 수
        n_page=document.num_pages(),     # 총 페이지 수
        reg_date=datetime.now().isoformat(timespec='seconds') + 'Z',
    )

    for chunk_idx, chunk in enumerate(chunks):
        # 청크의 텍스트 = 제목(heading) + 본문
        content = self.safe_join(chunk.meta.headings) + chunk.text

        # 빌더 패턴으로 GenOSVectorMeta 조립
        vector = (GenOSVectorMetaBuilder()
                  .set_text(content)
                  .set_page_info(...)
                  .set_chunk_index(chunk_idx)
                  .set_global_metadata(**global_metadata)
                  .set_chunk_bboxes(chunk.meta.doc_items, document)
                  .set_media_files(chunk.meta.doc_items)
                  ).build()
        vectors.append(vector)

        # 이미지 파일이 있으면 비동기로 업로드
        if upload_files:
            file_list = self.get_media_files(chunk.meta.doc_items)
            upload_tasks.append(asyncio.create_task(
                upload_files(file_list, request=request)
            ))

    # 모든 이미지 업로드 완료 대기
    if upload_tasks:
        await asyncio.gather(*upload_tasks)
```

**`safe_join()` 헬퍼**:

```python
def safe_join(self, iterable):
    if not isinstance(iterable, (list, tuple, set)):
        return ''                                    # None이면 빈 문자열
    return ''.join(map(str, iterable)) + '\n'        # 리스트면 합쳐서 반환
```

> 제목(heading) 리스트를 안전하게 하나의 문자열로 합칩니다. `None`이 들어와도 에러 없이 처리됩니다.

---

### 8.2 `HwpxProcessor`

```python
class HwpxProcessor:
```

**목적**: `.hwpx` (한글 오피스 XML) 파일을 Docling 엔진으로 파싱합니다.

`DocxProcessor`와 거의 동일한 구조이며, 차이점은 변환기 설정뿐입니다:

```python
self.converter = DocumentConverter(
    format_options={
        InputFormat.XML_HWPX: HwpxFormatOption(       # HWPX 전용 포맷 옵션
            pipeline_options=self.pipeline_options
        )
    }
)
```

| 비교 항목 | DocxProcessor | HwpxProcessor |
|-----------|--------------|---------------|
| 입력 포맷 | `InputFormat.DOCX` | `InputFormat.XML_HWPX` |
| 포맷 옵션 | `WordFormatOption` | `HwpxFormatOption` |
| 이미지 저장 | 기본값 | `save_images=False` (기본, kwargs로 변경 가능) |
| 나머지 로직 | 동일 | 동일 |

---

### 8.3 `DocumentProcessor` (메인 엔트리포인트)

```python
class DocumentProcessor:
```

**목적**: 모든 파일 포맷을 통합 관리하는 **메인 컨트롤러**입니다. 파일 확장자에 따라 적절한 로더/프로세서를 선택하여 처리합니다.

#### `__call__()` — 실행 진입점

```python
async def __call__(self, request: Request, file_path: str, **kwargs: dict):
```

**확장자별 분기 로직**:

```python
    ext = os.path.splitext(file_path)[-1].lower()

    if ext in ('.wav', '.mp3', '.m4a'):
        # ── 오디오 파일 ──
        loader = AudioLoader(...)
        vectors = loader.return_vectormeta_format()
        return vectors

    elif ext in ('.csv', '.xlsx'):
        # ── 정형 데이터 ──
        loader = TabularLoader(file_path, ext)
        vectors = loader.return_vectormeta_format()
        return vectors

    elif ext == '.hwp':
        # ── 한글(HWP) ──
        documents = self.load_documents(file_path)    # HwpLoader 사용
        chunks = self.split_documents(documents)       # RecursiveCharacterTextSplitter
        vectors = self.compose_vectors(file_path, chunks)
        return vectors

    elif ext == '.hwpx':
        # ── 한글(HWPX) ── Docling 기반 프로세서에 위임
        return await self.hwpx_processor(request, file_path, **kwargs)

    elif ext == '.docx':
        # ── Word(DOCX) ── Docling 기반 프로세서에 위임
        return await self.docx_processor(request, file_path, **kwargs)

    else:
        # ── 기타 모든 포맷 (PDF, PPT, DOC, 이미지, TXT 등) ──
        documents = self.load_documents(file_path)
        chunks = self.split_documents(documents)
        vectors = self.compose_vectors(file_path, chunks)
        return vectors
```

#### `get_loader()` — 로더 선택기

파일 확장자에 따라 적절한 LangChain 로더를 반환합니다.

```python
def get_loader(self, file_path: str):
    ext = os.path.splitext(file_path)[-1].lower()
    real_type = self.get_real_file_type(file_path)  # 매직 바이트로 실제 타입 확인

    # 확장자와 실제 타입이 다르면 실제 타입 우선 (확장자 조작/오류 대응)
    if ext != real_type and real_type == 'pdf':
        return PyMuPDFLoader(file_path)
```

**확장자-로더 매핑 총정리**:

| 확장자 | 로더 | 비고 |
|--------|------|------|
| `.pdf` | `PyMuPDFLoader` | 가장 빠른 PDF 텍스트 추출 |
| `.doc` | `UnstructuredWordDocumentLoader` | LibreOffice로 PDF 변환 후 처리 |
| `.ppt`, `.pptx` | `UnstructuredPowerPointLoader` | LibreOffice로 PDF 변환 후 처리 |
| `.jpg`, `.jpeg`, `.png` | `UnstructuredImageLoader` | OCR로 텍스트 추출 (한/영) |
| `.txt`, `.json` | `TextLoader` (커스텀) | 인코딩 자동 감지 |
| `.hwp` | `HwpLoader` (커스텀) | hwp5html → PDF 경로 |
| `.md` | `UnstructuredMarkdownLoader` | 마크다운 구조 파싱 |
| 기타 | `UnstructuredFileLoader` | 범용 폴백 로더 |

#### `get_real_file_type()` — 파일 매직 바이트 검사

```python
def get_real_file_type(self, file_path: str) -> str:
    with open(file_path, 'rb') as f:
        header = f.read(8)         # 파일의 처음 8바이트 읽기
    if header.startswith(b'%PDF-'):      # PDF 매직 바이트
        return 'pdf'
    elif header.startswith(b'\x89PNG'):   # PNG 매직 바이트
        return 'png'
    elif header.startswith(b'\xff\xd8\xff'):  # JPEG 매직 바이트
        return 'jpg'
    return os.path.splitext(file_path)[-1].lower()  # 판단 불가 시 확장자 사용
```

> **왜 필요한가?**: 사용자가 확장자를 잘못 지정하거나(예: PDF 파일인데 `.txt`로 저장), 시스템에서 확장자가 손실된 경우에도 올바른 로더를 선택할 수 있게 합니다.

#### `split_documents()` — 텍스트 분할

```python
def split_documents(self, documents, **kwargs: dict) -> list[Document]:
    chunk_size = kwargs.get('chunk_size', 1000)      # 기본 1000자
    chunk_overlap = kwargs.get('chunk_overlap', 100)  # 기본 100자 오버랩
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
```

**`RecursiveCharacterTextSplitter` 동작 원리**:

```
입력 텍스트 (3000자):
─────────────────────────────────────────────────────────────

chunk_size=1000, chunk_overlap=100 이면:

청크 1: [0 ─────────────── 1000]
청크 2:              [900 ─────────────── 1900]     ← 100자 오버랩
청크 3:                            [1800 ─────────────── 2800]
청크 4:                                          [2700 ──── 3000]

"Recursive"의 의미: 분할 구분자를 단계적으로 시도
  1순위: "\n\n" (빈 줄, 문단 경계)
  2순위: "\n" (줄 바꿈)
  3순위: " " (공백)
  4순위: "" (글자 단위)
```

#### `compose_vectors()` — 벡터 메타데이터 조립

이 메서드는 `HwpxProcessor`/`DocxProcessor`와 달리 빌더 패턴을 사용하지 않고, 딕셔너리를 직접 구성합니다:

```python
def compose_vectors(self, file_path, chunks, **kwargs):
    # ...

    for chunk_idx, chunk in enumerate(chunks):
        page = chunk.metadata.get('page', 1)
        if ext not in ['.hwpx', '.docx']:
            page += 1    # LangChain 로더는 페이지를 0부터, 
                         # Docling은 1부터 시작하므로 보정

        vectors.append(GenOSVectorMeta.model_validate({
            'text': text,
            'n_char': len(text),
            'n_word': len(text.split()),
            'n_line': len(text.splitlines()),
            'i_page': page,
            'e_page': page,
            'i_chunk_on_page': chunk_index_on_page,
            'n_chunk_of_page': self.page_chunk_counts[page],
            'i_chunk_on_doc': chunk_idx,
            **global_metadata
        }))
```

> **첨부용 전처리기의 특징**: 주석에서도 언급되듯이, bbox(바운딩 박스) 정보 추출은 생략됩니다. 이는 속도를 위한 의도적 생략입니다.

#### `setup_logging()` — 로깅 레벨 설정

```python
def setup_logging(self, level_num: int):
    # 레벨 매핑:
    # 5 → DEBUG   (가장 상세)
    # 4 → INFO    (기본값)
    # 3 → WARNING
    # 2 → ERROR
    # 1 → CRITICAL
    # 0 → NOLOG   (모든 로그 비활성화)
```

---

## 9. 예외 클래스

```python
class GenosServiceException(Exception):
    def __init__(self, error_code: str, error_msg: Optional[str] = None, 
                 msg_params: Optional[dict] = None) -> None:
        self.code = 1
        self.error_code = error_code
        self.error_msg = error_msg or "GenOS Service Exception"
        self.msg_params = msg_params or {}
```

**목적**: GenOS 플랫폼과의 의존성을 제거하기 위한 독립적 예외 클래스입니다. 원래 GenOS 프레임워크에 내장된 예외를 대체합니다.

**사용 위치**:
- `DocxProcessor.__call__()`: 청크가 0개일 때 발생
- `HwpxProcessor.__call__()`: 청크가 0개일 때 발생

```python
if len(chunks) >= 1:
    vectors = await self.compose_vectors(...)
else:
    raise GenosServiceException(1, f"chunk length is 0")
```

---

## 10. 실행 흐름 요약

### 사용 방법 (Genos facade에 통합 시)

```python
# 1. DocumentProcessor 인스턴스 생성
processor = DocumentProcessor()

# 2. 파일 처리 호출 (비동기)
vectors = await processor(
    request=request,           # FastAPI Request 객체
    file_path="/path/to/file.pdf",
    chunk_size=1000,           # 선택: 청크 크기 (기본값 1000)
    chunk_overlap=100,         # 선택: 청크 오버랩 (기본값 100)
    log_level=4,               # 선택: 로그 레벨 (기본값 4=INFO)
)

# 3. 결과: List[GenOSVectorMeta]
for v in vectors:
    print(v.text)              # 청크 텍스트
    print(v.i_page)            # 페이지 번호
    print(v.i_chunk_on_doc)    # 문서 내 청크 순서
```

### 전체 처리 흐름 (PDF 파일 예시)

```
processor(request, "/data/report.pdf")
    │
    ▼
__call__()
    │  ext = '.pdf' → else 분기
    │
    ├── load_documents("/data/report.pdf")
    │   │
    │   ├── get_loader() → PyMuPDFLoader 선택
    │   └── loader.load() → List[Document]
    │       [
    │         Document(page_content="1페이지 텍스트...", metadata={'page': 0}),
    │         Document(page_content="2페이지 텍스트...", metadata={'page': 1}),
    │         ...
    │       ]
    │
    ├── split_documents(documents, chunk_size=1000)
    │   │
    │   └── RecursiveCharacterTextSplitter로 분할
    │       → List[Document] (청크 단위)
    │
    └── compose_vectors("/data/report.pdf", chunks)
        │
        └── 각 청크를 GenOSVectorMeta로 변환
            [
              GenOSVectorMeta(text="제1조...", i_page=1, i_chunk_on_doc=0, ...),
              GenOSVectorMeta(text="제2조...", i_page=1, i_chunk_on_doc=1, ...),
              GenOSVectorMeta(text="제3조...", i_page=2, i_chunk_on_doc=2, ...),
              ...
            ]
```

---

## 11. 지원 파일 포맷 총정리

| 카테고리 | 확장자 | 처리 경로 | 핵심 도구 |
|----------|--------|-----------|-----------|
| **PDF** | `.pdf` | 직접 텍스트 추출 | PyMuPDF |
| **한글** | `.hwp` | hwp5html → PDF → 추출 | pyhwp, WeasyPrint |
| **한글 XML** | `.hwpx` | Docling 구조 파싱 | Docling + HybridChunker |
| **Word** | `.docx` | Docling 구조 파싱 | Docling + HybridChunker |
| **Word 레거시** | `.doc` | LibreOffice → PDF 변환 | LangChain Unstructured |
| **프레젠테이션** | `.ppt`, `.pptx` | LibreOffice → PDF 변환 | LangChain Unstructured |
| **스프레드시트** | `.csv` | pandas DataFrame 파싱 | pandas + DataFrameLoader |
| **스프레드시트** | `.xlsx` | pandas DataFrame 파싱 (다중 시트) | pandas + openpyxl |
| **이미지** | `.jpg`, `.jpeg`, `.png` | OCR 텍스트 추출 | Unstructured + Tesseract |
| **텍스트** | `.txt`, `.json` | 인코딩 감지 + 직접 읽기 | chardet |
| **마크다운** | `.md` | 마크다운 구조 파싱 | Unstructured Markdown |
| **오디오** | `.mp3`, `.wav`, `.m4a` | STT (Speech-to-Text) | pydub + Whisper API |
| **기타** | 그 외 | 범용 텍스트 추출 시도 | UnstructuredFileLoader |

---

> **참고**: 이 전처리기는 **속도를 최우선**으로 설계되어, AI 기반 레이아웃 분석(Layout Detection)이나 TableFormer 같은 고급 기능은 포함하지 않습니다. 고품질 구조 분석이 필요한 경우 `intelligent_processor.py`를, PDF 표준화가 필요한 경우 `convert_processor.py`를 사용하세요.