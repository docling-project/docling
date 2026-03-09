# `convert_processor.py` 코드 상세 설명서

---

## 📌 목차

1. [개요](#1-개요)
2. [전체 아키텍처](#2-전체-아키텍처)
3. [attachment_processor와의 핵심 차이점](#3-attachment_processor와의-핵심-차이점)
4. [임포트 및 초기 설정](#4-임포트-및-초기-설정)
5. [유틸리티 함수](#5-유틸리티-함수)
   - 5.1 [`convert_to_pdf()`](#51-convert_to_pdf)
   - 5.2 [`_get_pdf_path()`](#52-_get_pdf_path)
6. [청커: `GenosBucketChunker`](#6-청커-genosbucketchunker)
   - 6.1 [기존 Chunker 와의 비교](#61-기존-Chunker와의-비교)
   - 6.2 [`preprocess()` — 문서 아이템 수집](#62-preprocess--문서-아이템-수집)
   - 6.3 [`_split_document_by_tokens()` — 4단계 분할·병합 파이프라인](#63-_split_document_by_tokens--4단계-분할병합-파이프라인)
   - 6.4 [헬퍼 메서드들](#64-헬퍼-메서드들)
   - 6.5 [`chunk()` — 최종 진입점](#65-chunk--최종-진입점)
7. [데이터 모델](#7-데이터-모델)
   - 7.1 [`GenOSVectorMeta`](#71-genosvectormeta)
   - 7.2 [`GenOSVectorMetaBuilder`](#72-genosvectormetabuilder)
8. [메인 프로세서: `DocumentProcessor`](#8-메인-프로세서-documentprocessor)
   - 8.1 [`__init__()` — 컨버터 및 파이프라인 초기화](#81-__init__--컨버터-및-파이프라인-초기화)
   - 8.2 [문서 로딩 메서드들](#82-문서-로딩-메서드들)
   - 8.3 [문서 분할 메서드들](#83-문서-분할-메서드들)
   - 8.4 [GLYPH 감지 및 선택적 OCR](#84-glyph-감지-및-선택적-ocr)
   - 8.5 [문서 Enrichment (LLM 기반 메타데이터 보강)](#85-문서-enrichment-llm-기반-메타데이터-보강)
   - 8.6 [벡터 조립 메서드들](#86-벡터-조립-메서드들)
   - 8.7 [`__call__()` — 실행 진입점](#87-__call__--실행-진입점)
9. [예외 클래스 및 유틸리티](#9-예외-클래스-및-유틸리티)
10. [Enrichment 프롬프트](#10-enrichment-프롬프트)
11. [실행 흐름 요약](#11-실행-흐름-요약)
12. [지원 파일 포맷 총정리](#12-지원-파일-포맷-총정리)

---

## 1. 개요

`convert_processor.py`는 **변환용 전처리기(Convert Processor)** 입니다. 문서의 시각적 형태(Layout)를 유지해야 하거나, 텍스트 추출이 까다로운 레거시 포맷을 처리하기 위한 전처리기로, 모든 문서를 **PDF로 우선 변환(Rendering)** 하여 포맷의 파편화를 해결합니다.

### 핵심 설계 철학

```
"호환성 중심: PDF 표준화 후 텍스트 추출"
```

| 특징 | 설명 |
|------|------|
| **PDF 표준화** | LibreOffice 엔진을 활용하여 PPT, DOCX 등 다양한 문서를 PDF 포맷으로 통일 |
| **시각적 정합성 유지** | 원본 문서의 폰트, 이미지 배치, 페이지 레이아웃을 그대로 보존 |
| **하이브리드 추출** | 변환된 PDF 레이어에서 텍스트와 이미지 정보를 결합하여 안정적인 정보 획득 |
| **Smart OCR** | GLYPH(인코딩 깨짐)가 감지된 영역만 선별적으로 OCR 수행 |
| **LLM Enrichment** | LLM으로 목차(TOC) 자동 생성 및 문서 메타데이터 추출 |

> **위치**: `preprocessor/facade/convert_processor.py`
> 
> 첨부용 전처리기(`attachment_processor.py`) 대용으로 쓸 수 있도록 고안된 **변형 전처리기**입니다.

---

## 2. 전체 아키텍처

아래 다이어그램은 파일이 입력되었을 때 확장자에 따라 어떤 경로로 처리되는지를 보여줍니다.

```
사용자가 파일 업로드
        │
        ▼
 ┌──────────────────┐
 │ DocumentProcessor│  ◄── 메인 엔트리포인트 (__call__)
 │   (라우터 역할)     │
 └──────┬───────────┘
        │
        │  확장자(ext)에 따라 분기
        │
        ├── .ppt ──────────────────► LangChain 경로
        │                            │
        │                            ├─ convert_to_pdf() → PDF 변환
        │                            ├─ UnstructuredPowerPointLoader
        │                            ├─ RecursiveCharacterTextSplitter
        │                            ├─ _extract_page_images() → 이미지 추출
        │                            └─ compose_vectors_langchain()
        │
        └── 기타 (.pdf, .docx, .pptx 등) ──► Genos 지능형 Doc Parser 경로
                                              │
                                              ├─ load_documents_with_docling()
                                              │    └─ 실패 시 second_converter로 폴백
                                              │
                                              ├─ [PDF만] GLYPH/품질 검사
                                              │    └─ 필요 시 OCR 재변환
                                              │    └─ 테이블 셀 단위 OCR
                                              │
                                              ├─ [DOCX/PPTX] convert_to_pdf()
                                              │
                                              ├─ _with_pictures_refs() → 이미지 참조
                                              │
                                              ├─ enrichment() → LLM 기반 보강
                                              │    ├─ TOC 자동 생성
                                              │    └─ 메타데이터 추출
                                              │
                                              ├─ GenosBucketChunker
                                              │    ├─ preprocess() → 아이템 수집
                                              │    └─ _split_document_by_tokens()
                                              │         ├─ 1단계: 섹션 헤더 분할
                                              │         ├─ 2단계: heading 텍스트 생성
                                              │         ├─ 2.5단계: 초과 청크 분할
                                              │         ├─ 3단계: 단독 타이틀 병합
                                              │         └─ 4단계: 토큰 기준 병합
                                              │
                                              └─ compose_vectors()
                                                   │
                                                   ▼
                                          ┌──────────────────────────┐
                                          │ List[GenOSVectorMeta]    │
                                          │ (최종 출력: 청크별 메타)      │
                                          │ + created_date, authors, │
                                          │   title, HEADER: ...     │
                                          └──────────────────────────┘
```

---

## 3. attachment_processor와의 핵심 차이점

`convert_processor`는 `attachment_processor`의 **상위 호환 변형**입니다. 아래 표에서 두 전처리기의 차이를 한눈에 비교할 수 있습니다.

| 비교 항목           | attachment_processor | convert_processor |
|-----------------|---------------------|-------------------|
| **설계 목표**       | 속도 중심 (즉각 응답) | 호환성·품질 중심 |
| **PDF 변환**      | 일부 포맷만 (HWP→PDF) | 거의 모든 포맷 PDF 통일 |
| **전처리 파이프라인**   | `SimplePipeline` (경량) | 전체 PDF 파이프라인 (레이아웃+표 구조 분석) |
| **청커(Chunker)** | `HybridChunker` (max_tokens=∞) | `GenosBucketChunker` (max_tokens=2000) |
| **OCR**         | 없음 | PaddleOCR (GLYPH 감지 시 선택적) |
| **Enrichment**  | 없음 | LLM 기반 TOC 생성 + 메타데이터 추출 |
| **표(Table) 처리** | 단순 마크다운 변환 | TableFormer(ACCURATE) + 셀 단위 OCR |
| **메타데이터**       | 기본(text, page 등) | 확장(created_date, authors, title, HEADER:) |
| **오디오/정형 데이터**  | AudioLoader, TabularLoader | 미지원 (문서 전용) |
| **컨버터 폴백**      | 없음 | 주 컨버터 실패 시 보조 컨버터 자동 전환 |

---

## 4. 임포트 및 초기 설정

### 주요 외부 라이브러리 그룹

| 그룹 | 라이브러리 | 용도 |
|------|-----------|------|
| **Docling 코어** | `docling.document_converter`, `docling.pipeline` | PDF 파싱, 레이아웃 분석, 표 구조 인식 |
| **Docling 데이터 모델** | `docling_core.types.doc` | DoclingDocument, DocItem, TableItem 등 문서 구조 타입 |
| **Docling 청킹** | `docling_core.transforms.chunker` | BaseChunker, DocChunk 등 청킹 기반 클래스 |
| **OCR** | `PaddleOcrOptions` | PaddleOCR 엔진 설정 |
| **PDF 조작** | `fitz` (PyMuPDF) | PDF 열기, 이미지 추출, 텍스트 검색 |
| **문서 로딩 (폴백)** | `langchain_community.document_loaders` | PPT 등 Docling 미지원 포맷의 폴백 처리 |
| **텍스트 분할 (폴백)** | `langchain_text_splitters` | LangChain 경로에서의 텍스트 분할 |
| **토크나이저** | `transformers.AutoTokenizer`, `semchunk` | 토큰 수 기반 청크 크기 관리 |
| **Enrichment** | `docling.utils.document_enrichment` | LLM 기반 목차/메타데이터 보강 |

### PDF 변환 대상 확장자 상수

```python
CONVERTIBLE_EXTENSIONS = ['.txt', '.json', '.md', '.docx', '.ppt', '.pptx']
```

> `attachment_processor`와 비교하면 `.hwp`가 빠져 있습니다. 
> 이 전처리기는 HWP를 별도로 처리하지 않고, Genos Doc Parser 에서 Layout 분석을 지원하는 포맷 위주로 동작합니다.

---

## 5. 유틸리티 함수

### 5.1 `convert_to_pdf()`

```python
def convert_to_pdf(file_path: str) -> str | None:
```

**목적**: LibreOffice를 활용하여 다양한 문서 포맷을 PDF로 변환합니다.

`attachment_processor`의 `convert_to_pdf()`와 거의 동일하지만 미세한 차이가 있습니다:

| 차이점 | attachment_processor | convert_processor |
|--------|---------------------|-------------------|
| 지원 필터 | `.xls/.xlsx/.csv` 포함 | `.xls/.xlsx/.csv` 미포함 |
| 예외 처리 | `except`에서 `None` 반환 | `except`에서 PDF가 이미 존재하면 경로 반환 |
| 로그 출력 | stderr만 | stderr + stdout |

```python
    except Exception as e:
        if pdf_path.exists():
            return str(pdf_path)    # 이미 PDF가 존재하면 그대로 반환
```

> **설계 의도**: 이전에 이미 변환된 PDF가 있을 경우 중복 변환을 방지합니다.

**처리 흐름**: (`attachment_processor`와 동일)

```
입력 파일 (PPT, DOCX 등)
    │
    ▼
확장자 판별 → 적절한 LibreOffice 필터 선택
    │
    ├── .ppt/.pptx → "pdf:impress_pdf_Export"
    ├── .doc/.docx → "pdf:writer_pdf_Export"
    └── 기타      → "pdf"
    │
    ▼
비ASCII 파일명 체크 → 필요 시 임시 ASCII 복사본 생성
    │
    ▼
soffice --headless --convert-to ... 실행
    │
    ├── 성공 → PDF 경로 반환
    └── 실패 → None 반환
```

---

### 5.2 `_get_pdf_path()`

```python
def _get_pdf_path(file_path: str) -> str:
```

파일 경로의 확장자를 `.pdf`로 단순 치환합니다.

---

## 6. 청커: `GenosBucketChunker`

```python
class GenosBucketChunker(BaseChunker):
    """토큰 제한을 고려하여 섹션별 청크를 분할하고 병합하는 청커 (v2)"""
```

### 6.1 기존 Chunker 와의 비교

`GenosBucketChunker`는 Genos Doc Parser 전처리기에서만 활용되는 **독자 구현** Chunker 입니다. `attachment_processor`의 `HybridChunker`가 "레이아웃 구조를 존중하되 토큰 제한은 사실상 무시"하는 방식이었다면, `GenosBucketChunker`는 **섹션 헤더 기반의 의미적 분할 + 토큰 수 기반의 정밀한 크기 제어**를 동시에 수행합니다.
- 주의
  - 현재 한시적으로 facade 에 구현되어있으며, 1.4 버전이후 container image (library) 에서 제공될 예정입니다.

**attachment_processor의 HybridChunker와 비교**:

| 비교 항목 | HybridChunker (attachment) | GenosBucketChunker (convert) |
|-----------|---------------------------|------------------------------|
| 분할 기준 | 문서 아이템 단위 | **섹션 헤더** 단위 |
| 토큰 제한 | `int(1e30)` (사실상 무제한) | **2000 토큰** (실질적 제한) |
| 병합 기준 | 동일 metadata(heading/caption) | **헤더 레벨 계층** + 토큰 제한 |
| 캡션 처리 | 없음 | 캡션을 부모 아이템에 **자동 병합** |
| 표 내 그림 | 없음 | 표 안의 그림을 표와 **자동 병합** |
| 초과 분할 | semchunk (시맨틱) | **균등 토큰 분할** (bisect 기반) |

**주요 설정값**:

```python
tokenizer: str = "/models/doc_parser_models/sentence-transformers-all-MiniLM-L6-v2"  # 토큰 카운팅용
max_tokens: int = 1024           # 기본값 (실제 사용 시 2000으로 오버라이드)
merge_peers: bool = True         # 동일 섹션 내 소규모 청크 병합
merge_list_items: bool = True    # 연속 리스트 아이템 병합
```

---

### 6.2 `preprocess()` — 문서 아이템 수집

```python
def preprocess(self, dl_doc: DLDocument, **kwargs: Any) -> Iterator[BaseChunk]:
```

**목적**: DoclingDocument의 모든 아이템을 순회하여, 각 아이템과 해당 시점의 **헤더 컨텍스트**를 쌍으로 수집합니다.

**처리 흐름**:

```
DoclingDocument
    │
    ▼
iterate_items()으로 모든 아이템 순회
    │
    │  각 아이템마다:
    │  ┌─────────────────────────────────────────┐
    │  │ 1. 리스트 아이템? → list_items에 누적        │
    │  │ 2. 섹션 헤더?    → heading_by_level 갱신   │
    │  │ 3. 텍스트/표/그림? → all_items에 추가        │
    │  │                                         │
    │  │ + 현재 heading_by_level 스냅샷을           │
    │  │   all_header_info에 저장                 │
    │  └─────────────────────────────────────────┘
    │
    ▼
누락된 테이블 검출 (iterate_items에서 빠진 테이블)
    │  → 문서 앞부분에 삽입
    │
    ▼
모든 아이템을 하나의 DocChunk로 포장하여 yield
    │  + _header_info_list 속성 부착
    │  + _header_short_info_list 속성 부착
    │
    ▼
_split_document_by_tokens()에서 실제 분할 수행
```

**핵심 데이터 구조 — 3개의 병렬 리스트**:

```python
all_items = []              # [DocItem, DocItem, ...]
all_header_info = []        # [{0: "제1장 총칙", 1: "제1절 목적"}, ...]  ← item.text 기반
all_header_short_info = []  # [{0: "총칙", 1: "목적"}, ...]            ← item.orig 기반 (짧은 형태)
```

이 세 리스트는 **동일한 인덱스**로 연결됩니다:
```
all_items[i]           → i번째 문서 아이템
all_header_info[i]     → i번째 아이템이 속한 헤더 컨텍스트 (전체 제목)
all_header_short_info[i] → i번째 아이템이 속한 헤더 컨텍스트 (짧은 제목)
```

**헤더 컨텍스트 추적 — `heading_by_level` 딕셔너리**:

```
문서 구조:                         current_heading_by_level 상태:
─────────                         ────────────────────────────
제1장 총칙                         {0: "제1장 총칙"}
  제1절 목적                       {0: "제1장 총칙", 1: "제1절 목적"}
    본문 텍스트 A                  {0: "제1장 총칙", 1: "제1절 목적"}
  제2절 범위                       {0: "제1장 총칙", 1: "제2절 범위"}
    ↑ level 1 변경 → 이전 level 1 삭제, 하위 레벨도 삭제
제2장 권리                         {0: "제2장 권리"}
    ↑ level 0 변경 → 모든 하위 레벨 삭제
```

```python
# 더 깊은 레벨의 헤더들 제거
keys_to_del = [k for k in current_heading_by_level if k > header_level]
for k in keys_to_del:
    current_heading_by_level.pop(k, None)
```

**`item.text` vs `item.orig`의 차이**:

```python
current_heading_by_level[header_level] = item.text   # "제1장 총칙" (전체 제목)
current_heading_short_by_level[header_level] = item.orig  # "총칙" (짧은 형태)
```

> `item.orig`는 Docling이 파싱할 때 원본에서 추출한 짧은 형태의 텍스트입니다. 이 짧은 헤더는 최종 청크의 `HEADER:` 접두어로 사용되어 벡터 검색 시 문맥 파악을 돕습니다.

**누락된 테이블 복구**:

```python
# iterate_items()에서 누락된 테이블들을 별도로 추가
missing_tables = []
for table in dl_doc.tables:
    table_ref = getattr(table, 'self_ref', None)
    if table_ref not in processed_refs:
        missing_tables.append(table)

# 누락된 테이블들을 문서 앞부분에 추가
if missing_tables:
    for missing_table in missing_tables:
        all_items.insert(0, missing_table)
```

> **왜 필요한가?**: Docling의 `iterate_items()`가 일부 특수한 위치의 테이블(예: 페이지 1의 헤더 테이블)을 누락하는 경우가 있습니다. `dl_doc.tables` 전체를 대조하여 빠진 테이블을 복구합니다.

---

### 6.3 `_split_document_by_tokens()` — 4단계 분할·병합 파이프라인

```python
def _split_document_by_tokens(self, doc_chunk: DocChunk, dl_doc: DoclingDocument) -> list[DocChunk]:
```

**목적**: `preprocess()`에서 만든 하나의 거대한 DocChunk를 **의미적으로 적절한 크기의 청크들로 분할**합니다. 이 과정은 4단계(+보정 단계)로 구성됩니다.

```
┌──────────────────────────────────────────────────────────────┐
│                     입력: 하나의 거대한 DocChunk                  │
│  [아이템1, 아이템2, 아이템3, ..., 아이템N]                          │
└──────────────┬───────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────┐
│  1단계: 섹션 헤더 기준 분할                                        │
│                                                              │
│  "제1장 총칙" 헤더를 만나면 → 새 섹션 시작                            │
│  [섹션A: 아이템1~3] [섹션B: 아이템4~8] [섹션C: 아이템9~N]             │
└──────────────┬───────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────┐
│  2단계: 각 섹션의 텍스트에 heading 붙이기                           │
│                                                              │
│  섹션A → "제1장 총칙, 제1절 목적, 본문텍스트..."                     │
│  섹션B → "제1장 총칙, 제2절 범위, 본문텍스트..."                     │
└──────────────┬───────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────┐
│  2.5단계: max_tokens 초과 섹션 분할                              │
│                                                              │
│  섹션B가 5000토큰이고 max_tokens=2000이면:                        │
│  → 캡션 조정 (caption을 부모에 붙임)                               │
│  → 표 안의 그림 조정 (그림을 표에 붙임)                              │
│  → 균등 토큰 분할: [섹션B-1: ~1700토큰] [섹션B-2: ~1700토큰]         │
│                   [섹션B-3: ~1600토큰]                         │
└──────────────┬───────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────┐
│  3단계: 단독 타이틀 → 다음 섹션으로 병합                             │
│                                                              │
│  ["제2장 권리" (10자)] + [섹션C: 1500토큰]                        │
│  → ["제2장 권리\n섹션C 내용..." (1510토큰)]                        │
│                                                              │
│  조건: 아이템 1개, 30자 이하, 다음 섹션 레벨이 더 낮지 않음              │
└──────────────┬───────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────┐
│  4단계: 토큰 기준 최종 병합                                       │
│                                                              │
│  [200토큰] + [300토큰] + [400토큰] = [900토큰] ← OK              │
│  + [1500토큰] = [2400토큰] → 초과! → 새 청크 시작                  │
│                                                              │
│  또는 헤더 레벨이 더 상위면 → 새 청크 시작                            │
└──────────────┬───────────────────────────────────────────────┘
               │
               ▼
        최종 List[DocChunk]
```

#### 1단계: 섹션 헤더 기준 분할

```python
for i, item in enumerate(items):
    if self._is_section_header(item):
        # 이전 섹션이 있으면 저장
        if cur_items:
            sections.append((cur_items, cur_h_infos, cur_h_short))
        # 새로운 섹션 시작
        cur_items = [item]
        cur_h_infos = [h_info]
        cur_h_short = [h_short]
    else:
        # 현재 섹션에 추가
        cur_items.append(item)
```

> **핵심**: 섹션 헤더(제목, 장/절/조 헤더)를 만날 때마다 새 섹션을 시작합니다. 이렇게 하면 의미적으로 독립된 단위로 문서가 분리됩니다.

#### 2단계: heading 텍스트 생성

```python
text = self._generate_section_text_with_heading(items, header_short_infos, dl_doc)
```

각 섹션의 텍스트 앞에 **계층적 헤딩 경로**를 붙입니다:

```
입력:
  header_short_infos[0] = {0: "총칙", 1: "목적"}
  items[0].text = "이 법은 국민의 권리를..."

출력:
  "총칙, 목적, 이 법은 국민의 권리를..."
```

#### 2.5단계: 초과 청크의 균등 분할

이 단계에서는 `max_tokens`를 초과하는 섹션을 처리합니다. 두 가지 사전 조정 후 균등 분할을 수행합니다:

**캡션 조정 (`adjust_captions`)**:

```
조정 전:                           조정 후:
──────────                        ──────────
아이템 0: 표 (caption → 아이템2)   아이템 0: [표, 캡션텍스트] ← 하나로 병합
아이템 1: 본문 텍스트              아이템 1: 본문 텍스트
아이템 2: "표 1의 설명" (캡션)      (제거됨)
```

> 캡션이 부모 아이템(표, 그림)과 다른 청크에 배치되면 의미가 끊어집니다. 캡션을 부모에 병합하여 이를 방지합니다.

**표 내 그림 조정 (`adjust_pictures_in_tables`)**:

```python
ios = pic_bbox.intersection_over_self(table_bbox)
if ios > 0.5:  # 그림이 50% 이상 표 안에 포함되면
    # → 표 아이템에 병합
```

> 표 안에 삽입된 그림이 별도 아이템으로 추출된 경우, 바운딩 박스의 겹침(IoS: Intersection over Self)을 계산하여 표에 자동 병합합니다.

**균등 토큰 분할 (`split_items_evenly_by_tokens`)**:

```python
def split_items_evenly_by_tokens(item_token_counts, max_tokens):
    # 총 토큰 수 / max_tokens → 필요한 청크 수(k) 계산
    k = math.ceil(total / max_tokens)
    target = total / k    # 각 청크의 목표 토큰 수

    # 접두합(prefix sum) 배열로 분할 지점 이진 탐색
    P = [0]
    for c in item_token_counts:
        P.append(P[-1] + c)

    # bisect_left로 목표 토큰에 가장 가까운 분할점 탐색
    ...
```

**시각적 예시**:

```
아이템별 토큰: [100, 200, 150, 300, 500, 400, 250, 100]
총 토큰: 2000, max_tokens: 800

k = ceil(2000/800) = 3
target = 2000/3 ≈ 667

접두합: [0, 100, 300, 450, 750, 1250, 1650, 1900, 2000]

분할점 탐색:
  목표 667 → bisect → 인덱스 4 (접두합 750) ← 분할1
  목표 1333 → bisect → 인덱스 6 (접두합 1650) ← 분할2

결과: [(0,4), (4,6), (6,8)]
  청크1: 아이템0~3 (750토큰)
  청크2: 아이템4~5 (900토큰)
  청크3: 아이템6~7 (350토큰)
```

#### 3단계: 단독 타이틀 병합

```python
for i in range(len(sections_with_text) - 2, -1, -1):  # 뒤에서부터 순회
    text, items, h_infos, h_short = sections_with_text[i]

    # 조건 확인:
    # 1. 아이템이 정확히 1개
    # 2. 그 아이템이 섹션 헤더
    # 3. 30자 이하 (문단이 아닌 순수 제목)
    # 4. 다음 섹션의 헤더 레벨이 현재보다 높지 않음(같거나 낮음)

    if len(items) == 1 and self._is_section_header(items[0]) and len(item_text) <= 30:
        # 다음 섹션과 병합
        sections_with_text[i] = (text + '\n' + n_text, items + n_items, ...)
        sections_with_text.pop(i + 1)
```

**시각적 예시**:

```
병합 전:                               병합 후:
──────────                            ──────────
섹션 3: "제2장 권리" (8자, 단독 타이틀)  섹션 3: "제2장 권리\n제3조 국민은..."
섹션 4: "제3조 국민은..."                (제거됨)
```

> **왜 뒤에서부터 순회하는가?**: `pop(i+1)` 연산이 뒤쪽 인덱스에 영향을 주지 않게 하기 위함입니다.

#### 4단계: 토큰 기준 최종 병합

```python
for text, items, header_infos, header_short_infos in sections_with_text:
    test_tokens = self._count_tokens("\n".join(merged_texts + [text]))

    b_new_chunk = False

    # 토큰 초과 시 새 청크
    if test_tokens > self.max_tokens and len(merged_texts) > 0:
        b_new_chunk = True

    # 현재 섹션의 헤더 레벨이 더 상위면 새 청크
    elif 0 <= section_level < merged_level:
        b_new_chunk = True
```

**병합 판단 로직 시각화**:

```
섹션들:  [A:200토큰, B:300토큰, C:1800토큰, D:500토큰, E:400토큰]
max_tokens = 2000

처리 과정:
  A(200) → 병합 시작
  A+B(500) → 토큰 여유 → 병합 계속
  A+B+C(2300) → 초과! → A+B를 청크1로 확정, C로 새 병합 시작
  C(1800) → 병합 시작
  C+D(2300) → 초과! → C를 청크2로 확정, D로 새 병합 시작
  D+E(900) → 토큰 여유 → 병합 계속
  [끝] → D+E를 청크3으로 확정

결과: [청크1: A+B (500토큰)] [청크2: C (1800토큰)] [청크3: D+E (900토큰)]
```

**헤더 레벨 기반 분할**:

```
merged_level = 1  (현재까지 병합된 것 중 가장 깊은 레벨: "제1절")
section_level = 0 (새로 만난 섹션: "제2장" → 상위 레벨)

→ 0 < 1 이므로 새 청크 시작! (장 단위로 청크를 분리)
```

---

### 6.4 헬퍼 메서드들

#### `_count_tokens()` — 안전한 토큰 카운팅

```python
def _count_tokens(self, text: str) -> int:
```

매우 긴 텍스트를 토크나이저에 한 번에 넣으면 메모리 문제가 발생할 수 있으므로, **줄 단위로 분할하여 300자씩** 토큰을 계산합니다:

```python
max_chunk_length = 300
lines = text.split('\n')

for line in lines:
    temp_chunk = current_chunk + '\n' + line
    if len(temp_chunk) <= max_chunk_length:
        current_chunk = temp_chunk        # 계속 누적
    else:
        total_tokens += len(self._tokenizer.tokenize(current_chunk))  # 계산 후
        current_chunk = line              # 새로 시작
```

> 토크나이저 호출이 실패할 경우 `단어 수 × 1.3`으로 근사 계산하는 폴백도 포함되어 있습니다.

#### `_extract_table_text()` — 표 텍스트 추출 (다중 폴백)

```python
def _extract_table_text(self, table_item: TableItem, dl_doc: DoclingDocument) -> str:
```

표에서 텍스트를 추출하는 과정은 3단계 폴백으로 구성됩니다:

```
1순위: export_to_markdown(dl_doc) → 마크다운 표
       │
       ├── 성공 → 반환
       └── 실패 ↓

2순위: table_cells 또는 grid에서 직접 추출
       │
       │  for cell in table_item.data.table_cells:
       │      cell_texts.append(cell.text)
       │  return ' '.join(cell_texts)
       │
       ├── 성공 → 반환
       └── 실패 ↓

3순위: item.text 그대로 사용
       └── 최후의 수단
```

#### `_is_section_header()` / `_get_section_header_level()` — 헤더 판별

```python
def _is_section_header(self, item: DocItem) -> bool:
    return (isinstance(item, SectionHeaderItem) or
            (isinstance(item, TextItem) and
             item.label in [DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE]))
```

| 반환값 | level |
|--------|-------|
| `SectionHeaderItem` | `item.level` (Docling이 분석한 실제 레벨) |
| `TITLE` | `0` (최상위) |
| `SECTION_HEADER` | `1` (기본 절 레벨) |

---

### 6.5 `chunk()` — 최종 진입점

```python
def chunk(self, dl_doc: DoclingDocument, **kwargs: Any) -> Iterator[BaseChunk]:
    doc_chunks = list(self.preprocess(dl_doc=dl_doc, **kwargs))      # 1개의 거대 청크
    doc_chunk = doc_chunks[0]
    final_chunks = self._split_document_by_tokens(doc_chunk, dl_doc)  # 분할·병합
    return iter(final_chunks)
```

전체 과정을 요약하면:

```
DoclingDocument
    → preprocess(): 모든 아이템 + 헤더 정보 수집 → 1개의 DocChunk
    → _split_document_by_tokens(): 4단계 파이프라인 → N개의 DocChunk
    → Iterator[DocChunk]
```

---

## 7. 데이터 모델

### 7.1 `GenOSVectorMeta`

```python
class GenOSVectorMeta(BaseModel):
    class Config:
        extra = 'allow'

    text: str = None
    n_char: int = None
    n_word: int = None
    n_line: int = None
    i_page: int = None           # 청크 시작 페이지
    e_page: int = None           # 청크 끝 페이지
    i_chunk_on_page: int = None
    n_chunk_of_page: int = None
    i_chunk_on_doc: int = None
    n_chunk_of_doc: int = None
    n_page: int = None
    reg_date: str = None
    chunk_bboxes: str = None
    media_files: str = None
    created_date: int = None     # ◄── 확장 필드: YYYYMMDD 정수
    authors: str = None          # ◄── 확장 필드: 작성자 JSON 문자열
    title: str = None            # ◄── 확장 필드: 문서 제목
```

**attachment_processor와의 차이점**:

| 필드 | attachment | convert | 설명 |
|------|-----------|---------|------|
| `created_date` | ❌ | ✅ | 문서 작성일 (YYYYMMDD 정수, 예: `20250115`) |
| `authors` | ❌ | ✅ | 작성자 이름 (JSON 배열 문자열, 예: `'["홍길동","김철수"]'`) |
| `title` | ❌ | ✅ | 문서 제목 (문자열) |
| 타입 기본값 | `None` (Optional) | `None` (비Optional) | convert는 `str = None` (차이 없음, 스타일 차이) |

---

### 7.2 `GenOSVectorMetaBuilder`

`attachment_processor`와 동일한 빌더 패턴이지만, 확장 필드(`created_date`, `authors`, `title`)를 추가로 지원합니다:

```python
class GenOSVectorMetaBuilder:
    def __init__(self):
        # ... (기존 필드들)
        self.created_date: Optional[int] = None
        self.authors: Optional[str] = None
        self.title: Optional[str] = None

    def build(self) -> GenOSVectorMeta:
        return GenOSVectorMeta(
            # ... (기존 필드들)
            created_date=self.created_date,
            authors=self.authors,
            title=self.title,
        )
```

나머지 메서드(`set_text`, `set_page_info`, `set_chunk_bboxes` 등)는 `attachment_processor`와 동일합니다.

---

## 8. 메인 프로세서: `DocumentProcessor`

### 8.1 `__init__()` — 컨버터 및 파이프라인 초기화

```python
class DocumentProcessor:
    def __init__(self):
```

이 프로세서의 초기화는 매우 정교한 파이프라인 설정을 포함합니다. `attachment_processor`와 달리 **4개의 서로 다른 컨버터**를 준비합니다.

#### OCR 엔진 설정
Genos 에서는 기본적으로 PaddleOCR v5 를 탑재합니다. 다만 이는 Site 마다 다르므로, PaddleOCR 사용하지 않으면 이부분은 skip 하시면 됩니다.

```python
self.ocr_endpoint = "http://192.168.73.172:48080/ocr" #Genos 에 설치된 OCR 주소. Site 환경에 따라 변경 
ocr_options = PaddleOcrOptions( 
    force_full_page_ocr=False,   # 전체 페이지 OCR은 비활성
    lang=['korean'],              # 한국어 OCR
    ocr_endpoint=self.ocr_endpoint,
    text_score=0.3                # 텍스트 신뢰도 임계값
)
```

#### PDF 파이프라인 옵션

```python
self.pipe_line_options = PdfPipelineOptions()
self.pipe_line_options.generate_page_images = True        # 페이지 이미지 생성
self.pipe_line_options.generate_picture_images = True      # 그림 이미지 생성
self.pipe_line_options.do_ocr = False                      # 기본적으로 OCR 비활성
self.pipe_line_options.do_table_structure = True            # 표 구조 분석 활성
self.pipe_line_options.images_scale = 2                    # 이미지 해상도 2배
self.pipe_line_options.table_structure_options.do_cell_matching = True
self.pipe_line_options.table_structure_options.mode = TableFormerMode.ACCURATE  # 정확도 우선
```

> **`TableFormerMode.ACCURATE`**: Genos Doc Parser 에 속한 TableFormer 모델을 정확도 우선 모드로 실행합니다. 병합된 셀, 다중 헤더 등 복잡한 표를 정확하게 분석하지만 처리 시간이 더 걸립니다.
- version up시 deprecate 될 예정

#### 4개의 컨버터 구성

```python
def _create_converters(self):
```

```
┌─────────────────────────────────────────────────────────────────┐
│                      4개의 컨버터 매트릭스                           │
│                                                                 │
│              │    OCR 비활성화         │    OCR 활성화              │
│ ─────────────┼───────────────────────┼────────────────────────  │
│ 주 백엔드      │ converter             │ ocr_converter            │
│ (Pypdfium2)  │ (PyPdfiumDocumentBack)│ (DoclingParseV4Backend)  │
│ ─────────────┼───────────────────────┼────────────────────────  │
│ 보조 백엔드     │ second_converter      │ ocr_second_converter     │
│ (Pypdfium2)  │ (PyPdfiumDocumentBack)│ (PyPdfiumDocumentBack)   │
│              │                       │                          │
└─────────────────────────────────────────────────────────────────┘
```

| 컨버터 | OCR | 백엔드 | 용도 |
|--------|-----|--------|------|
| `converter` | ❌ | PyPdfium | 일반 PDF 텍스트 추출 (1순위) |
| `second_converter` | ❌ | PyPdfium | `converter` 실패 시 폴백 |
| `ocr_converter` | ✅ 전체 | DoclingParseV4 | GLYPH 감지 시 OCR 재처리 (1순위) |
| `ocr_second_converter` | ✅ 전체 | PyPdfium | `ocr_converter` 실패 시 폴백 |

> **폴백 설계 이유**: PDF 파일마다 내부 구조가 다르므로, 하나의 백엔드가 실패할 수 있습니다. 두 번째 백엔드로 자동 전환하여 처리 실패율을 최소화합니다.

#### Enrichment 옵션 설정

```python
self.enrichment_options = DataEnrichmentOptions(
    do_toc_enrichment=True,         # TOC(목차) 자동 생성 활성화
    toc_doc_type="law",             # 문서 유형: 법률/규정
    extract_metadata=True,          # 메타데이터 추출 활성화
    toc_api_provider="custom",      # 커스텀 API 사용

    # LLM API 설정 (Mistral-Small-3.1-24B)
    toc_api_base_url="https://genos.genon.ai:3443/api/gateway/.../v1/chat/completions",
    toc_api_key="...",
    toc_model="model",
    toc_temperature=0.0,            # 결정론적 출력
    toc_top_p=0.00001,              # 거의 greedy decoding
    toc_seed=33,                    # 재현 가능성
    toc_max_tokens=10000,           # 충분한 출력 길이

    toc_system_prompt=toc_system_prompt,   # 시스템 프롬프트
    toc_user_prompt=toc_user_prompt,       # 사용자 프롬프트
)
```

---

### 8.2 문서 로딩 메서드들

#### `load_documents_with_docling()` — Docling 기본 로딩

```python
def load_documents_with_docling(self, file_path: str, **kwargs) -> DoclingDocument:
    try:
        conv_result = self.converter.convert(file_path, raises_on_error=True)
    except Exception as e:
        # 1순위 실패 → 2순위 컨버터로 재시도
        conv_result = self.second_converter.convert(file_path, raises_on_error=True)
    return conv_result.document
```

**흐름**:

```
PDF 파일
    │
    ├──► converter (PyPdfium 백엔드) ──► 성공 → DoclingDocument 반환
    │
    └──► [실패] second_converter (PyPdfium 백엔드) ──► DoclingDocument 반환
```

#### `load_documents_with_docling_ocr()` — OCR 기반 로딩

```python
def load_documents_with_docling_ocr(self, file_path: str, **kwargs) -> DoclingDocument:
    try:
        conv_result = self.ocr_converter.convert(file_path, raises_on_error=True)
    except Exception as e:
        conv_result = self.ocr_second_converter.convert(file_path, raises_on_error=True)
    return conv_result.document
```

> GLYPH가 감지되었거나 문서 품질이 낮을 때 호출됩니다. `force_full_page_ocr=True`로 전체 페이지를 OCR합니다.

#### `load_documents_langchain()` — LangChain 폴백 로딩

```python
def load_documents_langchain(self, file_path: str, **kwargs):
    loader = self.get_loader_langchain(file_path)
    documents = loader.load()
    return documents
```

> `.ppt` 파일 전용 경로입니다. 
> Genos Doc Prser 는 `.doc`, `.xls`, `.ppt`(레거시 MS office files)를 직접 지원하지 않으므로 LangChain의 `UnstructuredPowerPointLoader`를 사용합니다.

---

### 8.3 문서 분할 메서드들

#### `split_documents()` — Genos Doc Parser 경로 분할

```python
def split_documents(self, documents: DoclingDocument, **kwargs) -> List[DocChunk]:
    chunker = GenosBucketChunker(
        max_tokens=2000,        # 청크당 최대 2000 토큰
        merge_peers=True
    )
    chunks = list(chunker.chunk(dl_doc=documents, **kwargs))
    # 페이지별 청크 수 카운팅
    for chunk in chunks:
        self.page_chunk_counts[chunk.meta.doc_items[0].prov[0].page_no] += 1
    return chunks
```

#### `split_documents_langchain()` — LangChain 경로 분할

```python
def split_documents_langchain(self, documents, **kwargs):
    text_splitter = RecursiveCharacterTextSplitter(**splitter_params)
    chunks = text_splitter.split_documents(documents)
```

LangChain 경로에서는 페이지 번호 보정 로직이 포함되어 있습니다:

```python
    for chunk in chunks:
        page = chunk.metadata.get('page', 1)

        if file_ext in ['.jpg', '.jpeg', '.png']:
            # 이미지: 이미 1-based → 그대로
            if isinstance(page, int) and page <= 0:
                page = 1
        else:
            # 기타: 0-based → 1-based로 변환
            if isinstance(page, int) and page >= 0:
                page += 1

        chunk.metadata['page'] = page
```

---

### 8.4 GLYPH 감지 및 선택적 OCR

이 기능은 `convert_processor`만의 고유 기능으로, **문서 품질을 자동으로 판단하여 필요한 부분만 OCR**을 수행합니다.

#### `check_glyph_text()` — 텍스트 레벨 GLYPH 감지

```python
def check_glyph_text(self, text: str, threshold: int = 1) -> bool:
    matches = re.findall(r'GLYPH\w*', text)
    if len(matches) >= threshold:
        return True
    return False
```

> **GLYPH란?**: PDF에서 폰트 인코딩이 깨져 텍스트를 정상적으로 추출하지 못할 때, 추출 결과에 "GLYPH" 문자열이 포함됩니다. 이는 "이 영역의 텍스트를 읽을 수 없다"는 신호입니다.

#### `check_glyphs()` — 문서 레벨 GLYPH 감지

```python
def check_glyphs(self, document: DoclingDocument) -> bool:
    for item, level in document.iterate_items():
        if isinstance(item, TextItem):
            matches = re.findall(r'GLYPH\w*', item.text)
            if len(matches) > 10:    # 한 아이템에 10개 이상이면 "깨진 문서"
                return True
    return False
```

#### `ocr_all_table_cells()` — 테이블 셀 단위 선택적 OCR

```python
def ocr_all_table_cells(self, document: DoclingDocument, pdf_path) -> DoclingDocument:
```

**목적**: 전체 페이지가 아닌, **GLYPH가 감지된 테이블의 개별 셀만** OCR을 수행합니다. 이는 "전체 OCR" 대비 처리 시간과 정확도 모두에서 유리합니다.

**처리 흐름**:

```
문서의 모든 테이블 순회
    │
    ▼
각 테이블의 셀 텍스트에 GLYPH가 있는지 검사
    │
    ├── GLYPH 없음 → 건너뜀 (OCR 불필요)
    │
    └── GLYPH 발견! → 해당 테이블의 모든 셀을 OCR
         │
         ▼
    각 셀의 바운딩 박스 계산
         │
         ▼
    셀 영역만 PDF에서 이미지로 추출
    (zoom_factor 자동 계산: 최소 높이 20px 보장)
         │
         ▼
    PaddleOCR API에 이미지 전송
         │
         ▼
    OCR 결과로 셀 텍스트 교체
         │
         ▼
    cell.text = OCR 결과 텍스트
```

**줌 팩터(zoom_factor) 계산 로직**:

```python
# 셀 bbox의 높이 (PDF 좌표 단위)
bbox_height = cell_bbox.height

# 목표: OCR 정확도를 위해 최소 20픽셀 높이 보장
target_height = 20

# 줌 팩터 계산
zoom_factor = target_height / bbox_height if bbox_height > 0 else 1.0
zoom_factor = min(zoom_factor, 4.0)  # 최대 4배 확대
zoom_factor = max(zoom_factor, 1)    # 최소 1배 (축소 안 함)
```

> **왜 셀 단위 OCR인가?**: 
> 1. 전체 페이지 OCR은 시간이 오래 걸리고, 이미 정상인 텍스트까지 재처리합니다.
> 2. 셀 단위로 하면 GLYPH가 있는 테이블만 대상이므로 효율적입니다.
> 3. 셀 이미지를 적절히 확대하여 OCR 정확도를 높입니다.

**OCR API 호출**:

```python
def post_ocr_bytes(img_bytes: bytes, timeout=60) -> dict:
    payload = {
        "file": base64.b64encode(img_bytes).decode("ascii"),
        "fileType": 1,
        "visualize": False
    }
    r = requests.post(self.ocr_endpoint, json=payload, headers=HEADERS, timeout=timeout)
    return r.json()
```

---

### 8.5 문서 Enrichment (LLM 기반 메타데이터 보강)

```python
def enrichment(self, document: DoclingDocument, **kwargs) -> DoclingDocument:
    document = enrich_document(document, self.enrichment_options, **kwargs)
    return document
```

**목적**: LLM을 활용하여 문서에 추가 정보를 덧붙입니다.

**Enrichment가 수행하는 작업**:

```
DoclingDocument (원본)
    │
    ▼
enrich_document()
    │
    ├── TOC (목차) 자동 생성
    │   │
    │   ├── 문서 전문을 LLM에 전달
    │   ├── LLM이 계층적 목차 생성
    │   │   "1. 제1장 총칙"
    │   │   "1.1. 제1절 목적"
    │   │   "1.1.1. 제1조"
    │   └── TOC 정보를 문서에 부착
    │
    └── 메타데이터 추출
        ├── 작성일 (created_date)
        ├── 작성자 (authors)
        └── 키워드, 요약 등
    │
    ▼
DoclingDocument (보강됨)
```

> **`check_document()` 함수**: enrichment 전에 문서의 품질을 검사합니다. 텍스트가 너무 적거나 GLYPH가 많으면 OCR을 먼저 수행하도록 트리거합니다.

#### 메타데이터 파싱 헬퍼들

**`parse_created_date()`** — 작성일 파싱:

```python
def parse_created_date(self, date_text: str) -> Optional[int]:
    # "2024-01-15" → 20240115
    # "2024-01"    → 20240101
    # "2024"       → 20240101
    # 실패 시      → 0
```

**`parse_authors()`** — 작성자 파싱:

```python
def parse_authors(self, authors_data) -> list[str]:
    # [{"이름": "홍길동", "직책": "팀장"}] → ["홍길동"]
    # "홍길동, 김철수"                    → ["홍길동", "김철수"]
    # "홍길동/김철수"                      → ["홍길동", "김철수"]
    # 구분자: , ; / \n · •
```

---

### 8.6 벡터 조립 메서드들

#### `compose_vectors()` — Genos Doc Parser 결과물과 메타데이터 포함 적재될 내용 조립

```python
async def compose_vectors(self, document, chunks, file_path, request, **kwargs):
```

`attachment_processor`와 기본 구조는 동일하지만, 3가지 중요한 차이점이 있습니다:

**차이점 1: 확장 메타데이터 추출**

```python
# 작성일 추출 (Enrichment에서 채운 key_value_items에서)
if document.key_value_items and ...:
    date_text = document.key_value_items[0].graph.cells[1].text
    created_date = self.parse_created_date(date_text)

# 작성자 추출 (kwargs에서)
if "authors" in kwargs:
    authors = json.dumps(self.parse_authors(kwargs["authors"]))

# 문서 제목 추출 (TITLE 라벨의 첫 번째 아이템에서)
for item, _ in document.iterate_items():
    if item.label == DocItemLabel.TITLE:
        title = item.text.strip()
        break
```

**차이점 2: `HEADER:` 접두어**

```python
# header 앞에 헤더 마커 추가
headers_text = "HEADER: " + ", ".join(chunk.meta.headings) + '\n' if chunk.meta.headings else ''
content = headers_text + chunk.text
```

**결과 예시**:

```
HEADER: 제1장 총칙, 제1절 목적
제1조(목적) 이 법은 국민의 기본적 권리를 보장하고...
```

> **`HEADER:` 접두어의 의미**: 벡터 검색 시 LLM이 이 텍스트의 문서 내 위치(Context)를 즉시 파악할 수 있게 합니다. "이 내용은 '제1장 총칙 > 제1절 목적' 아래에 있다"는 것을 명시적으로 알려줍니다.

**차이점 3: 글로벌 메타데이터 확장**

```python
global_metadata = dict(
    n_chunk_of_doc=len(chunks),
    n_page=document.num_pages(),
    reg_date=datetime.now().isoformat(timespec='seconds') + 'Z',
    created_date=created_date,     # ◄── attachment에는 없음
    authors=authors,                # ◄── attachment에는 없음
    title=title                     # ◄── attachment에는 없음
)
```

#### `compose_vectors_langchain()` — LangChain 경로 벡터 조립

```python
def compose_vectors_langchain(self, chunks, file_path, **kwargs):
```

PPT 파일 전용 벡터 조립 메서드입니다. Docling 경로와 달리 **PyMuPDF를 사용하여 bbox 정보를 생성**합니다:

```python
if doc and total_pages > 0:
    fitz_page = doc.load_page(page_index)
    # 텍스트 검색으로 bbox 위치 추출
    for rect in fitz_page.search_for(text):
        bbox_data = {
            'page': page,
            'type': 'text',
            'bbox': {
                'l': rect[0] / fitz_page.rect.width,     # 정규화
                't': rect[1] / fitz_page.rect.height,
                'r': rect[2] / fitz_page.rect.width,
                'b': rect[3] / fitz_page.rect.height,
            }
        }
```

> `fitz_page.search_for(text)`는 PDF 페이지에서 텍스트를 검색하고 그 위치의 사각형(rect)을 반환합니다. 이를 통해 LangChain으로 추출한 텍스트의 원본 PDF 상 위치를 역추적합니다.

#### `_extract_page_images()` — 페이지 이미지 추출

```python
async def _extract_page_images(self, pdf_path: str, request: Request):
```

PDF의 각 페이지에서 **임베디드 이미지**를 추출합니다:

```python
for page_index in range(len(doc)):
    page = doc.load_page(page_index)
    for img_idx, img in enumerate(page.get_images(full=True)):
        xref = img[0]
        pix = fitz.Pixmap(doc, xref)

        # CMYK/RGBA/Grayscale → RGB 변환
        if pix.n >= 5:      pix = fitz.Pixmap(fitz.csRGB, pix)  # CMYK
        elif pix.n == 4:    pix = fitz.Pixmap(fitz.csRGB, pix)  # RGBA
        elif pix.alpha:     pix = fitz.Pixmap(fitz.csRGB, pix)
        elif pix.n < 3:     pix = fitz.Pixmap(fitz.csRGB, pix)  # Grayscale

        img_path = os.path.join("/tmp", f"{uuid.uuid4()}.png")
        pix.save(img_path)
```

---

### 8.7 `__call__()` — 실행 진입점

```python
async def __call__(self, request: Request, file_path: str, **kwargs: dict):
```

**확장자별 분기 로직**:

```python
ext = Path(file_path).suffix.lower()

if ext in ['.ppt']:
    # ═══════════════════════════════════════════
    # PPT 경로: LangChain + 이미지 추출
    # ═══════════════════════════════════════════
    documents = self.load_documents_langchain(file_path)
    chunks = self.split_documents_langchain(documents)

    # PDF에서 이미지 추출 시도
    pdf_path = _get_pdf_path(file_path)
    page_image_meta = await self._extract_page_images(pdf_path, request)

    vectors = self.compose_vectors_langchain(chunks, file_path)

    # 이미지 메타데이터를 벡터에 부착
    for v in vectors:
        if v.i_page in page_image_meta:
            v.media_files = json.dumps(page_image_meta[v.i_page])

    return vectors

else:
    # ═══════════════════════════════════════════
    # Docling 경로: PDF, DOCX, PPTX 등
    # ═══════════════════════════════════════════
    document = self.load_documents(file_path)

    # [PDF 전용] 품질 검사 및 OCR
    if ext in ['.pdf']:
        if not check_document(document, ...) or self.check_glyphs(document):
            document = self.load_documents_with_docling_ocr(file_path)
        document = self.ocr_all_table_cells(document, file_path)

    # [DOCX/PPTX] PDF 변환
    if ext in ['.docx', '.pptx']:
        convert_to_pdf(file_path)

    # 이미지 참조 설정
    document = document._with_pictures_refs(...)

    # LLM 기반 Enrichment
    document = self.enrichment(document)

    # GenosBucketChunker로 청킹
    chunks = self.split_documents(document)

    # 벡터 조립
    vectors = await self.compose_vectors(document, chunks, file_path, request)

    return vectors
```

**Docling 경로의 상세 파이프라인** (PDF 파일 처리 시):

```
① load_documents() ──► Docling으로 PDF 파싱
        │
        ▼
② 품질 검사
        │
        ├── check_document(): 텍스트 품질 OK? ──► 진행
        │                              │
        │                              NO → ③ OCR 재변환
        │                                   load_documents_with_docling_ocr()
        │
        ├── check_glyphs(): GLYPH 없음? ──► 진행
        │                          │
        │                          GLYPH 발견! → ③ OCR 재변환
        │
        ▼
④ ocr_all_table_cells()
        │  GLYPH가 있는 테이블의 개별 셀만 OCR
        │
        ▼
⑤ _with_pictures_refs()
        │  이미지 파일 경로 참조 설정
        │
        ▼
⑥ enrichment()
        │  LLM으로 TOC 생성 + 메타데이터 추출
        │
        ▼
⑦ split_documents()
        │  GenosBucketChunker(max_tokens=2000)
        │
        ▼
⑧ compose_vectors()
        │  HEADER: 접두어 + 확장 메타데이터
        │
        ▼
    List[GenOSVectorMeta]
```

---

## 9. 예외 클래스 및 유틸리티

### `GenosServiceException`

```python
class GenosServiceException(Exception):
    def __init__(self, error_code: str, error_msg: Optional[str] = None, ...):
```

`attachment_processor`와 동일합니다. GenOS 플랫폼과의 의존성을 제거하기 위한 독립적 예외 클래스입니다.

### `assert_cancelled()`

```python
async def assert_cancelled(request: Request):
    if await request.is_disconnected():
        raise GenosServiceException("1", f"Cancelled")
```

> `attachment_processor`에서는 주석 처리되어 있지만, `convert_processor`에서는 함수로 정의되어 있습니다 (코드 내에서는 호출 부분이 주석 처리됨). 클라이언트가 연결을 끊었을 때 처리를 중단하는 데 사용됩니다.

---

## 10. Enrichment 프롬프트

`convert_processor`의 하단에는 LLM 기반 TOC 생성을 위한 **프롬프트 템플릿**이 정의되어 있습니다.

### 시스템 프롬프트

```python
toc_system_prompt = """You are an expert at generating table of contents (목차) 
from Korean documents. You specialize in regulatory documents, terms of service, 
contracts, and mixed-format documents..."""
```

> LLM에게 한국어 문서의 목차 생성 전문가 역할을 부여합니다.

### 사용자 프롬프트

```python
toc_user_prompt = """
Here is the Korean document you need to analyze:

<document>
{{raw_text}}
</document>

Your task is to extract and organize all structural elements...

## Analysis Process
1. **Document Title Extraction**: ...
2. **Structural Marker Identification**: ...
3. **Systematic Section Extraction**: ...
4. **Hierarchy Building**: ...
5. **Structure Verification**: ...

## Output Requirements
<toc>
TITLE:<document title>
1. <first main section title>
1.1. <first subsection title>
...
</toc>
"""
```

**프롬프트의 구조**:

```
┌─────────────────────────────────────────┐
│ 입력: {{raw_text}} (문서 전문)             │
│                                         │
│ 분석 과정:                                │
│   1. 문서 제목 추출                        │
│   2. 구조적 마커 식별                       │
│      (제x장, 제x절, 제x조, 부칙, 별지 등)     │
│   3. 체계적 섹션 추출                       │
│   4. 계층 구조 구성                        │
│   5. 구조 검증                            │
│                                         │
│ 출력:                                    │
│   <toc>                                 │
│   TITLE:문서 제목                         │
│   1. 제1장 총칙                           │
│   1.1. 제1절 목적                         │
│   1.1.1. 제1조                           │
│   ...                                   │
│   </toc>                                │
└─────────────────────────────────────────┘
```

> **`{{raw_text}}` 플레이스홀더**: 실제 실행 시 문서의 전체 텍스트로 대체됩니다.

---

## 11. 실행 흐름 요약

### 사용 방법 (Genos facade에 통합 시)

```python
# 1. DocumentProcessor 인스턴스 생성
processor = DocumentProcessor()
# → 4개의 Docling 컨버터, OCR 엔진, Enrichment 옵션이 초기화됨

# 2. 파일 처리 호출 (비동기)
vectors = await processor(
    request=request,
    file_path="/path/to/document.pdf",
    log_level=4,               # 선택: 로그 레벨 (기본값 4=INFO)
    authors=[{"이름": "홍길동"}],  # 선택: 작성자 정보
)

# 3. 결과: List[GenOSVectorMeta]
for v in vectors:
    print(v.text)              # "HEADER: 제1장 총칙, 제1절 목적\n제1조(목적)..."
    print(v.title)             # "개인정보 보호법"
    print(v.created_date)      # 20240115
    print(v.authors)           # '["홍길동"]'
    print(v.chunk_bboxes)      # '[{"page":1,"bbox":{...},"type":"text"}]'
```

### 전체 처리 흐름 (PDF 파일 예시)

```
processor(request, "/data/privacy_law.pdf")
    │
    ▼
__call__()
    │  ext = '.pdf' → else 분기 (Docling 경로)
    │
    ├── load_documents_with_docling()
    │   └── Docling PDF 파이프라인
    │       ├── 레이아웃 분석 (Layout Detection)
    │       ├── 표 구조 분석 (TableFormer ACCURATE)
    │       └── → DoclingDocument
    │
    ├── [품질 검사]
    │   ├── check_document() → 텍스트 충분? 
    │   ├── check_glyphs() → GLYPH 없음?
    │   └── ocr_all_table_cells() → GLYPH 있는 테이블 셀만 OCR
    │
    ├── _with_pictures_refs() → 이미지 참조 설정
    │
    ├── enrichment()
    │   ├── LLM에 문서 전문 전달
    │   ├── TOC 자동 생성
    │   │   "1. 제1장 총칙"
    │   │   "1.1. 제1조(목적)"
    │   │   "1.2. 제2조(정의)"
    │   │   ...
    │   └── 메타데이터 추출 (작성일, 제목 등)
    │
    ├── split_documents()
    │   └── GenosBucketChunker(max_tokens=2000)
    │       ├── preprocess() → 아이템 + 헤더 수집
    │       └── _split_document_by_tokens()
    │           ├── 1단계: 섹션 헤더 분할
    │           ├── 2단계: heading 텍스트 생성
    │           ├── 2.5단계: 초과 분할
    │           ├── 3단계: 단독 타이틀 병합
    │           └── 4단계: 토큰 병합
    │
    └── compose_vectors()
        │
        └── [
              GenOSVectorMeta(
                text="HEADER: 제1장 총칙, 제1절 목적\n제1조(목적) 이 법은...",
                title="개인정보 보호법",
                created_date=20240115,
                authors='["홍길동"]',
                i_page=1, e_page=1,
                chunk_bboxes='[{"page":1,"bbox":{"l":0.1,"t":0.2,...}}]',
                ...
              ),
              GenOSVectorMeta(
                text="HEADER: 제1장 총칙, 제2절 범위\n제3조(적용 범위)...",
                ...
              ),
              ...
            ]
```

---

## 12. 지원 파일 포맷 총정리

| 카테고리 | 확장자 | 처리 경로 | 핵심 도구 | OCR | Enrichment |
|----------|--------|-----------|-----------|-----|------------|
| **PDF** | `.pdf` | Docling 경로 | Docling + TableFormer + PaddleOCR | ✅ 선택적 | ✅ |
| **Word** | `.docx` | Docling 경로 + PDF 변환 | Docling + LibreOffice | ❌ | ✅ |
| **프레젠테이션** | `.pptx` | Docling 경로 + PDF 변환 | Docling + LibreOffice | ❌ | ✅ |
| **프레젠테이션 (레거시)** | `.ppt` | LangChain 경로 | Unstructured + LibreOffice | ❌ | ❌ |
| **기타** | 그 외 | Docling 경로 시도 | Docling | 조건부 | ✅ |

> **attachment_processor와의 포맷 지원 비교**:
> - `convert_processor`는 **오디오(MP3, WAV, M4A)와 정형 데이터(CSV, XLSX)를 지원하지 않습니다**.
> - 대신 PDF에 대해 **레이아웃 분석, 표 구조 분석, 선택적 OCR, LLM Enrichment** 등 훨씬 깊은 처리를 수행합니다.
> - HWP, HWPX도 이 프로세서에서는 직접 처리하지 않습니다 (필요 시 PDF로 사전 변환 필요).

---

> **참고**: 이 전처리기는 **호환성과 품질을 최우선**으로 설계되어, AI 기반 레이아웃 분석과 LLM Enrichment가 포함되므로 처리 시간이 `attachment_processor`보다 길 수 있습니다. 실시간 채팅 첨부에는 `attachment_processor`를, 문서 DB 적재에는 `intelligent_processor`를 권장하며, 이 `convert_processor`는 그 중간 지점에서 **첨부용으로도 쓸 수 있는 고품질 대안**으로 활용됩니다.