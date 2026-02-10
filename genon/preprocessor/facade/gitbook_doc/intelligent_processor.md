# `intelligent_processor.py` 코드 상세 설명서

---

## 📌 목차

1. [개요](#1-개요)
2. [전체 아키텍처](#2-전체-아키텍처)
3. [세 전처리기의 핵심 비교](#3-세-전처리기의-핵심-비교)
4. [임포트 및 초기 설정](#4-임포트-및-초기-설정)
5. [청커: `GenosBucketChunker`](#5-핵심-청커-genosbucketchunker)
   - 5.1 [ Bucket Chunker와 max_tokens=0의 의미](#51-Bucket-Chunker와--max_tokens0의-의미)
   - 5.2 [`preprocess()` — 문서 아이템 수집](#52-preprocess--문서-아이템-수집)
   - 5.3 [`_split_document_by_tokens()` — 4단계 분할·병합 파이프라인](#53-_split_document_by_tokens--4단계-분할병합-파이프라인)
   - 5.4 [헬퍼 메서드들](#54-헬퍼-메서드들)
   - 5.5 [`chunk()` — 최종 진입점](#55-chunk--최종-진입점)
6. [데이터 모델](#6-데이터-모델)
   - 6.1 [`GenOSVectorMeta`](#61-genosvectormeta)
   - 6.2 [`GenOSVectorMetaBuilder`](#62-genosvectormetabuilder)
7. [메인 프로세서: `DocumentProcessor`](#7-메인-프로세서-documentprocessor)
   - 7.1 [`__init__()` — 컨버터 및 파이프라인 초기화](#71-__init__--컨버터-및-파이프라인-초기화)
   - 7.2 [문서 로딩 메서드들](#72-문서-로딩-메서드들)
   - 7.3 [문서 분할 — `split_documents()`](#73-문서-분할--split_documents)
   - 7.4 [GLYPH 감지 및 선택적 OCR](#74-glyph-감지-및-선택적-ocr)
   - 7.5 [문서 Enrichment (LLM 기반 메타데이터 보강)](#75-문서-enrichment-llm-기반-메타데이터-보강)
   - 7.6 [부록(Appendix) 자동 연결 기능](#76-부록appendix-자동-연결-기능)
   - 7.7 [벡터 조립 — `compose_vectors()`](#77-벡터-조립--compose_vectors)
   - 7.8 [빈 문서 처리 (Empty Document Fallback)](#78-빈-문서-처리-empty-document-fallback)
   - 7.9 [`__call__()` — 실행 진입점](#79-__call__--실행-진입점)
8. [예외 클래스 및 유틸리티](#8-예외-클래스-및-유틸리티)
9. [Enrichment 프롬프트](#9-enrichment-프롬프트)
10. [실행 흐름 요약](#10-실행-흐름-요약)

---

## 1. 개요

`intelligent_processor.py`는 **적재용 지능형 전처리기(Intelligent Processor)** 입니다. RAG(검색 증강 생성) 시스템의 **지식 베이스 구축**을 위해 설계된 고성능 전처리기로, 단순 텍스트 추출을 넘어 **딥러닝 기반의 Layout 분석**을 통해 문서의 논리적 구조를 정확하게 파악합니다.

### 핵심 설계 철학

```
"품질 중심: AI 기반 레이아웃 분석 및 고품질 데이터 적재"
```

| 특징 | 설명                                          |
|------|---------------------------------------------|
| **AI 레이아웃 분석** | 딥러닝 모델이 제목, 본문, 표, 그림, 캡션 등 11종 요소를 자동 식별   |
| **TableFormer** | 병합 셀, 다중 헤더 등 복잡한 표를 마크다운으로 완벽 복원           |
| **Smart OCR** | GLYPH(인코딩 깨짐)가 감지된 영역만 선별적 OCR 수행           |
| **LLM Enrichment** | LLM으로 목차(TOC) 자동 생성 및 문서 메타데이터 추출           |
| **부록 자동 연결** | 본문 내 '별지/별표' 참조를 실제 부록 파일과 자동 매칭 (Optional) |
| **섹션 기반 순수 분할** | 토큰 제한 없이 문서 구조(섹션 헤더)를 100% 존중하는 청킹         |

> **위치**: `preprocessor/facade/intelligent_processor.py`

### 왜 지능형 처리가 필요한가요?

복잡한 다단 구성, 표, 차트가 포함된 문서는 단순 추출 시 문맥이 파괴될 수 있습니다. 예를 들어:
- 2단 레이아웃의 학술 논문에서 왼쪽 단의 마지막 줄과 오른쪽 단의 첫 줄이 하나의 문장으로 이어짐
- 병합된 셀이 있는 복잡한 표에서 행/열 관계가 소실됨
- 캡션이 그림/표와 분리되어 별도 청크에 배치됨

지능형 전처리기는 이러한 문제를 **레이아웃 분석 + 구조적 청킹**으로 해결합니다.

---

## 2. 전체 아키텍처

```
PDF 파일 입력
    │
    ▼
┌─────────────────────┐
│  DocumentProcessor  │  ◄── 메인 엔트리포인트 (__call__)
│  (PDF 전용 파이프라인)  │
└──────┬──────────────┘
       │
       │  ═══════════════════════════════════════
       │  단일 경로: PDF → Docling (분기 없음)
       │  ═══════════════════════════════════════
       │
       ▼
① load_documents_with_docling()
       │  Docling PDF 파이프라인
       │  ├─ Layout Detection (딥러닝)
       │  ├─ TableFormer (ACCURATE)
       │  └─ → DoclingDocument
       │
       ▼
② 품질 검사 (GLYPH / 텍스트 품질)
       │
       ├── 정상 → 진행
       └── 이상 감지 → load_documents_with_docling_ocr()
                       (전체 페이지 OCR 재처리)
       │
       ▼
③ ocr_all_table_cells()
       │  GLYPH가 있는 테이블 셀만 선별적 OCR
       │
       ▼
④ _with_pictures_refs()
       │  이미지 파일 경로 참조 설정
       │
       ▼
⑤ enrichment()
       │  LLM 기반 보강
       │  ├─ TOC(목차) 자동 생성
       │  └─ 메타데이터 추출 (작성일 등)
       │
       ▼
⑥ 빈 문서 검사 (텍스트 아이템 존재 여부)
       │
       ├── 텍스트 있음 → 정상 처리
       └── 텍스트 없음 → 더미 텍스트 삽입 후 처리
       │
       ▼
⑦ GenosBucketChunker (max_tokens=0)
       │  ├─ preprocess() → 아이템 + 헤더 수집
       │  └─ _split_document_by_tokens()
       │       ├─ 1단계: 섹션 헤더 분할
       │       ├─ 2단계: heading 텍스트 생성
       │       ├─ 2.5단계: (스킵 — max_tokens=0)
       │       ├─ 3단계: 단독 타이틀 병합
       │       └─ 4단계: (스킵 — max_tokens=0)
       │
       ▼
⑧ compose_vectors()
       │  ├─ HEADER: 접두어
       │  ├─ 부록(appendix) 자동 매칭
       │  ├─ created_date, title
       │  └─ chunk_bboxes, media_files
       │
       ▼
┌──────────────────────────────────────┐
│  List[GenOSVectorMeta]               │
│  (최종 출력: 고품질 청크별 메타데이터)        │
│  + title, created_date               │
│  + appendix (별지/별표 자동 연결)         │
│  + HEADER: 계층적 위치 정보              │
│  + chunk_bboxes (정밀 위치 좌표)        │
│  + media_files (이미지 참조)            │
└──────────────────────────────────────┘
```

**핵심 특징 — 단일 경로 설계**: `attachment_processor`는 확장자별로 다양한 로더/프로세서로 분기하고, `convert_processor`는 PPT와 나머지로 분기합니다. 반면 `intelligent_processor`는 **PDF 파일만을 대상**으로 **단일 Docling 파이프라인**으로 처리합니다. 이는 "한 가지 포맷을 최고 품질로 처리"하는 설계 철학의 반영입니다.

---

## 3. 세 전처리기의 핵심 비교

| 비교 항목 | attachment | convert | **intelligent** |
|-----------|-----------|---------|-----------------|
| **설계 목표** | 속도 중심 | 호환성 중심 | **품질 중심** |
| **대상 포맷** | 다양 (PDF, HWP, 오디오, CSV 등) | 다양 (PPT, DOCX→PDF 변환) | **PDF 전용** |
| **Docling 파이프라인** | SimplePipeline (경량) | 전체 PDF 파이프라인 | **전체 PDF 파이프라인** |
| **청커** | HybridChunker (∞ 토큰) | GenosBucketChunker (2000 토큰) | **GenosBucketChunker (0 토큰=무제한)** |
| **청킹 전략** | 레이아웃 병합 | 섹션+토큰 병합 | **순수 섹션 분할 (병합 없음)** |
| **OCR** | 없음 | PaddleOCR (선택적) | **PaddleOCR (선택적)** |
| **Enrichment** | 없음 | LLM TOC + 메타데이터 | **LLM TOC + 메타데이터** |
| **부록 연결** | 없음 | 없음 | **✅ 별지/별표 자동 매칭** |
| **빈 문서 처리** | 예외 발생 | 예외 발생 | **더미 텍스트 삽입** |
| **이미지 옵션** | 고정 | 고정 | **동적 (save_images, include_wmf)** |
| **PDF 변환** | 일부 | convert_to_pdf() | **없음 (PDF 직접 입력)** |
| **`authors` 필드** | ❌ | ✅ | ❌ |
| **`appendix` 필드** | ❌ | ❌ | **✅** |
| **LangChain 폴백** | ✅ 다양 | ✅ PPT용 | **❌ (Docling 전용)** |

### `max_tokens`에 따른 청킹 동작 차이

이 차이가 세 전처리기의 출력 품질을 결정짓는 가장 핵심적인 요소입니다:

```
attachment (max_tokens=∞):
──────────────────────────
  모든 아이템이 하나의 청크로 병합됨
  → RecursiveCharacterTextSplitter로 후처리 분할

convert (max_tokens=2000):
──────────────────────────
  섹션 헤더 기반 분할 후, 2000 토큰 이내로 병합
  → 작은 섹션들이 합쳐져 적정 크기 유지

intelligent (max_tokens=0):
──────────────────────────
  ★ 섹션 헤더 기반 분할만 수행, 토큰 기반 병합 없음
  → 각 섹션이 독립적 청크로 유지
  → 문서의 논리적 구조를 100% 보존
  → RAG 검색 시 정확한 컨텍스트 제공
```

---

## 4. 임포트 및 초기 설정

### 주요 외부 라이브러리 그룹

| 그룹 | 라이브러리 | 용도 |
|------|-----------|------|
| **Docling 코어** | `docling.document_converter`, `docling.pipeline` | PDF 파싱, 레이아웃 분석, 표 구조 인식 |
| **Docling 데이터 모델** | `docling_core.types.doc` | DoclingDocument, DocItem, TableItem, ProvenanceItem 등 |
| **Docling 청킹** | `docling_core.transforms.chunker` | BaseChunker, DocChunk 등 |
| **OCR** | `PaddleOcrOptions` | PaddleOCR 엔진 설정 |
| **토크나이저** | `transformers.AutoTokenizer`, `semchunk` | 토큰 수 기반 분할 (2.5단계에서 사용) |
| **Enrichment** | `docling.utils.document_enrichment` | LLM 기반 목차/메타데이터 보강 |

### `convert_processor`와의 임포트 차이

| 항목 | convert_processor | intelligent_processor |
|------|-------------------|----------------------|
| `fitz` (PyMuPDF) | ✅ (bbox 추출, 이미지 추출) | ❌ (OCR 메서드 내에서만 지역 import) |
| LangChain 로더들 | ✅ (PPT 폴백용) | ❌ (불필요) |
| `RecursiveCharacterTextSplitter` | ✅ (PPT 폴백용) | ❌ (불필요) |
| `convert_to_pdf()` | ✅ | ❌ (PDF만 입력) |
| `ProvenanceItem`, `BoundingBox` | ❌ | ✅ (빈 문서 더미 삽입용) |

> `intelligent_processor`는 PDF 전용이므로 다른 포맷 지원을 위한 라이브러리가 대폭 줄어들었습니다.

---

## 5. 청커: `GenosBucketChunker`

```python
class GenosBucketChunker(BaseChunker):
    """토큰 제한을 고려하여 섹션별 청크를 분할하고 병합하는 청커 (v2)"""
```

### 5.1 Bucket Chunker와 `max_tokens`의 의미

`GenosBucketChunker`의 이름에서 **"Bucket"** 은 "바구니"를 의미합니다. 이 청커의 핵심 개념은 매우 직관적입니다:

```
Bucket = 바구니 (하나의 청크가 될 단위)
max_tokens = 바구니의 크기 (허용 용량)
섹션(item) = 바구니에 담을 물건
```

**동작 원리**: 섹션 헤더 기준으로 분할된 아이템(섹션)들을 순서대로 바구니에 담되, 바구니에 담긴 아이템들의 **콘텐츠 길이(토큰 수)의 합**이 `max_tokens`를 넘으면 새 바구니를 시작합니다. 여기서 측정 기준은 섹션 헤더 같은 메타데이터가 아니라, **실제 콘텐츠에 해당하는 아이템의 텍스트 길이의 합**입니다.

**중요 규칙**: 아무리 `max_tokens`가 작더라도, **최소 1개의 아이템은 반드시 바구니에 담깁니다**. 즉, 단일 아이템이 `max_tokens`를 초과하더라도 그 자체로 하나의 청크가 됩니다. 빈 바구니는 생성되지 않습니다.

```
바구니 규칙 정리:
─────────────
1. 섹션 헤더를 만나면 새 아이템(섹션)이 시작됨
2. 바구니에 아이템을 넣되, 넣었을 때 총 토큰이 max_tokens를 넘으면 → 새 바구니
3. 단, 바구니가 비어있으면 무조건 1개는 넣음 (at least 1)
4. 상위 레벨 헤더를 만나면 토큰과 무관하게 새 바구니 시작
```

이 규칙은 코드의 4단계에서 정확히 구현되어 있습니다:

```python
# 4단계: 토큰 기준 병합
for text, items, header_infos, header_short_infos in sections_with_text:

    test_tokens = self._count_tokens("\n".join(merged_texts + [text]))

    # 토큰 수 초과 시 새로운 청크 생성
    #                     ↓↓↓ "이미 바구니에 뭔가 있을 때만" 새 바구니 시작
    if test_tokens > self.max_tokens and len(merged_texts) > 0:
        b_new_chunk = True
    #   ↑ 바구니가 비어있으면(len==0) 이 조건이 False → 무조건 담김 (at least 1)

    elif 0 <= section_level < merged_level:
        b_new_chunk = True   # 상위 레벨 헤더 → 무조건 새 바구니
```

#### `max_tokens` 값에 따른 동작 변화

`max_tokens`는 사용자가 자유롭게 조절할 수 있는 설정값입니다. 이 값을 바꾸면 청킹 결과가 달라집니다. 동일한 문서를 두 가지 설정으로 처리하는 예시를 통해 차이를 확인해 보겠습니다.

**예시 문서 구조**:

```
제1장 총칙                    (섹션 헤더)
  제1조(목적) 이 법은...       → 아이템 A: 500 토큰
  제2조(정의) 이 법에서...     → 아이템 B: 300 토큰
  제3조(적용범위) 이 법은...   → 아이템 C: 200 토큰

제2장 권리                    (섹션 헤더, 상위 레벨)
  제4조(기본원칙) 개인정보는... → 아이템 D: 400 토큰
  제5조(동의) 개인정보를...    → 아이템 E: 1800 토큰 (긴 조항)
  제6조(수집제한) 처리자는...  → 아이템 F: 600 토큰
```

---

##### 경우 1: `max_tokens=2000` (convert_processor 기본값)

바구니 크기가 2000토큰입니다. 작은 섹션들이 하나의 바구니에 합쳐져 **적정 크기의 청크**가 만들어집니다.

```
바구니 1 (비어있음):
  A(500) 담기 → 총 500 < 2000 → OK, 계속
  B(300) 담기 → 총 800 < 2000 → OK, 계속
  C(200) 담기 → 총 1000 < 2000 → OK, 계속
  [제2장: 상위 레벨 헤더] → 레벨 변경! → 바구니 확정

  ✅ 청크 1: A+B+C (1000토큰) "제1조~제3조"

바구니 2 (비어있음):
  D(400) 담기 → 총 400 < 2000 → OK, 계속
  E(1800) 담기 → 총 2200 > 2000 → 초과!
    → 하지만 바구니에 이미 D가 있음 → D만으로 바구니 확정

  ✅ 청크 2: D (400토큰) "제4조"

바구니 3 (비어있음):
  E(1800) 담기 → 바구니 비어있으므로 무조건 담기 (at least 1)
  F(600) 담기 → 총 2400 > 2000 → 초과! → E만으로 확정

  ✅ 청크 3: E (1800토큰) "제5조"

바구니 4 (비어있음):
  F(600) 담기 → 끝 → 확정

  ✅ 청크 4: F (600토큰) "제6조"

최종 결과: 4개 청크
  [A+B+C] [D] [E] [F]
   1000    400  1800  600 토큰
```

> **특징**: 작은 조항(A, B, C)이 하나의 청크로 합쳐집니다. 벡터 DB의 청크 수가 줄어들어 저장 효율이 좋고, 검색 시 관련 조항들이 함께 반환됩니다. 단, "제2조만 찾고 싶을 때" 제1조, 제3조까지 함께 딸려옵니다.

**적합한 상황**:
- 일반적인 문서 검색 (관련 내용을 넓게 찾고 싶을 때)
- 벡터 DB 저장 비용을 줄이고 싶을 때
- 개별 조항보다 **섹션 단위**의 맥락이 중요할 때

---

##### 경우 2: `max_tokens=0` (intelligent_processor 기본값)

바구니 크기가 0입니다. 어떤 아이템이든 담으면 즉시 0을 초과하므로, **각 아이템이 독립된 바구니**가 됩니다. 단, "at least 1" 규칙에 의해 빈 바구니는 생성되지 않습니다.

```
바구니 1 (비어있음):
  A(500) 담기 → 바구니 비어있으므로 무조건 담기 (at least 1) → 확정

  ✅ 청크 1: A (500토큰) "제1조"

바구니 2 (비어있음):
  B(300) 담기 → at least 1 → 확정

  ✅ 청크 2: B (300토큰) "제2조"

바구니 3 (비어있음):
  C(200) 담기 → at least 1 → 확정

  ✅ 청크 3: C (200토큰) "제3조"

바구니 4 (비어있음):
  D(400) 담기 → at least 1 → 확정

  ✅ 청크 4: D (400토큰) "제4조"

바구니 5 (비어있음):
  E(1800) 담기 → at least 1 → 확정

  ✅ 청크 5: E (1800토큰) "제5조"

바구니 6 (비어있음):
  F(600) 담기 → at least 1 → 확정

  ✅ 청크 6: F (600토큰) "제6조"

최종 결과: 6개 청크
  [A] [B] [C] [D] [E] [F]
  500  300  200  400  1800  600 토큰
```

> **특징**: 모든 조항이 독립 청크로 유지됩니다. "제2조"를 검색하면 정확히 제2조만 반환됩니다. 문서의 논리적 구조가 100% 보존됩니다.

**적합한 상황**:
- **RAG 지식베이스 구축** (정확한 조항/조문 검색이 필요할 때)
- 법률, 규정, 약관 등 **조항별 독립성**이 중요한 문서
- "제5조에 대해 설명해줘"와 같은 **정밀 질의**에 대응해야 할 때

---

##### 시각적 비교

```
원본 문서:
┌─────────────────────────────────────────────┐
│ 제1장 총칙                                    │
│   제1조(목적) 이 법은... (500토큰)               │
│   제2조(정의) 이 법에서... (300토큰)              │
│   제3조(적용범위) 이 법은... (200토큰)            │
│ 제2장 권리                                    │
│   제4조(기본원칙) 개인정보는... (400토큰)          │
│   제5조(동의) 개인정보를... (1800토큰)            │
│   제6조(수집제한) 처리자는... (600토큰)            │
└─────────────────────────────────────────────┘

max_tokens=2000:                    max_tokens=0:
┌──────────────────┐                ┌───────────────┐
│ 청크1: 제1~3조     │ 1000토큰        │ 청크1: 제1조     │ 500토큰
├──────────────────┤                ├───────────────┤
│ 청크2: 제4조       │ 400토큰         │ 청크2: 제2조     │ 300토큰
├──────────────────┤                ├───────────────┤
│ 청크3: 제5조       │ 1800토큰        │ 청크3: 제3조     │ 200토큰
├──────────────────┤                ├───────────────┤
│ 청크4: 제6조       │ 600토큰         │ 청크4: 제4조     │ 400토큰
└──────────────────┘                ├───────────────┤
4개 청크, 평균 950토큰                  │ 청크5: 제5조    │ 1800토큰
                                    ├───────────────┤
                                    │ 청크6: 제6조    │ 600토큰
                                    └───────────────┘
                                    6개 청크, 평균 633토큰
```

#### 설정 변경 방법

`max_tokens` 값은 `split_documents()` 메서드에서 설정됩니다. 사용자의 요구에 따라 이 값을 변경할 수 있습니다:

```python
# intelligent_processor.py 내 split_documents() 메서드

def split_documents(self, documents: DoclingDocument, **kwargs: dict) -> List[DocChunk]:
    chunker = GenosBucketChunker(
        max_tokens=0,           # ← 이 값을 변경
        merge_peers=True
    )
    ...
```

| 설정값 | 동작 | 권장 상황 |
|--------|------|-----------|
| `max_tokens=0` | 각 섹션이 독립 청크 (기본값) | 법률/규정 문서, 정밀 검색이 필요한 RAG |
| `max_tokens=512` | 작은 섹션은 병합, 512토큰 이내 유지 | 짧은 조항이 많은 문서 |
| `max_tokens=1024` | 중간 크기 병합 | 일반적인 RAG 지식베이스 |
| `max_tokens=2000` | 큰 단위 병합 (convert_processor 기본값) | 문맥 연속성이 중요한 문서 |

> **팁**: 법률/규정 문서처럼 조항별 독립성이 중요한 경우 `0`이 적합하고, 매뉴얼/보고서처럼 내용의 연속성이 중요한 경우 `1024~2000` 범위가 적합합니다.

#### 2.5단계와의 관계

`max_tokens`는 4단계(병합)뿐 아니라 **2.5단계(초과 분할)**에도 영향을 줍니다:

```python
# 2.5단계: 너무 긴 청크는 분할
if self.max_tokens > 0:          # max_tokens=0이면 이 블록 전체가 스킵됨
    for i in range(len(sections_with_text)):
        token_count = self._count_tokens(text)
        if token_count < self.max_tokens:
            continue
        # ... 캡션 조정, 표 내 그림 조정, 균등 분할 ...
```

| `max_tokens` | 2.5단계 동작 | 4단계 동작 |
|--------------|-------------|-----------|
| `0` | **스킵** (분할 안 함) | **모든 섹션 독립** (병합 안 함) |
| `2000` | 2000토큰 초과 섹션을 균등 분할 | 2000토큰 이내로 섹션 병합 |

`max_tokens=0`일 때 2.5단계가 스킵되는 것은 의도된 동작입니다: 바구니 크기가 0이면 "큰 아이템을 쪼갤" 이유도 없기 때문입니다. 각 아이템이 아무리 크더라도 "at least 1" 규칙에 의해 그대로 하나의 바구니(청크)가 됩니다.

### 5.2 `preprocess()` — 문서 아이템 수집

```python
def preprocess(self, dl_doc: DLDocument, **kwargs: Any) -> Iterator[BaseChunk]:
```

`convert_processor`의 `preprocess()`와 **완전히 동일한 코드**입니다. 핵심 동작을 요약합니다:

**처리 흐름**:

```
DoclingDocument
    │
    ▼
iterate_items()으로 모든 아이템 순회
    │  (ContentLayer.BODY + ContentLayer.FURNITURE 포함)
    │
    │  각 아이템마다:
    │  ├── ListItem → list_items에 누적 (나중에 일괄 추가)
    │  ├── SectionHeader → heading_by_level 갱신 + all_items에 추가
    │  └── Text/Table/Picture/Code → all_items에 추가
    │
    │  + 각 시점의 heading_by_level 스냅샷을 all_header_info에 저장
    │  + 각 시점의 heading_short_by_level 스냅샷을 all_header_short_info에 저장
    │
    ▼
누락된 테이블 복구 (iterate_items에서 빠진 테이블 보정)
    │
    ▼
모든 아이템을 하나의 DocChunk로 포장하여 yield
    ├── _header_info_list 속성 부착
    └── _header_short_info_list 속성 부착
```

**3개의 병렬 리스트**:

```python
all_items[i]              → i번째 문서 아이템 (DocItem)
all_header_info[i]        → i번째 아이템의 헤더 컨텍스트 (item.text 기반, 전체 제목)
all_header_short_info[i]  → i번째 아이템의 짧은 헤더 (item.orig 기반)
```

---

### 5.3 `_split_document_by_tokens()` — 4단계 분할·병합 파이프라인

코드 자체는 `convert_processor`와 동일하지만, `max_tokens=0`에 의해 **동작이 크게 달라집니다**.

#### 1단계: 섹션 헤더 기준 분할 (동일)

```python
for i, item in enumerate(items):
    if self._is_section_header(item):
        if cur_items:
            sections.append((cur_items, cur_h_infos, cur_h_short))
        cur_items = [item]          # 새 섹션 시작
    else:
        cur_items.append(item)       # 현재 섹션에 추가
```

> 섹션 헤더를 만날 때마다 새 섹션을 시작합니다. 이 단계는 세 전처리기 모두 동일합니다.

#### 2단계: heading 텍스트 생성 (동일)

각 섹션의 텍스트 앞에 계층적 헤딩 경로를 붙입니다.

#### 2.5단계: 초과 청크 분할 — ★ 스킵됨!

```python
if self.max_tokens > 0:          # max_tokens=0이므로 이 블록 전체가 실행되지 않음
    for i in range(len(sections_with_text)):
        # ... 캡션 조정, 표 내 그림 조정, 균등 분할 ...
```

> **`convert_processor`와의 결정적 차이**: `max_tokens=0`이므로 `if self.max_tokens > 0:` 조건이 `False`가 되어, 아무리 긴 섹션이라도 **분할하지 않습니다**. 이는 RAG에서 하나의 조항이나 표가 중간에 잘리는 것을 방지합니다.

#### 3단계: 단독 타이틀 병합 (동일)

```
"제2장 권리" (8자, 단독 타이틀) + "제3조 국민은..." → 하나의 청크로 병합
```

> 이 단계는 `max_tokens` 값과 무관하게 동작합니다. 30자 이하의 단독 제목은 다음 섹션에 병합됩니다.

#### 4단계: 토큰 기준 병합 — ★ 사실상 병합 없음!

```python
# 토큰 수 초과 시 새로운 청크 생성
if test_tokens > self.max_tokens and len(merged_texts) > 0:
    b_new_chunk = True
# max_tokens=0이면: 어떤 test_tokens든 > 0 이므로 항상 True
# 결과: 모든 섹션이 개별 청크로 생성됨
```

**시각적 비교**:

```
입력 섹션들:
  [A: "제1조 목적..." (500토큰)]
  [B: "제2조 범위..." (300토큰)]
  [C: "제3조 적용..." (200토큰)]

convert_processor (max_tokens=2000):
  merged_texts=["A"] → test_tokens=500 < 2000 → 병합 계속
  merged_texts=["A","B"] → test_tokens=800 < 2000 → 병합 계속
  merged_texts=["A","B","C"] → test_tokens=1000 < 2000 → 병합 계속
  결과: [청크1: A+B+C] ← 3개 섹션이 하나로 합쳐짐

intelligent_processor (max_tokens=0):
  merged_texts=["A"] → test_tokens=500 > 0 → 새 청크!
  결과: [청크1: A] → 즉시 확정
  merged_texts=["B"] → test_tokens=300 > 0 → 새 청크!
  결과: [청크2: B] → 즉시 확정
  merged_texts=["C"] → 끝 → 확정
  결과: [청크3: C]
  
  최종: [A] [B] [C] ← 각 섹션이 독립 청크!
```

---

### 5.4 헬퍼 메서드들

`convert_processor`의 헬퍼 메서드들과 완전히 동일합니다. 주요 메서드를 간단히 정리합니다:

| 메서드 | 역할 |
|--------|------|
| `_count_tokens(text)` | 줄 단위 분할 후 300자씩 안전하게 토큰 수 계산 |
| `_generate_text_from_items_with_headers(...)` | 아이템 리스트에서 헤더 포함 텍스트 생성 |
| `_extract_table_text(table_item, dl_doc)` | 표 텍스트 추출 (마크다운→셀데이터→item.text 폴백) |
| `_extract_used_headers(header_info_list)` | 헤더 리스트에서 중복 없는 헤더 추출 |
| `_is_section_header(item)` | 아이템이 섹션 헤더인지 판별 |
| `_generate_section_text_with_heading(...)` | 섹션 텍스트 앞에 계층적 heading 경로 추가 |
| `adjust_captions(items_group)` | 캡션을 부모 아이템에 병합 |
| `adjust_pictures_in_tables(items_group)` | 표 안의 그림을 표에 병합 (IoS>0.5 기준) |
| `split_items_evenly_by_tokens(...)` | 접두합+이진탐색으로 균등 토큰 분할 |

> **참고**: `adjust_captions`와 `adjust_pictures_in_tables`는 2.5단계에서만 호출되므로, `max_tokens=0`인 intelligent_processor에서는 **실제로 실행되지 않습니다**.

---

### 5.5 `chunk()` — 최종 진입점

```python
def chunk(self, dl_doc: DoclingDocument, **kwargs: Any) -> Iterator[BaseChunk]:
    doc_chunks = list(self.preprocess(dl_doc=dl_doc, **kwargs))    # 1개의 거대 청크
    doc_chunk = doc_chunks[0]
    final_chunks = self._split_document_by_tokens(doc_chunk, dl_doc)  # 섹션 기반 분할
    return iter(final_chunks)
```

---

## 6. 데이터 모델

### 6.1 `GenOSVectorMeta`

```python
class GenOSVectorMeta(BaseModel):
    class Config:
        extra = 'allow'

    text: str = None
    n_char: int = None
    n_word: int = None
    n_line: int = None
    i_page: int = None
    e_page: int = None
    i_chunk_on_page: int = None
    n_chunk_of_page: int = None
    i_chunk_on_doc: int = None
    n_chunk_of_doc: int = None
    n_page: int = None
    reg_date: str = None
    chunk_bboxes: str = None
    media_files: str = None
    title: str = None              # 문서 제목
    created_date: int = None       # 작성일 (YYYYMMDD 정수)
    appendix: str = None           # ◄── 고유 필드: 매칭된 부록 파일명
```

**세 전처리기의 필드 비교**:

| 필드 | attachment | convert | intelligent | 설명 |
|------|-----------|---------|-------------|------|
| 기본 필드들 | ✅ | ✅ | ✅ | text, n_char, pages 등 |
| `title` | ❌ | ✅ | ✅ | 문서 제목 |
| `created_date` | ❌ | ✅ | ✅ | 작성일 (YYYYMMDD) |
| `authors` | ❌ | ✅ | ❌ | 작성자 |
| **`appendix`** | ❌ | ❌ | **✅** | 매칭된 부록 파일명 |

**`appendix` 필드 예시**:

```python
# 청크 텍스트에 "별지 제1호 서식"이 언급되고,
# appendix_list에 "별지 제1호 서식.pdf"가 있으면:
appendix = "별지 제1호 서식.pdf"

# 여러 부록이 매칭되면 쉼표로 연결:
appendix = "별지 제1호 서식.pdf, 별표 제2호.pdf"

# 매칭되는 부록이 없으면:
appendix = ""
```

---

### 6.2 `GenOSVectorMetaBuilder`

`convert_processor`와 동일한 빌더 패턴이지만, `authors` 대신 `appendix` 필드를 지원합니다:

```python
class GenOSVectorMetaBuilder:
    def __init__(self):
        # ... (기본 필드들)
        self.title: Optional[str] = None
        self.created_date: Optional[int] = None
        self.appendix: Optional[str] = None  # ◄── intelligent 전용

    def build(self) -> GenOSVectorMeta:
        return GenOSVectorMeta(
            # ... (기본 필드들)
            title=self.title,
            created_date=self.created_date,
            appendix=self.appendix or ""      # None이면 빈 문자열로 변환
        )
```

---

## 7. 메인 프로세서: `DocumentProcessor`

### 7.1 `__init__()` — 컨버터 및 파이프라인 초기화

```python
class DocumentProcessor:
    def __init__(self):
```

`convert_processor`의 `__init__()`과 구조가 **매우 유사**합니다. 동일한 부분은 간략히, 차이점은 상세히 설명합니다.

#### 공통 설정 (convert_processor와 동일)

```python
# OCR 엔진 설정
self.ocr_endpoint = "http://192.168.73.172:48080/ocr" ##Genos OCR Endpoint, Site 마다 변경
ocr_options = PaddleOcrOptions( ##PaddleOCR v5 활용시 필요한 정보, 상용 OCR 사용시 변경 필요
    force_full_page_ocr=False,
    lang=['korean'],
    ocr_endpoint=self.ocr_endpoint,
    text_score=0.3
)

# PDF 파이프라인 옵션
self.pipe_line_options = PdfPipelineOptions()
self.pipe_line_options.generate_page_images = True
self.pipe_line_options.generate_picture_images = True
self.pipe_line_options.do_ocr = False
self.pipe_line_options.do_table_structure = True
self.pipe_line_options.images_scale = 2
self.pipe_line_options.table_structure_options.do_cell_matching = True
self.pipe_line_options.table_structure_options.mode = TableFormerMode.ACCURATE

# 4개의 컨버터 매트릭스 (동일)
self._create_converters()

# Enrichment 옵션 (동일)
self.enrichment_options = DataEnrichmentOptions(...)
```

#### 차이점 — Simple 파이프라인 옵션

```python
# convert_processor:
self.simple_pipeline_options = PipelineOptions()
self.simple_pipeline_options.save_images = False    # 항상 False

# intelligent_processor:
self.simple_pipeline_options = PipelineOptions()
self.simple_pipeline_options.save_images = False    # 초기값 False, kwargs로 변경 가능
```

> `intelligent_processor`에서는 `save_images`와 `include_wmf` 옵션이 **런타임에 동적으로 변경**될 수 있습니다. 이는 RAG 지식베이스 구축 시 이미지를 함께 저장할지 여부를 유연하게 결정할 수 있게 합니다.

#### 4개의 컨버터 구성 (`_create_converters()`)

`convert_processor`와 완전히 동일한 구조입니다:

```
┌─────────────────────────────────────────────────────────┐
│              │  OCR 비활성화        │  OCR 활성화           │
│ ─────────────┼───────────────────┼──────────────────    │
│ 주 백엔드      │ converter         │ ocr_converter        │
│              │ (PyPdfium)        │ (DoclingParseV4)     │
│ ─────────────┼───────────────────┼──────────────────    │
│ 보조 백엔드     │ second_converter  │ ocr_second_converter │
│              │ (PyPdfium)        │ (PyPdfium)           │
└─────────────────────────────────────────────────────────┘
```

---

### 7.2 문서 로딩 메서드들

#### `load_documents_with_docling()` — 기본 로딩

```python
def load_documents_with_docling(self, file_path: str, **kwargs) -> DoclingDocument:
    # ★ intelligent 고유: 동적 옵션 갱신
    save_images = kwargs.get('save_images', True)    # 기본값이 True!
    include_wmf = kwargs.get('include_wmf', False)

    # 옵션이 변경되면 컨버터 재생성
    if (self.simple_pipeline_options.save_images != save_images or
        getattr(self.simple_pipeline_options, 'include_wmf', False) != include_wmf):
        self.simple_pipeline_options.save_images = save_images
        self.simple_pipeline_options.include_wmf = include_wmf
        self._create_converters()

    try:
        conv_result = self.converter.convert(file_path, raises_on_error=True)
    except Exception:
        conv_result = self.second_converter.convert(file_path, raises_on_error=True)
    return conv_result.document
```

**`convert_processor`와의 핵심 차이**:

| 항목 | convert_processor | intelligent_processor |
|------|-------------------|----------------------|
| `save_images` 기본값 | ❌ (미사용) | `True` (이미지 저장 기본 활성화) |
| `include_wmf` 지원 | ❌ | ✅ (WMF 이미지 포맷 포함 가능) |
| 동적 컨버터 재생성 | ❌ | ✅ (옵션 변경 시 `_create_converters()` 재호출) |

**`include_wmf`란?**: WMF(Windows Metafile)는 Windows 환경의 벡터 그래픽 포맷으로, 일부 한국어 공문서에서 사용됩니다. 이 옵션을 `True`로 설정하면 WMF 이미지도 추출 대상에 포함됩니다.

**컨버터 재생성 흐름**:

```
첫 번째 호출: save_images=True, include_wmf=False
  → 컨버터 생성 (이미지 저장 활성화)

두 번째 호출: save_images=False, include_wmf=True
  → 옵션 변경 감지 → _create_converters() 재호출
  → 새 설정으로 컨버터 재생성
```

#### `load_documents_with_docling_ocr()` — OCR 기반 로딩

기본 로딩과 동일한 동적 옵션 갱신 로직을 포함합니다. GLYPH 감지 시 호출됩니다.

#### `load_documents()` — 진입점

```python
def load_documents(self, file_path: str, **kwargs) -> DoclingDocument:
    return self.load_documents_with_docling(file_path, **kwargs)
```

> `convert_processor`와 동일한 위임 패턴입니다.

---

### 7.3 문서 분할 — `split_documents()`

```python
def split_documents(self, documents: DoclingDocument, **kwargs) -> List[DocChunk]:
    chunker = GenosBucketChunker(
        max_tokens=0,           # ★ 핵심: 토큰 제한 없음!
        merge_peers=True
    )
    chunks = list(chunker.chunk(dl_doc=documents, **kwargs))
    for chunk in chunks:
        self.page_chunk_counts[chunk.meta.doc_items[0].prov[0].page_no] += 1
    return chunks
```

**세 전처리기의 분할 비교**:

```python
# attachment_processor:
# → HybridChunker(max_tokens=int(1e30))  # 무제한 토큰, 문서 아이템 기반

# convert_processor:
# → GenosBucketChunker(max_tokens=2000)   # 2000 토큰 제한, 섹션+토큰 병합

# intelligent_processor:
# → GenosBucketChunker(max_tokens=0)      # 토큰 무제한, 순수 섹션 분할
```

---

### 7.4 GLYPH 감지 및 선택적 OCR

`convert_processor`와 **완전히 동일한 코드**입니다.

#### `check_glyph_text()` — 텍스트 레벨 감지

```python
def check_glyph_text(self, text: str, threshold: int = 1) -> bool:
    matches = re.findall(r'GLYPH\w*', text)
    return len(matches) >= threshold
```

#### `check_glyphs()` — 문서 레벨 감지

```python
def check_glyphs(self, document: DoclingDocument) -> bool:
    for item, level in document.iterate_items():
        if isinstance(item, TextItem):
            matches = re.findall(r'GLYPH\w*', item.text)
            if len(matches) > 10:    # 10개 이상이면 "깨진 문서"
                return True
    return False
```

#### `ocr_all_table_cells()` — 테이블 셀 단위 선택적 OCR

```
문서의 모든 테이블 순회
    │
    ├── 셀 텍스트에 GLYPH 없음 → 건너뜀
    └── GLYPH 발견 → 해당 테이블의 모든 셀을 OCR
         │
         ├── 셀 바운딩 박스로 이미지 추출
         ├── zoom_factor 자동 계산 (최소 20px 높이)
         ├── PaddleOCR API에 전송
         └── 셀 텍스트를 OCR 결과로 교체
```

> `convert_processor`와 동일한 로직입니다. 자세한 설명은 `convert_processor` 설명서를 참조하세요.

---

### 7.5 문서 Enrichment (LLM 기반 메타데이터 보강)

```python
def enrichment(self, document: DoclingDocument, **kwargs) -> DoclingDocument:
    document = enrich_document(document, self.enrichment_options, **kwargs)
    return document
```

`convert_processor`와 동일합니다. LLM(Mistral-Small-3.1-24B)을 활용하여:
- **계층적 목차(TOC)** 자동 생성
- **작성일** 등 문서 메타데이터 추출

---

### 7.6 부록(Appendix) 자동 연결 기능

```python
def check_appendix_keywords(self, content: str, appendix_list: list) -> str:
```

**이 기능은 `intelligent_processor`에만 존재하는 고유 기능**입니다. 문서 본문에서 "별지", "별표", "장부" 등의 참조를 탐지하고, 실제 부록 파일과 자동으로 매칭합니다.

**왜 필요한가?**

한국어 법률/규정 문서에는 본문에서 부록을 참조하는 패턴이 자주 등장합니다:
```
"...본 약관의 세부 사항은 (별지 제1호 서식)에 따른다..."
"...수수료는 [별표 제2호]에 명시된 바에 따른다..."
```

이때 RAG 시스템이 "별지 제1호 서식"에 대한 질문을 받으면, 해당 부록 파일을 찾아야 합니다. `appendix` 필드에 매칭된 파일명이 저장되어 있으면 검색 품질이 크게 향상됩니다.

**처리 흐름**:

```
입력:
  content = "...세부 사항은 별지 제1호 서식에 따른다..."
  appendix_list = ["별지 제1호 서식.pdf", "별표 제2호.pdf", "장부 양식.pdf"]

    │
    ▼
1단계: 공백 제거
    content = "...세부사항은별지제1호서식에따른다..."

    │
    ▼
2단계: 복합 패턴 탐색
    정규식: (별지|별표|장부)(?:제)?([^<>()\[\]]+?)(?=(?:호|서식)|...)
    
    매칭: ("별지", "제1") → 변형 생성:
      ["별지 제1", "별지 제제1호", "별지제1", "별지제제1호"]

    │
    ▼
3단계: 독립 패턴 탐색
    정규식: [\(\[](별지|별표|장부)[\)\]]
    
    예: "(별표)" → ["별표"]

    │
    ▼
4단계: 부록 목록과 매칭
    appendix_list의 각 파일명에서 .pdf 제거 후 비교:
    
    "별지 제1호 서식" 에 "별지 제1" 포함? → YES!
    → matched: ["별지 제1호 서식.pdf"]

    │
    ▼
출력: "별지 제1호 서식.pdf"
```

**코드의 정규식 패턴 분석**:

```python
# 공백을 모두 제거하여 "별지 제 1 호"와 "별지제1호"를 동일하게 처리
content = re.sub(r"\s+", "", content)

# 복합 패턴: "별지 제Ⅰ-1호 서식" 같은 복잡한 패턴도 캡처
complex_patterns = re.findall(
    r'(별지|별표|장부)(?:제)?([^<>()\[\]]+?)(?=(?:호|서식)|[<>\)\]]|$)', 
    content
)
# → [("별지", "1"), ("별표", "2-1")] 등

# 독립 패턴: "(별지)", "[별표]" 같은 괄호로 감싼 단독 참조
standalone_patterns = re.findall(
    r'[\(\[]+(별지|별표|장부)[\)\]]+', 
    content
)
# → ["별지", "별표"] 등
```

**매칭 예시 모음**:

```
본문 텍스트                    appendix_list 파일              매칭 결과
─────────────                 ─────────────────              ──────────
"별지 제1호 서식 참조"         ["별지 제1호 서식.pdf"]         "별지 제1호 서식.pdf"
"(별표) 참조"                  ["별표.pdf", "별표 제1호.pdf"]  "별표.pdf"
"[별지 제Ⅰ-1호 서식]"         ["별지 제Ⅰ-1호 서식.pdf"]      "별지 제Ⅰ-1호 서식.pdf"
"별지 및 별표 참조"            ["별지.pdf", "별표.pdf"]        "별지.pdf, 별표.pdf"
"일반 텍스트"                  ["별지.pdf"]                   "" (빈 문자열)
```

---

### 7.7 벡터 조립 — `compose_vectors()`

```python
async def compose_vectors(self, document, chunks, file_path, request, **kwargs):
```

`convert_processor`의 `compose_vectors()`와 유사하지만, **부록 매칭** 로직이 추가되고 **`authors` 추출이 없습니다**.

**차이점 1: 부록(appendix) 매칭**

```python
# kwargs에서 부록 파일 목록 추출
appendix_info = kwargs.get('appendix', '')
appendix_list = []
if isinstance(appendix_info, str):
    appendix_list = [item.strip() for item in json.loads(appendix_info) if item.strip()]
elif isinstance(appendix_info, list):
    appendix_list = appendix_info

# 각 청크마다 부록 매칭 수행
for chunk_idx, chunk in enumerate(chunks):
    content = headers_text + chunk.text

    # ★ 이 청크의 텍스트에서 부록 참조를 탐색
    matched_appendices = self.check_appendix_keywords(content, appendix_list)

    chunk_global_metadata = global_metadata.copy()
    chunk_global_metadata['appendix'] = matched_appendices  # 청크별 부록 정보
```

> **핵심**: `global_metadata`를 청크마다 **복사(`copy()`)**하여 `appendix` 값을 개별 설정합니다. 청크 A에서는 "별지 제1호"가, 청크 B에서는 "별표 제2호"가 매칭될 수 있기 때문입니다.

**차이점 2: `authors` 필드 없음**

```python
# convert_processor:
global_metadata = dict(
    n_chunk_of_doc=len(chunks),
    n_page=document.num_pages(),
    reg_date=...,
    created_date=created_date,
    authors=authors,          # ◄── convert에만 있음
    title=title
)

# intelligent_processor:
global_metadata = dict(
    n_chunk_of_doc=len(chunks),
    n_page=document.num_pages(),
    reg_date=...,
    created_date=created_date,
    title=title
    # authors 없음! appendix는 chunk별로 별도 설정
)
```

**`HEADER:` 접두어** (`convert_processor`와 동일):

```python
headers_text = "HEADER: " + ", ".join(chunk.meta.headings) + '\n' if chunk.meta.headings else ''
content = headers_text + chunk.text
```

결과 예시:
```
HEADER: 제1장 총칙, 제1절 목적
제1조(목적) 이 법은 국민의 기본적 권리를 보장하고...
```

---

### 7.8 빈 문서 처리 (Empty Document Fallback)
- 빈 문서는 예외로 처리할 수도 있고, 아래 예제와 같이 에러를 방지하게끔 dummy text 를 넣어서 처리할 수도 있습니다.

```python
# 텍스트 아이템 존재 여부 확인
has_text_items = False
for item, _ in document.iterate_items():
    if (isinstance(item, (TextItem, ListItem, CodeItem, SectionHeaderItem))
        and item.text and item.text.strip()) or \
       (isinstance(item, TableItem) and item.data and len(item.data.table_cells) == 0):
        has_text_items = True
        break
```

**텍스트가 있는 경우**: 정상 처리

**텍스트가 없는 경우**: 더미 텍스트 아이템 삽입

```python
if not has_text_items:
    # ProvenanceItem 생성 (위치 정보)
    prov = ProvenanceItem(
        page_no=1,                                      # 1페이지
        bbox=BoundingBox(l=0, t=0, r=1, b=1),          # 전체 페이지 영역
        charspan=(0, 1)                                 # 최소 문자 범위
    )

    # 더미 텍스트 아이템 추가
    document.add_text(
        label=DocItemLabel.TEXT,
        text=".",                                       # 최소 텍스트
        prov=prov
    )

    chunks = self.split_documents(document, **kwargs)
```

**시각적 비교 — 빈 문서 처리**:

```
attachment_processor:
  빈 문서 → split_documents() → chunks=[] → Exception('Empty document')
  
convert_processor:
  빈 문서 → split_documents() → chunks=[] → GenosServiceException("chunk length is 0")

intelligent_processor:
  빈 문서 → has_text_items=False → 더미 "." 삽입 → split_documents()
  → 최소 1개의 청크 보장 → 벡터 생성 성공!
```

> **왜 필요한가?**: RAG 지식베이스에 이미지만으로 구성된 문서(예: 스캔된 문서에서 OCR 실패)가 적재될 수 있습니다. 이 경우 예외를 발생시키면 전체 배치 작업이 중단됩니다. 더미 텍스트를 삽입하면 문서가 최소한 "존재하는 것"으로 등록되어, 이미지 메타데이터(media_files)를 통한 후속 처리가 가능해집니다.

---

### 7.9 `__call__()` — 실행 진입점

```python
async def __call__(self, request: Request, file_path: str, **kwargs: dict):
```

**`convert_processor`와의 핵심 차이**: 확장자 분기가 **없습니다**. PDF 전용이므로 단일 경로로 처리합니다.

```python
async def __call__(self, request: Request, file_path: str, **kwargs: dict):
    self.setup_logging(kwargs.get('log_level', 4))

    # ① 문서 로딩 (Docling)
    document = self.load_documents(file_path, **kwargs)

    # ② 품질 검사 및 OCR
    if not check_document(document, self.enrichment_options) or self.check_glyphs(document):
        document = self.load_documents_with_docling_ocr(file_path, **kwargs)
    document = self.ocr_all_table_cells(document, file_path)

    # ③ 이미지 참조 설정
    output_path, output_file = os.path.split(file_path)
    filename, _ = os.path.splitext(output_file)
    artifacts_dir = Path(f"{output_path}/{filename}")
    ...
    document = document._with_pictures_refs(...)

    # ④ LLM Enrichment
    document = self.enrichment(document, **kwargs)

    # ⑤ 빈 문서 검사 및 폴백
    has_text_items = False
    for item, _ in document.iterate_items():
        if (isinstance(item, (TextItem, ListItem, CodeItem, SectionHeaderItem))
            and item.text and item.text.strip()) or ...:
            has_text_items = True
            break

    if has_text_items:
        chunks = self.split_documents(document, **kwargs)
    else:
        # 더미 텍스트 삽입
        document.add_text(label=DocItemLabel.TEXT, text=".", prov=prov)
        chunks = self.split_documents(document, **kwargs)

    # ⑥ 벡터 조립
    if len(chunks) >= 1:
        vectors = await self.compose_vectors(document, chunks, file_path, request, **kwargs)
    else:
        raise GenosServiceException(1, f"chunk length is 0")

    return vectors
```

**세 전처리기의 `__call__()` 구조 비교**:

```
attachment_processor.__call__():
├── .wav/.mp3/.m4a → AudioLoader
├── .csv/.xlsx → TabularLoader
├── .hwp → HwpLoader → RecursiveTextSplitter
├── .hwpx → HwpxProcessor (Docling)
├── .docx → DocxProcessor (Docling)
└── 기타 → LangChain 로더 → RecursiveTextSplitter

convert_processor.__call__():
├── .ppt → LangChain 경로
└── 기타 → Docling 경로
              ├── [PDF] 품질 검사 + OCR
              ├── [DOCX/PPTX] PDF 변환
              ├── Enrichment
              └── GenosBucketChunker(2000)

intelligent_processor.__call__():
└── [단일 경로]                            ◄── 분기 없음!
     ├── Docling 로딩
     ├── 품질 검사 + OCR
     ├── Enrichment
     ├── 빈 문서 폴백                      ◄── 고유 기능
     └── GenosBucketChunker(0)             ◄── 순수 섹션 분할
```

---

## 8. 예외 클래스 및 유틸리티

### `GenosServiceException`

```python
class GenosServiceException(Exception):
    def __init__(self, error_code: str, error_msg: Optional[str] = None, ...):
```

세 전처리기 모두 동일합니다.

### `assert_cancelled()`

```python
async def assert_cancelled(request: Request):
    if await request.is_disconnected():
        raise GenosServiceException(1, f"Cancelled")
```

> `convert_processor`와 동일하게 함수로 정의되어 있지만, `__call__()` 내에서 호출 부분은 주석 처리되어 있습니다.

### `setup_logging()`

```python
def setup_logging(self, level_num: int):
    # 5→DEBUG, 4→INFO, 3→WARNING, 2→ERROR, 1→CRITICAL, 0→NOLOG
```

세 전처리기 모두 동일합니다.

---

## 9. Enrichment 프롬프트

`convert_processor`와 **완전히 동일한 프롬프트**를 사용합니다.

### 시스템 프롬프트

```python
toc_system_prompt = """You are an expert at generating table of contents (목차) 
from Korean documents..."""
```

### 사용자 프롬프트

```python
toc_user_prompt = """
<document>
{{raw_text}}     ← 문서 전문이 여기에 삽입됨
</document>

## Analysis Process
1. Document Title Extraction
2. Structural Marker Identification (제x장/절/관/조, 부칙, 별지, 별표...)
3. Systematic Section Extraction
4. Hierarchy Building
5. Structure Verification

## Output: <toc>TITLE:... 1. ... 1.1. ...</toc>
"""
```

** 주의 **
> LLM이 한국어 법률/규정/약관/공공 문서 구조를 분석하여 계층적 목차를 자동 생성하도록 구성되어 있습니다.
> Site 문서 특성에 따라 Prompt 수정이 필요할 수 있습니다.
> 1.4 이후 버전에서는, Prompt 는 Resource file 을 통해서 읽도록 조정되고 하드코딩에서는 삭제됩니다.
---

## 10. 실행 흐름 요약

### 사용 방법 (Genos facade에 통합 시)

```python
# 1. DocumentProcessor 인스턴스 생성
processor = DocumentProcessor()
# → 4개의 Docling 컨버터, OCR 엔진, Enrichment 옵션 초기화

# 2. 파일 처리 호출 (비동기)
vectors = await processor(
    request=request,
    file_path="/path/to/document.pdf",
    log_level=4,                           # 선택: 로그 레벨
    save_images=True,                      # 선택: 이미지 저장 여부 (기본 True)
    include_wmf=False,                     # 선택: WMF 이미지 포함 여부
    appendix='["별지 제1호.pdf", "별표.pdf"]',  # 선택: 부록 파일 목록
)

# 3. 결과: List[GenOSVectorMeta]
for v in vectors:
    print(v.text)              # "HEADER: 제1장 총칙, 제1절 목적\n제1조(목적)..."
    print(v.title)             # "개인정보 보호법"
    print(v.created_date)      # 20240115
    print(v.appendix)          # "별지 제1호.pdf" 또는 ""
    print(v.chunk_bboxes)      # '[{"page":1,"bbox":{"l":0.1,...}}]'
    print(v.media_files)       # '[{"name":"img1.png","type":"image"}]'
```

### 전체 처리 흐름 (상세)

```
processor(request, "/data/privacy_law.pdf", appendix='["별지 제1호.pdf"]')
    │
    ▼
__call__()
    │
    ├── ① load_documents_with_docling()
    │   │  save_images=True (기본) → 이미지도 추출
    │   └── Docling PDF 파이프라인
    │       ├── Layout Detection (딥러닝 비전 모델)
    │       │   → 제목, 본문, 표, 그림, 캡션 등 11종 요소 식별
    │       ├── TableFormer (ACCURATE 모드)
    │       │   → 병합 셀, 다중 헤더 표를 마크다운으로 복원
    │       └── → DoclingDocument 생성
    │
    ├── ② 품질 검사
    │   ├── check_document(): 텍스트 품질 검사
    │   ├── check_glyphs(): GLYPH 감지
    │   │   └── [필요 시] OCR 전체 재처리
    │   └── ocr_all_table_cells(): 깨진 테이블 셀만 OCR
    │
    ├── ③ _with_pictures_refs(): 이미지 참조 설정
    │
    ├── ④ enrichment()
    │   ├── LLM에 문서 전문 전달
    │   ├── TOC 자동 생성: "1. 제1장 총칙 / 1.1. 제1조(목적) / ..."
    │   └── 메타데이터 추출: created_date 등
    │
    ├── ⑤ 빈 문서 검사
    │   ├── has_text_items=True → 정상 진행
    │   └── has_text_items=False → 더미 "." 삽입 후 진행
    │
    ├── ⑥ GenosBucketChunker(max_tokens=0)
    │   ├── preprocess(): 모든 아이템 + 헤더 수집
    │   └── _split_document_by_tokens():
    │       ├── 1단계: 섹션 헤더 기준 분할
    │       │   "제1조" → 새 섹션, "제2조" → 새 섹션, ...
    │       ├── 2단계: heading 텍스트 생성
    │       │   "총칙, 목적, 제1조(목적) 이 법은..."
    │       ├── 2.5단계: [스킵] max_tokens=0
    │       ├── 3단계: 단독 타이틀 병합
    │       │   "제2장"(8자) + "제3조..." → 하나로
    │       └── 4단계: [병합 없음] max_tokens=0
    │           각 섹션이 독립 청크로 유지
    │
    └── ⑦ compose_vectors()
        │
        │  각 청크마다:
        │  ├── HEADER: 접두어 추가
        │  ├── check_appendix_keywords() → 부록 매칭
        │  │   "...별지 제1호 서식에 따른다..."
        │  │   → appendix = "별지 제1호.pdf"
        │  ├── GenOSVectorMetaBuilder로 조립
        │  └── 이미지 업로드 (있으면)
        │
        ▼
    [
      GenOSVectorMeta(
        text="HEADER: 제1장 총칙, 제1절 목적\n제1조(목적) 이 법은...",
        title="개인정보 보호법",
        created_date=20240115,
        appendix="별지 제1호.pdf",
        i_page=1, e_page=1,
        chunk_bboxes='[{"page":1,"bbox":{"l":0.12,"t":0.08,...},"type":"text"}]',
        media_files='[]',
        i_chunk_on_doc=0, n_chunk_of_doc=45, ...
      ),
      GenOSVectorMeta(
        text="HEADER: 제1장 총칙, 제2절 범위\n제3조(적용 범위)...",
        appendix="",          ← 이 청크에는 부록 참조 없음
        i_chunk_on_doc=1, ...
      ),
      ...
    ]
```

### 지원 파일 포맷

| 카테고리 | 확장자 | 처리 경로 | 핵심 도구 | OCR | Enrichment | 부록 연결 |
|----------|--------|-----------|-----------|-----|------------|-----------|
| **PDF** | `.pdf` | Docling 전체 파이프라인 | Layout Detection + TableFormer | ✅ 선택적 | ✅ | ✅ |

> `intelligent_processor`는 **PDF 전용**입니다. 다른 포맷(DOCX, PPT, HWP 등)을 처리하려면 사전에 PDF로 변환하는 구성을 전제로 합니다.

---

> **참고**: 이 전처리기는 **품질을 최우선**으로 설계되어 있어, 딥러닝 기반 레이아웃 분석, LLM Enrichment, 선택적 OCR 등 가장 많은 AI 처리 단계를 포함합니다. 따라서 처리 시간이 세 전처리기 중 **가장 길 수 있습니다**. 실시간 채팅 첨부에는 `attachment_processor`를, PDF 변환 중심의 빠른 적재에는 `convert_processor`를, **RAG 지식베이스 구축을 위한 최고 품질의 전처리**에는 이 `intelligent_processor`를 사용하세요.