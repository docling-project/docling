# Parser 전처리기 사용자 매뉴얼

> **역할:** 다양한 문서 포맷을 파싱하여 element 단위 구조화 데이터를 반환하는 전용 파서.
> 청킹·벡터 조합은 수행하지 않습니다.

---

## 목차

1. [개요](#개요)
2. [API 엔드포인트](#api-엔드포인트)
3. [config.yaml 설정](#configyaml-설정)
4. [지원 파일 형식 및 전제조건](#지원-파일-형식-및-전제조건)
5. [API 요청 파라미터](#api-요청-파라미터)
6. [출력 데이터 구조](#출력-데이터-구조)
7. [예외 처리](#예외-처리)
8. [사용 예시](#사용-예시)

---

## 개요

Parser 전처리기는 문서를 파싱하여 **element 목록** 형태로 반환하는 서비스입니다.
원문 구조(제목·단락·표·그림 등)를 element 단위로 분해하여 반환합니다.

**주요 특징:**
- PDF의 레이아웃 인식, OCR, TOC 추출은 `IntelligentDocumentProcessor`가 담당
- HWP/HWPX는 전용 SDK 백엔드 사용, 실패 시 단계별 폴백 적용
- 오디오는 Whisper API를 통한 음성 전사
- CSV/XLSX는 표 구조 그대로 반환
- 출력 형식은 `config.yaml`의 `output.format`으로 제어 (`json` / `html` / `markdown`)

---

## API 엔드포인트

전처리기는 Gateway 서버를 통해 외부에서 호출할 수 있습니다. Gateway는 전처리기 ID 기준으로 실제 K8s Service로 라우팅하며, AuthKey 기반 인증을 통해 승인된 요청만 전달합니다.

**주요 특징:**
- 전처리기 리소스를 Gateway 외부 API로 호출 (`/preprocessor/{id}/healthcheck`, `/preprocessor/{id}/run`)
- AuthKey 기반 인증/승인 흐름 연동 — 승인된 AuthKey로만 호출 가능하며, 회수된 AuthKey는 차단됨
- admin-api에 전처리기 Gateway 라우팅 정보 제공 — 전처리기 ID 기준으로 실제 K8s Service로 라우팅
- 화면에서 배포한 전처리기 코드를 Gateway를 통해 실행 가능 — 파일 경로와 실행 파라미터를 받아 전처리 결과 반환

### 접속 정보 예시

| 항목 | 값 |
|------|----|
| Base URL | `http://192.168.82.185:30908` |
| Preprocessor ID | `1` |
| Authorization Key | `<key>` |
| Smoke 파일 경로 | `/nfs-root/preprocessor-gateway-smoke.txt` |

### GET `/preprocessor/{id}/healthcheck`

전처리기 상태를 확인합니다.

```http
GET http://192.168.82.185:30908/preprocessor/1/healthcheck
Authorization: Bearer <key>
```

### POST `/preprocessor/{id}/run`

전처리기를 실행하여 파싱 결과를 반환합니다. 파일 경로와 실행 파라미터를 전달합니다.

```http
POST http://192.168.82.185:30908/preprocessor/1/run
Authorization: Bearer <key>
Content-Type: application/json

{
  "file_path": "/path/to/document.pdf",
  "params": {
    "log_level": 4,
  }
}
```

**성공 응답:**
```json
{
  "code": 0,
  "data": {
    "content": "...",
    "elements": [...],
    "usage": {"pages": 10}
  }
}
```

> Gateway URL 형식: `http://<gateway-host>:<port>/preprocessor/<id>/healthcheck` 또는 `.../run`
> `Authorization: Bearer <authKey>` 헤더가 없거나 회수된 키를 사용하면 요청이 차단됩니다.

---

## config.yaml 설정

`DocumentProcessor`는 초기화 시 `config.yaml`을 읽습니다. 기본 경로: `facade/config.yaml`

아래는 전체 설정 스키마입니다.

```yaml
# ───────────────────────────────────────────────
# OCR 설정 (PDF 파이프라인에서 사용)
# ───────────────────────────────────────────────
ocr:
  ocr_endpoint: "http://<ocr-server>/ocr"
  # OCR 모드: auto | force | disable
  #   auto    — 문서 품질 자동 감지 후 필요 시 OCR 수행 (기본값)
  #   force   — 항상 OCR 수행
  #   disable — OCR 수행 안 함
  ocr_mode: "auto"

# ───────────────────────────────────────────────
# 레이아웃 모델 설정
# ───────────────────────────────────────────────
layout:
  # 레이아웃 모델 종류: genos_layout | docling_layout
  layout_model_type: "genos_layout"
  genos_layout:
    endpoint: "http://<layout-server>/v1/chat/completions"
    api_key: ""
    page_batch_size: 32       # 배치당 처리 페이지 수 (기본값: 32)

# ───────────────────────────────────────────────
# Enrichment (TOC 추출 / 메타데이터 추출)
# ───────────────────────────────────────────────
enrichment:
  api_url: "http://<llm-server>/v1/chat/completions"
  api_key: ""
  model: "model"
  do_toc: true               # 목차(TOC) 추출 여부
  do_metadata: true          # 메타데이터 추출 여부 (작성일)
  toc:
    temperature: 0.0
    top_p: 0.00001
    seed: 33
    max_tokens: 10000

# ───────────────────────────────────────────────
# Whisper 음성 전사 설정 (오디오 파일 처리)
# ───────────────────────────────────────────────
whisper:
  url: "http://<whisper-server>/v1/audio/transcriptions"
  model: "model"
  language: "ko"
  response_format: "json"
  temperature: "0"
  stream: "false"
  timestamp_granularities: "word"
  chunk_sec: 29              # 오디오 분할 단위 (초, 기본값: 29)

# ───────────────────────────────────────────────
# 출력 형식
# ───────────────────────────────────────────────
output:
  # 출력 형식: json | html | markdown
  #   json     — element 목록 형태 (기본값)
  #   html     — 전체 문서를 HTML 문자열로 반환 (content 필드)
  #   markdown — 전체 문서를 Markdown 문자열로 반환 (content 필드)
  format: "json"
  # 테이블 표현 형식: html | markdown (json, markdown 포맷의 테이블에 적용)
  table_format: "html"
```

### 설정 항목 상세

| 섹션 | 키 | 기본값 | 설명 |
|------|----|--------|------|
| `ocr` | `ocr_endpoint` | `""` | PaddleOCR 서버 URL |
| `ocr` | `ocr_mode` | `"auto"` | OCR 수행 정책. `auto` / `force` / `disable` |
| `layout` | `layout_model_type` | `"genos_layout"` | 레이아웃 모델 선택 |
| `layout.genos_layout` | `endpoint` | `""` | Genos Layout API URL |
| `layout.genos_layout` | `api_key` | `""` | API 인증 키 |
| `layout.genos_layout` | `page_batch_size` | `32` | 배치당 처리 페이지 수. 양의 정수, 유효하지 않으면 32로 대체 |
| `enrichment` | `api_url` | `""` | LLM API URL (TOC·메타데이터 추출용) |
| `enrichment` | `do_toc` | `true` | 목차 추출 활성화 여부 |
| `enrichment` | `do_metadata` | `true` | 메타데이터 추출 활성화 여부 |
| `whisper` | `url` | `""` | Whisper 전사 API URL |
| `whisper` | `chunk_sec` | `29` | 오디오 분할 크기(초). 유효하지 않으면 29로 대체 |
| `output` | `format` | `"json"` | 응답 포맷. `json` / `html` / `markdown`. 유효하지 않으면 `json`으로 대체 |
| `output` | `table_format` | `"html"` | 표 변환 포맷. `html` / `markdown`. 유효하지 않으면 `html`로 대체 |

---

## 지원 파일 형식 및 전제조건

### 형식별 분류표

| 확장자 | 처리 경로 | 출력 category | 전제조건 |
|--------|-----------|---------------|----------|
| `.pdf`, `.html`, `.htm` | IntelligentDocumentProcessor | 문서 구조 그대로 | Layout 모델 서버, OCR 서버(선택) |
| `.hwp`, `.hwpx` | HwpDocumentLoader | 문서 구조 그대로 | GenosHwpDocumentBackend (HWP SDK) |
| `.docx` | DocxDocumentLoader | 문서 구조 그대로 | GenosMsWordDocumentBackend |
| `.wav`, `.mp3`, `.m4a` | AudioLoader → Whisper | `paragraph` | pydub, Whisper API 서버 |
| `.csv`, `.xlsx` | TabularLoader | `table` (HTML) | pandas, openpyxl, chardet |
| `.doc`, `.ppt`, `.pptx`, `.txt`, `.json`, `.md`, `.jpg`, `.jpeg`, `.png` | GenericDocumentLoader (Langchain) | `paragraph` | LibreOffice (`soffice`), unstructured 라이브러리 |

---

### PDF / HTML / HTM

**처리 클래스:** `IntelligentDocumentProcessor`

**전제조건:**
- **Layout 모델 서버:** `config.yaml`의 `layout.genos_layout.endpoint` 로 접근 가능한 vLLM 호환 서버가 실행 중이어야 함
- **OCR 서버:** `ocr_mode`가 `disable`이 아닌 경우 `ocr.ocr_endpoint`에 PaddleOCR 서버가 실행 중이어야 함 (단, `ocr_mode=auto`이면 OCR 품질 감지 후 필요할 때만 접근)
- **Enrichment LLM:** `enrichment.do_toc=true` 또는 `do_metadata=true`인 경우 `enrichment.api_url`에 LLM API가 실행 중이어야 함
- **Python 패키지:** `docling`, `docling-core`, `pymupdf`

---

### HWP / HWPX

**처리 클래스:** `HwpDocumentLoader`

**전제조건:**
- **GenosHwpDocumentBackend:** HWP 처리용 SDK 백엔드. `use_hwp_sdk=true`(기본값)일 때 사용
- SDK 실패 시 자동으로 `HwpDocumentBackend` / `HwpxDocumentBackend` 폴백 시도
- 모든 백엔드 실패 시 LibreOffice로 PDF 변환 후 PDF 경로로 재처리
- **Python 패키지:** `docling`(hwp 백엔드 포함)

**폴백 순서:**
```
GenosHwpDocumentBackend  →  HwpDocumentBackend/HwpxDocumentBackend  →  LibreOffice PDF 변환 → PDF 경로로 재처리
```

---

### DOCX

**처리 클래스:** `DocxDocumentLoader`

**전제조건:**
- **GenosMsWordDocumentBackend:** DOCX 처리용 커스텀 백엔드
- **Python 패키지:** `docling`(msword 백엔드 포함)
- 이미지 좌표(coordinates)는 출력에서 제외됨 (`clear_coordinates=True`)

---

### WAV / MP3 / M4A (오디오)

**처리 클래스:** `AudioLoader`

**전제조건:**
- **Whisper API 서버:** `config.yaml`의 `whisper.url`에 OpenAI Whisper 호환 API가 실행 중이어야 함
- **Python 패키지:** `pydub`
- **ffmpeg / ffprobe:** pydub의 오디오 디코딩에 필요. 시스템에 설치되어 있어야 함
- 오디오 파일은 `chunk_sec`(기본 29초) 단위로 분할 후 병렬 전사

---

### CSV / XLSX

**처리 클래스:** `TabularLoader`

**전제조건:**
- **Python 패키지:** `pandas`, `openpyxl`(xlsx용), `chardet`(csv 인코딩 감지용)
- CSV는 인코딩 자동 감지 (chardet → utf-8 → cp949 → euc-kr 순)
- XLSX는 시트 단위로 처리하며 시트별 element 생성

---

### 기타 포맷 (doc, ppt, pptx, txt, json, md, jpg, jpeg, png)

**처리 클래스:** `GenericDocumentLoader`

**전제조건:**
- **LibreOffice (`soffice`):** `.doc`, `.ppt`, `.pptx`, `.jpg`, `.jpeg`, `.png` → PDF 변환에 필요. `PATH`에 등록되어 있어야 함
  - 비ASCII 파일명의 경우 임시 디렉토리에 ASCII 이름으로 복사 후 변환 시도
- **WeasyPrint:** `.txt`, `.json`, `.md` → HTML → PDF 변환에 필요 (없으면 PDF 변환 없이 텍스트만 반환)
- **Python 패키지:** `unstructured`, `chardet`, `markdown2`
- **이미지 파일:** 텍스트가 전혀 없으면 `"."` (점 하나)를 content로 반환

**실제 파일 타입 감지:**
- 확장자와 파일 헤더(`%PDF-`, `\x89PNG`, `\xff\xd8\xff`)가 다른 경우 실제 타입으로 처리

---

## API 요청 파라미터

`params` 딕셔너리를 통해 파일 형식별로 아래 파라미터를 전달할 수 있습니다.

### 공통 파라미터

| 파라미터 | 타입 | 기본값 | 적용 대상 | 설명 |
|----------|------|--------|-----------|------|
| `log_level` | `int` | `4` | 전체 | 로그 레벨. `5`=DEBUG, `4`=INFO, `3`=WARNING, `2`=ERROR, `1`=CRITICAL, `0`=비활성화 |

### HWP / HWPX 전용 파라미터

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `save_images` | `bool` | `true` | 이미지 파일 저장 여부 |
| `use_hwp_sdk` | `bool` | `true` | `true`: GenosHwpDocumentBackend 사용. `false`: 내장 백엔드(HwpDocumentBackend/HwpxDocumentBackend) 강제 사용 |
| `dump_sdk_output` | `bool` | `false` | HWP SDK 내부 출력 덤프 여부 (`use_hwp_sdk=true`일 때만 유효) |

---

## 출력 데이터 구조

모든 응답은 파일 형식·출력 설정에 관계없이 **항상 `content`, `elements`, `usage` 세 필드를 포함**합니다. 해당 경로에서 생성하지 않는 필드는 빈 값(`""` 또는 `[]`)으로 채워집니다.

### 공통 응답 스키마

```json
{
  "elements": [...],
  "usage": {"pages": 10},
  "content": ""
}
```

#### 최상위 필드

| 필드 | 타입 | 항상 존재 | 빈 값 형태 | 설명 |
|------|------|-----------|------------|------|
| `elements` | `array` | O | `[]` | 파싱된 element 목록. 문서 순서대로 정렬됨 |
| `usage` | `object` | O | `{"pages": 0}` | 문서 사용량 정보 |
| `usage.pages` | `int` | O | `0` | 문서 전체 페이지(또는 시트) 수 |
| `content` | `str` | O | `""` | `output.format`이 `html` 또는 `markdown`일 때 전체 문서 문자열. `json` 포맷이거나 Docling 외 경로에서는 `""` |

---

### `elements` 배열 항목 필드

```json
{
  "id": 0,
  "page": 1,
  "category": "paragraph",
  "content": "텍스트 내용 또는 HTML 테이블",
  "coordinates": [
    {"x": 0.1, "y": 0.1},
    {"x": 0.9, "y": 0.1},
    {"x": 0.9, "y": 0.3},
    {"x": 0.1, "y": 0.3}
  ]
}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| `id` | `int` | element 순번. 0-based, 문서 전체에서 고유함 |
| `page` | `int` | element가 위치한 페이지 번호. 1-based. CSV/XLSX는 시트 순번, 오디오는 항상 `1` |
| `category` | `str` | element의 의미적 분류. 아래 [category 값 목록](#category-값-목록) 참고 |
| `content` | `str` | element의 실제 내용. category에 따라 형식이 다름 (아래 표 참고) |
| `coordinates` | `array` | 페이지 내 위치. 정규화된 4-꼭짓점 좌표 (0.0~1.0). 좌표를 제공하지 않는 포맷은 `[]` |

#### `content` 필드의 형식 (category별)

| category | `content` 형식 | 예시 |
|----------|----------------|------|
| `title` | 제목 텍스트 문자열 | `"문서 제목"` |
| `section_header` | 헤더 텍스트 문자열 | `"1. 개요"` |
| `paragraph` | 본문 텍스트 문자열 | `"이 문서는 ..."` |
| `list_item` | 목록 항목 텍스트 문자열 | `"• 항목 내용"` |
| `table` | HTML 또는 Markdown 형식의 표 (`output.table_format` 설정에 따라 결정) | `"<table>...</table>"` |
| `picture` | 빈 문자열 `""` (이미지 자체는 별도 파일로 저장) | `""` |
| `caption` | 그림/표 캡션 텍스트 | `"그림 1. 시스템 구조도"` |
| `footnote` | 각주 텍스트 | `"1) 출처: ..."` |
| `page_header` | 페이지 상단 반복 텍스트 | `"2025 연간 보고서"` |
| `page_footer` | 페이지 하단 반복 텍스트 | `"- 3 -"` |
| `code` | 코드 블록 텍스트 | `"def hello(): ..."` |

### `output.format: html` 또는 `markdown`

Docling 경로(PDF, HTML, HWP, HWPX, DOCX)에서 `config.yaml`의 `output.format`을 `html` 또는 `markdown`으로 설정하면, 전체 문서를 하나의 문자열로 직렬화하여 `content`에 담고 `elements`는 `[]`로 반환됩니다.

```json
{
  "elements": [],
  "usage": {"pages": 10},
  "content": "<html>...</html>"
}
```

> `output.format=markdown`, `output.table_format=html`인 경우: Markdown 문서 내의 테이블 블록이 HTML 테이블로 치환되어 반환됩니다.

---

### `category` 값 목록

Docling 파이프라인(PDF/HTML/HWP/HWPX/DOCX) 출력 기준:

| category 값 | 의미 | 비고 |
|-------------|------|------|
| `title` | 문서 제목 | |
| `section_header` | 섹션 헤더 | level 1~6에 따라 `<h1>`~`<h6>` |
| `paragraph` | 일반 단락 | |
| `list_item` | 목록 항목 | |
| `table` | 표 | `output.table_format`에 따라 HTML 또는 Markdown |
| `picture` | 이미지 | `content`는 빈 문자열 |
| `caption` | 그림/표 캡션 | |
| `footnote` | 각주 | |
| `page_header` | 페이지 헤더 | |
| `page_footer` | 페이지 푸터 | |
| `code` | 코드 블록 | |

오디오·Langchain 경로:

| category 값 | 의미 |
|-------------|------|
| `paragraph` | 전사 텍스트 또는 일반 텍스트 |
| `table` | CSV/XLSX 시트 데이터 (HTML 형식) |

---

### `coordinates` 필드

페이지 너비/높이를 기준으로 정규화된 4개의 꼭짓점 좌표 (top-left 기준, 0.0~1.0).

```json
[
  {"x": 0.1, "y": 0.1},   // 좌상
  {"x": 0.9, "y": 0.1},   // 우상
  {"x": 0.9, "y": 0.3},   // 우하
  {"x": 0.1, "y": 0.3}    // 좌하
]
```

- 좌표 정보가 없는 포멧은 `[]` 반환
  - DOCX, 오디오, CSV/XLSX 등

---

### CSV/XLSX `content` 형식

시트 데이터가 HTML 테이블 형태로 `content`에 저장됩니다.

```html
<table>
  <tr><th>컬럼1</th><th>컬럼2</th></tr>
  <tr><td>값1</td><td>값2</td></tr>
</table>
```

- XLSX는 시트 단위로 element가 생성되며, `page` 값은 시트 순서(1-based)
- `usage.pages`는 시트 수

---

## 예외 처리

### FastAPI 레벨 예외

| 예외 타입 | 발생 상황 | HTTP 응답 | 응답 형식 |
|-----------|-----------|-----------|-----------|
| `GenosServiceException` | 내부 서비스 처리 오류 | 200 | `{"code": 1, "errMsg": "...", "error_code": "..."}` |
| `RequestValidationError` | 잘못된 요청 본문 (필수 필드 누락 등) | 200 | `{"code": 1, "errMsg": "..."}` |
| `Exception` (기타) | 예상치 못한 서버 오류 | 200 | `{"code": 1, "errMsg": "..."}` |

모든 오류는 HTTP 200으로 반환되며 `code: 1`로 실패를 표현합니다.

---

### 파일 형식별 예외 및 폴백

#### PDF / HTML

| 예외 | 원인 | 동작 |
|------|------|------|
| `converter.convert()` 실패 | 파일 손상, 백엔드 오류 | `second_converter`로 재시도 (동일 설정, 다른 인스턴스) |
| 레이아웃 서버 연결 오류 | `layout.genos_layout.endpoint` 미응답 | 예외 발생, 처리 중단 |
| OCR 서버 연결 오류 | `ocr.ocr_endpoint` 미응답 | `ocr_all_table_cells`에서 예외를 캐치하고 OCR 없이 진행 |
| Enrichment LLM 연결 오류 | `enrichment.api_url` 미응답 | enrichment 단계에서 예외 발생 |

#### HWP / HWPX

| 예외 | 원인 | 동작 |
|------|------|------|
| SDK 백엔드 오류 | GenosHwpDocumentBackend 실패 | `use_hwp_sdk=False`로 내장 백엔드 재시도 |
| 내장 백엔드 오류 | HwpDocumentBackend/HwpxDocumentBackend 실패 | LibreOffice로 PDF 변환 후 PDF 경로로 재처리 |
| PDF 변환 실패 | LibreOffice 미설치 또는 파일 손상 | 원본 SDK 예외 재발생 |

#### 오디오 (WAV/MP3/M4A)

| 예외 | 원인 | 동작 |
|------|------|------|
| Whisper API 연결 오류 | `whisper.url` 미응답 | `requests.exceptions.ConnectionError` 등 예외 발생 |
| `pydub` 오류 | ffmpeg 미설치 또는 파일 손상 | 예외 발생 |
| 임시 파일 정리 실패 | 디스크 권한 오류 | 무시하고 계속 진행 |

#### CSV / XLSX

| 예외 | 원인 | 동작 |
|------|------|------|
| `ValueError: Any columns cannot be converted...` | 모든 컬럼을 문자열로 변환할 수 없는 경우 | 예외 발생 |
| 인코딩 오류 | chardet 감지 실패 | utf-8 → cp949 → euc-kr → iso-8859-1 순서로 재시도 |

#### 기타 포맷 (GenericDocumentLoader)

| 예외 | 원인 | 동작 |
|------|------|------|
| `TypeError: __str__ returned non-string` | UnstructuredImageLoader NoneType 오류 | `partition_image` 직접 호출로 폴백 |
| LibreOffice 변환 실패 | `soffice` 미설치 | `None` 반환, 이후 처리에서 오류 가능 |
| WeasyPrint 미설치 | txt/md → PDF 변환 불가 | PDF 변환 없이 텍스트를 `Document`로 직접 반환 |
| 이미지 파일 텍스트 없음 | OCR 결과 없음 | `"."` 단일 문자를 content로 반환 |

---

### 주요 오류 코드 및 조치 방법

| 상황 | 오류 메시지 예시 | 조치 |
|------|-----------------|------|
| Layout 서버 미응답 | Connection refused to `<endpoint>` | `config.yaml`의 `layout.genos_layout.endpoint` 확인 |
| OCR 서버 미응답 | `OCR HTTP 502` | `config.yaml`의 `ocr.ocr_endpoint` 및 서버 상태 확인 |
| Enrichment LLM 오류 | LLM API timeout | `config.yaml`의 `enrichment.api_url` 확인 |
| HWP SDK 실패 | `HWP SDK 실패: ...` | 로그 확인 후 `use_hwp_sdk=false` 파라미터로 재시도 |
| Whisper 미설정 | Connection error | `config.yaml`의 `whisper.url` 설정 확인 |
| `soffice` 미설치 | `FileNotFoundError: soffice` | LibreOffice 설치 후 PATH 등록 |
| 파일 없음 | `FileNotFoundError` | `file_path` 경로 및 파일 존재 여부 확인 |

---

## 사용 예시

### 헬스체크 (curl)

```bash
curl http://192.168.82.185:30908/preprocessor/1/healthcheck \
  -H "Authorization: Bearer <key>"
```

### PDF 파싱 (curl)

```bash
curl -X POST http://192.168.82.185:30908/preprocessor/1/run \
  -H "Authorization: Bearer <key>" \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "/data/documents/report.pdf",
    "params": {"log_level": 4}
  }'
```

### PDF 파싱 (Python requests)

```python
import requests

BASE_URL = "http://192.168.82.185:30908"
PREPROCESSOR_ID = 1
AUTH_KEY = "<key>"

response = requests.post(
    f"{BASE_URL}/preprocessor/{PREPROCESSOR_ID}/run",
    headers={"Authorization": f"Bearer {AUTH_KEY}"},
    json={
        "file_path": "/data/documents/report.pdf",
        "params": {
            "log_level": 3
        }
    }
)
result = response.json()
if result["code"] == 0:
    for element in result["data"]["elements"]:
        print(f"[Page {element['page']}][{element['category']}] {str(element['content'])[:80]}")
```

### HTML 포맷으로 출력 (`config.yaml: output.format: html`)

```python
# config.yaml에 output.format: html 설정 후
response = requests.post(
    f"{BASE_URL}/preprocessor/{PREPROCESSOR_ID}/run",
    headers={"Authorization": f"Bearer {AUTH_KEY}"},
    json={
        "file_path": "/data/documents/report.pdf",
        "params": {}
    }
)
result = response.json()
html_content = result["data"]["content"]  # 전체 문서 HTML
```
