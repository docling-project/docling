# 파싱용 전처리기 매뉴얼

> **역할:** 다양한 문서 포맷을 파싱하여 element 단위 구조화 데이터를 반환하는 전용 파서.
> 청킹·벡터 조합은 수행하지 않습니다.

---

## 목차

1. [개요](#개요)
2. [API 엔드포인트](#api-엔드포인트)
3. [parser_processor_config.yaml 설정](#parser_processor_configyaml-설정)
4. [지원 파일 형식 및 전제조건](#지원-파일-형식-및-전제조건)
5. [API 요청 파라미터](#api-요청-파라미터)
6. [출력 데이터 구조](#출력-데이터-구조)
7. [예외 처리](#예외-처리)
8. [사용 예시](#사용-예시)

---

## 개요

파싱용 전처리기는 문서를 파싱하여 **element 목록** 형태로 반환하는 서비스입니다.
원문 구조(제목·단락·표·그림 등)를 element 단위로 분해하여 반환합니다.

**주요 특징:**
- PDF의 레이아웃 인식, OCR, TOC 추출은 `IntelligentDocumentProcessor`가 담당
- HWP/HWPX는 전용 SDK 백엔드 사용, 실패 시 단계별 폴백 적용
- 오디오는 Whisper API를 통한 음성 전사
- CSV/XLSX는 표 구조 그대로 반환
- 출력 형식은 `parser_processor_config.yaml`의 `output.format`으로 제어 (`json` / `html` / `markdown`)

---

## API 엔드포인트

전처리기는 Gateway 서버를 통해 외부에서 호출할 수 있습니다. Gateway는 전처리기 ID 기준으로 실제 K8s Service로 라우팅하며, AuthKey 기반 인증을 통해 승인된 요청만 전달합니다.

**주요 특징:**
- 전처리기 리소스를 Gateway 외부 API로 호출 (`/preprocessor/{id}/healthcheck`, `/preprocessor/{id}/run`)
- AuthKey 기반 인증/승인 흐름 연동 — 승인된 AuthKey로만 호출 가능하며, 회수된 AuthKey는 차단됨
- admin-api에 전처리기 Gateway 라우팅 정보 제공 — 전처리기 ID 기준으로 실제 K8s Service로 라우팅
- 화면에서 배포한 전처리기 코드를 Gateway를 통해 실행 가능 — 파일 경로와 실행 파라미터를 받아 전처리 결과 반환

### GET `/preprocessor/{id}/healthcheck`

전처리기 상태를 확인합니다.

```http
GET http://<base_url:port>/preprocessor/{id}/healthcheck
Authorization: Bearer <key>
```

- 요청에 필요한 주요 요소는 아래와 같습니다.
  - `<base_url:port>`: Genos 기본 주소
  - `{id}`: Genos에 등록된 전처리기의 ID
  - `<key>`: Genos에서 발급한 AuthKey, 등록된 전처리기에서 발급 가능


### POST `/preprocessor/{id}/run`

전처리기를 실행하여 파싱 결과를 반환합니다. 파일 경로와 실행 파라미터를 전달합니다.

```http
POST http://<base_url:port>/preprocessor/{id}/run
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

## parser_processor_config.yaml 설정

`파싱용 전처리기`는 `parser_processor_config.yaml` 파일을 읽어서 전처리기 수행에 필요한 정보를 설정합니다.
`parser_processor_config.yaml` 파일은 Genos 전처리기의 `resource` 탭에 등록됩니다.

Genos에서 전처리기를 배포할 때 `parser_processor_config.yaml`파일도 함께 배포되어 전처리기 컨테이너에 적용됩니다. 따라서 `parser_processor_config.yaml`파일의 내용이 수정된 경우에는 전처리기를 재배포해야만 수정된 내용이 적용됩니다.

> **전처리기 최초 등록시 아래의 config 수정가이드를 참고해서 수정해 주세요.**

### config 스키마

아래는 전체 설정 스키마입니다.

```yaml
# ───────────────────────────────────────────────
# OCR 설정 (PDF 파이프라인에서 사용)
# ───────────────────────────────────────────────
ocr:
  # OCR 모드: auto | force | disable
  #   auto    — 문서 품질 자동 감지 후 필요 시 OCR 수행 (기본값)
  #   force   — 항상 OCR 수행
  #   disable — OCR 수행 안 함
  ocr_mode: "auto"

  # OCR 엔진 선택: paddle | upstage
  # docker image 재빌드 없이 yaml 만으로 전환 가능
  engine: "paddle"

  # engine: "paddle" 일 때만 사용
  # <OCR_ENDPOINT>: PaddleOCR 서버 주소로 변경 필요
  ocr_endpoint: "http://<OCR_ENDPOINT>/ocr"

  # engine: "upstage" 일 때만 사용 (Upstage Document Digitization REST API)
  upstage:
    api_endpoint: "https://api.upstage.ai/v1/document-digitization"
    # api_key 가 비어있으면 UPSTAGE_API_KEY 환경변수에서 fallback (시크릿 운영용)
    api_key: ""
    model: "ocr"
    timeout: 60
    text_score: 0.5

# ───────────────────────────────────────────────
# 레이아웃 모델 설정
# ───────────────────────────────────────────────
layout:
  # 레이아웃 모델 종류: genos_layout | docling_layout
  layout_model_type: "genos_layout"
  genos_layout:
    # <LAYOUT_SERVING_ID>: Genos에 등록한 layout 모델서빙 ID로 변경 필요
    # api_key는 k8s 내부 통신 기반 모델 호출 시 불필요.
    endpoint: "http://llmops-gateway-api-service:8080/rep/serving/<LAYOUT_SERVING_ID>/v1/chat/completions"
    api_key: ""
    page_batch_size: 32

# ───────────────────────────────────────────────
# Enrichment (TOC 추출 / 메타데이터 추출)
# ───────────────────────────────────────────────
enrichment:
  do_toc: true
  do_metadata: true

  # <ENRICHMENT_SERVING_ID>: Genos에 등록한 enrichment 모델서빙 ID로 변경 필요
  # api_key는 k8s 내부 통신 기반 모델 호출 시 불필요.
  api_url: "http://llmops-gateway-api-service:8080/rep/serving/<ENRICHMENT_SERVING_ID>/v1/chat/completions"
  api_key: ""
  model: "model"
  precheck:
    # true 로 설정하면 LLM 호출 전 입력 토큰을 추정하여 초과 시 즉시 400 에러 반환
    enabled: false
    # 모델 전체 컨텍스트 한도 (입력 + 출력 합산)
    max_context_tokens: 128000
    # 출력용 예약 토큰 수. 허용 입력 = max_context_tokens - completion_reserved_tokens
    completion_reserved_tokens: 12000
  toc:
    temperature: 0.0
    top_p: 0.00001
    seed: 33
    max_tokens: 10000

  # facade 후처리 기반 이미지 설명 생성 (문맥 포함)
  image_description:
    enabled: false
    # 필요 시만 지정 (미지정 시 enrichment.api_url 상속)
    api_url: ""
    # 필요 시만 지정 (미지정 시 enrichment.api_key 상속)
    api_key: ""
    # 필요 시만 지정 (미지정 시 enrichment.model 상속)
    model: "model"
    # 이미지 설명 요청 병렬 수 (기본 4)
    concurrency: 4
    before_items: 3
    after_items: 2
    max_context_chars: 1500
    prompt_template: |
      문서의 일부 이미지를 설명해줘. 아래 문맥을 참고해서 핵심 정보를 2~4문장으로 간결하게 작성해줘.
      [앞 문맥]
      {before_context}
      [캡션]
      {caption}
      [뒤 문맥]
      {after_context}

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
| `ocr` | `ocr_mode` | `"auto"` | OCR 수행 정책. `auto` / `force` / `disable` |
| `ocr` | `engine` | `"paddle"` | OCR 엔진 선택. `paddle` / `upstage` (유효하지 않으면 `paddle`) |
| `ocr` | `ocr_endpoint` | `""` | PaddleOCR 서버 URL (engine=paddle 일 때만 사용) |
| `ocr.upstage` | `api_endpoint` | `"https://api.upstage.ai/v1/document-digitization"` | Upstage OCR API URL |
| `ocr.upstage` | `api_key` | `""` | Upstage API 키. 비어있으면 `UPSTAGE_API_KEY` 환경변수 사용 |
| `ocr.upstage` | `model` | `"ocr"` | Upstage 모델명 |
| `ocr.upstage` | `timeout` | `60` | HTTP 타임아웃 (초) |
| `ocr.upstage` | `text_score` | `0.5` | word confidence 필터링 임계값 |
| `layout` | `layout_model_type` | `"genos_layout"` | 레이아웃 모델 선택. `genos_layout` / `docling_layout` (유효하지 않으면 `genos_layout`) |
| `layout.genos_layout` | `endpoint` | `""` | Genos Layout API URL |
| `layout.genos_layout` | `api_key` | `""` | API 인증 키 |
| `layout.genos_layout` | `page_batch_size` | `32` | 배치당 처리 페이지 수. 양의 정수, 유효하지 않으면 32로 대체 |
| `enrichment` | `do_toc` | `true` | 목차(TOC) 추출 활성화 여부 |
| `enrichment` | `do_metadata` | `true` | 메타데이터 추출 활성화 여부 |
| `enrichment` | `api_url` | `""` | LLM API URL (TOC/메타데이터 추출용) |
| `enrichment` | `api_key` | `""` | LLM API 인증 키 |
| `enrichment` | `model` | `"model"` | TOC/메타데이터 추출에 사용할 모델명 |
| `enrichment.precheck` | `enabled` | `false` | 사전 토큰 검사 활성화 여부. `true`이면 LLM 호출 전 토큰을 추정하여 초과 시 즉시 에러 반환 |
| `enrichment.precheck` | `max_context_tokens` | `128000` | 모델 전체 컨텍스트 한도 (입력 + 출력 합산) |
| `enrichment.precheck` | `completion_reserved_tokens` | `12000` | 출력용 예약 토큰 수. 실제 허용 입력 = `max_context_tokens` − `completion_reserved_tokens` |
| `enrichment.toc` | `temperature` | `0.0` | TOC 생성 temperature |
| `enrichment.toc` | `top_p` | `0.00001` | TOC 생성 top-p |
| `enrichment.toc` | `seed` | `33` | TOC 생성 seed |
| `enrichment.toc` | `max_tokens` | `10000` | TOC 생성 최대 토큰 수 |
| `enrichment.image_description` | `enabled` | `false` | facade 이미지 설명 enrichment 활성화 여부 |
| `enrichment.image_description` | `api_url` | `""` | 이미지 설명 VLM API URL. 비어 있으면 `enrichment.api_url` 상속 |
| `enrichment.image_description` | `api_key` | `""` | 이미지 설명 VLM API 키. 비어 있으면 `enrichment.api_key` 상속 |
| `enrichment.image_description` | `model` | `"model"` | 이미지 설명에 사용할 모델명. 비어 있으면 `enrichment.model` 상속 |
| `enrichment.image_description` | `concurrency` | `16` | 이미지 설명 VLM 요청 병렬 처리 수 (`ThreadPoolExecutor`의 `max_workers`) |
| `enrichment.image_description` | `before_items` | `3` | 이미지 앞 문맥으로 넣을 텍스트 item 수 |
| `enrichment.image_description` | `after_items` | `2` | 이미지 뒤 문맥으로 넣을 텍스트 item 수 |
| `enrichment.image_description` | `max_context_chars` | `1500` | 프롬프트 전체 최대 문자 수 (초과 시 절단) |
| `enrichment.image_description` | `prompt_template` | 내장 기본 프롬프트 | 이미지 설명 요청 프롬프트 템플릿 (`{before_context}`, `{caption}`, `{after_context}` 치환) |
| `output` | `format` | `"json"` | 응답 포맷. `json` / `html` / `markdown`. 유효하지 않으면 `json`으로 대체 |
| `output` | `table_format` | `"html"` | 표 변환 포맷. `html` / `markdown`. 유효하지 않으면 `html`로 대체 |

### 파싱용 전처리기 최초 등록시 config 수정가이드

아래 설정은 사이트환경에 맞게 수정이 필요합니다.
- ocr
  - endpoint: `<OCR_ENDPOINT>` 는 ocr server 를 서비스 하는 주소로 변경해야 합니다.
- layout.genos_layout
  - endpoint: `<LAYOUT_SERVING_ID>` 는 Genos에 등록한 layout 모델서빙 ID 로 변경해야 합니다.
- enrichment
  - api_url: `<ENRICHMENT_SERVING_ID>`는 Genos에 등록한 enrichment 모델서빙 ID로 변경해야 합니다.
  - image_description.api_url: 별도 VLM endpoint 사용 시 변경해야 합니다. 비워두면 `enrichment.api_url` 상속

---

## 지원 파일 형식 및 전제조건

### 형식별 분류표

| 확장자 | 처리 경로 | 출력 category | 전제조건 |
|--------|-----------|---------------|----------|
| `.pdf`, `.html`, `.htm` | IntelligentDocumentProcessor | 문서 구조 그대로 | Layout 모델 서버, OCR 서버(선택) |
| `.hwp`, `.hwpx` | HwpDocumentLoader | 문서 구조 그대로 | GenosHwpDocumentBackend (HWP SDK) |
| `.docx` | DocxDocumentLoader | 문서 구조 그대로 | GenosMsWordDocumentBackend |
| `.csv`, `.xlsx` | TabularLoader | `table` (HTML) | pandas, openpyxl, chardet |
| `.doc`, `.ppt`, `.pptx`, `.txt`, `.json`, `.md`, `.jpg`, `.jpeg`, `.png` | GenericDocumentLoader (Langchain) | `paragraph` | LibreOffice (`soffice`), unstructured 라이브러리 |

---

### PDF / HTML / HTM

**처리 클래스:** `IntelligentDocumentProcessor`

**전제조건:**
- **Layout 모델 서버:** `parser_processor_config.yaml`의 `layout.genos_layout.endpoint` 로 접근 가능한 vLLM 호환 서버가 실행 중이어야 함
- **OCR 서버:** `ocr_mode`가 `disable`이 아닌 경우 `ocr.ocr_endpoint`에 PaddleOCR 서버가 실행 중이어야 함 (단, `ocr_mode=auto`이면 OCR 품질 감지 후 필요할 때만 접근)
- **Enrichment LLM:** `enrichment.do_toc=true` 또는 `do_metadata=true`인 경우 `enrichment.api_url`에 LLM API가 실행 중이어야 함
- **Image Description VLM(선택):** `enrichment.image_description.enabled=true`인 경우 `image_description.api_url`(또는 상속된 `enrichment.api_url`)에 이미지+텍스트 입력 가능한 VLM API가 실행 중이어야 함
- **Python 패키지:** `docling`, `docling-core`, `pymupdf`

---

### HWP / HWPX

**처리 클래스:** `HwpDocumentLoader`

**전제조건:**
- **GenosHwpDocumentBackend:** HWP 처리용 SDK 백엔드. `use_hwp_sdk=true`(기본값)일 때 사용
- SDK 실패 시 자동으로 `HwpDocumentBackend` / `HwpxDocumentBackend` 폴백 시도
- 모든 백엔드 실패 시 LibreOffice로 PDF 변환 후 PDF 경로로 재처리
- **Python 패키지:** `docling`(hwp 백엔드 포함)

**수식(LaTeX) 추출 (이슈 #195):** 신규 HWP SDK가 수식 추출 기능을 지원합니다. `GenosHwpDocumentBackend`는 SDK가 emit하는 base64 인코딩 LaTeX를 디코드하여 docling의 `DocItemLabel.FORMULA` 노드로 변환하며, 표 셀 HTML 내부에 임베드된 `<latex>` 태그는 셀 텍스트에 `<math>{decoded}</math>`로 인라인 치환합니다 (chandra OCR prompt 컨벤션 정합). 별도 파라미터 없이 기본 동작입니다. 자세한 처리 흐름은 [`attachment_processor.md` §8.2](attachment_processor.md#82-hwpprocessor) 참고.

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
- **Whisper API 서버:** `parser_processor_config.yaml`의 `whisper.url`에 OpenAI Whisper 호환 API가 실행 중이어야 함
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

> 이미지 설명 문맥 추출(`enrichment.image_description.*`)은 현재 `parser_processor_config.yaml` 설정값으로 제어합니다.

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
  "category": "picture",
  "content": "문맥 기반 이미지 설명",
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
| `picture` | 기본은 빈 문자열 `""` (이미지 자체는 별도 파일로 저장). 이미지 설명 활성화 시 `content`에 설명 텍스트가 포함됨 | `"이미지 설명 텍스트"` |
| `caption` | 그림/표 캡션 텍스트 | `"그림 1. 시스템 구조도"` |
| `footnote` | 각주 텍스트 | `"1) 출처: ..."` |
| `page_header` | 페이지 상단 반복 텍스트 | `"2025 연간 보고서"` |
| `page_footer` | 페이지 하단 반복 텍스트 | `"- 3 -"` |
| `code` | 코드 블록 텍스트 | `"def hello(): ..."` |

### `output.format: html` 또는 `markdown`

Docling 경로(PDF, HTML, HWP, HWPX, DOCX)에서 `parser_processor_config.yaml`의 `output.format`을 `html` 또는 `markdown`으로 설정하면, 전체 문서를 하나의 문자열로 직렬화하여 `content`에 담고 `elements`는 `[]`로 반환됩니다.

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
| `picture` | 이미지 | 기본 `content`는 빈 문자열, 옵션 활성화 시 `content`에 설명 텍스트가 포함됨 |
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
| Enrichment 입력 토큰 초과 | `precheck.enabled: true` 상태에서 추정 토큰 수가 `max_context_tokens - completion_reserved_tokens` 초과 | LLM 호출 없이 즉시 `GenosServiceException` 발생. `errMsg`에 JSON 페이로드 포함 (아래 참고) |

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
| Layout 서버 미응답 | Connection refused to `<endpoint>` | `parser_processor_config.yaml`의 `layout.genos_layout.endpoint` 확인 |
| OCR 서버 미응답 | `OCR HTTP 502` | `parser_processor_config.yaml`의 `ocr.ocr_endpoint` 및 서버 상태 확인 |
| Enrichment LLM 오류 | LLM API timeout | `parser_processor_config.yaml`의 `enrichment.api_url` 확인 |
| Enrichment 입력 토큰 초과 | `프롬프트 입력 토큰 (N) 초과 하였습니다. (128000 - reserved 12000).` | 문서 크기를 줄이거나 `precheck.max_context_tokens` 값 조정. 비활성화는 `precheck.enabled: false` |
| HWP SDK 실패 | `HWP SDK 실패: ...` | 로그 확인 후 `use_hwp_sdk=false` 파라미터로 재시도 |
| Whisper 미설정 | Connection error | `parser_processor_config.yaml`의 `whisper.url` 설정 확인 |
| `soffice` 미설치 | `FileNotFoundError: soffice` | LibreOffice 설치 후 PATH 등록 |
| 파일 없음 | `FileNotFoundError` | `file_path` 경로 및 파일 존재 여부 확인 |

#### Enrichment 토큰 초과 에러 페이로드

`precheck.enabled: true` 상태에서 입력 토큰이 허용치를 초과하면, `GenosServiceException.errMsg`에 아래 JSON이 문자열로 담겨 반환됩니다.

```json
{
  "object": "error",
  "message": "프롬프트 입력 토큰 (169000) 초과 하였습니다. (128000 - reserved 12000).",
  "type": "BadRequestError",
  "param": "prompt",
  "code": 400
}
```

> 이 에러는 LLM을 호출하지 않고 로컬에서 생성됩니다. DB 적재 상태란에는 위 JSON 문자열이 에러 메시지로 저장됩니다.

---

## 사용 예시

### 헬스체크 (curl)

```bash
curl http://<base_url:port>/preprocessor/{id}/healthcheck \
  -H "Authorization: Bearer <key>"
```

### PDF 파싱 (curl)

```bash
curl -X POST http://<base_url:port>/preprocessor/{id}/run \
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

BASE_URL = "http://<base_url:port>"
PREPROCESSOR_ID = "{id}"
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

### HTML 포맷으로 출력 (`parser_processor_config.yaml: output.format: html`)

```python
# parser_processor_config.yaml에 output.format: html 설정 후
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
