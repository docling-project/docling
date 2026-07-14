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
- 출력 형식은 `parser_processor_config.yaml`의 `output.format`으로 제어 (`json` / `html` / `markdown` / `docling`)
- `docling` 포맷은 복원 가능한 DoclingDocument 원본 JSON을 반환하며, Chunk API(`/chunker`)의 입력으로 사용

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
# 공통 기본값 (로깅 레벨 등)
# ───────────────────────────────────────────────
defaults:
  # 5=DEBUG, 4=INFO, 3=WARNING, 2=ERROR, 1=CRITICAL, 0=NOLOG
  log_level: 4

# ───────────────────────────────────────────────
# 포맷별 처리 옵션 (xlsx/csv — "지원 파일 형식 > CSV / XLSX" 참고)
# ───────────────────────────────────────────────
formats:
  xlsx:
    processing_mode: "tabular"    # "tabular"(default) | "docling"
    tabular:                      # tabular 모드에서만 사용
      header_row: 0               # 0=자동판정 | >0=단일헤더 강제
      multi_table: false          # true 면 1시트 복수표(빈 행 분리) 분리

# ───────────────────────────────────────────────
# OCR 설정 (PDF 파이프라인에서 사용)
# ───────────────────────────────────────────────
ocr:
  # OCR 모드: auto | force | disable
  #   auto    — 글리프 깨짐 등 휴리스틱 감지 후 필요 시 재OCR 수행 (기본값)
  #   force   — 항상 OCR 수행
  #   disable — OCR 수행 안 함
  ocr_mode: "auto"

  # OCR 엔진 선택: paddle | upstage
  # docker image 재빌드 없이 yaml 만으로 전환 가능
  engine: "paddle"

  # 글리프 깨진 테이블 셀 재OCR 시 HTTP timeout(초)
  table_cell_ocr_timeout: 60

  # engine: "paddle" 일 때만 사용
  paddle:
    # <OCR_ENDPOINT>: PaddleOCR 서버 주소로 변경 필요
    ocr_endpoint: "http://<OCR_ENDPOINT>/ocr"
    text_score: 0.3

  # 글리프 깨짐 기반 auto-OCR 재트리거 임계값
  glyph_detection:
    table_cell_threshold: 1   # 셀 GLYPH 토큰 N개 이상이면 재OCR
    document_threshold: 10    # 문서 GLYPH 토큰 N개 초과면 OCR 경로 재시도

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
    page_batch_size: 128
    max_completion_tokens: 16384
    model: "dots-mocr"          # 서빙 모델명
    timeout: 3600               # VLM 요청 타임아웃(초)
    retry_count: 2              # 비정상 VLM 응답 재시도 횟수
    temperature: 0.1            # 생성 temperature
    top_p: 0.9                  # 생성 top_p (0<top_p<=1)
    repetition_penalty: 1.15    # >1.0, 토큰 반복(degeneration) 억제 (아래 가이드 참조)

# ───────────────────────────────────────────────
# PDF 파이프라인 (docling PDF 파싱 성능/품질 노브)
# ───────────────────────────────────────────────
pdf_pipeline:
  num_threads: 8                     # accelerator 스레드 수
  device: "auto"                     # auto(기본값) | cpu | cuda | mps
  images_scale: 2                    # 페이지/그림 이미지 렌더 배율
  generate_page_images: true
  generate_picture_images: true      # false 면 이미지 설명 enrichment 비활성화됨
  table_structure_mode: "accurate"   # accurate(기본값) | fast

# ───────────────────────────────────────────────
# Enrichment (TOC 추출 / 메타데이터 추출 / 이미지 설명 / 커스텀 필드)
# 각 항목은 {이름: {옵션}} 형식의 list 입니다.
#   이름 ∈ {toc, metadata, image_description, custom_fields}
#   비활성화: ① 항목 삭제  ② 항목 주석 처리  ③ enable: false
#   모든 url 의 <ENRICHMENT_SERVING_ID> / <IMAGE_DESCRIPTION_SERVING_ID> 는
#   Genos에 등록한 모델서빙 ID로 변경 필요. api_key 는 k8s 내부 통신 시 불필요.
#   프롬프트는 별도 .md 파일로 분리합니다(아래 "프롬프트 파일 분리 & 변수 치환" 참고).
#   *_file 경로는 이 config 파일과 같은 디렉토리 기준이며, 파일명만 적습니다.
# ───────────────────────────────────────────────
enrichment:
  - toc:
      enable: true
      url: "http://llmops-gateway-api-service:8080/rep/serving/<ENRICHMENT_SERVING_ID>/v1/chat/completions"
      api_key: ""
      model: "model"
      temperature: 0.0
      top_p: 0.00001
      seed: 33
      max_tokens: 10000
      precheck:
        # true 로 설정하면 LLM 호출 전 입력 토큰을 추정하여 초과 시 즉시 400 에러 반환
        enabled: true
        max_context_tokens: 128000        # 모델 전체 컨텍스트 한도 (입력 + 출력 합산)
        completion_reserved_tokens: 12000 # 출력용 예약 토큰 수
      system_prompt_file: prompt_toc_default_system.md
      user_prompt_file: prompt_toc_default_user.md   # 파일 안에서 {{raw_text}} 치환

  - metadata:
      enable: true
      url: "http://llmops-gateway-api-service:8080/rep/serving/<ENRICHMENT_SERVING_ID>/v1/chat/completions"
      api_key: ""
      model: "model"
      max_tokens: 10000
      temperature: 0.0
      timeout: 3600
      pages: [1, 4]                       # null/미지정 시 첫 4페이지
      # output_fields: 프롬프트의 JSON 키와 일치해야 함
      output_fields: [created_date, authors]
      parser:
        type: json                        # json(기본값) | python
      # field_transforms: 추출 키 → 벡터 메타 필드/타입 변환 (선택)
      #   미지정 시 created_date(date_int / doc_text_scan) 기본 변환 적용
      field_transforms:
        - source: [created_date, 작성일]
          target: created_date
          type: date_int                  # YYYYMMDD 정수 변환
          fallback: doc_text_scan         # 비었을 때 본문 휴리스틱 추출
      system_prompt_file: prompt_metadata_default_system.md
      user_prompt_file: prompt_metadata_default_user.md   # 파일 안에서 {{raw_text}} 치환
      precheck:
        enabled: true
        max_context_tokens: 128000
        completion_reserved_tokens: 12000

  # facade 후처리 기반 이미지 설명 생성 (문맥 포함)
  - image_description:
      enable: true
      # <IMAGE_DESCRIPTION_SERVING_ID>: 별도 VLM 서빙 ID
      url: "http://llmops-gateway-api-service:8080/rep/serving/<IMAGE_DESCRIPTION_SERVING_ID>/v1/chat/completions"
      api_key: ""
      model: "model"
      concurrency: 16        # 이미지 설명 요청 병렬 수
      before_items: 3
      after_items: 2
      max_context_chars: 1500
      # 파일 안에서 {{before_context}} / {{caption}} / {{after_context}} / {{doc_summary}} 치환
      prompt_template_file: prompt_image_description_default.md
      # 본문요약: 이미지·차트 description 공통 컨텍스트({{doc_summary}} 로 주입)
      doc_summary:
        enable: false
        prompt_file: prompt_doc_summary.md
        max_chars: 6000
      # 차트 처리
      chart:
        enable: false          # true 면 차트 처리 수행(아니면 일반 image description)
        detection: auto         # auto=docling 자동판별(차트만 차트 프롬프트) | all=모든 이미지를 차트로
        chart_prompt_file: prompt_chart_description_default.md

  # 커스텀 필드 추출 (여러 개 지정 가능). 인라인 옵션 또는 외부 config_file 사용.
  # - custom_fields:
  #     enable: true
  #     url: "http://llmops-gateway-api-service:8080/rep/serving/<ENRICHMENT_SERVING_ID>/v1/chat/completions"
  #     api_key: ""
  #     model: "model"
  #     output_fields: [doc_category, keywords]
  #     system_prompt_file: prompt_custom_fields_xxx_system.md  # 생략 시 built-in default
  #     user_prompt_file: prompt_custom_fields_xxx_user.md      # 파일 안에서 {{raw_text}} 치환
  #     # 또는 외부 config 파일로 분리:
  #     # config_file: "custom_fields_xxx.yaml"

# ───────────────────────────────────────────────
# 출력 형식
# ───────────────────────────────────────────────
output:
  # 출력 형식: json | html | markdown | docling
  #   json     — element 목록 형태 (기본값)
  #   html     — 전체 문서를 HTML 문자열로 반환 (content 필드)
  #   markdown — 전체 문서를 Markdown 문자열로 반환 (content 필드)
  #   docling  — 복원 가능한 DoclingDocument 원본 JSON 을 data.document 로 반환 (Chunk API 입력용)
  format: "json"
  # 테이블 표현 형식: html | markdown (json, markdown 포맷의 테이블에 적용; docling 포맷에서는 무시)
  table_format: "html"

# ───────────────────────────────────────────────
# Whisper 설정 (오디오 음성인식이 필요한 경우에만 설정)
# ───────────────────────────────────────────────
whisper:
  url: ""
  model: "model"
  language: "ko"
  response_format: "json"
  temperature: "0"
  stream: "false"
  timestamp_granularities: "word"
  chunk_sec: 29
  chunk_overlap_ms: 300  # 청크 간 겹침(ms)
```

> **호환성 안내:** `enrichment`는 위와 같은 **list 형식(권장)** 외에 구버전 **dict 형식**(`enrichment: {do_toc, do_metadata, api_url, toc, image_description, ...}`)도 그대로 수용됩니다. 신규 등록 시에는 list 형식을 사용해 주세요.

### 설정 항목 상세

**OCR / 레이아웃 / PDF 파이프라인**

| 섹션 | 키 | 기본값 | 설명 |
|------|----|--------|------|
| `defaults` | `log_level` | `4` (INFO) | 로깅 레벨. `5`=DEBUG / `4`=INFO / `3`=WARNING / `2`=ERROR / `1`=CRITICAL / `0`=NOLOG(전체 비활성화). 누락/오류 시 4 폴백. 실행 시 `params.log_level` 이 주어지면 그 값이 우선 |
| `ocr` | `ocr_mode` | `"auto"` | OCR 수행 정책. `auto`(휴리스틱 감지 후 필요 시 재OCR) / `force` / `disable`. 유효하지 않으면 `auto` |
| `ocr` | `engine` | `"paddle"` | OCR 엔진 선택. `paddle` / `upstage` (유효하지 않으면 `paddle`) |
| `ocr` | `table_cell_ocr_timeout` | `60` | 글리프 깨진 테이블 셀 재OCR 시 HTTP 타임아웃(초). 양의 정수, 유효하지 않으면 60 |
| `ocr.paddle` | `ocr_endpoint` | `""` | PaddleOCR 서버 URL (engine=paddle 일 때만 사용). 구버전 `ocr.ocr_endpoint`(상위) 위치도 호환 인식 |
| `ocr.paddle` | `text_score` | `0.3` | PaddleOCR word confidence 필터링 임계값 |
| `ocr.glyph_detection` | `table_cell_threshold` | `1` | 셀 GLYPH 토큰 N개 이상이면 셀 재OCR. 양의 정수, 유효하지 않으면 1 |
| `ocr.glyph_detection` | `document_threshold` | `10` | 문서 GLYPH 토큰 N개 초과면 OCR 경로 재시도. 양의 정수, 유효하지 않으면 10 |
| `ocr.upstage` | `api_endpoint` | `"https://api.upstage.ai/v1/document-digitization"` | Upstage OCR API URL |
| `ocr.upstage` | `api_key` | `""` | Upstage API 키. 비어있으면 `UPSTAGE_API_KEY` 환경변수 사용 |
| `ocr.upstage` | `model` | `"ocr"` | Upstage 모델명 |
| `ocr.upstage` | `timeout` | `60` | HTTP 타임아웃 (초) |
| `ocr.upstage` | `text_score` | `0.5` | word confidence 필터링 임계값 |
| `layout` | `layout_model_type` | `"genos_layout"` | 레이아웃 모델 선택. `genos_layout` / `docling_layout` (유효하지 않으면 `genos_layout`) |
| `layout.genos_layout` | `endpoint` | `""` | Genos Layout API URL |
| `layout.genos_layout` | `api_key` | `""` | API 인증 키 |
| `layout.genos_layout` | `page_batch_size` | `128` | 배치당 처리 페이지 수. 양의 정수, 유효하지 않으면 128 대체 |
| `layout.genos_layout` | `max_completion_tokens` | `16384` | Layout LLM 최대 생성 토큰. 양의 정수, 유효하지 않거나 0 이하이면 16384로 대체 |
| `layout.genos_layout` | `model` | `"dots-mocr"` | 서빙 모델명. 비어있으면 `dots-mocr` 폴백 |
| `layout.genos_layout` | `timeout` | `3600` | VLM 요청 HTTP 타임아웃(초). 유효하지 않거나 0 이하이면 3600 폴백 |
| `layout.genos_layout` | `retry_count` | `2` | 비정상(스키마 불일치 등) VLM 응답 재시도 횟수. 음수/오류 시 2 폴백 |
| `layout.genos_layout` | `temperature` | `0.1` | 생성 샘플링 temperature. 음수/오류 시 0.1 폴백 |
| `layout.genos_layout` | `top_p` | `0.9` | nucleus 샘플링 top_p. `0<top_p<=1` 아니면 0.9 폴백 |
| `layout.genos_layout` | `repetition_penalty` | `1.15` | 토큰 반복(degeneration) 억제. `>0`, 유효하지 않으면 1.15 폴백. **자세한 사용은 아래 가이드 참조** |
| `pdf_pipeline` | `num_threads` | `8` | accelerator 스레드 수. 양의 정수, 유효하지 않으면 8 |
| `pdf_pipeline` | `device` | `"auto"` | accelerator 디바이스. `auto` / `cpu` / `cuda` / `mps` (유효하지 않으면 `auto`) |
| `pdf_pipeline` | `images_scale` | `2` | 페이지/그림 이미지 렌더 배율. 양의 정수, 유효하지 않으면 2 |
| `pdf_pipeline` | `generate_page_images` | `true` | 페이지 이미지 생성 여부 |
| `pdf_pipeline` | `generate_picture_images` | `true` | 그림 이미지 생성 여부. `false`면 이미지 설명 enrichment 비활성화 |
| `pdf_pipeline` | `table_structure_mode` | `"accurate"` | 표 구조 인식 모드. `accurate` / `fast` (유효하지 않으면 `accurate`) |

#### `repetition_penalty` 사용 가이드

**무엇을 막는가** — DotsOCR(VLM)이 OCR 도중 특정 토큰·구절에 갇혀, 같은 내용을 `max_completion_tokens`(기본 16384)에 도달할 때까지 끝없이 반복 생성하는 현상(*degeneration*, 반복 붕괴)을 억제합니다. 증상은 DEBUG 로그(`defaults.log_level: 5`)의 `dotsocr raw response (page=N, ...)` 출력에서 한 `text` 필드가 `"휴식, 휴식, 휴식, ..."`처럼 무한 반복되는 형태로 나타납니다. 이 반복이 **유효한 JSON 문자열 안**에 들어가면 파싱이 성공해 `retry_count` 재시도로도 걸러지지 않으므로, 생성 단계에서 억제하는 `repetition_penalty`가 1차 방어선입니다.

**동작 원리** — 이미 생성된 토큰이 다시 나올 확률(logit)을 나눠 낮춥니다. `1.0`이면 페널티 없음(원본 모델 그대로), `1.0`보다 클수록 반복 억제가 강해집니다.

| 값 | 효과 | 사용 상황 |
|----|------|-----------|
| `1.0` | 억제 없음 | dots.ocr 원본 동작. 반복이 없더라도 비권장 |
| `1.05`~`1.1` | 약한 억제 | 가끔 반복이 보일 때 |
| **`1.15` (기본)** | 표준 억제 | 대부분의 문서에 권장 |
| `1.2`~`1.3` | 강한 억제 | 표/반복 어절이 많아 반복이 끈질긴 문서 |
| `>1.3` | 과도 | 정상 텍스트까지 변형·누락될 위험, 비권장 |

**튜닝 절차**
1. 반복이 보이는 문서로 DEBUG 로그(`defaults.log_level: 5`)를 켜고 실행합니다.
2. `dotsocr raw response`에 무한 반복이 남아 있으면 `repetition_penalty`를 `0.05`씩 상향합니다 (1.15 → 1.2 → 1.25).
3. 반대로 정상 텍스트에서 글자 누락·치환이 생기면 값을 하향합니다.

**주의**
- 한국어처럼 조사·어절이 자연스럽게 반복되는 문서는 과한 값(`>1.3`)에서 정상 반복까지 깎여 품질이 떨어질 수 있습니다.
- `temperature`(기본 0.1)를 약간 올리면(예: 0.2~0.3) 결정성이 낮아져 반복 탈출에 도움이 되기도 하지만 OCR 정확도와 trade-off이므로, `repetition_penalty`를 우선 조정하세요.
- 동일 항목이 여러 번 나오는 표는 과한 penalty가 실제 반복 데이터를 누락시킬 수 있으니 문서별로 결과를 검증하세요.

**Enrichment (list 형식)** — 각 항목은 `{이름: {옵션}}`. 이름 ∈ `toc` / `metadata` / `image_description` / `custom_fields`. 각 항목은 `enable: true/false`(미지정 시 `true`)로 켜고 끕니다.

| 항목 | 키 | 기본값 | 설명 |
|------|----|--------|------|
| (공통) | `enable` | `true` | 해당 enricher 활성화 여부 |
| `toc` | `url` | `""` | TOC 추출 LLM API URL |
| `toc` | `api_key` | `""` | LLM API 인증 키. k8s 내부 통신 시 불필요 |
| `toc` | `model` | `"model"` | TOC 추출 모델명 |
| `toc` | `temperature` | `0.0` | TOC 생성 temperature |
| `toc` | `top_p` | `0.00001` | TOC 생성 top-p |
| `toc` | `seed` | `33` | TOC 생성 seed |
| `toc` | `max_tokens` | `10000` | TOC 생성 최대 토큰 수 |
| `toc` | `system_prompt_file` / `user_prompt_file` | — | TOC 프롬프트 `.md` 파일 경로(config 디렉토리 기준). 권장 방식 |
| `toc` | `system_prompt` / `user_prompt` | — | inline 프롬프트(`*_file` 미지정 시 fallback). `user_prompt` 의 `{{raw_text}}` 치환 |
| `toc.precheck` | `enabled` | (미설정) | 사전 토큰 검사. `true`면 LLM 호출 전 토큰 추정하여 초과 시 즉시 에러 |
| `toc.precheck` | `max_context_tokens` | `128000` | 모델 전체 컨텍스트 한도 (입력 + 출력 합산) |
| `toc.precheck` | `completion_reserved_tokens` | `12000` | 출력용 예약 토큰 수. 허용 입력 = `max_context_tokens` − `completion_reserved_tokens` |
| `toc.split` | `enabled` | `false` | 긴 문서 **분할(Split) TOC 추출**(carry-over refine) 수행 여부(아래 참고) |
| `toc.split` | `pages_per_chunk` / `page_overlap` | `100` / `1` | 청크당 페이지 수 / 청크 경계 중복 페이지 수 |
| `toc.split` | `carryover_max_tokens` | `1500` | 다음 청크에 주입할 누적 목차(outline) 토큰 상한 |
| `toc` | `repetition_penalty` | — | 토큰 반복(degeneration) 억제(>1.0). 게이트웨이/vLLM 지원 시에만, 미설정 시 미전송 |
| `toc` | `thinking` / `thinking_dialect` | `off` / `standard` | 추론(thinking) 모드 / 방언. 아래 [thinking(추론) 모드](#thinking추론-모드) 참고 |
| `metadata` | `url` | `""` | 메타데이터 추출 LLM API URL |
| `metadata` | `api_key` | `""` | LLM API 인증 키 |
| `metadata` | `model` | `"model"` | 메타데이터 추출 모델명 |
| `metadata` | `max_tokens` | `10000` | 최대 토큰 수 |
| `metadata` | `temperature` | `0.0` | temperature |
| `metadata` | `timeout` | `3600` | HTTP 타임아웃 (초) |
| `metadata` | `pages` | `null` | 추출 대상 페이지 목록. 미지정/빈 값이면 첫 4페이지 |
| `metadata` | `output_fields` | `[]` | 추출할 필드 목록. 프롬프트 JSON 키와 일치해야 함. 비어 있으면 파싱 결과 전체 사용 |
| `metadata` | `system_prompt_file` / `user_prompt_file` | — | 메타데이터 프롬프트 `.md` 파일 경로(권장). `system_prompt_file` 생략 시 built-in default system prompt 사용 |
| `metadata` | `system_prompt` / `user_prompt` | — | inline 프롬프트(`*_file` 미지정 시 fallback). `user_prompt` 의 `{{raw_text}}` 치환 |
| `metadata.parser` | `type` | `"json"` | 파서 종류. `json`(자동 fallback 파싱) / `python`(외부 파일). `python`이면 `file`/`callable` 필요 |
| `metadata.field_transforms` | — | 내장 기본값 | 추출 키 → 벡터 메타 필드/타입 변환 목록(선택). 아래 [메타데이터 enricher](#메타데이터-enricher) 참고 |
| `metadata` | `thinking` / `thinking_dialect` | `off` / `standard` | 추론(thinking) 모드 / 방언. 아래 [thinking(추론) 모드](#thinking추론-모드) 참고 |
| `metadata.precheck` | `enabled` / `max_context_tokens` / `completion_reserved_tokens` | `/ 128000 / 12000` | TOC와 동일 의미 |
| `image_description` | `url` | `""` | 이미지 설명 VLM API URL |
| `image_description` | `api_key` | `""` | 이미지 설명 VLM API 키 |
| `image_description` | `model` | `"model"` | 이미지 설명 모델명 |
| `image_description` | `concurrency` | `16` | 이미지 설명 VLM 요청 병렬 처리 수 |
| `image_description` | `before_items` | `3` | 이미지 앞 문맥으로 넣을 텍스트 item 수 |
| `image_description` | `after_items` | `2` | 이미지 뒤 문맥으로 넣을 텍스트 item 수 |
| `image_description` | `max_context_chars` | `1500` | 프롬프트 전체 최대 문자 수 (초과 시 절단) |
| `image_description` | `prompt_template_file` | — | 프롬프트 템플릿 `.md` 파일 경로(권장). 미지정 시 inline `prompt_template` → 내장 기본 프롬프트 |
| `image_description` | `prompt_template` | 내장 기본 프롬프트 | inline 프롬프트 템플릿(`*_file` 미지정 시 fallback). `{{before_context}}`, `{{caption}}`, `{{after_context}}`, `{{doc_summary}}` 치환 |
| `image_description` | `doc_summary.enable` / `.prompt_file` / `.max_chars` | `false` / `prompt_doc_summary.md` / `6000` | 문서 본문요약 생성(공통 `{{doc_summary}}` 컨텍스트). `{{full_text}}` 치환 |
| `image_description` | `chart.enable` | `false` | 차트 처리 활성화(false 면 일반 image description 만) |
| `image_description` | `chart.detection` | `auto` | `auto`=docling 자동판별(차트로 분류된 이미지만) / `all`=모든 이미지를 차트로 |
| `image_description` | `chart.chart_prompt_file` | `prompt_chart_description_default.md` | 차트 전용 프롬프트 `.md` |
| `custom_fields` | (인라인 옵션 또는 `config_file`) | — | 커스텀 필드 추출 enricher. 여러 개 지정 가능. 아래 [커스텀 필드 enricher](#커스텀-필드-enricher) 참고 |

> `chart.enable: true` 면 변환 단계에서 docling 그림 분류가 자동 활성화됩니다. **런타임 kwargs**(호출 `params`, 0/1)로 오버라이드 가능: `img_desc`→`image_description.enable`, `chart_desc`(별칭 `chart_convert`)→`chart.enable`, `chart_detection`(1=auto/0=all), `doc_summary`→`doc_summary.enable`. `chart_detection=1`(auto) 은 `chart.enable: true` 로 분류가 켜져 있어야 하며 아니면 `all` 로 강등됩니다.

> 이미지 설명 enrichment는 `pdf_pipeline.generate_picture_images: false`인 경우 동작하지 않습니다 (그림 이미지가 생성되지 않으므로).

#### 분할(Split) TOC 추출 — 긴 문서 대응

기본 TOC 추출은 **문서 전체를 한 번에** LLM에 보냅니다. 문서가 길면 서빙 모델의 `max-model-len` 을 초과해
토큰 에러가 날 수 있습니다. `toc.split.enabled: true` 면 문서를 **페이지 단위 청크**로 나눠 순차 추출하고,
앞 청크까지 누적된 목차를 다음 청크 프롬프트에 컨텍스트로 주입(**carry-over refine**)해 계층/번호 일관성을
유지합니다. (parser 는 내부적으로 `IntelligentDocumentProcessor` 의 TOC 경로를 사용하므로 동작은 동일합니다.)

```yaml
- toc:
    enable: true
    # ... url/model/precheck/프롬프트 파일 등 기존 설정
    system_prompt_file: prompt_toc_default_system.md
    user_prompt_file: prompt_toc_default_user.md
    split:
      enabled: false            # true 면 길이와 무관하게 항상 분할 추출
      pages_per_chunk: 100      # 청크당 페이지 수
      page_overlap: 1           # 청크 경계 중복 페이지 수(경계 누락 완화용)
      carryover_max_tokens: 1500
    # repetition_penalty: 1.1   # 반복 억제가 필요할 때만 주석 해제
```

- **OFF(기본)**: 단일 호출. 동작 변화 없음(컨텍스트 초과 시 기존처럼 에러).
- **ON**: 페이지 N개씩 청크화 → 첫 청크는 설정 프롬프트로, 이후 청크는 설정 프롬프트 앞에 누적 목차를 덧붙여
  순차 추출 → 매 스텝 병합(경계 중복 제거·번호 재부여). `<toc>` 블록이 없거나(분석문/절단) 컨텍스트를 초과하는
  청크는 건너뛰어 부분 결과를 보존합니다.
- **통합 프롬프트**: `prompt_toc_default_user.md` 가 `{{prior_toc}}` 자리표시자 + 작업 모드(Operating Mode)로
  첫 추출/이어쓰기를 모두 처리합니다(`<previous_outline>` 비면 전체 추출, 있으면 새 항목만 이어쓰기).
- **주의**: 컨텍스트에 들어가는 문서는 OFF 유지 권장. `page_overlap>0` 은 경계 누락을 줄이나 중복이 일부 남을 수
  있어(중복이 문제면 `0` 권장), 매우 긴 청크는 토큰 소진 시 스킵될 수 있습니다.

#### thinking(추론) 모드

`thinking` / `thinking_dialect` 는 `toc`·`metadata` enricher에 동일하게 지정합니다. 모델별로 추론 토글 키 이름이 달라 `thinking_dialect` 로 흡수합니다.

- `off`(기본) → `chat_template_kwargs` 에 `{"enable_thinking": false}`(hcx 면 `{"skip_reasoning": true}`) 전송 — 추론 비활성화, 빠른 응답.
- `on` → `{"enable_thinking": true}`(hcx 면 `{"force_reasoning": true}`) 전송 — 추론 활성화.
- `auto` → `chat_template_kwargs` 미전송 — 모델 기본 동작에 맡김(기존 동작 보존).
- `thinking_dialect`: HyperCLOVAX-SEED-Think 등 **hcx 계열 서빙은 `hcx`**, Qwen3/GLM/DeepSeek 등 그 외는 `standard`(기본).
- 추론이 켜진 응답에 섞인 `<think>...</think>` 블록은 자동 제거되어 본문만 저장됩니다.

**출력 / Whisper**

| 섹션 | 키 | 기본값 | 설명 |
|------|----|--------|------|
| `output` | `format` | `"json"` | 응답 포맷. `json` / `html` / `markdown` / `docling`. 유효하지 않으면 `json`으로 대체. `docling`은 docling 경로(pdf/html/htm/docx/hwp/hwpx)에서만 `data.document` 생성, 그 외 포맷은 항상 parse-format(`elements`) |
| `output` | `table_format` | `"html"` | 표 변환 포맷. `html` / `markdown`. 유효하지 않으면 `html`로 대체 |
| `whisper` | `url` | `""` | OpenAI Whisper 호환 API URL (오디오 처리 시 필수) |
| `whisper` | `model` | `"model"` | Whisper 모델명 |
| `whisper` | `language` | `"ko"` | 전사 언어 |
| `whisper` | `response_format` | `"json"` | 응답 포맷 |
| `whisper` | `chunk_sec` | `29` | 오디오 분할 단위(초) |
| `whisper` | `chunk_overlap_ms` | `300` | 청크 간 겹침(ms) |

#### 메타데이터 enricher

`metadata` 항목에 커스텀 신호(`system_prompt`/`user_prompt`/`system_prompt_file`/`user_prompt_file`/`output_fields`/`parser`) 중 하나라도 지정되면 YAML 설정 기반 커스텀 `MetadataEnricher`가 동작합니다(`has_custom_metadata`). 아무 신호도 없으면 docling 내장 enricher가 동작합니다(하위 호환). `system_prompt` 를 따로 지정하지 않아도 built-in default system prompt 가 채워지므로, `output_fields` 만 지정해도 커스텀 enricher 가 활성화됩니다. 동작 방식:

- `pages`로 지정한 페이지(미지정 시 첫 4페이지)의 텍스트를 추출하고 `<!-- image -->` 태그를 제거한 뒤 user 프롬프트의 `{{raw_text}}`에 주입하여 LLM을 1회 호출합니다.
- 응답은 `parser.type`(`json` 기본값 / `python`)으로 파싱하며, `output_fields`로 지정한 키만 추출합니다 (비어 있으면 파싱 결과 전체).
- 추출 결과는 docling `KeyValueItem`으로 문서에 저장되고, enrichment 컨텍스트의 `metadata`에도 병합됩니다.
- **`field_transforms`(선택):** 추출 결과의 키를 벡터 메타 필드로 매핑·변환하는 선언형 목록입니다.

`output_fields` 로 추출한 키 이름은 프롬프트가 정하기 나름이고(`created_date`, `작성일`, `doc_date` …) 값도 문자열이라, 검색에 쓰이는 최종 벡터 메타 필드와 이름·타입이 다를 수 있습니다. `field_transforms` 는 **"추출 결과의 어떤 키를 → 어떤 벡터 필드에 → 어떤 변환을 거쳐 넣을지"** 를 YAML 로 선언하는 다리 역할입니다. 프롬프트를 바꿔 추출 키 이름이 달라져도 `source` 만 고치면 되고, 코드 수정 없이 설정만으로 매핑을 바꿀 수 있습니다.

```yaml
metadata:
  field_transforms:
    - source: [created_date, 작성일]   # 추출 결과에서 순서대로 탐색할 후보 키
      target: created_date            # 값을 넣을 벡터 메타 필드 (생략 시 source 첫 키)
      type: date_int                  # 값 변환기: 날짜 텍스트 → YYYYMMDD 정수
      fallback: doc_text_scan         # 추출이 비면 본문에서 보조 추출
```

각 항목(spec)의 필드:

| 필드 | 필수 | 의미 |
|------|------|------|
| `source` | ✅ | 추출 결과에서 찾을 후보 키. 문자열 1개 또는 목록(목록이면 순서대로 탐색해 **첫 비어있지 않은 값** 사용) |
| `target` | 선택 | 값을 넣을 벡터 메타 필드명. 생략 시 `source` 첫 키 |
| `type` | 선택 | 값 변환기. 현재 지원: `date_int` — 날짜 텍스트/정수를 `YYYYMMDD` 정수로 변환(`YYYY-MM-DD`, `YYYY년 MM월 DD일`, `YYYY-MM`(일자 `01`), `YYYY`(`0101`) 인식, 실패 시 `0`). 생략 시 값을 그대로 사용 |
| `fallback` | 선택 | 추출값이 비었을 때 쓸 보조 추출. 현재 지원: `doc_text_scan` — 본문에서 `기준일`/`작성일`/`최초 작성일`/`보고자료` 키워드 주변 날짜를 휴리스틱 탐색 |

**동작 흐름** (각 spec 마다): ① `source` 후보 키를 순서대로 보며 첫 비어있지 않은 값 선택 → ② `type` 변환기 적용 → ③ 값이 비면 `fallback` 으로 본문에서 보조 추출 → ④ 결과를 `target` 에 주입(사용한 키는 passthrough 제외) → ⑤ 변환 대상이 아닌 나머지 추출 키는 그대로 벡터 메타에 통과(passthrough), 중첩 객체는 JSON 문자열로 직렬화.

**입력 → 출력 예시** (위 기본 설정 기준)

```text
추출 결과:  { "created_date": "2025-01-15", "department": "IT" }
→ 벡터 메타: created_date = 20250115 (date_int 변환),  department = "IT" (passthrough)
```

- **키 이름 변경**: 프롬프트가 `doc_date` 로 추출하면 `source: [doc_date]` 로만 바꿔도 `created_date` 가 동일하게 채워집니다.
- **fallback**: 추출이 비고 본문에 `"보고자료 2024-01-15 기준"` 이 있으면 `created_date: 20240115` 로 보강됩니다.
- `field_transforms` 미지정 시 위 `created_date` 기본 변환(`DEFAULT_METADATA_FIELD_TRANSFORMS`)이 적용되어 기존 동작을 보존합니다. 신규 변환기/보조추출은 `field_transforms.py` 의 `VALUE_TRANSFORMS`/`FALLBACK_STRATEGIES` 에 등록하면 YAML 에서 바로 사용할 수 있습니다.

#### 커스텀 필드 enricher

`custom_fields` 항목은 문서 단위 커스텀 메타데이터를 추출하는 enricher로, 여러 개를 지정할 수 있습니다. 각 항목은 독립적으로 LLM을 1회 호출하며, 추출 결과는 enrichment 컨텍스트의 `metadata`에 병합됩니다.

- **인라인 옵션:** `url`, `api_key`, `model`, `system_prompt_file`/`user_prompt_file`(또는 inline `system_prompt`/`user_prompt`), `output_fields`, `parser`, `pages`, `max_tokens`, `temperature`, `timeout`를 항목에 직접 지정합니다. `system_prompt`(파일·inline 모두) 미지정 시 built-in default system prompt 가 사용됩니다.
- **외부 config:** `config_file: "이름.yaml"`로 분리할 수 있습니다. 상대 경로는 config 파일과 동일한 디렉터리(resource_path) 기준으로 해석됩니다. 외부 파일에도 `system_prompt_file`/`user_prompt_file`(또는 `system_prompt`/`user_prompt`, `prompt.system`/`prompt.user`)/`url`/`model`/`output_fields`/`parser`/`pages` 등을 둘 수 있으며, 항목에 직접 지정한 값이 외부 config보다 우선합니다. `*_file` 경로는 외부 config 파일이 위치한 디렉터리 기준으로 해석됩니다.
- `parser.type`은 `json`(기본값) 또는 `python`(외부 파일 위임)을 지원합니다.

#### 프롬프트 파일 분리 & 변수 치환

enrichment 프롬프트는 YAML 안에 inline 으로 박지 않고 **별도 `.md` 파일**로 분리합니다. 운영 시 프롬프트만 교체하기 쉽고, 두 전처리기(parser/intelligent/convert)가 동일 프롬프트를 공유할 때 파일 경로 한 줄만 같게 두면 됩니다.

**프롬프트 파일 지정**

| 키 | 대상 |
|----|------|
| `system_prompt_file` / `user_prompt_file` | toc / metadata / custom_fields |
| `prompt_template_file` | image_description |

- **경로 규칙:** 상대경로는 **해당 config 파일이 위치한 디렉토리** 기준으로 해석합니다(파일명만 적으면 됨). 절대경로도 허용하지만, 상대경로가 base 디렉토리를 벗어나면(`../…`) 거부합니다.
- **우선순위:** `*_file` > inline(`system_prompt`/`user_prompt`/`prompt_template`) > built-in default. system prompt 는 미지정 시 enricher 별 built-in default 가 채워지므로, **`user_prompt_file` 만 지정**해도 동작합니다(system 은 도메인 안에서 거의 고정이고 자주 바뀌는 것은 user prompt 이기 때문).
- 출하 config 는 `prompt_toc_default_system.md` / `prompt_metadata_default_user.md` 등 중립적 파일명을 사용합니다.

**변수 치환(`{{var}}`)**

프롬프트 본문의 `{{변수}}` 는 런타임 값으로 치환됩니다. JSON 예시(`{"key": "value"}`)의 단일 중괄호와 충돌하지 않도록 **이중 중괄호** 로 통일했습니다.

- **escape:** 리터럴 중괄호가 필요하면 4중 중괄호 `{{{{ ... }}}}` 로 적습니다(→ `{{ ... }}` 로 출력). 치환된 값 안에 `{{...}}` 가 들어와도 다시 치환되지 않습니다(1-pass).
- **단일 중괄호 호환:** 과거 표기 `{raw_text}` 도 값이 주입된 변수에 한해 당분간 치환되지만 deprecation 경고가 남습니다. 신규 프롬프트는 `{{raw_text}}` 를 사용하세요.

**(1) reserved 변수 — 1차 변환 결과 `DoclingDocument` 에서 자동 추출**

문서 단위 (toc / metadata / custom_fields 프롬프트):

| 변수 | 의미 | 추출 소스 | 예시 값 |
|------|------|-----------|---------|
| `{{raw_text}}` | enricher 가 추출한 본문 텍스트(프롬프트의 핵심 입력) | metadata: `pages`(미지정 시 첫 4p)의 markdown — `<!-- image -->` 제거 + 맨 앞에 `filename: …` 주입 / custom_fields: 전체 plain text(또는 `pages` markdown) | `filename: 보고서.pdf\n\n# 1장 총칙 …` |
| `{{full_text}}` | 문서 전체 plain text | `document.export_to_text()` | `보고서\n1장 총칙 …` (전문) |
| `{{filename}}` | 원본 파일명 | `document.origin.filename` | `보고서.pdf` |
| `{{mimetype}}` | MIME 타입 | `document.origin.mimetype` | `application/pdf` |
| `{{page_count}}` | 총 페이지 수 | `document.num_pages()` | `12` |
| `{{table_count}}` | 표 개수 | `len(document.tables)` | `3` |
| `{{picture_count}}` | 이미지 개수 | `len(document.pictures)` | `5` |
| `{{section_headers}}` | 섹션 헤더·제목 목록(줄바꿈 구분) | `texts` 중 label ∈ {section_header, title} | `1장 총칙\n2장 정의 …` |

이미지 item 단위 (image_description 프롬프트 — 이미지마다 값이 달라짐):

| 변수 | 의미 | 추출 소스 |
|------|------|-----------|
| `{{before_context}}` | 이미지 앞 문맥 텍스트(`before_items` 개) | 이미지 직전 텍스트 아이템들 |
| `{{after_context}}` | 이미지 뒤 문맥 텍스트(`after_items` 개) | 이미지 직후 텍스트 아이템들 |
| `{{caption}}` | 이미지 캡션 | `PictureItem.caption_text(document)` |
| `{{section_header}}` | 이미지 직전 섹션 헤더 | 이미지 위쪽에서 가장 가까운 section_header/title |
| `{{doc_summary}}` | 문서 본문요약(공통 컨텍스트). `doc_summary.enable: true` 일 때만 채워짐 | 문서 BODY 텍스트 LLM 1회 요약 |

> 값이 없는 reserved 변수는 **빈 문자열**로 치환됩니다. 카운트(`page_count` 등)는 정수지만 문자열로 렌더링됩니다. `raw_text`/`full_text` 는 토큰이 매우 클 수 있으니 보통 둘 중 하나만 사용합니다.

**(2) 사용자 정의 변수 (`variables:`)**

reserved 외에 운영자가 직접 변수를 추가할 수 있습니다. enricher 항목에 `variables:` 블록으로 이름·값을 선언하고 프롬프트에서 `{{이름}}` 으로 참조합니다. 회사명·도메인처럼 프롬프트에 고정 주입할 값에 유용합니다.

```yaml
# config (예: metadata 항목)
- metadata:
    user_prompt_file: prompt_metadata_default_user.md
    variables:
      company_name: "GenON"
      domain: "금융"
```

```text
<!-- prompt_metadata_default_user.md 본문 -->
당신은 {{company_name}} 의 {{domain}} 문서 분석가입니다. (총 {{page_count}}페이지)
아래 문서에서 작성일·작성자를 추출하세요.

문서:
---
{{raw_text}}
---
```

- `variables` 의 이름은 strict 검증에서 reserved 와 함께 **허용 목록**에 포함됩니다.
- 이름이 reserved 와 겹치면 `variables` 값이 **우선**합니다(override).

**(3) 미정의 변수 처리 (`template.mode`)**

| mode | 동작 |
|------|------|
| `strict` (기본) | 프롬프트에 (reserved ∪ user-defined)에 없는 `{{foo}}` 가 있으면 **config 로드 시점에 에러**로 즉시 실패(오타·누락 조기 발견) |
| `lenient` | 미정의 변수를 빈 문자열로 치환하고 경고 로그만 남김 |

```yaml
- metadata:
    user_prompt_file: prompt_metadata_default_user.md
    template:
      mode: strict   # strict(기본) | lenient
```

**(4) 워크플로우 예시 — 입력 문서 → 렌더링된 프롬프트**

`보고서.pdf`(12페이지, 표 3개)를 metadata enricher 로 처리:

```yaml
# config
- metadata:
    user_prompt_file: prompt_metadata_default_user.md
    variables: { company_name: "GenON" }
```
```text
# prompt_metadata_default_user.md 본문
{{company_name}} 문서 분석 — 파일 {{filename}} ({{page_count}}p, 표 {{table_count}}개)
---
{{raw_text}}
---
```
→ 런타임 치환 후 실제 LLM 에 전달되는 user 메시지:
```text
GenON 문서 분석 — 파일 보고서.pdf (12p, 표 3개)
---
filename: 보고서.pdf

# 1장 총칙 …
---
```

> TOC 프롬프트는 docling 레이어에서 처리되어 `{{raw_text}}` 만 지원하며, 위 reserved 카탈로그·`variables`·`template.mode` 는 facade enricher(metadata / custom_fields / image_description)에 적용됩니다.

### 파싱용 전처리기 최초 등록시 config 수정가이드

아래 설정은 사이트환경에 맞게 수정이 필요합니다.
- `ocr.paddle.ocr_endpoint`
  - `<OCR_ENDPOINT>` 는 PaddleOCR server 를 서비스 하는 주소로 변경해야 합니다.
- `layout.genos_layout.endpoint`
  - `<LAYOUT_SERVING_ID>` 는 Genos에 등록한 layout 모델서빙 ID 로 변경해야 합니다.
- `enrichment` (list 형식)
  - `toc.url` / `metadata.url`: `<ENRICHMENT_SERVING_ID>` 는 Genos에 등록한 enrichment 모델서빙 ID로 변경해야 합니다.
  - `image_description.url`: `<IMAGE_DESCRIPTION_SERVING_ID>` 는 별도 VLM 모델서빙 ID로 변경해야 합니다.
- `whisper.url`
  - 오디오 처리가 필요한 경우 OpenAI Whisper 호환 API 주소로 설정해야 합니다.

### 민감정보 분류/마스킹 (개인정보 비식별화, `guardrail_masking`)

파서 전처리기는 이 기능의 **대상이 아닙니다.** 민감정보 분류/마스킹은 청킹 후 각 청크에서
`quote_origin` 을 매칭해 라벨을 붙이고 치환하는 후처리인데, 파서는 **파스 전용(청킹·벡터 없음)** 이라
매칭할 청크가 없기 때문입니다. 따라서 파서 config 에는 `guardrail_masking` 블록이 없습니다.

- 파서 결과(JSON/docling)를 **chunking API 로 넘기면 chunking 전처리기가 문서 전체를 1회 분류하고
  청크별로 `content_category` 라벨을 부착**합니다(옵션으로 마스킹 치환).
- 파서 단계에서 직접 문서를 적재(청크 생성)하려면 청크를 만드는 전처리기(intelligent/attachment/convert)를 사용하세요.

> 요청/응답 형식·매칭 규칙·출력 필드(`content_category`)의 **상세 설명은 지능형 전처리기 매뉴얼의
> "민감정보 분류/마스킹" 절**을 참고하세요.

---

## 지원 파일 형식 및 전제조건

### 형식별 분류표

| 확장자 | 처리 경로 | 출력 category | 전제조건 |
|--------|-----------|---------------|----------|
| `.pdf`, `.html`, `.htm` | IntelligentDocumentProcessor | 문서 구조 그대로 | Layout 모델 서버, OCR 서버(선택) |
| `.hwp`, `.hwpx` | HwpDocumentLoader | 문서 구조 그대로 | GenosHwpDocumentBackend (HWP SDK) |
| `.docx` | DocxDocumentLoader | 문서 구조 그대로 | GenosMsWordDocumentBackend |
| `.csv`, `.xlsx`, `.xlsm` | `load_tables`(openpyxl 병합셀) / `formats.xlsx.processing_mode` (`tabular` 기본 \| `docling`) | `table` (HTML) | openpyxl, chardet |
| `.ppt`, `.pptx` | PDF→경량 docling 재라우팅 + `formats.ppt.page_description`(페이지 설명) | 문서 구조 + 페이지 설명 element | LibreOffice (`soffice`), VLM 서빙(선택) |
| `.doc`, `.txt`, `.json`, `.md`, `.jpg`, `.jpeg`, `.png` | GenericDocumentLoader (Langchain) | `paragraph` | LibreOffice (`soffice`), unstructured 라이브러리 |

#### PPT 페이지 설명 (`formats.ppt.page_description`)

`.ppt`/`.pptx` 는 **PDF→경량 docling** 으로 재라우팅해 파싱하며, `formats.ppt.page_description.enable: true` 이면 각 페이지를 **이미지로 렌더링해 VLM 으로 설명**하고 그 텍스트를 **페이지별 element(TextItem)로 주입**합니다(파스 출력 `data.elements`/`data.document` 에 그대로 포함). 파서는 **파스 전용**이라 청킹은 수행하지 않습니다. PDF 변환이 불가하면 레거시 langchain 경로로 폴백합니다. 페이지 native text 는 프롬프트의 **`{{page_text}}`** 변수로 전달됩니다.

| 키 | 의미 | 기본값 |
|----|------|--------|
| `enable` | PPT 페이지 설명 활성화 | `false` |
| `url` / `api_key` / `model` | VLM 서빙 endpoint / 키 / 모델명 | `<PAGE_DESCRIPTION_SERVING_ID>` / `""` / `model` |
| `timeout` / `concurrency` | VLM 타임아웃(초) / 병렬 요청 수 | `360` / `16` |
| `images_scale` / `max_image_side` / `max_tokens` / `params` | 렌더 배율 / 이미지 최대 변(px) / 출력 토큰 상한 / 추가 VLM 파라미터 | `2.0` / `0` / `0` / `{}` |
| `prompt_template_file` | 프롬프트 `.md` 경로. `{{page_text}}` 치환 | `prompt_page_image_description_default.md` |

> **주의** — 페이지 설명은 **파싱 단계 전용**입니다. Chunk API(`/chunker`)는 파서 출력(JSON)만 받으며 렌더된 페이지 이미지가 없어 페이지 설명을 수행하지 않습니다.

> **청킹(Chunk API) 연계**: docling 경로(`.pdf/.html/.htm/.docx/.hwp/.hwpx`, `.ppt/.pptx`)는 `output.format: docling`
> 일 때 `data.document` 를 만들어 구조 인식 청킹(GenosSmartChunker)에 쓰인다. 그 외 포맷은 항상
> parse-format(`data.elements`)으로 반환되며, chunker 가 이를 공통 청킹한다 — 오디오는 `[AUDIO]`
> 단일 벡터, csv/xlsx(`table`)는 `[DA]` 단일 벡터, 그 외 텍스트(`paragraph`)는 문자 기반 splitter
> (`chunking.generic`). 즉 비-docling 포맷도 추가 변환 없이 그대로 청킹 가능하다.

---

### PDF / HTML / HTM

**처리 클래스:** `IntelligentDocumentProcessor`

**전제조건:**
- **Layout 모델 서버:** `parser_processor_config.yaml`의 `layout.genos_layout.endpoint` 로 접근 가능한 vLLM 호환 서버가 실행 중이어야 함
- **OCR 서버:** `ocr_mode`가 `disable`이 아닌 경우 `ocr.paddle.ocr_endpoint`에 PaddleOCR 서버가 실행 중이어야 함 (단, `ocr_mode=auto`이면 글리프 깨짐 등 휴리스틱 감지 후 필요할 때만 접근)
- **Enrichment LLM:** `enrichment`의 `toc` 또는 `metadata` 항목이 활성화된 경우 해당 항목의 `url`에 LLM API가 실행 중이어야 함
- **Image Description VLM(선택):** `enrichment`의 `image_description` 항목이 활성화된 경우 해당 항목의 `url`에 이미지+텍스트 입력 가능한 VLM API가 실행 중이어야 함
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

**처리 함수:** `xlsx_processor.load_tables` (converters) — 출력은 시트/표별 HTML `table` element (parse-JSON).

`.xlsx`/`.xlsm`/`.csv` 는 PDF 변환 없이 직접 처리하며, `formats.xlsx.processing_mode` 로 방식을 선택합니다.

- **tabular (기본)** — openpyxl 로 병합셀을 **unmerge + forward-fill** 하여 병합 헤더 유실을 방지하고, 시트(또는 표)별로 HTML `table` element 를 생성합니다.
  - **멀티헤더 자동 판정**: 전열 병합 제목행은 표 상단 컨텍스트로만 두고, 부분 병합 계층 헤더는 `상위_하위` 로 flatten 하여 컬럼 헤더(`<th>`)로 사용합니다.
  - **시트명·제목 컨텍스트**: 각 표 `content` 앞에 `시트명: <시트명>`(+ 제목행이 있으면 제목)을 붙입니다.
  - `formats.xlsx.tabular.multi_table: true` 면 한 시트의 복수 표(빈 행 분리)를 표별 element 로 분리합니다. `header_row`(0=자동/>0=단일헤더 강제)로 헤더 판정을 제어합니다.
- **docling** — docling MsExcel/Csv 백엔드로 `DoclingDocument` 를 만든 뒤 parse-JSON 으로 직렬화합니다(`output.format: docling` 이면 원본 DoclingDocument JSON 반환).

**전제조건:**
- **Python 패키지:** `openpyxl`(xlsx 병합셀 처리), `chardet`(csv 인코딩 자동 감지)
- CSV 는 chardet 로 인코딩 자동 감지 후 utf-8 폴백
- XLSX 는 시트(및 `multi_table` 시 표) 단위로 element 생성

> 참고: parser 출력은 parse-JSON(HTML 표)이라 벡터 메타가 없습니다. 컬럼 단위 필터용 stable-key/`column_map`
> 는 적재용(intelligent/convert)의 tabular 벡터에만 적용됩니다. parser 의 HTML `<th>` 는 원본 헤더명(한글 포함)을 그대로 사용합니다.

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
| `log_level` | `int` | `defaults.log_level` (없으면 `4`) | 전체 | 로그 레벨. `5`=DEBUG, `4`=INFO, `3`=WARNING, `2`=ERROR, `1`=CRITICAL, `0`=비활성화. 미전달 시 config `defaults.log_level` 값을 사용 |

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

### `output.format: docling`

`output.format`을 `docling`으로 설정하면, 파싱된 **DoclingDocument 원본을 손실 없이 직렬화한 JSON**(`DoclingDocument.model_dump(mode="json")`)을 `data.document`에 담아 반환합니다. `content`/`elements`는 비어 있습니다(`""` / `[]`).

```json
{
  "document": { "schema_name": "DoclingDocument", "texts": [...], "tables": [...], "pictures": [...], "body": {...}, "..." : "..." },
  "usage": {"pages": 10}
}
```

- 이 JSON은 `DoclingDocument.model_validate(data["document"])`로 **무손실 복원**할 수 있어, GenOS Temporal 파이프라인의 다음 단계인 **Chunk API(`/chunker`)의 입력**으로 그대로 전달합니다.
- `output.table_format`은 `docling` 포맷에서는 무시됩니다(표 구조가 원본 그대로 보존되므로).

#### 연계 — Chunk API (`POST /preprocessor/{id}/chunker`)

파싱 결과를 입력받아 청킹만 수행하고 청크 목록(`GenOSVectorMeta`)을 반환하는 별도 전처리기(`chunking_processor`)가 처리합니다. 해당 전처리기는 `IS_CHUNKER` 마커로 식별되며, 요청 본문은 다음과 같습니다.

```json
{
  "file_path": "/data/sample.pdf",
  "params": { "document": { /* 위 data.document 그대로 */ }, "chunk_size": 0 }
}
```

`params.document` 에는 docling JSON(`data.document`)뿐 아니라 비-docling 포맷의 parse-format
응답(`{"elements":[...]}` = `data` 전체)도 그대로 넣을 수 있습니다. chunker 가 입력 형태를 자동
판별해, docling 은 구조 인식 청킹(GenosSmartChunker)을, parse-format 은 공통 청킹(audio→`[AUDIO]`
단일, csv/xlsx→`[DA]` 단일, 그 외 텍스트→`chunking.generic` splitter)을 수행합니다.

E2E 흐름:
- docling 포맷: `POST /parser (output.format=docling)` → `data.document` → `POST /chunker (params.document=…)` → 청크 목록.
- 비-docling 포맷: `POST /parser` → `data.elements`(parse-format) → `POST /chunker (params.document=data)` → 청크 목록.

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

시트(또는 표) 데이터가 **시트명 접두 + HTML 테이블** 형태로 `content`에 저장됩니다. 병합셀은 unmerge + forward-fill 되어 병합 헤더가 보존됩니다.

```text
시트명: Sheet1
<table><tr><th>컬럼1</th><th>컬럼2</th></tr><tr><td>값1</td><td>값2</td></tr></table>
```

- 상단에 제목행(전열 병합)이 있으면 시트명 다음 줄에 제목이 컨텍스트로 함께 포함됩니다.
- 계층(부분 병합) 헤더는 `상위_하위` 로 flatten 되어 `<th>` 로 렌더됩니다.
- 기본은 시트 단위로 element 가 생성되며 `page` 는 element 순번(1-based). `formats.xlsx.tabular.multi_table: true` 면 **한 시트에서 표별로 여러 element** 가 생성됩니다.

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
| OCR 서버 연결 오류 | `ocr.paddle.ocr_endpoint` 미응답 | `ocr_all_table_cells`에서 예외를 캐치하고 OCR 없이 진행 |
| Enrichment LLM 연결 오류 | `enrichment` 항목의 `url` 미응답 | enrichment 단계에서 예외 발생 |
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
| openpyxl 로드 실패 | 파일 손상, 비-xlsx(예: `.xls`) | 예외 발생(구형 `.xls`/`.xlsb` 는 미지원) |
| CSV 인코딩 오류 | chardet 감지 실패 | utf-8 로 폴백(`errors="ignore"`) |

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
| OCR 서버 미응답 | `OCR HTTP 502` | `parser_processor_config.yaml`의 `ocr.paddle.ocr_endpoint` 및 서버 상태 확인 |
| Enrichment LLM 오류 | LLM API timeout | `parser_processor_config.yaml`의 `enrichment` 항목 `url` 확인 |
| Enrichment 입력 토큰 초과 | `프롬프트 입력 토큰 (N) 초과 하였습니다. (128000 - reserved 12000).` | 문서 크기를 줄이거나 해당 항목 `precheck.max_context_tokens` 값 조정. 비활성화는 `precheck.enabled: false` |
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
