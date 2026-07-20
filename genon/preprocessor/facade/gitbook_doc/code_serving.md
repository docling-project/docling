# 코드 서빙(Code Serving) 사용자 매뉴얼

doc_parser 전처리기를 GenOS **코드 서빙** 플랫폼에 배포하면, 게이트웨이를 통해
문서 **파싱(`/parser`)** 과 **청킹(`/chunker`)** 기능을 HTTP API로 호출할 수 있습니다.
이 문서는 배포된 코드 서빙을 호출하는 방법을 설명합니다.

## 목차

- [개요](#개요)
- [사전 준비](#사전-준비)
- [엔드포인트](#엔드포인트)
- [사용 예시](#사용-예시)
- [설정 참고](#설정-참고)
- [주의사항 / FAQ](#주의사항--faq)

## 개요

- **코드 서빙**은 doc_parser 전처리기를 코드 서빙 컨테이너로 배포해, 단일 서빙이
  적재/첨부/변환(`/preprocess*`), 파싱(`/parser`), 청킹(`/chunker`), 헬스 체크(`/health`)
  엔드포인트를 제공하는 형태입니다.
- 적재/첨부/변환은 문서를 한 번에 처리해 적재용 결과를 만드는 **단일 단계** API입니다.
- 파싱과 청킹은 **분리된 2단계**로 동작합니다.
  - **파싱(`/parser`)**: 원본 문서(PDF/HWP/DOCX 등)를 입력받아 파싱 결과 JSON 을 생성합니다.
    포맷에 따라 docling(`data.document`) 또는 parse-format(`data.elements`)으로 반환됩니다.
  - **청킹(`/chunker`)**: 파싱 결과 JSON(docling 또는 parse-format)을 입력받아
    벡터 적재용 **청크 리스트**(`GenOSVectorMeta`)를 생성합니다. chunker 가 입력 형태를 자동 판별합니다.

처리 흐름(E2E):

```
원본 문서                  파싱 결과 JSON                       청크 리스트
(report.pdf)  ──POST /parser──▶  data.document(docling)   ──POST /chunker──▶  data[ {...}, ... ]
(sheet.csv)   ──POST /parser──▶  data.elements(parse-format) ─┘
```

> 청킹 단계는 파싱 결과만 입력으로 받습니다. OCR·레이아웃·enrichment 등 무거운
> 처리는 모두 파싱 단계에서 끝나므로, 청킹은 가볍고 빠르게 반복 호출할 수 있습니다.
> docling 포맷(pdf/html/htm/docx/hwp/hwpx)은 구조 인식 청킹(GenosSmartChunker)을, 그 외
> 비-docling 포맷은 parse-format 공통 청킹(audio→`[AUDIO]` 단일, csv/xlsx→`[DA]` 단일,
> 그 외 텍스트→문자 기반 splitter)을 수행합니다.

## 사전 준비

호출에 필요한 정보:

| 항목 | 설명 | 예시 |
| --- | --- | --- |
| `base URL` | 게이트웨이 base URL | `https://genos.genon.ai` |
| `serving_id` | 배포된 코드 서빙 ID | `139` |
| `auth_key` | 게이트웨이 인증 토큰(Bearer) | `b8c0b48f7b4d410699ed1aa8f2c0da8a` |

전제 조건:

- **`/parser` 의 `file_path` 는 서빙 컨테이너 내부의 로컬 경로**입니다(MinIO 키 아님).
  게이트웨이로 파싱을 호출하려면 서버가 접근 가능한 경로를 넣어야 합니다.
- docling 포맷 파일(pdf/html/htm/docx/hwp/hwpx)은 파싱 서빙의 `parser_processor_config.yaml` 이
  **`output.format: "docling"`** 이어야 응답에 `data.document` 가 생성됩니다.
- 그 외 비-docling 포맷(csv/xlsx/txt/md/ppt/pptx/doc/이미지/오디오)은 `output.format` 과 무관하게
  항상 parse-format(`data.elements`)으로 반환되며, chunker 가 이를 그대로 청킹합니다(별도 설정 불필요).

## 엔드포인트

**공통 URL 패턴**

```
{base}/api/gateway/code_serving/{serving_id}/{route}
```

**공통 헤더**

```
Content-Type: application/json
Authorization: Bearer {auth_key}
```

**공통 요청 본문** (모든 POST 엔드포인트 공통)

```json
{ "file_path": "<문서 경로>", "params": { } }
```

- `file_path` : 처리할 문서 경로(서버 기준). `/chunker` 는 기본값 `""`(메타데이터 용도).
- `params` : 엔드포인트별 추가 옵션(객체). 비우면 config 기본값을 사용합니다.

**공통 응답 envelope** (모든 POST 엔드포인트 공통)

```json
{ "code": 0, "errMsg": "success", "data": { } }
```

- `code` 가 `0` 이면 성공, 그 외에는 실패이며 `errMsg` 에 사유가 담깁니다.
- 서버 내부 예외가 발생해도 HTTP 상태는 `200` 이며 `code` 가 `0` 이 아닌 값으로 반환됩니다.
  성공 여부는 HTTP 상태가 아니라 **`code` 값으로 판단**하세요.
- 실패 응답에는 `error_code` 가 함께 담기며, `error_policy: "strict"`(#329) 또는 요청
  deadline 초과 시엔 `stage`(실패 단계)·`error_kind`(`transient`/`permanent`/`timeout`) 필드가
  추가됩니다. 자세한 내용은 아래 「LLM 캐시 / 실패 정책 / 요청 deadline (#329)」 절 참고.

엔드포인트 요약:

| 메서드 | 경로 | 용도 | 비고 |
| --- | --- | --- | --- |
| `GET` | `/health` | 헬스 체크 | `{"status":"ok"}` |
| `POST` | `/preprocess` | 적재용(지능형) 전처리 | `/preprocess_intelligent` 의 하위호환 별칭 |
| `POST` | `/preprocess_attachment` | 첨부용 전처리 | |
| `POST` | `/preprocess_intelligent` | 적재용(지능형) 전처리 | |
| `POST` | `/preprocess_convert` | 변환용 전처리 | |
| `POST` | `/parser` | 문서 파싱 → DoclingDocument JSON | `IS_PARSER` 지원 전처리기 필요 |
| `POST` | `/chunker` | DoclingDocument JSON → 청크 리스트 | `IS_CHUNKER` 지원 전처리기 필요 |

> **경로는 단일 세그먼트만 사용:** 게이트웨이는 `{serving_id}` 뒤의 `route` 를 **단일 세그먼트로만**
> 포워딩한다. 따라서 슬래시가 포함된 중첩 경로(`/preprocess/intelligent` 등)는 호출되지 않고
> HTTP 500 이 발생한다. 적재/첨부/변환은 평탄 경로(`/preprocess_intelligent`,
> `/preprocess_attachment`, `/preprocess_convert`)를 사용해야 한다.

> **마커 가드:** `/parser` 와 `/chunker` 는 설치된 전처리기가 해당 기능을 지원할
> 때만 동작합니다. 미지원 전처리기에 호출하면 `code: 1` 과 함께
> `"현재 설치된 전처리기는 /parser API를 지원하지 않습니다."` 같은 안내가 반환됩니다.
> 즉, 배포된 전처리기 종류에 따라 활성화되는 엔드포인트가 다를 수 있습니다.

### GET /health

서빙 동작 여부 확인. envelope 가 아닌 단순 응답을 반환합니다.

```json
{ "status": "ok" }
```

### POST /preprocess, /preprocess_attachment, /preprocess_intelligent, /preprocess_convert

문서를 한 번에 처리해 적재용 결과를 반환하는 전처리 엔드포인트입니다.
경로에 따라 사용하는 프로세서가 다릅니다.

| 경로 | 프로세서 | 용도 |
| --- | --- | --- |
| `/preprocess` | intelligent | 적재용(지능형). `/preprocess_intelligent` 의 하위호환 별칭 |
| `/preprocess_attachment` | attachment | 첨부용 |
| `/preprocess_intelligent` | intelligent | 적재용(지능형) |
| `/preprocess_convert` | convert | 변환용 |

**요청 본문** (네 엔드포인트 공통)

```json
{
  "file_path": "/app/src/service/genon/preprocessor/sample_files/pdf_sample.pdf",
  "params": {}
}
```

**응답 본문**

```json
{ "code": 0, "errMsg": "success", "data": { } }
```

- `data` 의 구조는 프로세서별로 다릅니다. 각 프로세서의 동작·옵션 상세는
  [intelligent_processor.md](intelligent_processor.md),
  [attachment_processor.md](attachment_processor.md),
  [convert_processor.md](convert_processor.md) 를 참고하세요.

### POST /parser

원본 문서를 파싱해 DoclingDocument JSON을 반환합니다.
`IS_PARSER` 를 지원하는 전처리기에서만 동작합니다(미지원 시 `code: 1`).

**요청 본문**

```json
{
  "file_path": "/app/src/service/genon/preprocessor/sample_files/pdf_sample.pdf",
  "params": {}
}
```

**응답 본문**

```json
{
  "code": 0,
  "errMsg": "success",
  "data": {
    "document": { "schema_name": "DoclingDocument", "...": "..." },
    "usage": { "pages": 10 }
  }
}
```

- `data.document` : 청킹 입력으로 사용할 DoclingDocument JSON
- `data.usage.pages` : 처리한 페이지 수

### POST /chunker

파싱 결과 JSON 을 입력받아 청크 리스트를 반환합니다. 앞단계 파싱 결과는 `params.document` 로
인라인 전달합니다(docling `{"document":...}` 또는 parse-format `{"elements":[...]}` 모두 허용 —
chunker 가 형태를 자동 판별). `IS_CHUNKER` 를 지원하는 전처리기에서만 동작합니다(미지원 시 `code: 1`).

**요청 본문**

```json
{
  "file_path": "report.pdf",
  "params": {
    "document": { "schema_name": "DoclingDocument", "...": "..." },
    "chunk_size": 0
  }
}
```

**응답 본문**

```json
{
  "code": 0,
  "errMsg": "success",
  "data": [
    {
      "schema_name": "GenOSVectorMeta",
      "i_chunk_on_doc": 0,
      "i_page": 1,
      "text": "청크 텍스트 ...",
      "chunk_token_count": 128
    }
  ]
}
```

**주요 파라미터**

| 파라미터 | 위치 | 기본값 | 설명 |
| --- | --- | --- | --- |
| `document` | `params` | (필수) | 파싱이 반환한 결과 JSON. docling(`{"document":...}`) 또는 parse-format(`{"elements":[...]}`) 모두 허용(자동 판별) |
| `chunk_size` | `params` | `0` | 청크 최대 크기. `0` 이면 토큰/문자 기반 분할 안 함. 청킹 config 기본값을 덮어씀 |
| `log_level` | `params` | config 값 | 런타임 로깅 레벨(5=DEBUG ~ 1=CRITICAL, 0=NOLOG) |

> `file_path` 는 청킹 단계에서는 청크 메타데이터에 기록되는 경로 용도이며,
> 실제 입력은 `params.document` 입니다.

## 사용 예시

아래 예시는 `genon/preprocessor/examples/code_serving/` 의 테스트 스크립트와 동등합니다.

### curl

공통 변수:

```bash
BASE="https://genos.genon.ai"
SERVING_ID="139"
AUTH="b8c0b48f7b4d410699ed1aa8f2c0da8a"
GW="${BASE}/api/gateway/code_serving/${SERVING_ID}"
FILE_PATH="/app/src/service/genon/preprocessor/sample_files/pdf_sample.pdf"
```

**1) health**

```bash
curl --location "${GW}/health" \
  --header 'Content-Type: application/json' \
  --header "Authorization: Bearer ${AUTH}"
```

**2) /parser — 파싱 후 `data.document` 만 `doc.json` 으로 저장**

```bash
curl --location "${GW}/parser" \
  --header 'Content-Type: application/json' \
  --header "Authorization: Bearer ${AUTH}" \
  --data "$(jq -nc --arg fp "${FILE_PATH}" '{file_path:$fp, params:{}}')" \
  | jq '.data.document' > doc.json
```

**3) /chunker — 저장한 `doc.json` 을 `params.document` 로 실어 청킹**

```bash
jq -nc --slurpfile doc "doc.json" \
  '{file_path:"report.pdf", params:{document:$doc[0], chunk_size:0}}' \
| curl --location "${GW}/chunker" \
    --header 'Content-Type: application/json' \
    --header "Authorization: Bearer ${AUTH}" \
    --data-binary @- \
  | jq '.data | length as $n | "chunks: \($n)"'
```

> **주의:** `doc.json` 은 수 MB가 될 수 있습니다. `--data "$(...)"` 처럼 인자로
> 직접 전달하면 셸 `ARG_MAX`(약 1MB)를 초과해 `Argument list too long` 오류가
> 납니다. 위처럼 `jq` 출력을 파이프로 넘겨 `--data-binary @-` 로 stdin 에서 읽으세요.

### Python

표준 라이브러리(urllib)만 사용하는 `serving_gateway_test.py` 로 동일하게 호출할 수 있습니다.

```bash
# 1) 헬스 체크
python serving_gateway_test.py --mode health

# 2) 파싱 → 청킹 E2E (서버 접근 가능 경로 필요)
python serving_gateway_test.py --mode e2e \
  --file-path /data/documents/report.pdf \
  --out /tmp/chunks.json

# 3) 파싱만 (docling JSON 저장)
python serving_gateway_test.py --mode parser \
  --file-path /data/documents/report.pdf --out-doc /tmp/doc.json

# 4) 청킹만 (저장해둔 docling JSON 사용)
python serving_gateway_test.py --mode chunker --doc-json /tmp/doc.json

# 5) LLM 캐시(#329) — 별도 모드가 아니라 parser 에 --param 으로 opt-in.
#    같은 스코프로 2회 파싱 → 1회차 MISS(저장), 2회차 HIT(재사용). 로그의 hit/miss 및 두 doc 비교.
python serving_gateway_test.py --mode parser --file-path /data/documents/report.pdf \
  --param llm_cache=1 --param interim_root=/nfs-root/interim \
  --param workflow_id=wf-123 --param run_id=run-1 --out-doc /tmp/doc_run1.json
python serving_gateway_test.py --mode parser --file-path /data/documents/report.pdf \
  --param llm_cache=1 --param interim_root=/nfs-root/interim \
  --param workflow_id=wf-123 --param run_id=run-1 --out-doc /tmp/doc_run2.json
```

주요 CLI 인자:

| 인자 | 기본값 | 설명 |
| --- | --- | --- |
| `--mode` | `e2e` | `health` / `parser` / `parser_upload` / `chunker` / `e2e` |
| `--base-url` | `https://genos.genon.ai` | 게이트웨이 base URL |
| `--serving-id` | `139` | 코드 서빙 ID |
| `--auth-key` | (스크립트 기본값) | `Authorization: Bearer <key>` |
| `--file-path` | `""` | 파싱할 문서 경로(서버 기준). `parser`/`e2e` 에 필요 |
| `--chunk-size` | `0` | 청크 최대 크기(0=분할 안 함). 청킹 config 기본값을 덮어씀 |
| `--doc-json` | `None` | `chunker` 모드의 입력 docling JSON 파일 경로 |
| `--out` | `None` | 청크 결과 JSON 저장 경로/디렉터리(옵션) |
| `--out-doc` | `None` | `parser` 모드의 docling JSON 저장 경로/디렉터리(옵션) |
| `--timeout` | `3600` | 요청 타임아웃(초) |
| `--param KEY=VALUE` | (없음) | 임의 `params` 오버라이드(반복). #329 캐시/정책도 이걸로: `llm_cache=1`·`interim_root=..`·`workflow_id=..`·`run_id=..`·`error_policy=strict`·`request_deadline=..` |

## LLM 캐시 / 실패 정책 / 요청 deadline (#329)

모든 POST 엔드포인트(`/preprocess*`·`/parser`·`/chunker`)에서 `params` 로 opt-in 할 수 있는
공통 옵션입니다. 대용량 배치 중 문서가 중간 실패해도 그때까지 성공한 LLM 호출을 캐시해 재시도 시
재사용하고, 실패/행잉을 응답으로 돌려줍니다. **미지정 시 기존과 완전히 동일하게 동작**합니다.

| 파라미터 | 위치 | 기본값 | 설명 |
| --- | --- | --- | --- |
| `llm_cache` | `params` | `false` | `true` 일 때만 LLM 호출 입출력을 파일 캐시(0/1 표기도 수용) |
| `interim_root` | `params` | env `INTERIM_ROOT`(미설정 시 `/nfs-root/interim`) | 캐시 루트 경로. 우선순위 요청값 > env > 기본 |
| `workflow_id` | `params` | (없음) | 캐시 스코프. **없으면 캐시 비활성** |
| `run_id` | `params` | `"default"` | 캐시 스코프 |
| `error_policy` | `params` | `"lenient"` | `"strict"` 면 enrichment 실패를 `code:1` 로 응답(+`stage`/`error_kind`) |
| `request_deadline` | `params` | (없음) | 요청 전체 상한(초). 초과 시 timeout 응답 |

- 캐시 활성 조건: `llm_cache` **AND** `workflow_id`. interim root 는 요청 `interim_root` > env
  `INTERIM_ROOT` > 기본 `/nfs-root/interim` 순으로 항상 확보되므로, 사실상 이 둘만 있으면 켜집니다.
  경로는 `<interim_root>/<workflow_id>/<run_id>/llm_cache/<key>.json`.
- `/chunker` 는 `interim_ref`(=`"<workflow_id>/<run_id>"`)로도 스코프를 유도할 수 있습니다
  (청킹은 LLM 호출이 없어 실질 no-op).

요청 예시(`/parser`):
```json
{
  "file_path": "/app/src/service/genon/preprocessor/sample_files/pdf_sample.pdf",
  "params": {
    "llm_cache": true,
    "interim_root": "/nfs-root/interim",
    "workflow_id": "wf-123",
    "run_id": "run-1"
  }
}
```

### 캐시 키(`<key>`) 생성 방식

LLM 호출 1건 = 파일 1개(`<key>.json`). 키는 **그 호출의 지문(fingerprint)** 으로, 엔드포인트와
요청 payload 를 함께 해싱해 만든다(`docling/utils/llm_cache.py` `cache_key`).

```python
def cache_key(endpoint, payload):
    h = hashlib.sha256()
    h.update(str(endpoint).encode("utf-8"))          # ① 호출 엔드포인트 URL
    h.update(b"\x00")                                # ② 구분자(널바이트)
    h.update(_canonical(payload).encode("utf-8"))    # ③ 정규화된 요청 payload
    return h.hexdigest()                             # 64자 hex → <key>.json

def _canonical(payload):   # 결정적 정규화
    return json.dumps(payload, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), default=str)
```

| 구성 | 내용 |
| --- | --- |
| ① `endpoint` | 그 호출의 URL(모델 서빙 URL). **모델/URL 이 다르면 키가 갈림**(충돌 방지) |
| ② `\x00` | endpoint 와 payload 경계 구분자 |
| ③ `payload` | 그 호출의 **전체 요청 본문**: `model`, `messages`(텍스트/이미지 base64), 샘플링 파라미터(`temperature`/`top_p`/`max_tokens`/`seed`/`repetition_penalty`), `chat_template_kwargs` 등 |

- **결정적**: 같은 `(endpoint, payload)` → 항상 같은 64자 hex → 재실행 시 같은 파일을 찾아 **HIT**.
- **키 순서 무관**: `sort_keys=True` 라 payload dict 키 순서가 달라도 같은 키.
- 입력이 하나라도 바뀌면(프롬프트·이미지·모델·샘플링 등) 다른 키 → **MISS**(새로 호출·저장).

### 저장되는 데이터 (파일 내용)

각 `<key>.json` 은 forward-compat envelope 로 감싼다.
```json
{ "v": 1, "value": <LLM 응답 직렬화값> }
```
- `v`: envelope 스키마 버전(현재 1). 값이 바뀌면 기존 캐시는 miss 처리.
- `value`: 그 호출의 **결과**. 호출 유형별 형태:

| LLM 작업 | 공유 헬퍼 | `value` 형태 |
| --- | --- | --- |
| OCR / layout VLM(DotsOCR, 페이지별) | `call_vlm_server` | `{"content": <텍스트>, "usage": <토큰정보\|null>, "finish_reason": <문자열\|null>}` |
| 텍스트 프롬프트(품질검사·TOC 등) | `call_ai_model` | 문자열(응답 본문, thinking 제거 후) |
| 이미지 설명 · 표 설명 | `api_image_request` | 문자열(생성 텍스트) |
| 문서 본문요약(doc_summary) | `summarize_body` | 문자열(요약) |
| 메타데이터 · 커스텀 필드 | `_call_llm`(async) | 문자열(정규화 content) |

- 저장·읽기 안전성: temp 파일 기록 후 **atomic rename**(부분 쓰기 방지). 읽을 때 JSON 파싱 실패면
  miss 로 처리해 재호출·덮어쓴다. 같은 키 동시 쓰기는 내용이 같아 락이 필요 없다.
- **요청(payload)은 파일에 저장하지 않는다** — payload 는 키(해시)로만 반영되고, 파일엔 결과(`value`)만 담긴다.

### 저장 제외 (캐시 안 됨)
- `None` / 빈 문자열 결과 — "실패성 성공"은 저장 안 함(재호출 기회 보존).
- 예외로 실패한 호출(402/500 등) — 저장 안 되고 miss 로 남아 **재시도 시 다시 호출**(캐시 오염 없음).
- 활성 조건 미충족(캐시 off).
- PaddleOCR `/ocr` 텍스트 엔드포인트(강제 재OCR) — 현재 캐시 범위 밖.
- `/chunker` — LLM 호출 자체가 없음.

### 로그
호출마다 캐시 사용 여부가 로그에 남는다.
```
[llm_cache] MISS — 캐시 없음, LLM 실제 호출 (endpoint=... key=...)
[llm_cache] STORE — LLM 결과 캐시 저장 (key=...)
[llm_cache] HIT  — 캐시 재사용, LLM 호출 안 함 (endpoint=... key=...)
```
요청 종료 시 요약: `[llm_cache] hit=.. miss=.. save_fail=.. dir=...`

### 재사용 단위(스코프)
캐시는 `<workflow_id>/<run_id>` 디렉토리 단위다. Temporal activity 재시도는 같은
workflow_id/run_id 를 유지하므로 **재시도 간 성공분을 그대로 재사용**한다. 서로 다른
문서/워크플로우는 다른 디렉토리라 격리된다.

### 실패 정책 (`error_policy`)

| 값 | 동작 |
| --- | --- |
| `lenient`(기본) | enrichment 실패를 삼키고 null 로 채운 뒤 `code:0` — 기존 하위호환 동작 |
| `strict` | enrichment 실패 시 삼키지 않고 `code:1` 로 응답 |

`strict`/timeout 응답 envelope 에는 실패 맥락이 추가된다.

| 필드 | 설명 |
| --- | --- |
| `stage` | 실패 단계(`doc_summary`/`image_description`/`table_description`/`metadata`/`custom_fields`/`enrichment`/`request`) |
| `error_kind` | 실패 성격 — `transient`(연결/5xx) / `permanent`(4xx/파싱) / `timeout` |

> **주의**: `strict` 는 캐시가 선행되어야 실용적이다. 캐시 없이 하드페일하면 재시도마다 성공분이 전부 재과금된다.

### 요청 deadline (`request_deadline`)
- **요청 전체 상한**(초): 초과 시 timeout 응답(`error_kind: "timeout"`, `stage: "request"`).
- **per-call timeout**: 개별 LLM 호출도 남은 deadline 으로 소켓 timeout 을 좁혀 조기 종료한다.
- 미지정 시 상한 없음.

### 배포 요건
- 코드서빙 컨테이너에 **`INTERIM_ROOT` env** 가 있고(또는 요청 `interim_root` 전달), 그 경로가
  **Temporal worker 와 공유되는 NFS** 여야 재시도 간 재사용이 성립한다. 요청 `interim_root` 가 env 보다 우선.
- 셋 중 하나라도 없으면 캐시는 켜지지 않고(안전 no-op) 기존대로 동작한다.

### 테스트 방법
```bash
# 게이트웨이(HTTP): parser 를 같은 스코프로 2회 호출 → 1회차 MISS(저장), 2회차 HIT(재사용)
python serving_gateway_test.py --mode parser --file-path /data/report.pdf \
  --param llm_cache=1 --param interim_root=/nfs-root/interim \
  --param workflow_id=wf-123 --param run_id=run-1 --out-doc /tmp/doc_run1.json
python serving_gateway_test.py --mode parser --file-path /data/report.pdf \
  --param llm_cache=1 --param interim_root=/nfs-root/interim \
  --param workflow_id=wf-123 --param run_id=run-1 --out-doc /tmp/doc_run2.json

# in-process 파싱→청킹: examples/parse_chunk/parse_chunk_test.sh
python parse_chunk_test.py --llm_cache --interim_root <경로> \
  --workflow_id wf-1 --run_id run-1 <input.pdf> <out>/

# in-process 적재(/run): shkim_labs/test.sh
python test.py --llm_cache --interim_root <경로> --workflow_id <id> --run_id <id> <input> <out>
```
로그의 1회차 `MISS→STORE`, 2회차 `HIT` 및 요약 `hit=.. miss=..` 로 재사용을 확인한다.

> **참고 — `/parse → /chunk == /run` 정합**: 캐시 도입과 함께 분리 경로(`/parser`→`/chunker`)가
> 모놀리식(`/run`)과 동일 결과를 내도록 `/parse` 의 auto-OCR 재시도 휴리스틱(빈 텍스트 페이지 감지 포함)과
> picture 이미지 참조를 `/run` 과 일치시켰다.

## 설정 참고

서빙 동작은 배포된 전처리기의 resource config 로 결정됩니다. 호출 시 주요 항목:

| 단계 | config 파일 | 핵심 항목 | 설명 |
| --- | --- | --- | --- |
| 파싱 | `parser_processor_config.yaml` | `output.format` | docling 포맷 파일은 `"docling"` 이어야 `data.document` 생성. 그 외 `json`/`html`/`markdown`. 비-docling 포맷은 항상 parse-format(`data.elements`) |
| 청킹 | `chunking_processor_config.yaml` | `chunking.chunk_size` | docling 청킹(GenosSmartChunker) 최대 크기(0=분할 안 함). 호출 `chunk_size` 가 우선 |
| 청킹 | `chunking_processor_config.yaml` | `chunking.tokenizer_type` | `"char"`(문자 수) 또는 `"huggingface"`(토크나이저 기준) |
| 청킹 | `chunking_processor_config.yaml` | `chunking.generic.chunk_size` / `chunk_overlap` | 비-docling(parse-format) 일반 텍스트 splitter 기본값(문자 단위, 기본 1000/100). 호출 `chunk_size`/`chunk_overlap` 가 우선. audio/csv·xlsx 단일 벡터엔 미적용 |

> 파싱 옵션(OCR·레이아웃·enrichment 등)의 상세 설명은 [parser_processor.md](parser_processor.md) 를 참고하세요.

## 주의사항 / FAQ

- **`/parser` 의 `file_path` 는 서버 로컬 경로**입니다(MinIO 키 아님). 서버가 접근
  가능한 경로를 넣어야 파싱이 됩니다.
- **응답에 `data.document` 가 없을 때**: docling 포맷 파일(pdf/html/htm/docx/hwp/hwpx)인데
  `data.document` 가 없으면 파싱 서빙 config 의 `output.format` 이 `"docling"` 인지 확인하세요.
  비-docling 포맷(csv/xlsx/txt/md/ppt/이미지/오디오)은 `data.elements`(parse-format)로 반환되며,
  이 역시 `params.document` 에 그대로 넣어 청킹할 수 있습니다(audio→`[AUDIO]` 단일, csv/xlsx→`[DA]`
  단일, 그 외 텍스트→`chunking.generic` splitter).
- **`Argument list too long`**: 큰 `doc.json` 을 curl `--data` 인자로 직접 넘긴 경우입니다.
  `jq | --data-binary @-` 방식(stdin)으로 전달하세요.
- **`code` 가 0이 아님**: 요청이 실패한 것이며 `errMsg` 에 사유가 들어 있습니다.
- **타임아웃**: 큰 문서 파싱은 시간이 걸릴 수 있습니다. 클라이언트 타임아웃을
  넉넉히(예: 3600초) 설정하세요.
- **LLM 캐시(#329)가 안 켜져요**: 캐시는 `llm_cache=true` **AND** `workflow_id` 가 있어야 동작합니다
  (interim root 는 요청 `interim_root` > env `INTERIM_ROOT` > 기본 `/nfs-root/interim` 순으로 항상 확보).
  재사용은 그 경로가 재시도 간 공유되는 NFS 일 때 성립합니다. 로그의 `[llm_cache] hit=.. miss=..`
  요약으로 확인하세요. 상세: 위 「LLM 캐시 / 실패 정책 / 요청 deadline (#329)」 절.
