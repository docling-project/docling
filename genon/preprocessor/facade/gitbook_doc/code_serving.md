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
  - **파싱(`/parser`)**: 원본 문서(PDF/HWP/DOCX 등)를 입력받아 복원 가능한
    **DoclingDocument JSON**(`data.document`)을 생성합니다.
  - **청킹(`/chunker`)**: 파싱이 만든 DoclingDocument JSON을 입력받아
    벡터 적재용 **청크 리스트**(`GenOSVectorMeta`)를 생성합니다.

처리 흐름(E2E):

```
원본 문서                  DoclingDocument JSON              청크 리스트
(report.pdf)  ──POST /parser──▶  data.document  ──POST /chunker──▶  data[ {...}, ... ]
```

> 청킹 단계는 파싱 결과만 입력으로 받습니다. OCR·레이아웃·enrichment 등 무거운
> 처리는 모두 파싱 단계에서 끝나므로, 청킹은 가볍고 빠르게 반복 호출할 수 있습니다.

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
- 파싱 서빙의 `parser_processor_config.yaml` 이 **`output.format: "docling"`** 이어야
  응답에 `data.document` 가 생성되어 청킹 입력으로 쓸 수 있습니다.

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

파싱 결과(DoclingDocument JSON)를 입력받아 청크 리스트를 반환합니다.
앞단계 파싱 결과는 `params.document` 로 인라인 전달합니다.
`IS_CHUNKER` 를 지원하는 전처리기에서만 동작합니다(미지원 시 `code: 1`).

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
| `document` | `params` | (필수) | 파싱이 반환한 DoclingDocument JSON |
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
```

주요 CLI 인자:

| 인자 | 기본값 | 설명 |
| --- | --- | --- |
| `--mode` | `e2e` | `health` / `parser` / `chunker` / `e2e` |
| `--base-url` | `https://genos.genon.ai` | 게이트웨이 base URL |
| `--serving-id` | `139` | 코드 서빙 ID |
| `--auth-key` | (스크립트 기본값) | `Authorization: Bearer <key>` |
| `--file-path` | `""` | 파싱할 문서 경로(서버 기준). `parser`/`e2e` 에 필요 |
| `--chunk-size` | `0` | 청크 최대 크기(0=분할 안 함). 청킹 config 기본값을 덮어씀 |
| `--doc-json` | `None` | `chunker` 모드의 입력 docling JSON 파일 경로 |
| `--out` | `None` | 청크 결과 JSON 저장 경로/디렉터리(옵션) |
| `--out-doc` | `None` | `parser` 모드의 docling JSON 저장 경로/디렉터리(옵션) |
| `--timeout` | `3600` | 요청 타임아웃(초) |

## 설정 참고

서빙 동작은 배포된 전처리기의 resource config 로 결정됩니다. 호출 시 주요 항목:

| 단계 | config 파일 | 핵심 항목 | 설명 |
| --- | --- | --- | --- |
| 파싱 | `parser_processor_config.yaml` | `output.format` | `"docling"` 이어야 `data.document` 생성(청킹 입력용). 그 외 `json`/`html`/`markdown` |
| 청킹 | `chunking_processor_config.yaml` | `chunking.chunk_size` | 청크 최대 크기(0=분할 안 함). 호출 `chunk_size` 가 우선 |
| 청킹 | `chunking_processor_config.yaml` | `chunking.tokenizer_type` | `"char"`(문자 수) 또는 `"huggingface"`(토크나이저 기준) |

> 파싱 옵션(OCR·레이아웃·enrichment 등)의 상세 설명은 [parser_processor.md](parser_processor.md) 를 참고하세요.

## 주의사항 / FAQ

- **`/parser` 의 `file_path` 는 서버 로컬 경로**입니다(MinIO 키 아님). 서버가 접근
  가능한 경로를 넣어야 파싱이 됩니다.
- **응답에 `data.document` 가 없을 때**: 파싱 서빙 config 의 `output.format` 이
  `"docling"` 인지 확인하세요. docling 이 아니면 청킹 입력을 만들 수 없습니다.
- **`Argument list too long`**: 큰 `doc.json` 을 curl `--data` 인자로 직접 넘긴 경우입니다.
  `jq | --data-binary @-` 방식(stdin)으로 전달하세요.
- **`code` 가 0이 아님**: 요청이 실패한 것이며 `errMsg` 에 사유가 들어 있습니다.
- **타임아웃**: 큰 문서 파싱은 시간이 걸릴 수 있습니다. 클라이언트 타임아웃을
  넉넉히(예: 3600초) 설정하세요.
