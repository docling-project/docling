# 민감정보 분류 워크플로우 구성·배포 안내

전처리기의 `guardrail_masking` 기능이 호출하는 **GenOS 분류 워크플로우** 를 GenOS 에서 만들고
배포해 전처리기와 연결하는 절차입니다.

- 전처리기 쪽 사용법(config/동작)은 각 전처리기 매뉴얼의 「민감정보 분류/마스킹」 절을 참고하세요.
- 이 문서는 그 절이 전제하는 **워크플로우 자체를 어떻게 준비하는가** 를 다룹니다.

---

## 0. 전체 그림

```
문서 업로드(guardrail_masking: true)
   → 전처리기가 청킹 직전 문서 전체를 워크플로우에 1회 POST
     POST {url}/workflow/{workflow_id}/run/v2   (Authorization: Bearer {api_key})
     body: {"question": "<문서 전체 텍스트>"}
   → 워크플로우(Python 단계)가 정규식 + LLM 분류 → sensitive_infos[] 반환
   → 전처리기가 청킹 후 quote_origin 매칭 → content_category 부착(+옵션 마스킹)
```

운영이 준비할 것은 **가운데의 워크플로우 하나** 입니다. 나머지(호출·매칭·부착)는 전처리기가 합니다.

---

## 1. 워크플로우 입출력 계약 (전처리기와의 약속)

전처리기 코드(`_gr_classify_document`)가 기대하는 형식입니다. 반드시 지켜야 합니다.

**입력** (전처리기 → 워크플로우)

```json
{ "question": "<문서 전체 텍스트>" }
```

**출력** (워크플로우 → 전처리기)

```json
{
  "code": 0,
  "data": {
    "text": "...",
    "sensitive_infos": [
      {
        "category": "인사 정보",
        "specific_category": "주민번호",
        "quote_origin": "홍길동 900101-1234567",
        "quote_masked": "홍길동 [주민등록번호]"
      }
    ]
  }
}
```

- `code` 는 `0` 이 성공입니다. 그 외 값이면 전처리기는 실패로 보고 fail-open(원문 통과)합니다.
- 결과 배열의 키 이름은 정확히 `sensitive_infos` 여야 합니다.
  - (호환) `data.sensitive_infos` 가 없고 `data.text` 에 `{"sensitive_infos": [...]}` JSON 문자열이
    실려오면 전처리기가 파싱을 시도합니다. 워크플로우 엔진이 dict 반환 시 `text` 키를 요구하는
    제약(에러코드 `09050003`) 때문에 스텝 코드가 `text` 에도 같은 JSON 을 실어 보냅니다.

**각 항목 필드**

| 필드 | 의미 | 전처리기 사용 |
|---|---|---|
| `category` | 소분류_h1 (부동산 정보 / 인사 정보 / 민감 정보 …) | `content_category` 라벨로 부착. **비어있으면 그 항목 skip** |
| `specific_category` | 소분류_h2 (주민번호 / 휴대전화 …) | 사용 안 함(버림) |
| `quote_origin` | 원문 그대로의 발췌 | 청크 매칭 키 |
| `quote_masked` | 마스킹본 | `masking_enabled` on 일 때 치환값. 의미 범주는 `quote_origin` 과 동일하게 두면 치환 안 됨 |

> `quote_origin` 은 **원문 그대로** 여야 합니다. 프롬프트가 요약·재작성하면 청크 매칭이 실패합니다.
> (전처리기는 공백 차이 정도는 fuzzy 매칭으로 흡수하지만, 표현이 바뀌면 매칭 불가.)

---

## 2. 워크플로우 Python 단계 코드

`run(data)` 하나로 정규식 + LLM 분류를 수행하는 스텝 코드가 저장소에 있습니다:
`00_system/issue_branch/issue_315/workflow_run_step.py`. 핵심 구조만 발췌합니다.

```python
import os, re, json, requests

# LLM 접속(모델서빙). 인증키는 코드 하드코딩 대신 워크플로우 "환경 변수"로 주입한다.
_LLM_URL   = os.environ.get("GUARDRAIL_LLM_URL", "https://genos.genon.ai/api/gateway/rep/serving/776/v1/chat/completions")
_LLM_KEY   = os.environ.get("GUARDRAIL_LLM_KEY", "")
_LLM_MODEL = os.environ.get("GUARDRAIL_LLM_MODEL", "model")

# 단어 범위(정규식) — 매칭부 전체를 마스킹 토큰으로 치환
_REGEX_RULES = [ ("주민번호", r"...", "[주민등록번호]"), ("휴대전화", r"...", "[휴대전화번호]"),
                 ("전자메일", r"...", "[전자메일]"), ... ]

# 의미 범위(LLM) — 부동산/인사 등. quote_masked == quote_origin (치환 안 함, 라벨만)
_SYSTEM_PROMPT = """... 부동산 정보 / 인사 정보 를 의미로 판별. quote_origin 은 원문 그대로 복사 ..."""

async def run(data: dict) -> dict:
    text = data.get("text") or data.get("question", "") or ""
    if not text:
        return {"text": "", "sensitive_infos": []}   # dict 반환 시 text 키 필수(09050003 방지)
    infos = _run_regex(text) + _run_llm(text)
    return {"text": json.dumps({"sensitive_infos": infos}, ensure_ascii=False),
            "sensitive_infos": infos}
```

- **정규식류**(주민번호·전화·이메일 등)는 코드에서 바로 잡아 `quote_masked` 에 토큰을 채웁니다.
- **의미류**(부동산·인사)는 LLM(모델 776 등)에 넘겨 판별하고, 치환 없이 라벨만 달도록
  `quote_masked = quote_origin` 으로 둡니다.
- 프롬프트 품질·카테고리 정의는 운영이 조정합니다(전처리기는 구조만 소비).

---

## 3. GenOS 배포 절차

1. **모델서빙 확인** — 분류 LLM(예: 모델 776 qwen)이 서빙 중인지, 호출 URL/키를 확인합니다.
   나중에 GPT OSS 120B 등으로 교체할 경우 `GUARDRAIL_LLM_URL` 만 바꾸면 됩니다.
2. **워크플로우 생성** — GenOS 워크플로우에서 새 워크플로우를 만들고 **Python 단계** 를 추가해
   위 `run(data)` 코드를 붙여 넣습니다(정규식만/LLM만 쓰려면 해당 부분만 남깁니다).
3. **환경 변수 설정** — 워크플로우의 환경 변수에 인증키 등을 넣습니다(코드 하드코딩 금지).
   - `GUARDRAIL_LLM_URL` = 분류 LLM 서빙 chat/completions URL
   - `GUARDRAIL_LLM_KEY` = 그 서빙 인증키
   - `GUARDRAIL_LLM_MODEL` = 모델명(서빙 설정에 맞게)
4. **배포** 후 `workflow_id` 를 확인합니다(워크플로우 상세/주소에 노출되는 정수 ID).
5. **인증키(AuthKeyBearer) 발급** — 워크플로우 실행 라우트(`/workflow/{id}/run`)는 Bearer 인증을
   요구합니다. 이 키가 전처리기 config 의 `api_key` 로 들어갑니다.

### 배포 검증

```bash
# 헬스체크
curl -s "https://genos.genon.ai/api/gateway/workflow/{workflow_id}/healthcheck" \
  -H "Authorization: Bearer {api_key}"

# 실행 (run/v2) — 전처리기가 실제로 쓰는 경로/형식
curl -s -X POST "https://genos.genon.ai/api/gateway/workflow/{workflow_id}/run/v2" \
  -H "Authorization: Bearer {api_key}" -H "Content-Type: application/json" \
  -d '{"question": "홍길동 900101-1234567 서울 강남구 아파트를 5억에 매매"}'
# → data.sensitive_infos 에 주민번호(치환토큰) + 부동산 정보(라벨) 가 오면 정상
```

---

## 4. 전처리기와 연결

배포로 얻은 값 3개를 전처리기 config 의 `guardrail_masking` 에 넣습니다(전처리기별 config yaml).

```yaml
guardrail_masking:
  url: "https://genos.genon.ai/api/gateway"   # 코드가 /workflow/{workflow_id}/run/v2 를 붙임
  workflow_id: <배포 워크플로우 ID>
  api_key: "<워크플로우 Bearer 인증키>"
  timeout: 60
  masking_enabled: false   # 라벨 부착은 항상, quote_masked 치환만 이 스위치로 on/off
```

- `url` 은 게이트웨이 **베이스** 까지만 넣습니다(뒤의 `/workflow/{id}/run/v2` 는 코드가 붙임).
- 기능 자체의 on/off 는 config 가 아니라 **업로드 요청 kwargs `guardrail_masking`** 입니다(기본 off).
- `masking_enabled` 는 치환만 제어합니다. 끄면 `content_category` 라벨만 붙고 텍스트는 원문 유지.
- 대상 전처리기: intelligent / attachment / convert / chunking. parser 는 청크가 없어 대상 아님
  (파스 결과를 chunking API 로 넘기면 chunking 이 분류·부착).

---

## 5. 자주 겪는 문제

| 증상 | 원인 / 조치 |
|---|---|
| 배포 시 `09050003` (응답에 text/json 없음) | Python 단계 dict 반환에 `text` 키 누락 → 스텝 코드처럼 `text` 도 함께 반환 |
| 워크플로우 직접 호출 401 | admin 토큰이 아니라 **워크플로우 AuthKeyBearer** 필요. config `api_key` 확인 |
| 라벨이 하나도 안 붙음 | `category` 가 비어 옴 / `quote_origin` 이 원문과 불일치(프롬프트가 변형) → 프롬프트에서 원문 그대로 반환 강제 |
| 마스킹이 안 됨 | `masking_enabled: false` 이거나 요청 `guardrail_masking` off. 둘 다 on 이어야 치환 |
| 호출 자체가 안 감 | config `url`/`workflow_id`/`api_key` 중 빈 값 → 전처리기가 fail-open(원문 통과 + warning 로그) |
