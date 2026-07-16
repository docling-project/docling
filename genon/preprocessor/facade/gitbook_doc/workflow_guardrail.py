# GenOS 가드레일 분류 워크플로우 Python 단계 (#315) — run(data) 단독 구성
# (이 파일을 GenOS 워크플로우 Python 단계에 그대로 붙여넣어 배포한다. 배포 절차: guardrail_workflow_setup.md)
# 전처리기가 문서 전체 text 를 보내면, [가드레일 인스턴스 정규식 필터 + LLM 의미분류]로
# sensitive_infos[] JSON 을 만들어 반환한다.
#
# 계약(전처리기와 약속):
#   입력  : data = {"text": "<문서 전체>"} (또는 "question")
#   출력  : {"sensitive_infos": [
#              {"category": "민감 정보", "specific_category": "주민번호",
#               "quote_origin": "900101-1234567", "quote_masked": "[주민등록번호]"},
#              {"category": "부동산 정보", "specific_category": "",
#               "quote_origin": "강남구 ... 5억에 매매", "quote_masked": "강남구 ... 5억에 매매"},
#              ...
#           ]}
#
# 정규식은 이 코드에 하드코딩하지 않는다. 운영이 GenOS 가드레일 인스턴스(정규식 필터)에 정의한 것을
# 단일 소스로 삼아, 그 인스턴스의 dry-run 을 호출해 마스킹 결과를 받고 원문과 diff 해서 스팬을 복원한다.
# → 운영이 필터를 추가/수정하면 워크플로우 수정 없이 자동 반영된다.
#
# 환경 변수:
#   GUARDRAIL_DRYRUN_BASE  게이트웨이 베이스(클러스터 내부, 무인증). 예 http://llmops-gateway-api-service:8080
#   GUARDRAIL_ID           정규식 필터가 등록된 가드레일 인스턴스 ID (예: 99)
#   GUARDRAIL_LLM_URL / GUARDRAIL_LLM_KEY / GUARDRAIL_LLM_MODEL   의미분류 LLM 접속

import os
import json
import difflib
import requests

# ── 가드레일 인스턴스(정규식 필터) dry-run ────────────────────────────────────
# 클러스터 내부에서는 api_key 없이 직접 호출 가능(외부 genos.genon.ai 경유만 인증 필요).
_GR_BASE = os.environ.get("GUARDRAIL_DRYRUN_BASE", "http://llmops-gateway-api-service:8080")
_GR_ID = os.environ.get("GUARDRAIL_ID", "99")
_GR_TIMEOUT = int(os.environ.get("GUARDRAIL_TIMEOUT", "60"))
_REGEX_CATEGORY = "민감 정보"   # 정규식(단어 범위) 항목의 소분류_h1

# 치환 토큰 → 소분류_h2 매핑(알려진 것만; 모르는 토큰은 토큰 안쪽 텍스트를 그대로 사용).
# 운영이 가드레일 필터의 "치환 규칙"으로 무엇을 쓰든, category 는 항상 "민감 정보"로 매핑된다.
_TOKEN_SPEC = {
    "[주민등록번호]": "주민번호", "[휴대전화번호]": "휴대전화", "[휴대전화]": "휴대전화",
    "[전화번호]": "전화번호", "[카드번호]": "카드번호", "[여권번호]": "여권번호",
    "[운전면허번호]": "운전면허", "[전자메일]": "전자메일", "[이메일]": "전자메일",
}

# ── LLM 의미분류(부동산/인사) 접속 ────────────────────────────────────────────
# LLM 서빙은 내부 경로에서도 인증을 요구하므로(실측) GUARDRAIL_LLM_KEY 설정 필수.
# 기본 URL 은 외부 게이트웨이. 서빙 번호/주소가 다르면 GUARDRAIL_LLM_URL 로 교체.
_LLM_URL = os.environ.get(
    "GUARDRAIL_LLM_URL",
    "https://genos.genon.ai/api/gateway/rep/serving/776/v1/chat/completions",
)
_LLM_KEY = os.environ.get("GUARDRAIL_LLM_KEY", "")
_LLM_MODEL = os.environ.get("GUARDRAIL_LLM_MODEL", "model")
_LLM_TIMEOUT = int(os.environ.get("GUARDRAIL_LLM_TIMEOUT", "60"))

_SYSTEM_PROMPT = """You are a document sensitivity classifier for a Korean enterprise RAG pipeline.
Find spans that belong to these SEMANTIC categories, judging by meaning (not keywords):
- 부동산 정보: 특정 부동산의 소재지·면적·시세·매매/임대·등기·지번/동호수 등.
- 인사 정보: 특정 개인의 채용·평가·급여·직급·징계·근태·인사이동 등 조직 내 인사 정보.
Return JSON: {"items":[{"category":"부동산 정보|인사 정보","quote_origin":"<원문 그대로 발췌>"}]}.
Rules: quote_origin MUST be copied verbatim from the input text. If none, return {"items":[]}.
Use ONLY the two category values above. No extra text."""


def _call_guardrail_dryrun(text: str):
    """가드레일 인스턴스 dry-run 호출 → 마스킹된 전체 텍스트(str). 실패/미설정 시 None(fail-open)."""
    if not _GR_BASE or not _GR_ID:
        return None
    try:
        url = f"{_GR_BASE.rstrip('/')}/guardrail/{_GR_ID}/dry-run"
        resp = requests.post(url, json={"content": text}, timeout=_GR_TIMEOUT)
        resp.raise_for_status()
        body = resp.json()
        if body.get("code") != 0:
            raise RuntimeError(f"guardrail code={body.get('code')}")
        masked = (body.get("data") or {}).get("content")
        return masked if isinstance(masked, str) else None
    except Exception:
        return None   # 워크플로우 안 터지게 — 실패는 정규식 결과 없음으로


def _diff_to_infos(original: str, masked: str) -> list:
    """원문 vs 마스킹본 diff → 정규식 필터가 치환한 스팬을 sensitive_info 로 복원.
    replace 구간: 원문쪽 = quote_origin, 마스킹쪽 = quote_masked(치환토큰).
    구조화 탐지 API 가 없어 마스킹 텍스트만 오므로 diff 로 스팬을 되짚는다."""
    if not masked or masked == original:
        return []
    out = []
    sm = difflib.SequenceMatcher(None, original, masked, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        quote_origin = original[i1:i2]
        quote_masked = masked[j1:j2]
        if not quote_origin.strip():
            continue                       # 순수 삽입/공백 변화는 무시
        token = quote_masked.strip()
        spec = _TOKEN_SPEC.get(token, token.strip("[]") if token else "")
        out.append({
            "category": _REGEX_CATEGORY,
            "specific_category": spec,
            "quote_origin": quote_origin,
            "quote_masked": quote_masked if quote_masked else quote_origin,
        })
    return out


def _run_guardrail_regex(text: str) -> list:
    """가드레일 인스턴스(정규식 필터) 결과 → sensitive_info 리스트."""
    masked = _call_guardrail_dryrun(text)
    if masked is None:
        return []
    return _diff_to_infos(text, masked)


def _run_llm(text: str) -> list:
    """LLM 의미분류(부동산/인사) → sensitive_info 리스트. 실패/키미설정 시 [](fail-open).
    LLM 서빙은 인증 필요 — GUARDRAIL_LLM_KEY 없으면 의미분류 skip(정규식 결과만 반환)."""
    if not _LLM_KEY:
        return []
    try:
        resp = requests.post(
            _LLM_URL,
            headers={"Authorization": f"Bearer {_LLM_KEY}", "Content-Type": "application/json"},
            json={
                "model": _LLM_MODEL,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                "temperature": 0,
                "response_format": {"type": "json_object"},
            },
            timeout=_LLM_TIMEOUT,
        )
        resp.raise_for_status()
        parsed = json.loads(resp.json()["choices"][0]["message"]["content"])
    except Exception:
        return []

    out = []
    for it in (parsed.get("items") or []):
        cat = it.get("category")
        q = it.get("quote_origin")
        if not cat or not isinstance(q, str) or not q.strip():
            continue
        if cat not in ("부동산 정보", "인사 정보"):
            continue                       # enum 밖 값 드롭(환각 방지)
        out.append({
            "category": cat,
            "specific_category": "",       # 의미 카테고리는 소분류_h2 없음
            "quote_origin": q,
            "quote_masked": f"[{cat}]",    # 의미 카테고리도 마스킹 on 시 치환되도록 토큰 제공(정규식류와 동일 스타일)
        })
    return out


async def run(data: dict) -> dict:
    text = data.get("text") or data.get("question", "") or ""
    if not text:
        # 워크플로우 엔진 규약: dict 반환 시 "text" 키 필수(없으면 09050003).
        return {"text": "", "sensitive_infos": []}
    infos = _run_guardrail_regex(text) + _run_llm(text)
    return {"text": json.dumps({"sensitive_infos": infos}, ensure_ascii=False),
            "sensitive_infos": infos}
