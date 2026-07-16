"""청크 텍스트에 sensitive_infos 를 적용 (#315).

quote_origin 을 청크에서 찾아(정확→공백무시 fuzzy) 카테고리 라벨을 모으고,
masking=True 이면 quote_masked 로 치환한다.
"""
from __future__ import annotations

import re

_WS = re.compile(r"\s+")


def find_spans(text: str, quote: str) -> list:
    """청크 text 에서 quote 가 나타나는 모든 (start,end). 1차 정확, 실패 시 공백 무시(fuzzy).
    (LLM quote 의 '5 억에' vs 원문 '5억에' 불일치 대응)"""
    if not quote or not text:
        return []
    spans, start = [], 0
    while True:
        i = text.find(quote, start)
        if i == -1:
            break
        spans.append((i, i + len(quote)))
        start = i + len(quote)
    if spans:
        return spans
    kept = [(j, ch) for j, ch in enumerate(text) if not ch.isspace()]
    if not kept:
        return []
    stripped = "".join(ch for _, ch in kept)
    idx_map = [j for j, _ in kept]
    q = _WS.sub("", quote)
    if not q:
        return []
    s = 0
    while True:
        k = stripped.find(q, s)
        if k == -1:
            break
        spans.append((idx_map[k], idx_map[k + len(q) - 1] + 1))
        s = k + len(q)
    return spans


def apply_to_text(text: str, sensitive_infos: list, masking: bool):
    """청크 text 에 sensitive_infos 적용 → (새 text, 부착할 category 집합).
    category 부착은 항상, quote_masked 치환은 masking=True 일 때만. 매칭 실패는 skip."""
    cats = set()
    repl = []  # (start, end, masked)
    for info in sensitive_infos or []:
        cat = (info or {}).get("category")
        q = (info or {}).get("quote_origin")
        m = (info or {}).get("quote_masked")
        if not cat or not q:
            continue
        spans = find_spans(text, q)
        if not spans:
            continue
        cats.add(cat)
        if masking and isinstance(m, str) and m != q:
            for st, en in spans:
                repl.append((st, en, m))
    if repl:
        applied = []  # 이미 치환한 구간(겹침 방지)
        for st, en, m in sorted(repl, key=lambda x: -x[0]):
            if any(not (en <= a or st >= b) for a, b in applied):
                continue  # 겹치는 구간은 건너뜀
            text = text[:st] + m + text[en:]
            applied.append((st, en))
    return text, cats
