"""기존 OCR 결과와 B300 결과의 기능적 동일성 비교.

baseline.json (cu126 환경에서 미리 만든 것) 과 current.json (cu129/B300 결과) 을
받아 다음을 검사한다:
  - 인식 텍스트 집합이 정규화 기준으로 일치하는지
  - 각 텍스트의 score 차이가 허용 범위(--score-tol, 기본 0.05) 이내인지

부동소수 inference 특성상 score 가 완전히 같지는 않다.
"기능적으로 동일" 판정 기준이라 텍스트 일치 + score 근사 비교로 둔다.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _WS.sub("", text).strip().lower()


def _index_by_text(items: list[dict]) -> dict[str, list[float]]:
    # 같은 텍스트가 중복 인식될 수 있으므로 score 를 리스트로 모은다 (마지막 값으로 덮어쓰면 drift 누락 가능).
    out: dict[str, list[float]] = defaultdict(list)
    for it in items:
        out[_norm(it.get("text", ""))].append(float(it.get("score", 0.0)))
    for scores in out.values():
        scores.sort()
    return dict(out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--score-tol", type=float, default=0.05)
    args = parser.parse_args()

    baseline = json.loads(Path(args.baseline).read_text())
    current = json.loads(Path(args.current).read_text())

    b_items = _index_by_text(baseline.get("items", []))
    c_items = _index_by_text(current.get("items", []))

    b_count = Counter({k: len(v) for k, v in b_items.items()})
    c_count = Counter({k: len(v) for k, v in c_items.items()})
    missing = sorted((b_count - c_count).elements())
    added = sorted((c_count - b_count).elements())

    diffs: list[tuple[str, float, float]] = []
    for key in sorted(set(b_items) & set(c_items)):
        for b, c in zip(b_items[key], c_items[key]):
            if abs(b - c) > args.score_tol:
                diffs.append((key, b, c))

    ok = not missing and not added and not diffs

    b_total = sum(len(v) for v in b_items.values())
    c_total = sum(len(v) for v in c_items.values())
    print(f"[compare] baseline items: total={b_total} unique={len(b_items)}")
    print(f"[compare] current items : total={c_total} unique={len(c_items)}")
    if missing:
        print(f"[compare] MISSING in current ({len(missing)}):")
        for k in missing:
            print(f"  - {k!r}")
    if added:
        print(f"[compare] EXTRA in current ({len(added)}):")
        for k in added:
            print(f"  + {k!r}")
    if diffs:
        print(f"[compare] SCORE drift > {args.score_tol} ({len(diffs)}):")
        for k, b, c in diffs:
            print(f"  ~ {k!r}: baseline={b:.3f} current={c:.3f}")

    if ok:
        print("[compare] PASS: functionally equivalent")
        return 0
    print("[compare] FAIL: divergence detected", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
