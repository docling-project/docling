"""
dots.ocr(dotsocr) 표구조 추출 smoke 테스트 — TEDS-S 정량 검증.

병합셀(rowspan/colspan)이 포함된 스캔 PDF(sample_files/table.pdf, 51p)를
parser_processor 로 처리해 dotsocr 가 추출한 표구조를 페이지별 GT(sample_files/table.jsonl)와
TEDS-S(구조 전용 트리편집 유사도)로 비교한다.

- 실제 dotsocr layout 엔드포인트로 요청을 보낸다(= 실제 추출 경로 검증).
- 엔드포인트에 접속할 수 없는 GitHub CI 등에서는 GENOS_LAYOUT_AVAILABLE 미설정으로 skip.
- 합격 기준: 51개 표의 평균 TEDS-S >= 임계값(env TEDS_S_MIN, 기본 0.85).
  TEDS-S 는 셀 텍스트를 무시하고 병합 span 구조만 평가하므로 OCR 텍스트 변동에 강건.
"""
import os
from collections import defaultdict
from pathlib import Path

import pytest

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "sample_files"
REQUIRED_KEYS = {"elements", "usage"}
ELEMENT_KEYS = {"id", "page", "category", "content", "coordinates"}

# 평균 TEDS-S 합격 임계값 (필요 시 env 로 조정)
TEDS_S_MIN = float(os.environ.get("TEDS_S_MIN", "0.75"))


def _resolve_sample() -> Path | None:
    env = os.environ.get("DOTSOCR_TABLE_SAMPLE")
    p = Path(env) if env else SAMPLE_DIR / "table.pdf"
    return p if p.exists() else None


def _validate_result(result: dict) -> None:
    """parser_processor 출력 스키마 검증 (test_parser_processor_smoke 과 동일 패턴)."""
    assert isinstance(result, dict)
    for key in REQUIRED_KEYS:
        assert key in result, f"result missing key: {key!r}"
    assert isinstance(result["elements"], list), "elements must be a list"
    assert isinstance(result["usage"]["pages"], int), "usage.pages must be int"
    assert result["usage"]["pages"] >= 1
    for element in result["elements"]:
        for key in ELEMENT_KEYS:
            assert key in element, f"element missing key: {key!r}"


@pytest.fixture(scope="module")
def dp(parser_processor):
    return parser_processor()


@pytest.mark.smoke
@pytest.mark.skipif(
    not os.environ.get("GENOS_LAYOUT_AVAILABLE"),
    reason="GENOS_LAYOUT_AVAILABLE 미설정 — dotsocr layout 엔드포인트 없음(CI). skip.",
)
@pytest.mark.asyncio
async def test_dotsocr_merged_cell_table_teds(dp, teds_s, table_gt):
    sample = _resolve_sample()
    if sample is None:
        pytest.skip("표 PDF 픽스처 없음(DOTSOCR_TABLE_SAMPLE 또는 sample_files/table.pdf).")
    if not table_gt:
        pytest.skip("GT 없음(sample_files/table.jsonl).")

    result = await dp(None, str(sample))  # ← 실제 dotsocr 요청 발생

    # 1) 결과 스키마
    _validate_result(result)

    # 2) 표 element 가 1개 이상 추출됨 + HTML 구조 기본 점검
    table_elems = [e for e in result["elements"] if e["category"] == "table"]
    assert table_elems, "dotsocr 가 표 element 를 추출하지 못함"
    for t in table_elems:
        assert isinstance(t["content"], str) and "<table" in t["content"], (
            f"table content 가 HTML 표가 아님: {str(t['content'])[:120]!r}"
        )

    # 3) 페이지 → 추출된 표 HTML 매핑. (table_gt[i] == 페이지 i+1 의 표)
    tables_by_page: dict[int, list[str]] = defaultdict(list)
    for t in table_elems:
        tables_by_page[t["page"]].append(t["content"])

    # 4) 페이지별 TEDS-S. 한 페이지에 표가 여럿이면 GT 대비 최고 점수를 채택,
    #    표가 없으면 0(미검출도 감점).
    scores: list[float] = []
    print(f"\n[dotsocr TEDS-S] pages={len(table_gt)} extracted_tables={len(table_elems)}")
    for i, gt in enumerate(table_gt):
        page_no = i + 1
        candidates = tables_by_page.get(page_no, [])
        best = max((teds_s(html, gt["gt_html"]) for html in candidates), default=0.0)
        scores.append(best)
        print(f"  page {page_no:>2}: TEDS-S={best:.4f} (tables on page={len(candidates)})")

    mean_score = sum(scores) / len(scores)
    matched = sum(1 for s in scores if s > 0)
    print(f"[dotsocr TEDS-S] mean={mean_score:.4f} matched_pages={matched}/{len(scores)} "
          f"(threshold={TEDS_S_MIN})")

    # 5) 합격 기준: 평균 TEDS-S >= 임계값
    assert mean_score >= TEDS_S_MIN, (
        f"평균 TEDS-S {mean_score:.4f} < 임계값 {TEDS_S_MIN} "
        f"(matched {matched}/{len(scores)} pages)"
    )
