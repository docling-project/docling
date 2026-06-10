"""
병합셀 표구조 처리에 대한 결정적 단위테스트 (네트워크 불필요 → CI 상시 실행).

검증 대상:
1. TEDS-S 지표(teds_metric.teds_s)가 구조 동일/차이를 올바르게 점수화하는지.
2. dotsocr 가 내보내는 형식의 표 HTML(= table.jsonl 의 gt_html)을 실제 처리 함수
   `_parse_html_to_table_data()` 로 변환했을 때 병합셀(row_span/col_span/offset)이
   정확히 복원되고 격자가 정합(겹침 0, 구멍 0)한지.
"""
from __future__ import annotations

import re

import pytest


def _parse(html):
    """dotsocr 표 HTML → docling TableData (실제 파이프라인 처리 함수)."""
    from docling.models.genos_dots_ocr_layout_model import _parse_html_to_table_data

    return _parse_html_to_table_data(html)


def _grid_issues(table_data):
    """table_cells 를 span 으로 전개해 (겹침 좌표, 구멍 좌표) 반환."""
    occupied: dict[tuple[int, int], str] = {}
    overlaps: list[tuple[int, int]] = []
    for cell in table_data.table_cells:
        for r in range(cell.start_row_offset_idx, cell.end_row_offset_idx):
            for c in range(cell.start_col_offset_idx, cell.end_col_offset_idx):
                if (r, c) in occupied:
                    overlaps.append((r, c))
                occupied[(r, c)] = cell.text
    holes = [
        (r, c)
        for r in range(table_data.num_rows)
        for c in range(table_data.num_cols)
        if (r, c) not in occupied
    ]
    return overlaps, holes


def _cell_by_text(table_data, needle: str):
    for cell in table_data.table_cells:
        if needle in cell.text:
            return cell
    return None


# ─── (a) TEDS-S 지표 정확성 ──────────────────────────────────────────────────


@pytest.mark.unit
def test_teds_s_identical_is_one(teds_s, table_gt):
    if not table_gt:
        pytest.skip("table.jsonl GT 없음")
    # 동일 구조는 항상 만점.
    for row in table_gt[:10]:
        gt = row["gt_html"]
        assert teds_s(gt, gt) == pytest.approx(1.0), "동일 구조 TEDS-S 는 1.0 이어야 함"


@pytest.mark.unit
def test_teds_s_detects_structural_difference(teds_s, table_gt):
    if not table_gt:
        pytest.skip("table.jsonl GT 없음")

    # 병합(span)이 있는 GT 를 하나 고른다.
    gt = next((r["gt_html"] for r in table_gt if "span=" in r["gt_html"]), None)
    assert gt is not None, "병합셀 포함 GT 가 있어야 함"

    # 병합 속성을 모두 제거하면 구조가 달라져 점수가 1 미만으로 떨어져야 한다.
    no_merge = re.sub(r'\s+(col|row)span="\d+"', "", gt)
    score_no_merge = teds_s(no_merge, gt)
    assert 0.0 <= score_no_merge < 1.0, (
        f"병합 제거 시 TEDS-S < 1.0 이어야 함 (got {score_no_merge})"
    )

    # 행 하나를 삭제하면 더 큰 구조 변화 → 더 낮은 점수.
    drop_row = re.sub(r"<tr>.*?</tr>", "", gt, count=1, flags=re.S)
    score_drop_row = teds_s(drop_row, gt)
    assert score_drop_row < 1.0


@pytest.mark.unit
def test_teds_s_zero_for_non_table(teds_s, table_gt):
    if not table_gt:
        pytest.skip("table.jsonl GT 없음")
    assert teds_s("표가 아닌 텍스트", table_gt[0]["gt_html"]) == 0.0


# ─── (b) 병합셀 파싱 정확성 + 격자 정합성 ────────────────────────────────────


@pytest.mark.unit
def test_parse_merged_cells_exact_page1(table_gt):
    """table.jsonl 1번 표(=table.pdf 1페이지)의 알려진 병합 구조 정확 검증."""
    if not table_gt:
        pytest.skip("table.jsonl GT 없음")

    td = _parse(table_gt[0]["gt_html"])
    assert (td.num_rows, td.num_cols) == (6, 5)

    # 행 병합(rowspan)
    assert _cell_by_text(td, "일반직원").row_span == 3
    assert _cell_by_text(td, "해외근무자").row_span == 2
    # "자녀" 는 본문 첫 열 rowspan=3 셀
    janyeo = next(
        c for c in td.table_cells if c.text == "자녀" and c.start_col_offset_idx == 1
    )
    assert janyeo.row_span == 3

    # 열 병합(colspan)
    assert _cell_by_text(td, "부양가족 구분").col_span == 2
    assert _cell_by_text(td, "배우자").col_span == 2

    # 병합셀 텍스트는 정확히 1회만 등장(중복/유실 없음)
    texts = [c.text for c in td.table_cells]
    assert texts.count("일반직원") == 1
    assert texts.count("부양가족 구분") == 1

    # 격자 정합성: 겹침/구멍 없음
    overlaps, holes = _grid_issues(td)
    assert overlaps == [], f"격자 셀 겹침 발생: {overlaps[:5]}"
    assert holes == [], f"격자 빈 칸 발생: {holes[:5]}"


@pytest.mark.unit
def test_all_gt_tables_parse_with_consistent_grid(table_gt):
    """모든 GT 표가 파싱되고 격자가 정합(겹침/구멍 0)해야 한다."""
    if not table_gt:
        pytest.skip("table.jsonl GT 없음")

    merged_seen = 0
    for i, row in enumerate(table_gt):
        td = _parse(row["gt_html"])
        assert td is not None, f"GT[{i}] 파싱 실패"
        assert td.num_rows >= 1 and td.num_cols >= 1, f"GT[{i}] 크기 비정상"

        overlaps, holes = _grid_issues(td)
        assert overlaps == [], f"GT[{i}] 격자 셀 겹침: {overlaps[:5]}"
        assert holes == [], f"GT[{i}] 격자 빈 칸: {holes[:5]}"

        if any(c.row_span > 1 or c.col_span > 1 for c in td.table_cells):
            merged_seen += 1

    # 데이터셋에 병합셀 표가 다수 존재(33/51) — 파싱 결과에도 반영되어야 함.
    assert merged_seen >= 20, f"병합셀이 인식된 표가 너무 적음: {merged_seen}"
