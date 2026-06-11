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


# ─── (c) DotsOCR 한 아이템에 여러 <table> → 개별 분리 ────────────────────────


def _extract_all(text):
    """한 DotsOCR 아이템 text 에서 최상위 <table> 들을 모두 추출(실제 함수)."""
    from docling.models.genos_dots_ocr_layout_model import _extract_all_table_html

    return _extract_all_table_html(text)


def _split_bbox(bbox_values, count):
    from docling.models.genos_dots_ocr_layout_model import _split_bbox_vertically

    return _split_bbox_vertically(bbox_values, count)


# 사용자 제보 샘플: 한 아이템 text 에 <table> 2개가 연속으로 들어있는 경우
# (앞은 목차 표, 뒤는 회의록 표). 기존엔 첫 표만 가져오고 둘째 표가 통째로 유실됨.
_TWO_TABLE_HTML = (
    "<table>"
    "<tr><td>5. 2026년도 경상남도 제1회 추가경정예산안(계속)</td><td></td></tr>"
    "<tr><td>다. 관광개발국 소관</td><td>44면</td></tr>"
    "<tr><td>라. 보건의료국 소관</td><td>52면</td></tr>"
    "<tr><td>부록</td><td>59면</td></tr>"
    "</table>"
    "<table>"
    "<tr><td>(15시 51분 개의)</td><td>존경하는 박주언 위원장님!</td></tr>"
    "<tr><td>전기풍 의원입니다.</td><td>감사합니다.</td></tr>"
    "</table>"
)


@pytest.mark.unit
def test_extract_all_tables_returns_each_top_level_table():
    """여러 <table> 가 각각 분리되어 반환되는지(둘째 표 유실 없음)."""
    tables = _extract_all(_TWO_TABLE_HTML)
    assert len(tables) == 2
    # 각 추출 결과는 정확히 하나의 최상위 table.
    for html in tables:
        assert html.count("<table") == 1
    # 첫째는 목차, 둘째는 회의록 — 내용이 서로 섞이지 않음.
    assert "44면" in tables[0] and "44면" not in tables[1]
    assert "존경하는" in tables[1] and "존경하는" not in tables[0]


@pytest.mark.unit
def test_extract_all_tables_single_table_returns_one():
    """단일 table 입력은 1개만 반환(회귀 방지)."""
    single = "<table><tr><td>a</td><td>b</td></tr></table>"
    tables = _extract_all(single)
    assert len(tables) == 1
    assert "a" in tables[0] and "b" in tables[0]


@pytest.mark.unit
def test_extract_all_tables_ignores_nested_tables():
    """중첩 table 은 별도 항목이 아니라 외곽 table 안에 보존되어야 함."""
    nested = (
        "<table><tr><td>outer"
        "<table><tr><td>inner</td></tr></table>"
        "</td></tr></table>"
    )
    tables = _extract_all(nested)
    assert len(tables) == 1
    assert "inner" in tables[0]


@pytest.mark.unit
def test_extract_all_tables_empty_or_non_table():
    """table 이 없으면 빈 리스트."""
    assert _extract_all("") == []
    assert _extract_all("그냥 텍스트") == []
    assert _extract_all(None) == []


@pytest.mark.unit
def test_extract_table_html_first_only_backward_compat():
    """_extract_table_html 은 기존처럼 첫 table 만 반환."""
    from docling.models.genos_dots_ocr_layout_model import _extract_table_html

    html = _extract_table_html(_TWO_TABLE_HTML)
    assert html is not None
    assert "44면" in html and "존경하는" not in html


@pytest.mark.unit
def test_extracted_tables_parse_to_consistent_grids():
    """분리 추출한 각 표가 격자 정합(겹침/구멍 0)으로 파싱되는지."""
    for html in _extract_all(_TWO_TABLE_HTML):
        td = _parse(html)
        assert td is not None
        overlaps, holes = _grid_issues(td)
        assert overlaps == [], f"격자 셀 겹침: {overlaps[:5]}"
        assert holes == [], f"격자 빈 칸: {holes[:5]}"


@pytest.mark.unit
def test_different_column_counts_stay_consistent_when_split():
    """컬럼 수가 다른 두 표 — 분리하면 각자 정합 격자 유지(병합 시 구멍 발생 방지)."""
    mixed = (
        "<table><tr><td>a</td><td>b</td></tr></table>"
        "<table><tr><td>x</td><td>y</td><td>z</td></tr></table>"
    )
    tables = _extract_all(mixed)
    assert len(tables) == 2

    td0, td1 = _parse(tables[0]), _parse(tables[1])
    assert (td0.num_rows, td0.num_cols) == (1, 2)
    assert (td1.num_rows, td1.num_cols) == (1, 3)
    for td in (td0, td1):
        overlaps, holes = _grid_issues(td)
        assert overlaps == [] and holes == []


@pytest.mark.unit
def test_split_bbox_vertically_disjoint_and_ordered():
    """원본 bbox 를 세로 N등분 — 겹치지 않고 위→아래 순서, 원본 범위 내부."""
    bbox_values = (10.0, 100.0, 200.0, 400.0)  # l, t, r, b
    bands = _split_bbox(bbox_values, 3)
    assert len(bands) == 3

    for band in bands:
        # 좌우 폭은 원본과 동일.
        assert band["l"] == 10.0 and band["r"] == 200.0
        # 세로는 원본 범위 내부.
        assert 100.0 <= band["t"] <= band["b"] <= 400.0

    # 첫 밴드 상단 = 원본 상단, 마지막 밴드 하단 = 원본 하단.
    assert bands[0]["t"] == 100.0
    assert bands[-1]["b"] == 400.0
    # 인접 밴드는 경계만 공유(면적 겹침 0) → 후처리 overlap-dedup 비대상.
    for i in range(len(bands) - 1):
        assert bands[i]["b"] == bands[i + 1]["t"]
        assert bands[i]["t"] < bands[i]["b"]


@pytest.mark.unit
def test_split_bbox_vertically_single_band():
    """count<=1 이면 원본 bbox 그대로."""
    assert _split_bbox((0.0, 0.0, 10.0, 10.0), 1) == [
        {"l": 0.0, "t": 0.0, "r": 10.0, "b": 10.0}
    ]
