"""
table_description refine 재구성 HTML 의 구조 유효성 검증 단위 테스트.

재구성 표가 구조적으로 유효하지 않으면(<table> 없음 / grid 빈 / 2행 미만 / 첫 행 빈)
`is_valid_refined_html` 이 False 를 반환해야 한다(→ 부착 안 함 → 원본 표 폴백).

의존성(bs4/docling 등) 미가용 환경에서는 importorskip 으로 자동 skip 된다(CI gate).
"""

import pytest

_MOD = "facade.enrichment.table_description"


def _mod():
    return pytest.importorskip(_MOD)


@pytest.mark.unit
def test_valid_table_is_accepted():
    m = _mod()
    html = (
        "<table><thead><tr><th>구분</th><th>값</th></tr></thead>"
        "<tbody><tr><td>A</td><td>1</td></tr><tr><td>B</td><td>2</td></tr></tbody></table>"
    )
    assert m.is_valid_refined_html(html) is True
    assert m._parse_refined_table_data(html) is not None


@pytest.mark.unit
@pytest.mark.parametrize(
    "html",
    [
        "",                       # 빈 문자열
        "   ",                    # 공백만
        "<div>표 아님</div>",       # <table> 없음
        "일반 텍스트",              # 태그 없음
        "<table><tr><th>A</th><th>B</th></tr></table>",  # 헤더만(1행) → 2행 미만
        # </table> 없음(잘린 출력) — bs4 자동보정 방지 태그쌍 검사로 걸러야 함
        "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr>",
        # 여는 태그 2 / 닫는 태그 1 (불균형)
        "<table><tr><td>x</td></tr><table><tr><td>1</td><td>2</td></tr></table>",
    ],
)
def test_invalid_refined_html_is_rejected(html):
    m = _mod()
    assert m.is_valid_refined_html(html) is False
    assert m._parse_refined_table_data(html) is None


# ── _parse_refine_output: degenerate/잘린 응답 폐기 ────────────────────────────
@pytest.mark.unit
def test_parse_refine_output_valid_split():
    m = _mod()
    out = "[[[TABLE_HTML]]]\n<table><tr><td>1</td></tr></table>\n[[[TABLE_SUMMARY]]]\n표 요약입니다."
    summary, refined_html = m.TableDescriptionEnricher._parse_refine_output(out)
    assert summary == "표 요약입니다."
    assert "<table>" in refined_html and "[[[TABLE_HTML]]]" not in refined_html


@pytest.mark.unit
def test_parse_refine_output_truncated_marker_is_discarded():
    """[[[TABLE_HTML]]] 만 있고 [[[TABLE_SUMMARY]]] 누락(잘림/degeneration) → ("", "")."""
    m = _mod()
    out = "[[[TABLE_HTML]]]\n<table><tr><td>결정</td></tr>\n결정 결정 결정"  # </table>·SUMMARY 마커 없음
    assert m.TableDescriptionEnricher._parse_refine_output(out) == ("", "")


@pytest.mark.unit
def test_parse_refine_output_plain_prose_is_summary():
    m = _mod()
    out = "이 표는 부서별 권한을 나타낸다."
    summary, refined_html = m.TableDescriptionEnricher._parse_refine_output(out)
    assert summary == out and refined_html == ""


# ── is_valid_table_summary ────────────────────────────────────────────────────
@pytest.mark.unit
def test_valid_table_summary():
    m = _mod()
    assert m.is_valid_table_summary("이 표는 부서별 직무권한을 정리한 표이다.") is True


@pytest.mark.unit
@pytest.mark.parametrize(
    "summary",
    [
        "",
        "   ",
        "[[[TABLE_HTML]]]<table><tr><td>결정</td></tr>",  # 마커 잔재
        "요약 <table border='1'><tr><td>x</td></tr></table>",  # 원문 table 마크업
    ],
)
def test_invalid_table_summary(summary):
    m = _mod()
    assert m.is_valid_table_summary(summary) is False
