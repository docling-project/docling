"""xlsx 직접 처리(이슈 #288) 테스트.

- tabular 모드: 병합셀 unmerge+forward-fill, ASCII 키만 메타데이터, 한글 헤더 제외(값은 text 유지).
- docling 모드: 시트=1페이지로 변환되어 한 행이 페이지 경계로 쪼개지지 않음(버그 픽스).
- e2e: intelligent_processor 가 두 모드에서 non-empty 벡터를 반환(facade/fastapi 미가용 시 자동 skip).

mock 없이 실제 추출 경로를 호출한다. docling/facade 의존성 미가용 환경에서는 importorskip 으로 skip(CI gate).
"""

from pathlib import Path

import pytest
import yaml

# 실샘플(해진공 더미) — facade/excel 아래 위치
_PREPROC = Path(__file__).resolve().parents[2]  # genon/preprocessor
_SAMPLE = _PREPROC / "facade" / "excel" / "해진공_엑셀_샘플파일.xlsx"
_CONFIG = _PREPROC / "resource" / "intelligent_processor_config.yaml"


def _xp():
    """헬퍼 모듈 로드(openpyxl 등 미가용 시 skip)."""
    return pytest.importorskip("genon.preprocessor.converters.xlsx_processor")


def _make_xlsx(path: Path, rows, merges=None, sheet_name="Sheet1"):
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = sheet_name
    for r, row in enumerate(rows, start=1):
        for c, val in enumerate(row, start=1):
            ws.cell(row=r, column=c, value=val)
    for m in merges or []:
        ws.merge_cells(m)
    wb.save(str(path))
    return path


# --------------------------------------------------------------------------- #
# tabular 모드                                                                  #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_tabular_merged_title_and_ascii_keys(tmp_path):
    """병합 제목행 위, 실제 헤더행 아래 데이터. ASCII 헤더가 메타 KEY 로 부여된다."""
    xp = _xp()
    path = _make_xlsx(
        tmp_path / "t.xlsx",
        rows=[
            ["REPORT", None, None],          # 병합 제목행(A1:C1)
            ["name", "age", "dept"],         # 실제 헤더행
            ["Alice", "30", "eng"],
            ["Bob", "25", "sales"],
        ],
        merges=["A1:C1"],
    )
    vectors = xp.build_tabular_vectors(str(path), header_row=1)
    assert len(vectors) == 2

    v0 = vectors[0].model_dump()
    # ASCII 헤더 → 메타 KEY 로 부여
    assert v0["name"] == "Alice"
    assert v0["age"] == "30"
    assert v0["dept"] == "eng"
    # 페이지/청크 메타
    assert v0["i_page"] == 1 and v0["e_page"] == 1
    assert v0["n_chunk_of_doc"] == 2
    assert v0["i_chunk_on_doc"] == 0 and vectors[1].model_dump()["i_chunk_on_doc"] == 1
    # 값이 text 에도 포함
    assert "Alice" in v0["text"] and "name" in v0["text"]


@pytest.mark.unit
def test_tabular_merged_body_forward_fill(tmp_path):
    """본문 병합셀(그룹 컬럼)이 unmerge 후 forward-fill 되어 모든 행에 값이 채워진다."""
    xp = _xp()
    path = _make_xlsx(
        tmp_path / "g.xlsx",
        rows=[
            ["group", "name"],
            ["G1", "a"],
            [None, "b"],     # A3 은 A2 와 병합 → ffill 로 G1 채워짐
            ["G2", "c"],
        ],
        merges=["A2:A3"],
    )
    vectors = xp.build_tabular_vectors(str(path), header_row=0)
    assert len(vectors) == 3
    dumps = [v.model_dump() for v in vectors]
    assert dumps[0]["group"] == "G1" and dumps[0]["name"] == "a"
    assert dumps[1]["group"] == "G1" and dumps[1]["name"] == "b"   # forward-fill 확인
    assert dumps[2]["group"] == "G2" and dumps[2]["name"] == "c"


@pytest.mark.unit
def test_load_sheets_unmerge_forward_fill(tmp_path):
    """공개 load_sheets() — parser tabular 가 재사용. 병합 제목/그룹이 forward-fill 된다."""
    xp = _xp()
    path = _make_xlsx(
        tmp_path / "s.xlsx",
        rows=[
            ["REPORT", None, None],   # 병합 제목행(A1:C1)
            ["name", "age", "dept"],
            ["Alice", "30", "eng"],
            ["Bob", "25", "sales"],
        ],
        merges=["A1:C1"],
        sheet_name="S1",
    )
    sheets = xp.load_sheets(str(path))
    assert list(sheets.keys()) == ["S1"]
    rows = sheets["S1"]
    # 병합 제목행이 전 컬럼에 forward-fill
    assert rows[0] == ["REPORT", "REPORT", "REPORT"]
    assert rows[1] == ["name", "age", "dept"]
    assert rows[2] == ["Alice", "30", "eng"]


@pytest.mark.unit
@pytest.mark.skipif(not _SAMPLE.exists(), reason="해진공 샘플 xlsx 없음")
def test_tabular_korean_headers_dropped_from_keys():
    """한글 헤더는 Weaviate 키 규칙 위반이라 메타 KEY 에서 제외되고 값은 text 에 유지된다."""
    xp = _xp()
    vectors = xp.build_tabular_vectors(str(_SAMPLE))
    assert len(vectors) > 0

    v0 = vectors[0].model_dump()
    extra_keys = [k for k in v0 if k not in xp._RESERVED_FIELDS]
    # 한글 헤더가 메타 KEY 로 새지 않았는지(ASCII 규칙 위반 키 부재)
    assert all(xp._is_valid_key(k) for k in extra_keys)
    # 데이터는 text 에 살아있다
    assert "사번" in v0["text"] or "사원" in v0["text"]


# --------------------------------------------------------------------------- #
# docling 모드 (버그 픽스: 행이 페이지 경계로 쪼개지지 않음)                        #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
@pytest.mark.skipif(not _SAMPLE.exists(), reason="해진공 샘플 xlsx 없음")
def test_docling_single_page_no_row_split():
    pytest.importorskip("docling.backend.msexcel_backend")
    xp = _xp()
    doc = xp.build_docling_document(str(_SAMPLE))
    # 시트가 1개이므로 1페이지. PDF 변환 시 발생하던 행의 페이지 분할이 없다.
    assert doc.num_pages() == 1
    assert len(doc.tables) >= 1
    for t in doc.tables:
        pages = {p.page_no for p in t.prov}
        assert pages == {1}, f"표가 여러 페이지로 분할됨: {pages}"
        # 한 시트 전체 행이 단일 표로 유지
        assert t.data.num_rows >= 1


# --------------------------------------------------------------------------- #
# e2e (intelligent_processor 두 모드)                                           #
# --------------------------------------------------------------------------- #
def _make_e2e_config(tmp_path: Path, processing_mode: str) -> str:
    """출고 config 복사 + enrichment 비활성 + xlsx.processing_mode 지정."""
    cfg = yaml.safe_load(_CONFIG.read_text(encoding="utf-8"))
    cfg["enrichment"] = []  # 네트워크/LLM 호출 차단
    cfg.setdefault("xlsx", {})
    cfg["xlsx"]["processing_mode"] = processing_mode
    out = tmp_path / "intelligent_processor_config.yaml"
    out.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return str(out)


@pytest.mark.smoke
@pytest.mark.asyncio
@pytest.mark.skipif(not _SAMPLE.exists(), reason="해진공 샘플 xlsx 없음")
@pytest.mark.parametrize("mode", ["docling", "tabular"])
async def test_e2e_xlsx_modes(tmp_path, mode):
    mod = pytest.importorskip("facade.intelligent_processor")
    try:
        dp = mod.DocumentProcessor(config_path=_make_e2e_config(tmp_path, mode))
    except Exception as e:  # noqa: BLE001 - 모델/네트워크 등 환경 의존
        pytest.skip(f"DocumentProcessor init unavailable: {e}")

    vectors = await dp(None, str(_SAMPLE))
    assert isinstance(vectors, list) and len(vectors) >= 1
    v = vectors[0]
    if hasattr(v, "model_dump"):
        v = v.model_dump()
    assert isinstance(v.get("text"), str) and v["text"]
