"""xlsx 직접 처리(이슈 #288) 테스트.

- tabular 모드: 병합셀 unmerge+forward-fill, ASCII 키만 메타데이터, 한글 헤더 제외(값은 text 유지).
- docling 모드: 시트=1페이지로 변환되어 한 행이 페이지 경계로 쪼개지지 않음(버그 픽스).
- e2e: intelligent_processor 가 두 모드에서 non-empty 벡터를 반환(facade/fastapi 미가용 시 자동 skip).

mock 없이 실제 추출 경로를 호출한다. docling/facade 의존성 미가용 환경에서는 importorskip 으로 skip(CI gate).
"""

import json
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
    # ASCII 헤더 → 최상단 스칼라 property 로 부여(필터 가능)
    rf = v0
    assert rf["name"] == "Alice"
    assert rf["age"] == "30"
    assert rf["dept"] == "eng"
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
def test_tabular_auto_title_skip(tmp_path):
    """전열 병합 제목행은 자동으로 컨텍스트 처리되고 컬럼명행이 헤더가 된다(header_row 미지정)."""
    xp = _xp()
    path = _make_xlsx(
        tmp_path / "auto.xlsx",
        rows=[
            ["REPORT", None, None],   # 전열 병합 제목
            ["name", "age", "dept"],  # 컬럼명(leaf)
            ["Alice", "30", "eng"],
            ["Bob", "25", "sales"],
        ],
        merges=["A1:C1"],
    )
    vectors = xp.build_tabular_vectors(str(path))  # header_row 미지정 → 자동
    assert len(vectors) == 2
    v0 = vectors[0].model_dump()
    rf = v0
    assert rf["name"] == "Alice" and rf["age"] == "30" and rf["dept"] == "eng"
    assert "REPORT" in v0["text"]  # 제목은 컨텍스트로 포함
    assert v0["text"].count("REPORT") == 1  # 키로 flatten 되지 않음


@pytest.mark.unit
def test_tabular_multi_header_flatten(tmp_path):
    """부분 병합 계층 헤더는 '상위_하위' 로 flatten 되어 메타 키가 된다."""
    xp = _xp()
    path = _make_xlsx(
        tmp_path / "mh.xlsx",
        rows=[
            ["info", "info", "salary"],   # 상위(A1:B1 병합=info, C1=salary)
            ["name", "age", "base"],      # 하위(leaf)
            ["Alice", "30", "100"],
            ["Bob", "25", "200"],
        ],
        merges=["A1:B1"],
    )
    vectors = xp.build_tabular_vectors(str(path))
    assert len(vectors) == 2
    rf = vectors[0].model_dump()
    # 부분 병합 상위 + leaf → 상위_하위
    assert rf.get("info_name") == "Alice"
    assert rf.get("info_age") == "30"
    assert rf.get("salary_base") == "100"


@pytest.mark.unit
def test_tabular_stable_key_for_korean(tmp_path):
    """한글 헤더는 헤더 기반 안정 키(field_<hash>)로, ASCII 헤더는 그대로. 원본명은 column_map 보존."""
    xp = _xp()
    path = _make_xlsx(
        tmp_path / "kr.xlsx",
        rows=[
            ["이름", "age"],
            ["홍길동", "30"],
        ],
    )
    vectors = xp.build_tabular_vectors(str(path))
    assert len(vectors) == 1
    v0 = vectors[0].model_dump()
    assert v0["age"] == "30"                       # ASCII 헤더는 그대로(최상단)
    key = xp._stable_key("이름")                    # 한글 헤더의 안정 키
    assert key.startswith("field_")
    assert v0[key] == "홍길동"                       # 최상단 스칼라 property(필터 가능)
    column_map = json.loads(v0["column_map"])
    assert column_map[key] == "이름"                 # 원본 헤더명 보존
    # 같은 헤더 텍스트는 항상 같은 키(파일 간 안정)
    assert xp._stable_key("이름") == key


@pytest.mark.unit
def test_tabular_stable_key_cross_file(tmp_path):
    """서로 다른 파일이라도 같은 헤더('부서')는 같은 키가 되어 컬렉션 전체 필터가 안정적이다."""
    xp = _xp()
    p1 = _make_xlsx(tmp_path / "a.xlsx", rows=[["부서", "n"], ["AI전환팀", "1"]])
    p2 = _make_xlsx(tmp_path / "b.xlsx", rows=[["부서", "x"], ["재무팀", "2"]])
    cm1 = json.loads(xp.build_tabular_vectors(str(p1))[0].model_dump()["column_map"])
    v2 = xp.build_tabular_vectors(str(p2))[0].model_dump()
    cm2 = json.loads(v2["column_map"])
    # '부서' 의 키가 두 파일에서 동일
    key_a = next(k for k, name in cm1.items() if name == "부서")
    key_b = next(k for k, name in cm2.items() if name == "부서")
    assert key_a == key_b == xp._stable_key("부서")
    assert v2[key_b] == "재무팀"


@pytest.mark.unit
def test_load_tables_detection(tmp_path):
    """공개 load_tables — parser 등 비-벡터 소비자가 재사용하는 표 감지(멀티헤더/복수표/제목스킵)."""
    xp = _xp()
    # 제목행 + 부분병합 계층헤더 + 데이터, 그리고 빈 행으로 분리된 두번째 표
    path = _make_xlsx(
        tmp_path / "lt.xlsx",
        rows=[
            ["REPORT", None, None],       # 전열 병합 제목(A1:C1)
            ["info", "info", "salary"],   # 상위(A2:B2 병합)
            ["name", "age", "base"],      # leaf 컬럼명
            ["Alice", "30", "100"],
            [None, None, None],           # 빈 행 구분자
            ["city", "pop", None],
            ["Seoul", "900", None],
        ],
        merges=["A1:C1", "A2:B2"],
    )
    tables = xp.load_tables(str(path), multi_table=True)
    assert len(tables) == 2
    t0 = tables[0]
    assert t0["title"] == "REPORT"                       # 제목행은 title(컨텍스트)
    assert t0["headers"] == ["info_name", "info_age", "salary_base"]  # 계층 flatten
    assert t0["data_rows"] == [["Alice", "30", "100"]]
    t1 = tables[1]
    assert t1["headers"][:2] == ["city", "pop"]
    assert t1["data_rows"][0][:2] == ["Seoul", "900"]
    # multi_table=False 면 한 블록(두번째 표가 데이터로 섞임)
    assert len(xp.load_tables(str(path), multi_table=False)) == 1


@pytest.mark.unit
def test_tabular_multi_table_split(tmp_path):
    """multi_table=True 면 빈 행으로 분리된 표를 각각 헤더 재판정하여 별도 행 벡터로 만든다."""
    xp = _xp()
    path = _make_xlsx(
        tmp_path / "mt.xlsx",
        rows=[
            ["name", "age"],
            ["Alice", "30"],
            ["Bob", "25"],
            [None, None],        # 빈 행 구분자
            ["city", "pop"],
            ["Seoul", "900"],
        ],
    )
    off = xp.build_tabular_vectors(str(path), multi_table=False)
    on = xp.build_tabular_vectors(str(path), multi_table=True)
    # OFF: 단일 표(header=1행) → 두번째 표의 헤더/데이터가 데이터 행으로 섞임
    # ON: 표 2개 → 표1 데이터 2행 + 표2 데이터 1행 = 3
    assert len(on) == 3
    dumps = [v.model_dump() for v in on]
    assert dumps[0]["name"] == "Alice" and dumps[0]["age"] == "30"
    assert dumps[2].get("city") == "Seoul" and dumps[2].get("pop") == "900"
    assert len(off) != len(on)  # 분리 여부에 따라 벡터 수가 다름


@pytest.mark.unit
@pytest.mark.skipif(not _SAMPLE.exists(), reason="해진공 샘플 xlsx 없음")
def test_tabular_korean_headers_stable_key():
    """한글 헤더는 field_<hash> 안정 키로 최상단 property 부여, 원본명은 column_map 보존."""
    xp = _xp()
    vectors = xp.build_tabular_vectors(str(_SAMPLE))
    assert len(vectors) > 0

    v0 = vectors[0].model_dump()
    column_map = json.loads(v0["column_map"])
    # 컬럼 값 키는 모두 최상단 scalar property 이고 Weaviate 키 규칙에 맞는다(한글→field_<hash>)
    assert column_map and all(xp._is_valid_key(k) for k in column_map)
    assert any(k.startswith("field_") for k in column_map)
    # 각 키가 최상단에 실제 값으로 존재(필터 가능)
    for k in column_map:
        assert k in v0
    # column_map 에 원본 한글 헤더명이 보존된다
    assert "사번" in column_map.values() or "사원" in column_map.values()
    # 제목행(사원명부조회)은 컨텍스트로 text 에 포함(키 아님)
    assert "사원명부조회" in v0["text"]


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
    """출고 config 복사 + enrichment 비활성 + formats.xlsx.processing_mode 지정."""
    cfg = yaml.safe_load(_CONFIG.read_text(encoding="utf-8"))
    cfg["enrichment"] = []  # 네트워크/LLM 호출 차단
    cfg.setdefault("formats", {}).setdefault("xlsx", {})["processing_mode"] = processing_mode
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
