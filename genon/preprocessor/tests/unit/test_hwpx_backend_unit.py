"""
HWPX 백엔드 유효성 테스트 — 팀 컨벤션(test_docx_backend_unit.py)에 맞춤.
실제 샘플 파일로 InputDocument 생성 및 backend.is_valid()만 확인.
"""
from __future__ import annotations

from pathlib import Path
from typing import cast
import pytest


HWPX_SAMPLE = Path(__file__).resolve().parents[2] / "sample_files" / "hwpx_sample.hwpx"
HWP_SAMPLE = Path(__file__).resolve().parents[2] / "sample_files" / "hwp_sample.hwp"


@pytest.mark.unit
@pytest.mark.skipif(not HWPX_SAMPLE.exists(), reason="hwpx_sample.hwpx not found")
def test_hwpx_backend_valid_and_convert():
    """HwpxDocumentBackend가 hwpx 파일을 유효하게 인식하고 convert 가능한지 확인."""
    from docling.datamodel.document import InputDocument
    from docling.datamodel.base_models import InputFormat
    from docling.backend.xml.hwpx_backend import HwpxDocumentBackend

    in_doc = InputDocument(
        path_or_stream=HWPX_SAMPLE,
        format=InputFormat.XML_HWPX,
        backend=HwpxDocumentBackend,
        filename=HWPX_SAMPLE.name,
    )

    assert in_doc.valid is True
    assert in_doc._backend.is_valid() is True

    backend = cast(HwpxDocumentBackend, in_doc._backend)
    doc = backend.convert()
    assert doc is not None
    assert hasattr(doc, "texts")
    assert isinstance(doc.texts, list)
    assert len(doc.texts) >= 1


@pytest.mark.unit
@pytest.mark.skipif(not HWP_SAMPLE.exists(), reason="hwp_sample.hwp not found")
def test_genos_hwp_backend_valid():
    """GenosHwpDocumentBackend가 hwp 파일을 유효하게 인식하는지 확인."""
    from docling.datamodel.document import InputDocument
    from docling.datamodel.base_models import InputFormat
    from docling.backend.genos_hwp_backend import GenosHwpDocumentBackend

    in_doc = InputDocument(
        path_or_stream=HWP_SAMPLE,
        format=InputFormat.HWP,
        backend=GenosHwpDocumentBackend,
        filename=HWP_SAMPLE.name,
    )

    assert in_doc.valid is True
    assert in_doc._backend.is_valid() is True


# ---------------------------------------------------------------------------
# Issue #195: HWP SDK 수식(latex) 추출 → Docling 연결
# ---------------------------------------------------------------------------
# SDK가 emit하는 두 가지 수식 형태에 대한 처리를 격리 검증한다.
#   (1) 최상위 {"item": "latex", "value": "<base64>"} → DocItemLabel.FORMULA 노드
#   (2) 표 셀 HTML 안에 <latex value="<base64>"/> → 셀 텍스트 $<decoded>$로 치환
# 추가로 SDK가 base64 value에 줄바꿈을 끼우거나, <latex> 속성의 inner "를
# escape하지 않아 발생하는 비정상 JSON에 대한 보정 로직도 함께 검증한다.

def _make_backend_no_io():
    """SDK 실행/파일 I/O 없이 핸들러만 호출하기 위한 최소 초기화 인스턴스."""
    from pathlib import Path as _Path
    from docling.backend.genos_hwp_backend import GenosHwpDocumentBackend

    inst = GenosHwpDocumentBackend.__new__(GenosHwpDocumentBackend)
    inst.valid = True
    inst.source_path = _Path("/tmp/dummy.hwp")
    inst.original_path = inst.source_path
    inst.temp_input_path = None
    inst.include_wmf = False
    inst.save_images = False
    inst.dump_sdk_output = False
    inst._processed_hashes = set()
    inst.max_levels = 10
    inst.parents = {i: None for i in range(-1, inst.max_levels)}
    inst.history = {"names": [None], "levels": [None], "page_nos": [1]}
    inst.current_img_dir = None
    return inst


def _make_doc():
    from docling_core.types.doc import DoclingDocument, DocumentOrigin
    origin = DocumentOrigin(filename="test.hwp", mimetype="application/x-hwp", binary_hash=0)
    return DoclingDocument(name="test", origin=origin)


@pytest.mark.unit
def test_genos_hwp_backend_decode_latex_b64_basic():
    """base64로 인코딩된 latex value 가 LaTeX 원본으로 디코드된다."""
    from docling.backend.genos_hwp_backend import GenosHwpDocumentBackend

    # base64("\\sum_{x=0}^{\\infty}")
    raw = "XHN1bV97eD0wfV57XGluZnR5fQ=="
    assert GenosHwpDocumentBackend._decode_latex_b64(raw) == r"\sum_{x=0}^{\infty}"


@pytest.mark.unit
def test_genos_hwp_backend_decode_latex_b64_with_embedded_newlines():
    """SDK가 긴 base64 value에 줄바꿈을 끼워도 디코드가 정상 동작한다."""
    from docling.backend.genos_hwp_backend import GenosHwpDocumentBackend

    # 동일한 base64를 임의 위치에서 줄바꿈으로 쪼갬
    wrapped = "XHN1bV97e\nD0wfV57XGlu\nZnR5fQ=="
    assert GenosHwpDocumentBackend._decode_latex_b64(wrapped) == r"\sum_{x=0}^{\infty}"


@pytest.mark.unit
def test_genos_hwp_backend_decode_latex_b64_rejects_malformed_inputs(caplog):
    """손상된 입력은 garbage를 뱉지 않고 None + 경고 로그로 명시적 실패해야 한다.

    validate=False + errors="replace"로 관대하게 처리하면 깨진 base64나 깨진 UTF-8이
    조용히 통과되어 사일런트 버그가 되므로, strict 모드로 검증한다는 계약을 고정.
    """
    import base64
    import logging
    from docling.backend.genos_hwp_backend import GenosHwpDocumentBackend

    # 빈 입력 → None (경고 없음)
    assert GenosHwpDocumentBackend._decode_latex_b64("") is None
    assert GenosHwpDocumentBackend._decode_latex_b64(None) is None  # type: ignore[arg-type]

    # 1) base64 알파벳에 없는 문자가 섞인 입력
    with caplog.at_level(logging.WARNING, logger="docling.backend.genos_hwp_backend"):
        caplog.clear()
        assert GenosHwpDocumentBackend._decode_latex_b64("not_valid_base64!!!@@@") is None
        assert any("디코드 실패" in r.message for r in caplog.records)

    # 2) base64로는 디코드되지만 결과 바이트가 UTF-8이 아닌 경우
    # 0xFF, 0xFE 같은 단독 바이트는 UTF-8 시퀀스의 시작이 될 수 없다.
    invalid_utf8_b64 = base64.b64encode(b"\xff\xfe\xfd").decode("ascii")
    with caplog.at_level(logging.WARNING, logger="docling.backend.genos_hwp_backend"):
        caplog.clear()
        assert GenosHwpDocumentBackend._decode_latex_b64(invalid_utf8_b64) is None
        assert any("디코드 실패" in r.message for r in caplog.records)


@pytest.mark.unit
def test_genos_hwp_backend_normalize_sdk_json_escapes_latex_attr_quotes():
    """SDK 결과 JSON 안에 임베드된 <latex value="..."/> 의 inner "가 \"로 escape되고
    base64 안의 공백/줄바꿈이 제거되어, json.JSONDecoder가 outer 문자열을
    끝까지 파싱할 수 있도록 정규화된다.
    """
    import json
    from docling.backend.genos_hwp_backend import GenosHwpDocumentBackend

    # 정규화 전: outer JSON string 안에 <latex value="..."/>의 "가 raw로 들어 있어
    # 일반 json.loads로는 파싱 실패한다.
    raw = (
        '[{"item": "table", "value": "<table><tr><td>'
        '<latex value="XHN1bV97e\nD0wfV57XGlu\nZnR5fQ=="/>'
        '</td></tr></table>", "page": 1}]'
    )
    with pytest.raises(json.JSONDecodeError):
        json.loads(raw)

    normalized = GenosHwpDocumentBackend._normalize_sdk_json_text(raw)
    parsed = json.loads(normalized)
    assert isinstance(parsed, list) and len(parsed) == 1
    cell_html = parsed[0]["value"]
    # inner "는 escape 되고 base64는 한 줄로 합쳐졌어야 한다.
    assert '<latex value="XHN1bV97eD0wfV57XGluZnR5fQ=="/>' in cell_html


@pytest.mark.unit
def test_genos_hwp_backend_handle_latex_creates_formula_node():
    """최상위 {"item":"latex"} 아이템이 DocItemLabel.FORMULA 노드로 추가된다."""
    from docling_core.types.doc import DocItemLabel

    backend = _make_backend_no_io()
    doc = _make_doc()

    item = {
        "item": "latex",
        "value": "XHN1bV97eD0wfV57XGluZnR5fQ==",
        "font": {"name": "함초롬바탕", "size": 10.0},
        "page": 1,
    }
    before = len(doc.texts)
    backend._handle_latex(item, doc, page_no=1, parent=doc.body)
    added = doc.texts[before:]

    assert len(added) == 1
    assert added[0].label == DocItemLabel.FORMULA
    assert added[0].text == r"\sum_{x=0}^{\infty}"


@pytest.mark.unit
def test_genos_hwp_backend_handle_table_substitutes_embedded_latex():
    """표 셀 HTML 안의 <latex value="..."/> 가 $<decoded>$로 치환되어 cell text에 포함된다."""
    backend = _make_backend_no_io()
    doc = _make_doc()
    backend.active_main_parent = doc.body

    # base64("\\bar y_{T}") = "XGJhciB5X3tUfQ=="
    html = (
        "<table cols=2 rows=2>"
        "<tr><td>head1</td><td>head2</td></tr>"
        "<tr><td colspan=2 rowspan=1>"
        "(적률) <p style='font-size:13.0pt;'><latex value=\"XGJhciB5X3tUfQ==\"/></p> : T연도 소득"
        "</td></tr></table>"
    )
    before = len(doc.tables)
    backend._handle_table(html, doc, page_no=1, parent=doc.body)
    added = doc.tables[before:]

    assert len(added) == 1
    cell_with_latex = [
        c for c in added[0].data.table_cells if "T연도" in c.text
    ]
    assert len(cell_with_latex) == 1
    assert "$\\bar y_{T}$" in cell_with_latex[0].text


@pytest.mark.unit
def test_genos_hwp_backend_walk_emits_formula_for_top_level_latex_batch():
    """_walk_hwp_data가 latex 단독 batch를 만나면 FORMULA 노드를 만들고,
    이미지와 동일하게 누적된 텍스트 batch를 먼저 flush한다.
    """
    from docling_core.types.doc import DocItemLabel

    backend = _make_backend_no_io()
    doc = _make_doc()

    batches = [
        # 1) 일반 텍스트 paragraph
        [{"item": "text", "value": "수식 앞 문단", "font": {"size": 10.0}, "page": 1}],
        # 2) latex 단독 batch (실 SDK 출력의 전형적인 형태)
        [{"item": "latex", "value": "XHN1bV97eD0wfV57XGluZnR5fQ==",
          "font": {"size": 10.0}, "page": 1}],
        # 3) 일반 텍스트 paragraph
        [{"item": "text", "value": "수식 뒤 문단", "font": {"size": 10.0}, "page": 1}],
    ]
    backend._walk_hwp_data(batches, doc)

    labels = [t.label for t in doc.texts]
    assert labels.count(DocItemLabel.FORMULA) == 1
    formula = next(t for t in doc.texts if t.label == DocItemLabel.FORMULA)
    assert formula.text == r"\sum_{x=0}^{\infty}"
