"""PII 마스킹(#315) 헬퍼 단위 테스트.

검증 대상 (intelligent_processor 의 self-contained 블록; 4개 facade 에 동일 복제):
  - _pii_apply_masking: 가드레일 dry-run 응답 해석 (A형/동일/B형/실패 fail-open/설정누락)
  - _pii_chunk_status: 청크 단위 pii_status 집계 (우선순위 exposed > masked > unknown > none)

의존성(docling 등) 미가용 환경에서는 importorskip 으로 자동 skip 된다(CI gate).
requests.post 는 monkeypatch 로 대체해 실제 네트워크 없이 응답 형태만 검증한다.
"""

import pytest

_SEP = "\n<<<GRSEP>>>\n"


def _mod():
    return pytest.importorskip("facade.intelligent_processor")


class _FakeItem:
    def __init__(self, ref, text):
        self.self_ref = ref
        self.text = text


class _FakeDoc:
    """docling DoclingDocument 대역 — iterate_items 만 제공."""
    def __init__(self, items):
        self._items = items

    def iterate_items(self, *args, **kwargs):
        return [(it, None) for it in self._items]


def _items():
    return [
        _FakeItem("#/texts/0", "담당자 연락처 010-1234-5678"),
        _FakeItem("#/texts/1", "주민번호 850101-1234567"),
        _FakeItem("#/texts/2", "개인정보 없는 문장"),
    ]


class _Resp:
    def __init__(self, content):
        self._content = content

    def raise_for_status(self):
        pass

    def json(self):
        return {"code": 0, "errMsg": "success", "data": {"content": self._content}}


def _patch_requests(monkeypatch, responder):
    import requests
    monkeypatch.setattr(requests, "post", lambda *a, **k: responder(k.get("json") or a[1]))


def test_masking_a_type(monkeypatch):
    """A형(부분치환): PII 만 치환, 미탐지 문장 원문 보존, 아이템별 masked/none."""
    mod = _mod()

    def responder(payload):
        out = _SEP.join(
            p.replace("010-1234-5678", "[휴대전화]").replace("850101-1234567", "[주민등록번호]")
            for p in payload["content"].split(_SEP)
        )
        return _Resp(out)

    _patch_requests(monkeypatch, responder)
    items = _items()
    status = mod._pii_apply_masking(_FakeDoc(items), "http://gw", 96, 30)

    assert status == {"#/texts/0": "masked", "#/texts/1": "masked", "#/texts/2": "none"}
    assert items[0].text.endswith("[휴대전화]")
    assert items[2].text == "개인정보 없는 문장"  # 원문 보존


def test_masking_same(monkeypatch):
    """동일(미탐지): 응답이 원문과 같으면 전부 none, 텍스트 불변."""
    mod = _mod()
    _patch_requests(monkeypatch, lambda payload: _Resp(payload["content"]))
    items = _items()
    status = mod._pii_apply_masking(_FakeDoc(items), "http://gw", 96, 30)
    assert set(status.values()) == {"none"}
    assert items[0].text == "담당자 연락처 010-1234-5678"


def test_masking_b_type_block(monkeypatch):
    """B형(차단/통째교체): 구분자 소실 → 원문 보존, 전부 exposed."""
    mod = _mod()
    _patch_requests(monkeypatch, lambda payload: _Resp("해당 요청은 정책상 응답할 수 없습니다"))
    items = _items()
    status = mod._pii_apply_masking(_FakeDoc(items), "http://gw", 96, 30)
    assert set(status.values()) == {"exposed"}
    assert items[1].text == "주민번호 850101-1234567"  # 원문 보존(안내문구로 안 바뀜)


def test_masking_fail_open(monkeypatch):
    """호출 실패: fail-open — 원문 보존, 전부 unknown."""
    mod = _mod()

    def boom(*a, **k):
        raise RuntimeError("network down")

    import requests
    monkeypatch.setattr(requests, "post", boom)
    items = _items()
    status = mod._pii_apply_masking(_FakeDoc(items), "http://gw", 96, 30)
    assert set(status.values()) == {"unknown"}
    assert items[0].text == "담당자 연락처 010-1234-5678"


def test_masking_missing_config():
    """url/guardrail_id 미설정: 호출 없이 unknown(fail-open)."""
    mod = _mod()
    items = _items()
    status = mod._pii_apply_masking(_FakeDoc(items), "", None, 30)
    assert set(status.values()) == {"unknown"}


def test_masking_no_text_items():
    """마스킹 대상 텍스트 아이템이 없으면 빈 맵."""
    mod = _mod()
    status = mod._pii_apply_masking(_FakeDoc([]), "http://gw", 96, 30)
    assert status == {}


def test_chunk_status_aggregation():
    """청크 집계: 우선순위 exposed > masked > unknown > none, 대상 없으면 None."""
    mod = _mod()
    f = mod._pii_chunk_status
    assert f(["#/texts/0", "#/texts/2"], {"#/texts/0": "masked", "#/texts/2": "none"}) == "masked"
    assert f(["#/texts/2"], {"#/texts/2": "none"}) == "none"
    assert f(["a", "b"], {"a": "none", "b": "exposed"}) == "exposed"
    assert f(["a", "b"], {"a": "unknown", "b": "none"}) == "unknown"
    # 청크의 아이템이 마스킹 대상(맵)에 하나도 없으면 표기 안 함(None) — 표 전용 청크 등
    assert f(["#/tables/0"], {"#/texts/0": "masked"}) is None
    assert f(["x"], {}) is None
