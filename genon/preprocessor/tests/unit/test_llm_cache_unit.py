"""docling/utils/llm_cache.py 순수 로직 단위 테스트 (#329).

llm_cache 는 stdlib 만 의존하고 facade/무거운 docling 를 import 하지 않으므로,
로컬 vendored-docling 충돌과 무관하게 어디서든 import·실행된다(그래서 facade skip-gate
없이 직접 importorskip 한다). 실제 LLM/네트워크는 호출하지 않는다(produce 콜백을 mock).

검증 대상:
- cache_key: 결정성 / 키 순서 무관 / endpoint 분리
- cached_call: disabled→I/O 0 / miss→hit / None·빈문자열 미캐시 / 손상 파일→miss→덮어쓰기
- atomic write 파일 생성, 동일 키 동시 쓰기 안전
- resolve_context: llm_cache + workflow_id + INTERIM_ROOT 모두 있을 때만 enabled
- parse_interim_ref / classify_error / remaining_timeout(deadline)
- ContextVar + ThreadPoolExecutor 전파 회귀(in_current_context 없으면 무력화되는지)
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

m = pytest.importorskip("docling.utils.llm_cache")


@pytest.fixture
def enabled_ctx(tmp_path, monkeypatch):
    """INTERIM_ROOT 를 tmp 로 두고 enabled CacheContext 를 활성화하는 픽스처."""
    monkeypatch.setenv("INTERIM_ROOT", str(tmp_path))
    ctx = m.build_context(llm_cache=True, workflow_id="wf", run_id="run")
    token = m.set_context(ctx)
    try:
        yield ctx
    finally:
        m.reset_context(token)


# ── cache_key ──────────────────────────────────────────────────────────────

def test_cache_key_is_deterministic_and_order_independent():
    k1 = m.cache_key("http://x/chat", {"a": 1, "b": 2})
    k2 = m.cache_key("http://x/chat", {"b": 2, "a": 1})
    assert k1 == k2


def test_cache_key_includes_endpoint():
    assert m.cache_key("http://x", {"a": 1}) != m.cache_key("http://y", {"a": 1})


# ── disabled(=기본) 경로: 추가 I/O 없이 produce 그대로 ─────────────────────────

def test_disabled_context_calls_produce_every_time(tmp_path):
    assert m.current_context().enabled is False
    calls = {"n": 0}

    def produce():
        calls["n"] += 1
        return "R"

    assert m.cached_call("e", {"p": 1}, produce) == "R"
    assert m.cached_call("e", {"p": 1}, produce) == "R"
    assert calls["n"] == 2  # 캐시 안 탐


# ── enabled 경로: miss → hit ─────────────────────────────────────────────────

def test_enabled_miss_then_hit(enabled_ctx):
    calls = {"n": 0}

    def produce():
        calls["n"] += 1
        return {"text": "hello"}

    r1 = m.cached_call("http://x", {"q": 1}, produce)  # miss
    r2 = m.cached_call("http://x", {"q": 1}, produce)  # hit
    assert r1 == r2 == {"text": "hello"}
    assert calls["n"] == 1
    assert enabled_ctx._counters.hit == 1
    assert enabled_ctx._counters.miss == 1


def test_enabled_writes_cache_file(enabled_ctx):
    m.cached_call("http://x", {"q": 9}, lambda: "v")
    files = list(map(str, __import__("pathlib").Path(enabled_ctx.cache_dir).glob("*.json")))
    assert len(files) == 1


def test_none_and_empty_not_cached(enabled_ctx):
    calls = {"n": 0}

    def produce_none():
        calls["n"] += 1
        return None

    m.cached_call("http://x", {"q": 2}, produce_none)
    m.cached_call("http://x", {"q": 2}, produce_none)
    assert calls["n"] == 2  # None 은 저장 안 되므로 매번 재호출

    calls2 = {"n": 0}

    def produce_empty():
        calls2["n"] += 1
        return "   "

    m.cached_call("http://x", {"q": 3}, produce_empty)
    m.cached_call("http://x", {"q": 3}, produce_empty)
    assert calls2["n"] == 2


def test_corrupt_cache_file_is_miss_then_overwritten(enabled_ctx):
    from pathlib import Path

    def produce():
        return {"text": "ok"}

    m.cached_call("http://x", {"q": 4}, produce)  # miss → 저장
    f = next(Path(enabled_ctx.cache_dir).glob("*.json"))
    f.write_text("{not valid json", encoding="utf-8")

    calls = {"n": 0}

    def produce2():
        calls["n"] += 1
        return {"text": "ok2"}

    out = m.cached_call("http://x", {"q": 4}, produce2)  # 손상 → miss → 재호출
    assert calls["n"] == 1
    assert out == {"text": "ok2"}


def test_custom_serialize_deserialize_roundtrip(enabled_ctx):
    def produce():
        return ("content", {"tokens": 3}, "stop")

    ser = lambda t: {"content": t[0], "usage": t[1], "finish": t[2]}
    de = lambda d: (d["content"], d["usage"], d["finish"])
    r1 = m.cached_call("http://vlm", {"img": "b64"}, produce, serialize=ser, deserialize=de)
    r2 = m.cached_call("http://vlm", {"img": "b64"}, produce, serialize=ser, deserialize=de)
    assert r1 == r2 == ("content", {"tokens": 3}, "stop")


def test_concurrent_identical_writes_are_safe(enabled_ctx):
    def produce():
        return {"v": 1}

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(
            m.in_current_context(lambda i: m.cached_call("http://x", {"same": "key"}, produce)),
            range(16),
        ))
    assert all(r == {"v": 1} for r in results)


# ── resolve_context / build_context 게이팅 ───────────────────────────────────

def test_resolve_context_requires_all_of_flag_wf_and_root(tmp_path, monkeypatch):
    monkeypatch.setenv("INTERIM_ROOT", str(tmp_path))
    # 셋 다 있으면 enabled
    assert m.resolve_context({"llm_cache": True, "workflow_id": "w", "run_id": "r"}).enabled
    # llm_cache 없으면 disabled
    assert not m.resolve_context({"workflow_id": "w"}).enabled
    # workflow_id 없으면 disabled(= /run 등)
    assert not m.resolve_context({"llm_cache": True}).enabled
    # INTERIM_ROOT 없어도 기본값(/nfs-root/interim)으로 활성 — 캐시는 사실상 llm_cache+workflow_id 로 결정.
    monkeypatch.delenv("INTERIM_ROOT", raising=False)
    assert m.resolve_context({"llm_cache": True, "workflow_id": "w"}).enabled


def test_llm_cache_flag_accepts_0_1_and_strings(tmp_path, monkeypatch):
    monkeypatch.setenv("INTERIM_ROOT", str(tmp_path))
    for truthy in (True, 1, "1", "true", "on", "yes"):
        assert m.resolve_context({"llm_cache": truthy, "workflow_id": "w"}).enabled, truthy
    for falsy in (False, 0, "0", "false", "off", None, ""):
        assert not m.resolve_context({"llm_cache": falsy, "workflow_id": "w"}).enabled, falsy


def test_error_policy_defaults_lenient_and_parses_strict():
    assert m.resolve_context({}).error_policy == "lenient"
    assert m.resolve_context({"error_policy": "strict"}).error_policy == "strict"
    assert m.resolve_context({"error_policy": "STRICT"}).error_policy == "strict"
    assert m.resolve_context({"error_policy": "whatever"}).error_policy == "lenient"


def test_resolve_context_scope_override_from_interim_ref(tmp_path, monkeypatch):
    monkeypatch.setenv("INTERIM_ROOT", str(tmp_path))
    wf, run = m.parse_interim_ref("myflow/myrun")
    ctx = m.resolve_context({"llm_cache": True}, workflow_id=wf, run_id=run)
    assert ctx.enabled
    assert ctx.cache_dir.endswith("myflow/myrun/llm_cache")


# ── #329: interim_root 를 POST params 로 수신 ────────────────────────────────

def test_interim_root_from_request_params(tmp_path, monkeypatch):
    """요청 params.interim_root 만으로 (env 없이) 캐시 스코프가 잡힌다."""
    monkeypatch.delenv("INTERIM_ROOT", raising=False)
    root = str(tmp_path / "reqroot")
    ctx = m.resolve_context({"llm_cache": 1, "workflow_id": "w", "run_id": "r",
                             "interim_root": root})
    assert ctx.enabled
    assert ctx.cache_dir == os.path.join(root, "w", "r", "llm_cache")


def test_request_interim_root_overrides_env(tmp_path, monkeypatch):
    """요청 interim_root 가 env INTERIM_ROOT 보다 우선한다."""
    monkeypatch.setenv("INTERIM_ROOT", str(tmp_path / "envroot"))
    req = str(tmp_path / "reqroot")
    ctx = m.resolve_context({"llm_cache": 1, "workflow_id": "w", "interim_root": req})
    assert ctx.cache_dir == os.path.join(req, "w", "default", "llm_cache")


def test_blank_interim_root_falls_back_to_env(tmp_path, monkeypatch):
    """interim_root 가 공백/빈문자열이면 env fallback."""
    envroot = str(tmp_path / "envroot")
    monkeypatch.setenv("INTERIM_ROOT", envroot)
    ctx = m.resolve_context({"llm_cache": 1, "workflow_id": "w", "interim_root": "   "})
    assert ctx.cache_dir == os.path.join(envroot, "w", "default", "llm_cache")


def test_defaults_to_nfs_root_interim(monkeypatch):
    """요청·env 둘 다 없으면 기본 /nfs-root/interim 로 활성(NFS_ROOT_DIR 기반)."""
    monkeypatch.delenv("INTERIM_ROOT", raising=False)
    monkeypatch.delenv("NFS_ROOT_DIR", raising=False)
    ctx = m.resolve_context({"llm_cache": 1, "workflow_id": "w"})
    assert ctx.enabled
    assert ctx.cache_dir == os.path.join("/nfs-root", "interim", "w", "default", "llm_cache")
    # NFS_ROOT_DIR 오버라이드 시 그에 맞춰짐
    monkeypatch.setenv("NFS_ROOT_DIR", "/mnt/shared")
    ctx2 = m.resolve_context({"llm_cache": 1, "workflow_id": "w", "run_id": "r"})
    assert ctx2.cache_dir == os.path.join("/mnt/shared", "interim", "w", "r", "llm_cache")


# ── parse_interim_ref ────────────────────────────────────────────────────────

@pytest.mark.parametrize("ref,expected", [
    ("wf/run", ("wf", "run")),
    ("a/b/c/d", ("c", "d")),
    ("solo", ("solo", None)),
    ("/wf/run/", ("wf", "run")),
    ("", (None, None)),
    (None, (None, None)),
])
def test_parse_interim_ref(ref, expected):
    assert m.parse_interim_ref(ref) == expected


# ── classify_error ───────────────────────────────────────────────────────────

def test_classify_error():
    class _Timeout(Exception):
        pass

    class _ConnectionError(Exception):
        pass

    class _LLMApiError(Exception):
        def __init__(self, sc):
            self.status_code = sc

    assert m.classify_error(m.CacheDeadlineExceeded("x")) == "timeout"
    assert m.classify_error(_Timeout()) == "timeout"
    assert m.classify_error(_ConnectionError()) == "transient"
    assert m.classify_error(_LLMApiError(503)) == "transient"
    assert m.classify_error(_LLMApiError(404)) == "permanent"
    assert m.classify_error(ValueError("x")) == "permanent"


# ── remaining_timeout / deadline ─────────────────────────────────────────────

def test_remaining_timeout_no_deadline_returns_fallback():
    assert m.current_context().enabled is False
    assert m.remaining_timeout(123.0) == 123.0


def test_remaining_timeout_caps_to_deadline(tmp_path, monkeypatch):
    monkeypatch.setenv("INTERIM_ROOT", str(tmp_path))
    ctx = m.build_context(llm_cache=False, workflow_id=None, run_id=None,
                          deadline=time.monotonic() + 5)
    token = m.set_context(ctx)
    try:
        assert m.remaining_timeout(3600) <= 5.0
    finally:
        m.reset_context(token)


def test_remaining_timeout_raises_when_deadline_passed():
    ctx = m.build_context(llm_cache=False, workflow_id=None, run_id=None,
                          deadline=time.monotonic() - 1)
    token = m.set_context(ctx)
    try:
        with pytest.raises(m.CacheDeadlineExceeded):
            m.remaining_timeout(100)
    finally:
        m.reset_context(token)


# ── ContextVar + ThreadPoolExecutor 전파 회귀 ────────────────────────────────

def test_context_propagates_into_threadpool_workers(enabled_ctx):
    """in_current_context 로 감싸면 워커 스레드에서도 캐시가 동작해야 한다."""
    seen = []

    def worker(i):
        seen.append(m.current_context().enabled)
        return m.cached_call("http://x", {"i": i}, lambda: {"v": i})

    with ThreadPoolExecutor(max_workers=4) as ex:
        r1 = list(ex.map(m.in_current_context(worker), range(4)))
        r2 = list(ex.map(m.in_current_context(worker), range(4)))  # hit
    assert all(seen)
    assert r1 == r2
    assert enabled_ctx._counters.hit == 4
    assert enabled_ctx._counters.miss == 4


def test_bare_worker_does_not_see_context(enabled_ctx):
    """대조군: in_current_context 없이 넘기면 워커는 disabled 로 보인다(=caveat 실재)."""
    seen = []

    def worker(i):
        seen.append(m.current_context().enabled)
        return i

    with ThreadPoolExecutor(max_workers=2) as ex:
        list(ex.map(worker, range(2)))
    assert not any(seen)
