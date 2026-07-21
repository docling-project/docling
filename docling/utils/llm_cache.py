"""요청 스코프 LLM 파일 캐시 (#329, opt-in).

대용량 배치에서 문서 1건이 중간 실패하면 그때까지 성공한 LLM 호출(OCR/이미지설명/
TOC/메타데이터 등)이 전부 유실되고 재과금·재실행된다. 이 모듈은 LLM 호출의 입출력을
`<INTERIM_ROOT>/<workflow_id>/<run_id>/llm_cache/<key>.json` 에 저장해 재시도 시
재호출 없이 재사용하게 한다(Temporal worker 와 공유하는 NFS 전제).

설계 원칙:
- **opt-in**: 요청 `params.llm_cache: true` 이고 `workflow_id`/`INTERIM_ROOT` 가 있을 때만
  동작한다. 그 외에는 캐시 코드 경로 자체를 타지 않아(추가 I/O 0) 기존과 바이트 동일하다.
- **요청 스코프**: 프로세서가 싱글턴으로 재사용되고 LLM 호출이 ThreadPoolExecutor 워커
  스레드에서 실행되므로, 컨텍스트는 인스턴스 속성이 아니라 `contextvars.ContextVar` 로
  전달한다. executor 로 넘길 때는 호출 측에서 `contextvars.copy_context()` 로 전파해야 한다.
- **의존성 방향**: docling 리프 함수와 3개 facade 가 모두 import 하는 하위 계층이므로
  `genon.*` 를 import 하지 않는다. `INTERIM_ROOT` 는 `os.getenv` 로 지연 조회한다.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional, Tuple, TypeVar

_log = logging.getLogger(__name__)

T = TypeVar("T")

# 캐시 저장 envelope 스키마 버전 (forward-compat). 값이 바뀌면 기존 캐시는 miss 처리된다.
_ENVELOPE_VERSION = 1


class CacheDeadlineExceeded(Exception):
    """요청 전체 deadline 을 넘겨 남은 per-call timeout 이 소진됐을 때(→ error_type=timeout)."""


@dataclass
class _Counters:
    hit: int = 0
    miss: int = 0
    save_fail: int = 0

    def summary(self) -> str:
        return f"hit={self.hit} miss={self.miss} save_fail={self.save_fail}"


@dataclass(frozen=True)
class CacheContext:
    """요청 1건에 대한 캐시/타임아웃 컨텍스트.

    enabled=False 이면 어떤 캐시 I/O 도 하지 않는다(기존과 동일 동작).
    """

    enabled: bool = False
    cache_dir: Optional[str] = None
    # 요청 전체 deadline (time.monotonic() 기준 절대 시각). None 이면 요청 deadline 없음.
    deadline: Optional[float] = None
    # per-call 기본 timeout(초). remaining_timeout 의 fallback 이 없을 때 쓴다.
    default_timeout: float = 3600.0
    # "strict" | "lenient" — enrichment 실패 처리 정책(단일 소스). 리프/facade 가 공유.
    error_policy: str = "lenient"
    _counters: _Counters = field(default_factory=_Counters, compare=False)


_DISABLED = CacheContext()

_CTX: "contextvars.ContextVar[CacheContext]" = contextvars.ContextVar(
    "llm_cache_ctx", default=_DISABLED
)


# ---------------------------------------------------------------------------
# 컨텍스트 set/get (facade __call__ 에서 사용)
# ---------------------------------------------------------------------------


def set_context(ctx: CacheContext) -> "contextvars.Token[CacheContext]":
    return _CTX.set(ctx)


def reset_context(token: "contextvars.Token[CacheContext]") -> None:
    try:
        _CTX.reset(token)
    except (ValueError, LookupError):
        # 다른 컨텍스트에서 생성된 토큰(예: copy_context 경계)이면 무시.
        pass


def current_context() -> CacheContext:
    return _CTX.get()


def in_current_context(fn: Callable[..., T]) -> Callable[..., T]:
    """현재 CacheContext 를 캡처해, 다른 스레드에서 실행되어도 그 컨텍스트를 활성화하는 래퍼.

    LLM 호출은 대부분 ThreadPoolExecutor 워커 스레드에서 실행되는데, ContextVar 는
    스레드별 저장이라 워커 스레드에서는 기본값(disabled)로 보인다. 그러면 hot path
    (VLM OCR / 이미지·표 description)에서 캐시가 조용히 무력화된다. 그래서 submit/map
    전에 이 래퍼로 감싸 각 워커가 캡처된 컨텍스트를 set/reset 하도록 한다.

    NOTE: contextvars.copy_context() 를 여러 워커가 공유하면 "context already entered"
    가 나므로, Context 객체가 아니라 CacheContext 값을 스레드별로 set 한다.
    사용: ``executor.map(in_current_context(_process), items)``
    """
    captured = _CTX.get()

    def _wrapped(*args: Any, **kwargs: Any) -> T:
        token = _CTX.set(captured)
        try:
            return fn(*args, **kwargs)
        finally:
            reset_context(token)

    return _wrapped


def get_interim_root() -> Optional[str]:
    """interim root 조회(genon.* import 회피, env 지연 조회).

    우선순위: env `INTERIM_ROOT` > `NFS_ROOT_DIR`(GenOS 관례, 기본 `/nfs-root`) 기반 기본
    → 미설정 시 기본 `/nfs-root/interim`. (요청 params 의 interim_root 는 build_context 에서 최우선)
    """
    root = os.getenv("INTERIM_ROOT")
    if root:
        return root
    return os.path.join(os.getenv("NFS_ROOT_DIR", "/nfs-root"), "interim")


def build_context(
    *,
    llm_cache: bool,
    workflow_id: Optional[str],
    run_id: Optional[str],
    deadline: Optional[float] = None,
    default_timeout: float = 3600.0,
    error_policy: str = "lenient",
    interim_root: Optional[str] = None,
) -> CacheContext:
    """요청 파라미터로 CacheContext 를 만든다.

    llm_cache=True 이고 workflow_id 와 interim root 가 모두 있을 때만 enabled 이다.
    (run_id 가 비면 "default" 로 대체 — /run 처럼 run_id 가 없어도 workflow_id 만으로 유효.)

    interim root 우선순위: 요청 params 의 interim_root > env INTERIM_ROOT > 기본 `/nfs-root/interim`
    (NFS_ROOT_DIR 기반). 기본값이 항상 확보되므로, 캐시 활성은 사실상 llm_cache + workflow_id 로 결정된다.
    """
    root = (str(interim_root).strip() if interim_root else "") or get_interim_root()
    enabled = bool(llm_cache and workflow_id and root)
    cache_dir: Optional[str] = None
    if enabled:
        cache_dir = os.path.join(
            root, str(workflow_id), str(run_id or "default"), "llm_cache"
        )
    return CacheContext(
        enabled=enabled,
        cache_dir=cache_dir,
        deadline=deadline,
        default_timeout=default_timeout,
        error_policy=error_policy,
    )


_TRUE_STRINGS = {"1", "true", "yes", "y", "on"}


def _coerce_bool(value: Any) -> bool:
    """llm_cache 플래그 정규화. bool/int(0/1)/문자열("true"/"1"/"on" 등) 수용, 그 외 False."""
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in _TRUE_STRINGS
    return False


def _coerce_error_policy(value: Any) -> str:
    """error_policy 정규화. "strict" | "lenient"(기본). 미지정/미인식은 lenient(하위호환)."""
    return "strict" if str(value or "").strip().lower() == "strict" else "lenient"


def _coerce_deadline(value: Any) -> Optional[float]:
    """request_deadline(초) → time.monotonic() 기준 절대 deadline. 없거나 <=0 이면 None."""
    if value is None or value == "":
        return None
    try:
        secs = float(value)
    except (TypeError, ValueError):
        return None
    return time.monotonic() + secs if secs > 0 else None


def resolve_context(
    kwargs: Any, *, workflow_id: Any = None, run_id: Any = None
) -> CacheContext:
    """요청 kwargs 로 CacheContext 를 만든다(3개 facade 공용).

    - `llm_cache`(true/false 또는 0/1) AND `workflow_id` AND `INTERIM_ROOT` → 캐시 enabled.
    - `error_policy` 는 캐시 enabled 여부와 무관하게 항상 컨텍스트에 실린다(strict 처리용).
    - `request_deadline`(초)이 있으면 per-call/요청 deadline 으로 쓴다.
    workflow_id/run_id 를 명시로 넘기면(예: /chunk 의 interim_ref) kwargs 값보다 우선한다.
    """
    kw = kwargs if isinstance(kwargs, dict) else {}
    wf = workflow_id if workflow_id is not None else kw.get("workflow_id")
    run = run_id if run_id is not None else kw.get("run_id")
    return build_context(
        llm_cache=_coerce_bool(kw.get("llm_cache")),
        workflow_id=wf,
        run_id=run,
        deadline=_coerce_deadline(kw.get("request_deadline")),
        error_policy=_coerce_error_policy(kw.get("error_policy")),
        interim_root=kw.get("interim_root"),   # POST params 로 받은 interim root(없으면 env fallback)
    )


def parse_interim_ref(ref: Any) -> Tuple[Optional[str], Optional[str]]:
    """interim_ref("<workflow_id>/<run_id>") → (workflow_id, run_id).

    빈 값/형식 불일치는 (None, None) 또는 (workflow_id, None) 로 관대하게 처리.
    """
    if not ref or not isinstance(ref, str):
        return None, None
    parts = ref.strip().strip("/").split("/")
    parts = [p for p in parts if p]
    if not parts:
        return None, None
    if len(parts) == 1:
        return parts[0], None
    # 마지막 두 세그먼트를 workflow_id/run_id 로 본다.
    return parts[-2], parts[-1]


# ---------------------------------------------------------------------------
# 키 / timeout
# ---------------------------------------------------------------------------


def _canonical(payload: Any) -> str:
    return json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str
    )


def cache_key(endpoint: str, payload: Any) -> str:
    """hash(endpoint + canonical(payload)). endpoint 를 포함해 모델/URL 충돌을 막는다."""
    h = hashlib.sha256()
    h.update(str(endpoint).encode("utf-8"))
    h.update(b"\x00")
    h.update(_canonical(payload).encode("utf-8"))
    return h.hexdigest()


def classify_error(exc: BaseException) -> str:
    """예외를 error_type("timeout" | "transient" | "permanent")으로 분류.

    requests/httpx 를 import 하지 않고 예외 클래스명/status_code 로 휴리스틱 판정한다
    (llm_cache 는 경량 하위 유틸이라 무거운 의존성을 두지 않는다).
    - timeout: deadline 초과, *Timeout* 계열, asyncio.TimeoutError
    - transient: 연결/네트워크 오류, HTTP 5xx (재시도로 나을 수 있음)
    - permanent: HTTP 4xx, 파싱/값 오류 등 (재시도해도 동일)
    """
    if isinstance(exc, CacheDeadlineExceeded):
        return "timeout"
    name = type(exc).__name__.lower()
    if "timeout" in name:
        return "timeout"
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        if status >= 500:
            return "transient"
        if 400 <= status < 500:
            return "permanent"
    if any(k in name for k in ("connection", "connect", "network", "remoteprotocol",
                               "readerror", "writeerror", "pool")):
        return "transient"
    return "permanent"


def remaining_timeout(fallback: Optional[float] = None) -> float:
    """이 호출에 허용되는 per-call timeout(초).

    요청 deadline 이 있으면 min(fallback, 남은시간). 남은시간이 0 이하이면
    CacheDeadlineExceeded 를 던져 행잉 대신 timeout 응답으로 이어지게 한다.
    """
    ctx = _CTX.get()
    base = fallback if fallback is not None else ctx.default_timeout
    if ctx.deadline is None:
        return base
    remaining = ctx.deadline - time.monotonic()
    if remaining <= 0:
        raise CacheDeadlineExceeded("request deadline exceeded before LLM call")
    return min(base, remaining)


# ---------------------------------------------------------------------------
# 캐시 read/write (내부)
# ---------------------------------------------------------------------------


def _read_cache(path: str, deserialize: Callable[[Any], T]) -> Tuple[bool, Optional[T]]:
    """(hit, value). 파일 없음/파싱 실패/버전 불일치는 모두 miss 로 처리."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            envelope = json.load(f)
    except FileNotFoundError:
        return False, None
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        # 부분 쓰기/손상 → miss 후 재호출·덮어쓰기.
        _log.warning("[llm_cache] corrupt cache entry, treating as miss: %s (%s)", path, exc)
        return False, None
    if not isinstance(envelope, dict) or envelope.get("v") != _ENVELOPE_VERSION:
        return False, None
    try:
        return True, deserialize(envelope.get("value"))
    except Exception as exc:  # noqa: BLE001 - deserialize 실패도 miss 로 폴백
        _log.warning("[llm_cache] deserialize failed, treating as miss: %s (%s)", path, exc)
        return False, None


def _write_cache(
    cache_dir: str, path: str, value: T, serialize: Callable[[T], Any], counters: _Counters
) -> None:
    """temp 파일 쓰기 후 atomic rename. 실패해도 예외를 삼키고 save_fail 만 센다."""
    try:
        os.makedirs(cache_dir, exist_ok=True)
        envelope = {"v": _ENVELOPE_VERSION, "value": serialize(value)}
        data = json.dumps(envelope, ensure_ascii=False, default=str)
        # 같은 디렉터리에 temp 를 만들어 os.replace 가 동일 FS atomic rename 이 되게 한다.
        fd, tmp_path = tempfile.mkstemp(dir=cache_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(data)
            os.replace(tmp_path, path)  # 동일 키 동시 쓰기는 내용이 같아 락 불필요.
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
    except Exception as exc:  # noqa: BLE001 - 캐시 저장 실패는 best-effort
        counters.save_fail += 1
        _log.warning("[llm_cache] failed to write cache entry: %s (%s)", path, exc)


def _identity(x: Any) -> Any:
    return x


def _should_cache(value: Any) -> bool:
    """None/빈 문자열 같은 '실패성 성공'은 캐시하지 않는다(재호출 기회 보존)."""
    if value is None:
        return False
    if isinstance(value, str) and value.strip() == "":
        return False
    return True


# ---------------------------------------------------------------------------
# 공개 진입점: 6개 헬퍼가 사용
# ---------------------------------------------------------------------------


def cached_call(
    endpoint: str,
    payload: Any,
    produce: Callable[[], T],
    *,
    serialize: Callable[[T], Any] = _identity,
    deserialize: Callable[[Any], T] = _identity,
    should_cache: Callable[[Any], bool] = _should_cache,
) -> T:
    """캐시 경유 LLM 호출(동기).

    disabled 이면 produce() 를 그대로 반환(추가 I/O 0). enabled 이면 hit 시 재사용,
    miss 시 produce 후 atomic 저장. should_cache 로 결과별 저장 판정을 커스터마이즈할 수 있다
    (예: VLM 튜플의 content 가 비면 저장 안 함).
    """
    ctx = _CTX.get()
    if not ctx.enabled or not ctx.cache_dir:
        return produce()

    key = cache_key(endpoint, payload)
    path = os.path.join(ctx.cache_dir, f"{key}.json")

    hit, value = _read_cache(path, deserialize)
    if hit:
        ctx._counters.hit += 1
        _log.info("[llm_cache] HIT  — 캐시 재사용, LLM 호출 안 함 (endpoint=%s key=%s)", endpoint, key[:12])
        return value

    ctx._counters.miss += 1
    _log.info("[llm_cache] MISS — 캐시 없음, LLM 실제 호출 (endpoint=%s key=%s)", endpoint, key[:12])
    result = produce()
    if should_cache(result):
        _write_cache(ctx.cache_dir, path, result, serialize, ctx._counters)
        _log.info("[llm_cache] STORE — LLM 결과 캐시 저장 (key=%s)", key[:12])
    return result


async def async_cached_call(
    endpoint: str,
    payload: Any,
    produce: Callable[[], Awaitable[T]],
    *,
    serialize: Callable[[T], Any] = _identity,
    deserialize: Callable[[Any], T] = _identity,
    should_cache: Callable[[Any], bool] = _should_cache,
) -> T:
    """cached_call 의 async 변형. 파일 I/O 는 to_thread 로 이벤트 루프 블로킹을 피한다."""
    import asyncio

    ctx = _CTX.get()
    if not ctx.enabled or not ctx.cache_dir:
        return await produce()

    key = cache_key(endpoint, payload)
    path = os.path.join(ctx.cache_dir, f"{key}.json")

    hit, value = await asyncio.to_thread(_read_cache, path, deserialize)
    if hit:
        ctx._counters.hit += 1
        _log.info("[llm_cache] HIT  — 캐시 재사용, LLM 호출 안 함 (endpoint=%s key=%s)", endpoint, key[:12])
        return value

    ctx._counters.miss += 1
    _log.info("[llm_cache] MISS — 캐시 없음, LLM 실제 호출 (endpoint=%s key=%s)", endpoint, key[:12])
    result = await produce()
    if should_cache(result):
        await asyncio.to_thread(
            _write_cache, ctx.cache_dir, path, result, serialize, ctx._counters
        )
        _log.info("[llm_cache] STORE — LLM 결과 캐시 저장 (key=%s)", key[:12])
    return result


def log_summary(prefix: str = "llm_cache") -> None:
    """요청 종료 시 hit/miss/save_fail 요약 로그(enabled 일 때만)."""
    ctx = _CTX.get()
    if ctx.enabled:
        _log.info("[%s] %s dir=%s", prefix, ctx._counters.summary(), ctx.cache_dir)
