"""
pytest에서 자동 로드되는 공통 설정 파일.
여기 정의된 픽스처들은 다른 테스트에서 import 없이 바로 사용 가능.
"""

import sys
from pathlib import Path
import pytest

# 이슈 #199 — pytest sys.path 보강.
# pyproject.toml(rootdir=genon/preprocessor) 의 pythonpath 가 "src" 만이라
# 로컬/일부 CI 환경에서 다음 두 가지 절대 import 가 깨질 수 있어 두 경로를 prepend:
#   - `genon.preprocessor.converters.hwp_to_pdf.*` (신규 모듈, src/ 밖)  → repo root 필요
#   - `facade.*`                                        (기존 facade 모듈) → genon/preprocessor 필요
_PREPROC = Path(__file__).resolve().parents[1]   # genon/preprocessor
_REPO_ROOT = Path(__file__).resolve().parents[3]  # repo root
for _p in (_PREPROC, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


# 프로젝트 루트 경로 반환
# scope="session" → 테스트 전체 실행 동안 한 번만 계산
@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


# 샘플 파일 디렉터리 경로 반환
# 예: sample_dir / "sample.pdf"
@pytest.fixture(scope="session")
def sample_dir(repo_root: Path) -> Path:
    return repo_root / "sample_files"


# Regression test용 샘플 파일 디렉터리
@pytest.fixture(scope="session")
def regression_test_dir(repo_root: Path) -> Path:
    """Regression test 전용 샘플 파일 디렉터리"""
    return repo_root / "sample_files" / "regression_test"


# DocumentProcessor 클래스를 안전하게 로드
# 모듈이 없으면 해당 테스트를 skip 처리
@pytest.fixture(scope="session")
def basic_processor():
    mod = pytest.importorskip("facade.attachment_processor")
    return mod.DocumentProcessor


# intelligent_processor 픽스처 추가
@pytest.fixture(scope="session")
def intelligent_processor():
    mod = pytest.importorskip("facade.intelligent_processor")
    return mod.DocumentProcessor

@pytest.fixture(scope="session")
def attachment_processor():
    mod = pytest.importorskip("facade.attachment_processor")
    return mod.DocumentProcessor


@pytest.fixture(scope="session")
def parser_processor():
    mod = pytest.importorskip("facade.parser_processor")
    return mod.DocumentProcessor


# TEDS-S 지표 함수 노출 (tests/teds_metric.py)
@pytest.fixture(scope="session")
def teds_s():
    """구조 전용 TEDS(TEDS-S) 계산 함수: teds_s(pred_html, gt_html) -> float."""
    _tests_dir = Path(__file__).resolve().parent
    if str(_tests_dir) not in sys.path:
        sys.path.insert(0, str(_tests_dir))
    from teds_metric import teds_s as _f
    return _f


# table.pdf 각 페이지 표의 GT(ground-truth) 로드.
# table.jsonl: 한 줄당 한 표, index i == table.pdf 페이지 (i+1) 의 표.
@pytest.fixture(scope="session")
def table_gt(repo_root: Path):
    import json

    gt_path = repo_root / "sample_files" / "table.jsonl"
    if not gt_path.exists():
        return []
    lines = gt_path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


@pytest.fixture(autouse=True)
def _stub_vlm_for_unit_tests(request, monkeypatch):
    """unit 테스트에서는 외부 VLM 호출을 기본 차단한다."""
    if request.node.get_closest_marker("unit") is None:
        return

    try:
        import facade.parser_processor as parser_mod
    except Exception:
        # parser_processor를 사용하지 않는 unit 테스트도 있으므로 조용히 패스
        return

    monkeypatch.setattr(
        parser_mod,
        "api_image_request",
        lambda *args, **kwargs: "",
        raising=False,
    )
