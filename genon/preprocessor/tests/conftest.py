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
