"""
pytest에서 자동 로드되는 공통 설정 파일.
여기 정의된 픽스처들은 다른 테스트에서 import 없이 바로 사용 가능.
"""

import sys
from pathlib import Path
import pytest

# repo root 를 sys.path 에 prepend.
# pyproject.toml(rootdir=genon/preprocessor) 의 pythonpath 가 "src" 만이라
# `genon.preprocessor.*` 같은 절대 import 가 CI 에서 해석되지 않는 문제를 회피.
# (이슈 #199 — converters/hwp_to_pdf 모듈은 src/ 밖에 있어 src 만으로는 import 불가.)
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


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
