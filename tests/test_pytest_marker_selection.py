from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def load_marker_selection_module() -> ModuleType:
    module_path = (
        Path(__file__).resolve().parents[1]
        / ".github/scripts/pytest_marker_selection.py"
    )
    spec = importlib.util.spec_from_file_location(
        "pytest_marker_selection", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


marker_selection = load_marker_selection_module()


def write_test_file(repo_root: Path, relative_path: str, content: str) -> None:
    file_path = repo_root / relative_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")


def test_discover_test_markers_uses_module_level_pytestmark(tmp_path: Path) -> None:
    write_test_file(
        tmp_path,
        "tests/test_ocr.py",
        "import pytest\n\npytestmark = pytest.mark.ml_ocr\n\ndef test_ocr(): pass\n",
    )
    write_test_file(
        tmp_path,
        "tests/test_vlm.py",
        "import pytest\n\npytestmark = [pytest.mark.ml_vlm]\n\ndef test_vlm(): pass\n",
    )
    write_test_file(tmp_path, "tests/test_core.py", "def test_core(): pass\n")

    discovered = marker_selection.discover_test_markers(tmp_path)

    assert discovered["ml_ocr"] == [Path("tests/test_ocr.py")]
    assert discovered["ml_vlm"] == [Path("tests/test_vlm.py")]
    assert discovered["ml_pdf_model"] == []
    assert discovered["ml_asr"] == []


def test_build_ml_suites_combines_source_flags_and_changed_marked_tests(
    tmp_path: Path,
) -> None:
    write_test_file(
        tmp_path,
        "tests/test_audio.py",
        "import pytest\n\npytestmark = pytest.mark.ml_asr\n\ndef test_audio(): pass\n",
    )

    suites = marker_selection.build_ml_suites(
        repo_root=tmp_path,
        changed_test_files=[Path("tests/test_audio.py")],
        source_flags={
            "run_ocr": False,
            "run_pdf_model": True,
            "run_vlm": False,
            "run_asr": False,
        },
    )

    assert suites == ["pdf-model", "asr"]


def test_changed_unmarked_test_does_not_select_ml_suite(tmp_path: Path) -> None:
    write_test_file(tmp_path, "tests/test_core.py", "def test_core(): pass\n")

    suites = marker_selection.build_ml_suites(
        repo_root=tmp_path,
        changed_test_files=[Path("tests/test_core.py")],
        source_flags={
            "run_ocr": False,
            "run_pdf_model": False,
            "run_vlm": False,
            "run_asr": False,
        },
    )

    assert suites == []


def test_function_level_ml_marker_is_rejected(tmp_path: Path) -> None:
    write_test_file(
        tmp_path,
        "tests/test_mixed.py",
        "import pytest\n\n@pytest.mark.ml_vlm\ndef test_vlm(): pass\n",
    )

    with pytest.raises(ValueError, match="module-level `pytestmark`"):
        marker_selection.detect_ml_markers(tmp_path / "tests/test_mixed.py")
