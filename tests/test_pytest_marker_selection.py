from __future__ import annotations

import argparse
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
    write_test_file(
        tmp_path,
        "tests/test_smoke.py",
        "import pytest\n\npytestmark = pytest.mark.cross_platform\n\ndef test_smoke(): pass\n",
    )
    write_test_file(tmp_path, "tests/test_core.py", "def test_core(): pass\n")

    discovered = marker_selection.discover_test_markers(tmp_path)

    assert discovered["ml_ocr"] == [Path("tests/test_ocr.py")]
    assert discovered["ml_vlm"] == [Path("tests/test_vlm.py")]
    assert discovered["ml_pdf_model"] == []
    assert discovered["ml_asr"] == []
    assert discovered["cross_platform"] == [Path("tests/test_smoke.py")]


def test_build_ml_suites_returns_all_suites_when_ml_is_triggered() -> None:
    suites = marker_selection.build_ml_suites(run_all_ml=True)

    assert suites == ["ocr", "pdf-model", "vlm", "asr"]


def test_build_ml_suites_returns_empty_without_ml_trigger() -> None:
    suites = marker_selection.build_ml_suites(run_all_ml=False)

    assert suites == []


def test_function_level_ml_marker_is_rejected(tmp_path: Path) -> None:
    write_test_file(
        tmp_path,
        "tests/test_mixed.py",
        "import pytest\n\n@pytest.mark.ml_vlm\ndef test_vlm(): pass\n",
    )

    with pytest.raises(ValueError, match="module-level `pytestmark`"):
        marker_selection.detect_ml_markers(tmp_path / "tests/test_mixed.py")


def test_core_ignore_args_only_ignores_ml_marked_modules(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_test_file(
        tmp_path,
        "tests/test_ocr.py",
        "import pytest\n\npytestmark = pytest.mark.ml_ocr\n\ndef test_ocr(): pass\n",
    )
    write_test_file(
        tmp_path,
        "tests/test_smoke.py",
        "import pytest\n\npytestmark = pytest.mark.cross_platform\n\ndef test_smoke(): pass\n",
    )

    args = argparse.Namespace(repo_root=tmp_path)
    marker_selection.run_core_ignore_args(args)

    assert capsys.readouterr().out.splitlines() == ["--ignore=tests/test_ocr.py"]
