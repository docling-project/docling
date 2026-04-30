from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def load_check_needs_results_module() -> ModuleType:
    module_path = (
        Path(__file__).resolve().parents[1] / ".github/scripts/check_needs_results.py"
    )
    spec = importlib.util.spec_from_file_location("check_needs_results", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


check_needs_results = load_check_needs_results_module()


def test_collect_failures_allows_configured_skips(
    capsys: pytest.CaptureFixture[str],
) -> None:
    needs = {
        "changes": {"result": "success"},
        "run-tests-core": {"result": "skipped"},
    }

    failures = check_needs_results.collect_failures(needs, {"run-tests-core"})

    assert failures == []
    assert "Allowed skipped job::run-tests-core" in capsys.readouterr().out


def test_collect_failures_rejects_cancelled_jobs_even_when_skip_is_allowed() -> None:
    needs = {"run-tests-core": {"result": "cancelled"}}

    failures = check_needs_results.collect_failures(needs, {"run-tests-core"})

    assert failures == ["run-tests-core=cancelled"]


def test_collect_failures_rejects_unexpected_skips() -> None:
    needs = {"changes": {"result": "skipped"}}

    failures = check_needs_results.collect_failures(needs, set())

    assert failures == ["changes=skipped"]


def test_parse_needs_requires_json_object() -> None:
    with pytest.raises(ValueError, match="JSON object"):
        check_needs_results.parse_needs("[]")


def test_result_for_job_requires_string_result() -> None:
    with pytest.raises(ValueError, match="string result"):
        check_needs_results.result_for_job("changes", {"result": None})
