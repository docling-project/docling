# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""The ASR import path must stay free of optional dependencies and of MLX.

A leak on this path breaks every torch-free docling-slim install (PR 5072) or aborts
the process when Metal initializes in a restricted environment (PR 5090). Each test
runs in a fresh subprocess, so that it sees the real import order.
"""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

_IMPORT_ASR_PATH = """
import docling.document_converter
import docling.pipeline.asr_pipeline
import docling.pipeline.video_pipeline
from docling.datamodel import asr_model_specs

for preset in asr_model_specs.WHISPER_ASR_PRESETS.values():
    preset.to_options()
for preset in asr_model_specs.ADDITIONAL_ASR_PRESETS.values():
    preset.to_options()
    preset.to_auto_options()
print("IMPORT-OK")
"""


def _run_python(code: str, extra_path: Path | None = None) -> str:
    env = dict(os.environ)
    paths = [str(REPO_ROOT)]
    if extra_path is not None:
        paths.insert(0, str(extra_path))
    env["PYTHONPATH"] = os.pathsep.join([*paths, env.get("PYTHONPATH", "")])
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
        timeout=180,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    return result.stdout


def test_asr_path_imports_without_optional_dependencies() -> None:
    blocked = (
        "torch",
        "transformers",
        "packaging",
        "mlx",
        "mlx.core",
        "mlx_audio",
        "mlx_whisper",
    )
    stdout = _run_python(
        f"import sys; sys.modules.update(dict.fromkeys({blocked!r}))\n"
        + _IMPORT_ASR_PATH
    )
    assert "IMPORT-OK" in stdout


def test_asr_path_does_not_import_installed_mlx_libraries(tmp_path: Path) -> None:
    # A package that fails on import stands in for the native Metal abort.
    for name in ("mlx", "mlx_audio", "mlx_whisper"):
        package = tmp_path / name
        package.mkdir()
        (package / "__init__.py").write_text(
            f"raise RuntimeError('{name} was imported on the ASR import path')\n"
        )

    stdout = _run_python(_IMPORT_ASR_PATH, extra_path=tmp_path)

    assert "IMPORT-OK" in stdout


def test_importing_docling_does_not_load_transformers() -> None:
    stdout = _run_python(
        "import sys\n"
        "import docling.document_converter\n"
        "print('LOADED' if 'transformers' in sys.modules else 'NOT-LOADED')\n"
    )
    assert "NOT-LOADED" in stdout


def test_stopping_criteria_field_accepts_both_stopper_types() -> None:
    # Validation must still work now that the module does not import transformers.
    stdout = _run_python(
        """
from docling.datamodel.pipeline_options_vlm_model import (
    InlineVlmOptions,
    InferenceFramework,
    ResponseFormat,
)
from docling.models.utils.generation_utils import GenerationStopper
from transformers import StoppingCriteria


class Stopper(GenerationStopper):
    def should_stop(self, s):
        return False

    def lookback_tokens(self):
        return 1


class HfStopper(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs):
        return False


def options(criteria):
    return InlineVlmOptions(
        repo_id="r",
        prompt="p",
        response_format=ResponseFormat.MARKDOWN,
        inference_framework=InferenceFramework.TRANSFORMERS,
        custom_stopping_criteria=criteria,
    )


assert len(options([Stopper(), HfStopper()]).custom_stopping_criteria) == 2
try:
    options(["not a stopper"])
except ValueError as exc:
    assert "GenerationStopper" in str(exc)
else:
    raise AssertionError("invalid stopping criterion was accepted")
print("VALIDATION-OK")
"""
    )
    assert "VALIDATION-OK" in stdout
