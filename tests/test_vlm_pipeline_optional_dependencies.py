# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Remote VLM pipelines must not import optional local model providers."""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("enable_remote_services", [False, True])
def test_api_pipeline_does_not_load_local_providers(
    legacy: bool, enable_remote_services: bool
) -> None:
    # Use a fresh interpreter so providers loaded by other tests cannot hide an
    # eager import. This also runs unchanged in a slim install without local ML
    # dependencies; no replacement modules or model downloads are needed.
    script = f"""
import sys
import warnings

from docling.datamodel.pipeline_options import VlmConvertOptions, VlmPipelineOptions
from docling.datamodel.pipeline_options_vlm_model import ApiVlmOptions, ResponseFormat
from docling.datamodel.stage_model_specs import VlmModelSpec
from docling.datamodel.vlm_engine_options import ApiVlmEngineOptions
from docling.exceptions import OperationNotAllowed
from docling.models.inference_engines.vlm.base import VlmEngineType
from docling.pipeline.vlm_pipeline import VlmPipeline

url = "http://127.0.0.1:9/v1/chat/completions"
if {legacy!r}:
    options = ApiVlmOptions(
        url=url,
        params={{"model": "remote-test"}},
        prompt="Convert this page.",
        response_format=ResponseFormat.MARKDOWN,
    )
else:
    options = VlmConvertOptions(
        model_spec=VlmModelSpec(
            name="Remote test",
            default_repo_id="remote-test",
            prompt="Convert this page.",
            response_format=ResponseFormat.MARKDOWN,
        ),
        engine_options=ApiVlmEngineOptions(engine_type=VlmEngineType.API, url=url),
    )

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always", DeprecationWarning)
    try:
        pipeline = VlmPipeline(VlmPipelineOptions(
            vlm_options=options,
            enable_remote_services={enable_remote_services!r},
        ))
    except OperationNotAllowed as exc:
        assert not {enable_remote_services!r}
        assert "enable_remote_services=True" in str(exc)
    else:
        assert {enable_remote_services!r}, "Remote service policy was bypassed"
        assert len(pipeline.build_pipe) == 1
        if {legacy!r}:
            from docling.models.vlm_pipeline_models.api_vlm_model import ApiVlmModel
            assert isinstance(pipeline.build_pipe[0], ApiVlmModel)
        else:
            from docling.models.inference_engines.vlm.api_openai_compatible_engine import ApiVlmEngine
            from docling.models.stages.vlm_convert.vlm_convert_model import VlmConvertModel
            assert isinstance(pipeline.build_pipe[0], VlmConvertModel)
            assert isinstance(pipeline.build_pipe[0].engine, ApiVlmEngine)

if {legacy!r}:
    assert any("Using legacy VLM options" in str(w.message) for w in caught)

for module in (
    "docling.models.vlm_pipeline_models.hf_transformers_model",
    "docling.models.vlm_pipeline_models.mlx_model",
    "docling.models.vlm_pipeline_models.vllm_model",
    "docling.models.inference_engines.vlm.transformers_engine",
    "docling.models.inference_engines.vlm.mlx_engine",
):
    assert module not in sys.modules, module
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
