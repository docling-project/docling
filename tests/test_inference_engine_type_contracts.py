# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import json
import logging
from types import SimpleNamespace

import numpy as np
import pytest
import requests
from PIL import Image

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.pipeline_options_vlm_model import (
    ApiVlmOptions,
    InferenceFramework,
    InlineVlmOptions,
    ResponseFormat,
)
from docling.datamodel.vlm_engine_options import ApiVlmEngineOptions
from docling.models.inference_engines.common.kserve_v2_http import KserveV2HttpClient
from docling.models.inference_engines.vlm.api_openai_compatible_engine import (
    ApiVlmEngine,
)
from docling.models.inference_engines.vlm.base import VlmEngineInput
from docling.models.vlm_pipeline_models.api_vlm_model import ApiVlmModel
from docling.models.vlm_pipeline_models.mlx_model import HuggingFaceMlxModel

pytestmark = pytest.mark.cross_platform


def test_kserve_profiling_survives_a_log_level_change(monkeypatch, caplog):
    logger_name = "docling.models.inference_engines.common.kserve_v2_http"
    caplog.set_level(logging.INFO, logger=logger_name)

    def post(*args, **kwargs):
        logging.getLogger(logger_name).setLevel(logging.DEBUG)
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps(
            {
                "outputs": [
                    {"name": "out", "datatype": "FP32", "shape": [1], "data": [1.0]}
                ]
            }
        ).encode()
        return response

    monkeypatch.setattr(requests, "post", post)
    client = KserveV2HttpClient(
        base_url="http://test",
        model_name="model",
        model_version=None,
        timeout=1,
        headers={},
        use_binary_data=False,
    )
    result = client.infer(
        inputs={"input": np.zeros((1,), dtype=np.float32)}, output_names=["out"]
    )
    np.testing.assert_array_equal(result["out"], np.array([1.0], dtype=np.float32))


@pytest.mark.parametrize("key", ["usage_response_key", "token_extract_key"])
@pytest.mark.parametrize("engine", [False, True])
def test_api_response_keys_require_text_before_a_request(key, engine):
    image = Image.new("RGB", (2, 2))
    with pytest.raises(ValueError, match=key + " must be a string or None"):
        if engine:
            model = ApiVlmEngine(
                True, ApiVlmEngineOptions(url="http://test", params={key: 42})
            )
            model.predict_batch([VlmEngineInput(image=image, prompt="Read")])
        else:
            model = ApiVlmModel(
                True,
                True,
                ApiVlmOptions(
                    url="http://test",
                    params={key: 42},
                    prompt="Read",
                    response_format=ResponseFormat.PLAINTEXT,
                ),
            )
            list(model.process_images([image], "Read"))


@pytest.mark.parametrize(
    "logprobs", [None, [-0.25], np.array([[-0.25]], dtype=np.float32), "mlx-bfloat16"]
)
def test_mlx_stream_accepts_optional_and_list_logprobs(logprobs):
    if isinstance(logprobs, str):
        mx = pytest.importorskip("mlx.core")
        logprobs = mx.array([-0.25], dtype=mx.bfloat16)
    model = HuggingFaceMlxModel(
        False,
        None,
        AcceleratorOptions(),
        InlineVlmOptions(
            repo_id="test/model",
            prompt="Read",
            inference_framework=InferenceFramework.MLX,
            response_format=ResponseFormat.PLAINTEXT,
        ),
    )
    model.vlm_model = object()
    model.processor = object()
    model.config = {}
    model.apply_chat_template = lambda *args, **kwargs: "Read"
    model.stream_generate = lambda *args, **kwargs: iter(
        [
            SimpleNamespace(text="a", token=0, logprobs=logprobs),
        ]
    )
    (prediction,) = model.process_images([Image.new("RGB", (2, 2))], "Read")
    assert prediction.text == "a"
    if logprobs is None:
        assert prediction.generated_tokens == []
    else:
        assert prediction.generated_tokens[0].logprob == pytest.approx(-0.25)
