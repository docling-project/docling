# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""A KServe v2 REST inference route pack.

Serves the two endpoints the KServe client calls:

``GET  {base}/v2/models/{name}[/versions/{v}]``        -> model metadata
``POST {base}/v2/models/{name}[/versions/{v}]/infer``  -> inference

The client uses ``requests``, so this needs a real socket rather than an
httpx-only mock. Responses are built from the repo's own
``KserveV2ModelMetadataResponse`` and ``KserveV2InferResponse`` models, so
the fake cannot drift from the shapes the client validates against.

Output tensors are supplied per test by registering a handler keyed on the
requested output names, which keeps the fake a transport rather than a
reimplementation of any particular model.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response as RawResponse

from docling.models.inference_engines.common.kserve_v2_http import (
    KserveV2InferResponse,
    KserveV2OutputTensor,
)
from docling.models.inference_engines.common.kserve_v2_types import (
    KSERVE_V2_NUMPY_DATATYPES,
    NUMPY_KSERVE_V2_DATATYPES,
    KserveV2ModelMetadataResponse,
    KserveV2ModelTensorSpec,
)

#: Given the decoded request, return the output tensors to answer with.
InferHandler = Callable[[Mapping[str, Any]], Mapping[str, np.ndarray]]

_INFERENCE_HEADER_CONTENT_LENGTH = "Inference-Header-Content-Length"


def encode_tensor(name: str, array: np.ndarray) -> KserveV2OutputTensor:
    """Encode a numpy array as a KServe v2 output tensor."""
    return KserveV2OutputTensor(
        name=name,
        datatype=NUMPY_KSERVE_V2_DATATYPES[array.dtype],
        shape=list(array.shape),
        data=array.flatten().tolist(),
    )


@dataclass
class FakeKserveV2:
    """State plus an ``APIRouter`` serving the KServe v2 REST protocol."""

    model_name: str = "test-model"
    platform: str = "fake_backend"
    versions: list[str] = field(default_factory=lambda: ["1"])
    inputs: list[KserveV2ModelTensorSpec] = field(
        default_factory=lambda: [
            KserveV2ModelTensorSpec(name="input", datatype="FP32", shape=[-1, 3, 8, 8])
        ]
    )
    outputs: list[KserveV2ModelTensorSpec] = field(
        default_factory=lambda: [
            KserveV2ModelTensorSpec(name="output", datatype="FP32", shape=[-1, 2])
        ]
    )
    #: Answers inference; defaults to a single zero-filled ``output`` tensor.
    infer_handler: InferHandler | None = None
    #: Non-2xx status to answer inference with instead of a result.
    fail_status: int | None = None
    #: Answer with the binary extension (JSON header + raw tensor bytes).
    binary_response: bool = False
    router: APIRouter = field(init=False)
    service: Any = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.router = self._build_router()

    # -- payloads --------------------------------------------------------

    def _metadata(self) -> KserveV2ModelMetadataResponse:
        return KserveV2ModelMetadataResponse(
            name=self.model_name,
            versions=self.versions,
            platform=self.platform,
            inputs=self.inputs,
            outputs=self.outputs,
        )

    @staticmethod
    def _decode_request(body: bytes, header_len: int | None) -> Mapping[str, Any]:
        """Split a KServe v2 request into its JSON header.

        With the binary extension the body is a JSON header of
        ``Inference-Header-Content-Length`` bytes followed by raw tensor data,
        so the whole body is not valid JSON.
        """
        header = body if header_len is None else body[:header_len]
        return json.loads(header)

    def _infer(self, payload: Mapping[str, Any]) -> KserveV2InferResponse:
        if self.infer_handler is not None:
            tensors = self.infer_handler(payload)
        else:
            requested = [o["name"] for o in payload.get("outputs") or []] or ["output"]
            tensors = {name: np.zeros((1, 2), dtype=np.float32) for name in requested}
        return KserveV2InferResponse(
            outputs=[encode_tensor(name, array) for name, array in tensors.items()]
        )

    # -- routes ----------------------------------------------------------

    def _build_router(self) -> APIRouter:
        router = APIRouter()

        @router.get("/v2/models/{model_name}")
        async def metadata(model_name: str) -> Any:
            return JSONResponse(json.loads(self._metadata().model_dump_json()))

        @router.get("/v2/models/{model_name}/versions/{version}")
        async def metadata_versioned(model_name: str, version: str) -> Any:
            return JSONResponse(json.loads(self._metadata().model_dump_json()))

        async def _handle_infer(request: Request) -> Any:
            if self.fail_status is not None:
                return JSONResponse(
                    {"error": "fake inference failure"}, status_code=self.fail_status
                )
            raw_header_len = request.headers.get(_INFERENCE_HEADER_CONTENT_LENGTH)
            payload = self._decode_request(
                await request.body(),
                int(raw_header_len) if raw_header_len is not None else None,
            )
            body = self._infer(payload)
            if not self.binary_response:
                return JSONResponse(json.loads(body.model_dump_json()))

            # Binary extension: raw tensor bytes follow the JSON header, and
            # each tensor declares its length so the client can split them.
            chunks: list[bytes] = []
            for tensor in body.outputs:
                array = np.array(
                    tensor.data, dtype=KSERVE_V2_NUMPY_DATATYPES[tensor.datatype]
                )
                raw = np.ascontiguousarray(array).tobytes()
                tensor.data = None
                tensor.parameters = {"binary_data_size": len(raw)}
                chunks.append(raw)
            header = body.model_dump_json(exclude_none=True).encode()
            return RawResponse(
                content=header + b"".join(chunks),
                media_type="application/octet-stream",
                headers={_INFERENCE_HEADER_CONTENT_LENGTH: str(len(header))},
            )

        @router.post("/v2/models/{model_name}/infer")
        async def infer(model_name: str, request: Request) -> Any:
            return await _handle_infer(request)

        @router.post("/v2/models/{model_name}/versions/{version}/infer")
        async def infer_versioned(
            model_name: str, version: str, request: Request
        ) -> Any:
            return await _handle_infer(request)

        return router
