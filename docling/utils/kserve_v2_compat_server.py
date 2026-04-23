"""Lightweight KServe v2 HTTP compatibility server for Docling remote stages."""

from __future__ import annotations

import json
import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping, Optional
from urllib.parse import urlparse

import numpy as np

from docling.models.inference_engines.common.kserve_v2_types import (
    KSERVE_V2_NUMPY_DATATYPES,
    NUMPY_KSERVE_V2_DATATYPES,
)

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class KserveTensorSpec:
    name: str
    datatype: str
    shape: list[int | str]

    def to_metadata_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "datatype": self.datatype,
            "shape": self.shape,
        }


def _decode_input_tensor(raw_tensor: Mapping[str, Any]) -> np.ndarray:
    datatype = str(raw_tensor["datatype"])
    np_dtype = KSERVE_V2_NUMPY_DATATYPES.get(datatype)
    if np_dtype is None:
        raise ValueError(f"Unsupported KServe v2 input datatype: {datatype}")

    shape = tuple(int(dim) for dim in raw_tensor["shape"])
    values = raw_tensor.get("data", [])
    return np.asarray(values, dtype=np_dtype).reshape(shape)


def _encode_output_tensor(name: str, tensor: np.ndarray) -> dict[str, Any]:
    kserve_dtype = NUMPY_KSERVE_V2_DATATYPES.get(tensor.dtype)
    if kserve_dtype is None:
        raise ValueError(
            f"Unsupported numpy dtype for KServe v2 output: {tensor.dtype!s}"
        )

    return {
        "name": name,
        "datatype": kserve_dtype,
        "shape": list(tensor.shape),
        "data": tensor.reshape(-1).tolist(),
    }


def _extract_single_image(input_tensor: np.ndarray) -> np.ndarray:
    if input_tensor.ndim == 4:
        if input_tensor.shape[0] != 1:
            raise ValueError(
                "KServe compat server expects a single-image batch with shape "
                "(1, H, W, C)."
            )
        input_tensor = input_tensor[0]

    if input_tensor.ndim != 3 or input_tensor.shape[-1] != 3:
        raise ValueError(
            "KServe compat server expects RGB images with shape (H, W, 3) or "
            "(1, H, W, 3)."
        )

    return np.asarray(input_tensor, dtype=np.uint8)


def _json_compatible(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_compatible(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


class BaseCompatModel(ABC):
    """Base class for simple KServe-compatible model handlers."""

    model_name: str
    inputs: tuple[KserveTensorSpec, ...]
    outputs: tuple[KserveTensorSpec, ...]
    platform: str
    versions: list[str] | None

    def __init__(
        self,
        *,
        model_name: str,
        inputs: tuple[KserveTensorSpec, ...],
        outputs: tuple[KserveTensorSpec, ...],
        platform: str = "docling-compat",
        versions: list[str] | None = None,
    ) -> None:
        self.model_name = model_name
        self.inputs = inputs
        self.outputs = outputs
        self.platform = platform
        self.versions = versions or ["1"]
        self.infer_calls = 0

    def get_metadata(self) -> dict[str, Any]:
        return {
            "name": self.model_name,
            "versions": self.versions,
            "platform": self.platform,
            "inputs": [tensor.to_metadata_dict() for tensor in self.inputs],
            "outputs": [tensor.to_metadata_dict() for tensor in self.outputs],
        }

    def infer(
        self,
        *,
        inputs: Mapping[str, np.ndarray],
        output_names: list[str],
        request_parameters: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, np.ndarray]:
        self.infer_calls += 1
        outputs = self._infer(
            inputs=inputs,
            output_names=output_names,
            request_parameters=request_parameters,
        )

        if not output_names:
            return outputs

        missing = [name for name in output_names if name not in outputs]
        if missing:
            raise KeyError(
                f"Model {self.model_name!r} does not provide outputs: {missing}"
            )

        return {name: outputs[name] for name in output_names}

    @abstractmethod
    def _infer(
        self,
        *,
        inputs: Mapping[str, np.ndarray],
        output_names: list[str],
        request_parameters: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, np.ndarray]:
        """Run model-specific inference and return named output tensors."""


class DemoRapidOcrModel(BaseCompatModel):
    """Compatibility OCR model that emits one text cell per request."""

    def __init__(
        self,
        *,
        model_name: str = "rapidocr",
        text: str = "RemoteCell",
        score: float = 0.99,
    ) -> None:
        super().__init__(
            model_name=model_name,
            inputs=(KserveTensorSpec("image", "UINT8", [1, -1, -1, 3]),),
            outputs=(
                KserveTensorSpec("boxes", "FP32", [1, 4, 2]),
                KserveTensorSpec("txts", "BYTES", [1]),
                KserveTensorSpec("scores", "FP32", [1]),
            ),
        )
        self.text = text
        self.score = score

    def _infer(
        self,
        *,
        inputs: Mapping[str, np.ndarray],
        output_names: list[str],
        request_parameters: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, np.ndarray]:
        _ = output_names, request_parameters

        image = inputs["image"]
        _, height, width, _ = image.shape
        box = np.asarray(
            [
                [
                    [0.1 * width, 0.2 * height],
                    [0.9 * width, 0.2 * height],
                    [0.9 * width, 0.4 * height],
                    [0.1 * width, 0.4 * height],
                ]
            ],
            dtype=np.float32,
        )
        txts = np.asarray([self.text], dtype=object)
        scores = np.asarray([self.score], dtype=np.float32)
        return {"boxes": box, "txts": txts, "scores": scores}


class DemoLayoutModel(BaseCompatModel):
    """Compatibility layout model that emits one table detection per image."""

    def __init__(
        self,
        *,
        model_name: str = "layout-heron",
        score: float = 0.99,
    ) -> None:
        super().__init__(
            model_name=model_name,
            inputs=(KserveTensorSpec("image", "UINT8", [1, -1, -1, 3]),),
            outputs=(
                KserveTensorSpec("label_names", "BYTES", [1]),
                KserveTensorSpec("boxes", "FP32", [1, 4]),
                KserveTensorSpec("scores", "FP32", [1]),
            ),
        )
        self.score = score

    def _infer(
        self,
        *,
        inputs: Mapping[str, np.ndarray],
        output_names: list[str],
        request_parameters: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, np.ndarray]:
        _ = output_names, request_parameters

        image = inputs["image"]
        _, height, width, _ = image.shape

        label_names = np.asarray(["table"], dtype=object)
        scores = np.asarray([self.score], dtype=np.float32)
        boxes = np.asarray(
            [[0.05 * width, 0.05 * height, 0.95 * width, 0.95 * height]],
            dtype=np.float32,
        )

        return {
            "label_names": label_names,
            "boxes": boxes,
            "scores": scores,
        }


class DemoTableStructureModel(BaseCompatModel):
    """Compatibility table model that returns one single-cell table."""

    def __init__(
        self,
        *,
        model_name: str = "table-structure",
        default_text: str = "RemoteCell",
    ) -> None:
        super().__init__(
            model_name=model_name,
            inputs=(
                KserveTensorSpec("image", "UINT8", [1, -1, -1, 3]),
                KserveTensorSpec("request_json", "BYTES", [1]),
            ),
            outputs=(KserveTensorSpec("response_json", "BYTES", [1]),),
        )
        self.default_text = default_text
        self.last_request: dict[str, Any] | None = None

    def _infer(
        self,
        *,
        inputs: Mapping[str, np.ndarray],
        output_names: list[str],
        request_parameters: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, np.ndarray]:
        _ = output_names, request_parameters

        image = inputs["image"]
        _, height, width, _ = image.shape

        request_raw = inputs["request_json"].reshape(-1)[0]
        if isinstance(request_raw, bytes):
            request_json = request_raw.decode("utf-8")
        else:
            request_json = str(request_raw)

        request_payload = json.loads(request_json)
        self.last_request = request_payload

        token_text = self.default_text
        request_tokens = request_payload.get("tokens", [])
        if request_tokens:
            token_text = str(request_tokens[0].get("text", token_text))

        response_payload = {
            "otsl_seq": ["fcel"],
            "num_rows": 1,
            "num_cols": 1,
            "table_cells": [
                {
                    "bbox": {
                        "l": 0.0,
                        "t": 0.0,
                        "r": float(width),
                        "b": float(height),
                        "token": token_text,
                    },
                    "row_span": 1,
                    "col_span": 1,
                    "start_row_offset_idx": 0,
                    "end_row_offset_idx": 1,
                    "start_col_offset_idx": 0,
                    "end_col_offset_idx": 1,
                }
            ],
        }

        response_json = np.asarray([json.dumps(response_payload)], dtype=object)
        return {"response_json": response_json}


class RapidOcrCompatModel(BaseCompatModel):
    """KServe-compatible wrapper around Docling's local RapidOCR runtime."""

    def __init__(
        self,
        *,
        model_name: str = "rapidocr",
        artifacts_path: Path | None = None,
        accelerator_options: "AcceleratorOptions | None" = None,
        options: "RapidOcrOptions | None" = None,
    ) -> None:
        from docling.datamodel.accelerator_options import AcceleratorOptions
        from docling.datamodel.pipeline_options import RapidOcrOptions
        from docling.models.stages.ocr.rapid_ocr_model import RapidOcrModel

        super().__init__(
            model_name=model_name,
            inputs=(KserveTensorSpec("image", "UINT8", [1, -1, -1, 3]),),
            outputs=(
                KserveTensorSpec("boxes", "FP32", [-1, 4, 2]),
                KserveTensorSpec("txts", "BYTES", [-1]),
                KserveTensorSpec("scores", "FP32", [-1]),
            ),
            platform="docling-rapidocr",
        )
        self.options = options or RapidOcrOptions()
        self.accelerator_options = accelerator_options or AcceleratorOptions()
        self.stage_model = RapidOcrModel(
            enabled=True,
            artifacts_path=artifacts_path,
            options=self.options,
            accelerator_options=self.accelerator_options,
        )

    def _infer(
        self,
        *,
        inputs: Mapping[str, np.ndarray],
        output_names: list[str],
        request_parameters: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, np.ndarray]:
        _ = output_names, request_parameters

        image = _extract_single_image(inputs["image"])
        result = self.stage_model.reader(
            image,
            use_det=self.options.use_det,
            use_cls=self.options.use_cls,
            use_rec=self.options.use_rec,
        )
        if result is None or result.boxes is None:
            return {
                "boxes": np.zeros((0, 4, 2), dtype=np.float32),
                "txts": np.asarray([], dtype=object),
                "scores": np.asarray([], dtype=np.float32),
            }

        boxes = np.asarray(result.boxes, dtype=np.float32).reshape((-1, 4, 2))
        txts = np.asarray(list(result.txts), dtype=object)
        scores = np.asarray(list(result.scores), dtype=np.float32)
        return {
            "boxes": boxes,
            "txts": txts,
            "scores": scores,
        }


class LayoutCompatModel(BaseCompatModel):
    """KServe-compatible wrapper around Docling's local layout runtime."""

    def __init__(
        self,
        *,
        model_name: str = "layout-heron",
        artifacts_path: Path | None = None,
        accelerator_options: "AcceleratorOptions | None" = None,
        options: "LayoutOptions | None" = None,
    ) -> None:
        from docling.datamodel.accelerator_options import AcceleratorOptions
        from docling.datamodel.pipeline_options import LayoutOptions
        from docling.models.stages.layout.layout_model import LayoutModel

        super().__init__(
            model_name=model_name,
            inputs=(KserveTensorSpec("image", "UINT8", [1, -1, -1, 3]),),
            outputs=(
                KserveTensorSpec("label_names", "BYTES", [-1]),
                KserveTensorSpec("boxes", "FP32", [-1, 4]),
                KserveTensorSpec("scores", "FP32", [-1]),
            ),
            platform="docling-layout",
        )
        self.options = options or LayoutOptions()
        self.accelerator_options = accelerator_options or AcceleratorOptions()
        self.stage_model = LayoutModel(
            artifacts_path=artifacts_path,
            accelerator_options=self.accelerator_options,
            options=self.options,
        )

    def _infer(
        self,
        *,
        inputs: Mapping[str, np.ndarray],
        output_names: list[str],
        request_parameters: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, np.ndarray]:
        _ = output_names, request_parameters

        from PIL import Image

        image = Image.fromarray(_extract_single_image(inputs["image"]))
        predictions = self.stage_model.layout_predictor.predict_batch([image])
        page_predictions = predictions[0] if predictions else []

        label_names = np.asarray(
            [str(prediction["label"]) for prediction in page_predictions],
            dtype=object,
        )
        boxes = np.asarray(
            [
                [
                    float(prediction["l"]),
                    float(prediction["t"]),
                    float(prediction["r"]),
                    float(prediction["b"]),
                ]
                for prediction in page_predictions
            ],
            dtype=np.float32,
        )
        scores = np.asarray(
            [float(prediction["confidence"]) for prediction in page_predictions],
            dtype=np.float32,
        )

        if boxes.size == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)

        return {
            "label_names": label_names,
            "boxes": boxes,
            "scores": scores,
        }


class TableStructureCompatModel(BaseCompatModel):
    """KServe-compatible wrapper around Docling's local table runtime."""

    def __init__(
        self,
        *,
        model_name: str = "table-structure",
        artifacts_path: Path | None = None,
        accelerator_options: "AcceleratorOptions | None" = None,
        options: "TableStructureOptions | None" = None,
    ) -> None:
        from docling.datamodel.accelerator_options import AcceleratorOptions
        from docling.datamodel.pipeline_options import TableStructureOptions
        from docling.models.stages.table_structure.table_structure_model import (
            TableStructureModel,
        )

        super().__init__(
            model_name=model_name,
            inputs=(
                KserveTensorSpec("image", "UINT8", [1, -1, -1, 3]),
                KserveTensorSpec("request_json", "BYTES", [1]),
            ),
            outputs=(KserveTensorSpec("response_json", "BYTES", [1]),),
            platform="docling-table-structure",
        )
        self.options = options or TableStructureOptions()
        self.accelerator_options = accelerator_options or AcceleratorOptions()
        self.stage_model = TableStructureModel(
            enabled=True,
            artifacts_path=artifacts_path,
            options=self.options,
            accelerator_options=self.accelerator_options,
        )
        self.last_request: dict[str, Any] | None = None
        self.last_response: dict[str, Any] | None = None

    def _infer(
        self,
        *,
        inputs: Mapping[str, np.ndarray],
        output_names: list[str],
        request_parameters: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, np.ndarray]:
        _ = output_names, request_parameters

        image = _extract_single_image(inputs["image"])
        height, width, _ = image.shape

        request_raw = inputs["request_json"].reshape(-1)[0]
        if isinstance(request_raw, bytes):
            request_json = request_raw.decode("utf-8")
        else:
            request_json = str(request_raw)

        request_payload = json.loads(request_json)
        self.last_request = request_payload

        page_input = {
            "width": float(width),
            "height": float(height),
            "image": image,
            "tokens": request_payload.get("tokens", []),
        }
        tf_output = self.stage_model.tf_predictor.multi_table_predict(
            page_input,
            [[0.0, 0.0, float(width), float(height)]],
            do_matching=bool(
                request_payload.get("do_cell_matching", self.options.do_cell_matching)
            ),
        )
        table_out = tf_output[0] if tf_output else {}
        predict_details = table_out.get("predict_details", {})

        response_payload = {
            "otsl_seq": predict_details.get("prediction", {}).get("rs_seq", []),
            "num_rows": int(predict_details.get("num_rows", 0)),
            "num_cols": int(predict_details.get("num_cols", 0)),
            "table_cells": _json_compatible(table_out.get("tf_responses", [])),
        }
        self.last_response = response_payload

        response_json = np.asarray([json.dumps(response_payload)], dtype=object)
        return {"response_json": response_json}


@dataclass
class KserveV2CompatServer:
    """Small in-process KServe v2 HTTP server."""

    models: Mapping[str, BaseCompatModel]
    host: str = "127.0.0.1"
    port: int = 0
    _httpd: ThreadingHTTPServer | None = field(default=None, init=False, repr=False)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)

    @property
    def base_url(self) -> str:
        if self._httpd is None:
            return f"http://{self.host}:{self.port}"
        bound_host, bound_port = self._httpd.server_address[:2]
        return f"http://{bound_host}:{bound_port}"

    def _build_handler(self):
        server = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: Any) -> None:
                _log.debug("KServe compat server: " + fmt, *args)

            def _send_json(
                self,
                *,
                status_code: int,
                payload: Mapping[str, Any],
            ) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(status_code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _parse_model_route(self) -> tuple[str, bool] | None:
                path = urlparse(self.path).path
                parts = [part for part in path.split("/") if part]

                if len(parts) < 3 or parts[0] != "v2" or parts[1] != "models":
                    return None

                model_name = parts[2]
                is_infer = False
                if len(parts) == 4 and parts[3] == "infer":
                    is_infer = True
                elif len(parts) == 5 and parts[3] == "versions":
                    is_infer = False
                elif len(parts) == 6 and parts[3] == "versions" and parts[5] == "infer":
                    is_infer = True
                elif len(parts) != 3:
                    return None

                return model_name, is_infer

            def do_GET(self) -> None:
                route = self._parse_model_route()
                if route is None:
                    self._send_json(status_code=404, payload={"error": "Not found"})
                    return

                model_name, is_infer = route
                if is_infer:
                    self._send_json(status_code=405, payload={"error": "Method not allowed"})
                    return

                model = server.models.get(model_name)
                if model is None:
                    self._send_json(status_code=404, payload={"error": "Unknown model"})
                    return

                self._send_json(status_code=200, payload=model.get_metadata())

            def do_POST(self) -> None:
                route = self._parse_model_route()
                if route is None:
                    self._send_json(status_code=404, payload={"error": "Not found"})
                    return

                model_name, is_infer = route
                if not is_infer:
                    self._send_json(status_code=405, payload={"error": "Method not allowed"})
                    return

                model = server.models.get(model_name)
                if model is None:
                    self._send_json(status_code=404, payload={"error": "Unknown model"})
                    return

                try:
                    content_length = int(self.headers.get("Content-Length", "0"))
                    body = self.rfile.read(content_length) if content_length else b"{}"
                    payload = json.loads(body.decode("utf-8"))

                    input_tensors = {
                        raw_tensor["name"]: _decode_input_tensor(raw_tensor)
                        for raw_tensor in payload.get("inputs", [])
                    }
                    output_names = [
                        output["name"] for output in payload.get("outputs", [])
                    ]
                    request_parameters = payload.get("parameters")

                    outputs = model.infer(
                        inputs=input_tensors,
                        output_names=output_names,
                        request_parameters=request_parameters,
                    )
                except Exception as exc:
                    _log.exception("KServe compat server inference failed")
                    self._send_json(
                        status_code=500,
                        payload={"error": f"{type(exc).__name__}: {exc}"},
                    )
                    return

                encoded_outputs = [
                    _encode_output_tensor(name=name, tensor=tensor)
                    for name, tensor in outputs.items()
                ]
                self._send_json(status_code=200, payload={"outputs": encoded_outputs})

        return _Handler

    def _ensure_httpd(self) -> ThreadingHTTPServer:
        if self._httpd is None:
            handler = self._build_handler()
            self._httpd = ThreadingHTTPServer((self.host, self.port), handler)
            self.port = int(self._httpd.server_address[1])
        return self._httpd

    def start(self) -> None:
        if self._thread is not None:
            return

        httpd = self._ensure_httpd()
        self._thread = threading.Thread(
            target=httpd.serve_forever,
            name="kserve-v2-compat-server",
            daemon=True,
        )
        self._thread.start()

    def serve_forever(self) -> None:
        self._ensure_httpd().serve_forever()

    def stop(self) -> None:
        if self._httpd is None:
            return

        self._httpd.shutdown()
        self._httpd.server_close()
        self._httpd = None

        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def __enter__(self) -> KserveV2CompatServer:
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


def build_standard_pipeline_compat_server(
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    ocr_model_name: str = "rapidocr",
    layout_model_name: str = "layout-heron",
    table_model_name: str = "table-structure",
    cell_text: str = "RemoteCell",
) -> KserveV2CompatServer:
    """Build a compatibility server exposing OCR, layout, and table models."""

    models: dict[str, BaseCompatModel] = {
        ocr_model_name: DemoRapidOcrModel(
            model_name=ocr_model_name,
            text=cell_text,
        ),
        layout_model_name: DemoLayoutModel(
            model_name=layout_model_name,
        ),
        table_model_name: DemoTableStructureModel(
            model_name=table_model_name,
            default_text=cell_text,
        ),
    }
    return KserveV2CompatServer(models=models, host=host, port=port)


def build_standard_pipeline_runtime_server(
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    artifacts_path: str | Path | None = None,
    accelerator_options: "AcceleratorOptions | None" = None,
    ocr_options: "RapidOcrOptions | None" = None,
    layout_options: "LayoutOptions | None" = None,
    table_options: "TableStructureOptions | None" = None,
    ocr_model_name: str = "rapidocr",
    layout_model_name: str = "layout-heron",
    table_model_name: str = "table-structure",
) -> KserveV2CompatServer:
    """Build a KServe HTTP service backed by Docling's local model runtimes."""

    resolved_artifacts_path = (
        Path(artifacts_path) if artifacts_path is not None else None
    )
    models: dict[str, BaseCompatModel] = {
        ocr_model_name: RapidOcrCompatModel(
            model_name=ocr_model_name,
            artifacts_path=resolved_artifacts_path,
            accelerator_options=accelerator_options,
            options=ocr_options,
        ),
        layout_model_name: LayoutCompatModel(
            model_name=layout_model_name,
            artifacts_path=resolved_artifacts_path,
            accelerator_options=accelerator_options,
            options=layout_options,
        ),
        table_model_name: TableStructureCompatModel(
            model_name=table_model_name,
            artifacts_path=resolved_artifacts_path,
            accelerator_options=accelerator_options,
            options=table_options,
        ),
    }
    return KserveV2CompatServer(models=models, host=host, port=port)
