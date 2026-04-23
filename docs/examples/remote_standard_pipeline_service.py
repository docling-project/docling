# %% [markdown]
# Start a KServe v2-compatible HTTP service for the remote standard
# pipeline example.
#
# This service exposes three model endpoints:
# - `rapidocr`
# - `layout-heron`
# - `table-structure`
#
# By default the service wraps Docling's local RapidOCR, layout, and table
# runtimes so it can back a real remote GPU pool. Pass `--demo` to switch to
# deterministic compatibility shims for local wiring checks.

# %%

from __future__ import annotations

import argparse
from pathlib import Path

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.pipeline_options import (
    LayoutOptions,
    RapidOcrOptions,
    TableFormerMode,
    TableStructureOptions,
)
from docling.utils.kserve_v2_compat_server import (
    build_standard_pipeline_compat_server,
    build_standard_pipeline_runtime_server,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--artifacts-path")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--ocr-lang", default="english")
    parser.add_argument(
        "--rapidocr-backend",
        choices=["onnxruntime", "openvino", "paddle", "torch"],
        default="onnxruntime",
    )
    parser.add_argument(
        "--table-mode",
        choices=[mode.value for mode in TableFormerMode],
        default=TableFormerMode.ACCURATE.value,
    )
    parser.add_argument("--cell-text", default="RemoteCell")
    args = parser.parse_args()

    if args.demo:
        server = build_standard_pipeline_compat_server(
            host=args.host,
            port=args.port,
            cell_text=args.cell_text,
        )
    else:
        artifacts_path = Path(args.artifacts_path) if args.artifacts_path else None
        accelerator_options = AcceleratorOptions(
            device=args.device,
            num_threads=args.num_threads,
        )
        server = build_standard_pipeline_runtime_server(
            host=args.host,
            port=args.port,
            artifacts_path=artifacts_path,
            accelerator_options=accelerator_options,
            ocr_options=RapidOcrOptions(
                lang=[args.ocr_lang],
                backend=args.rapidocr_backend,
            ),
            layout_options=LayoutOptions(),
            table_options=TableStructureOptions(
                mode=TableFormerMode(args.table_mode),
                do_cell_matching=True,
            ),
        )

    mode = "demo" if args.demo else "runtime"
    print(f"KServe v2 {mode} server listening on {server.base_url}")
    print(f"OCR model: {server.base_url}/v2/models/rapidocr")
    print(f"Layout model: {server.base_url}/v2/models/layout-heron")
    print(f"Table model: {server.base_url}/v2/models/table-structure")
    server.serve_forever()


if __name__ == "__main__":
    main()
