# Model Backends

This page complements the [Model Catalog](model_catalog.md) by organizing Docling's
models along a different axis: the runtime **backend** each one executes on. Use it
to answer questions like "which models need `onnxruntime`?" or "which stages are pure
PyTorch vs. calling out to an external engine/API?"

Backends fall into five groups:

- **Pure PyTorch** — local weights loaded via `torch`/`transformers`/`safetensors`.
- **ONNX** — local weights executed via `onnxruntime.InferenceSession`.
- **MLX** — Apple Silicon-only, via `mlx`/`mlx_vlm`/`mlx_whisper`.
- **vLLM** — high-throughput local serving engine (itself PyTorch-based).
- **Other** — a non-Python inference engine (Tesseract, macOS Vision, CTranslate2), or
  a remote call (HTTP API, KServe/Triton gRPC).

!!! note "AMD ROCm"

    "Pure PyTorch / Transformers" backends also run on AMD GPUs via a
    ROCm-built PyTorch — `device="cuda"` is correct on ROCm too, since ROCm
    PyTorch aliases HIP under the same `torch.cuda` namespace. This has been
    verified for the default layout model and TableFormer. The ONNX backend
    has no ROCm execution provider yet and falls back to CPU on AMD GPUs. See
    [GPU Support](gpu.md#amd-rocm-support) for install instructions and
    caveats.

## Standard PDF pipeline

| Model | Purpose | Backend |
|---|---|---|
| Layout model (default) | Page-layout object detection | Pure PyTorch (`docling-ibm-models`) |
| Layout model — new pluggable path (opt-in) | RT-DETR-style layout detector | ONNX by default; Transformers or remote KServe selectable |
| TableFormer v1 (default) | Table structure recognition | Pure PyTorch (`docling-ibm-models`) |
| TableFormer v2 (opt-in) | Table structure, OTSL sequence output | Pure PyTorch (`docling-ibm-models`) |
| GraniteVision table structure (opt-in) | VLM-based table structure | Pure PyTorch / Transformers |
| OCR — Auto (default) | Picks best OCR engine for the platform | Delegates to one of the engines below |
| EasyOCR | OCR | Pure PyTorch |
| Tesseract (`tesserocr` / CLI) | OCR | Other — Tesseract C++ engine (binding or subprocess) |
| RapidOCR | OCR | ONNX by default; torch / OpenVINO / Paddle selectable |
| OcrMac | OCR | Other — macOS native Vision framework |
| Nemotron OCR | OCR | Pure PyTorch, CUDA-only |
| KServe v2 OCR | Remote OCR | Other — Triton/KServe gRPC/HTTP server |
| Document Picture Classifier | Picture type classification | Transformers by default; ONNX or remote KServe selectable |
| Code/Formula model | Code/formula transcription | Transformers by default; MLX/vLLM/API selectable |
| Chart extraction (GraniteVision) | Chart → CSV/summary/code | Pure PyTorch / Transformers |
| Picture description — local VLM | Image captioning | Transformers by default; MLX on Apple Silicon, vLLM selectable |
| Picture description — API | Remote image captioning | Other — HTTP API |

## VLM pipeline (Granite-Docling, SmolDocling, etc.)

| Model | Purpose | Backend |
|---|---|---|
| VlmConvertModel (`AUTO_INLINE`) | Full-page document → DocTags/Markdown | Transformers on Linux/Windows/CUDA, MLX on Apple Silicon; vLLM or remote API selectable |
| Legacy `HuggingFaceTransformersVlmModel` | Same, legacy path | Pure PyTorch / Transformers |
| Legacy `HuggingFaceMlxModel` | Same, Apple Silicon | MLX |
| Legacy `VllmVlmModel` | Same, high-throughput serving | vLLM |
| Legacy `ApiVlmModel` | Same, remote model | Other — HTTP API |

## Extraction pipeline

| Model | Purpose | Backend |
|---|---|---|
| `TransformersExtractionModel` | Structured data extraction from document images | Pure PyTorch / Transformers |
| `NuExtractTransformersModel` | Template-guided extraction | Pure PyTorch / Transformers |

## ASR and video pipelines

| Model | Purpose | Backend |
|---|---|---|
| Native Whisper (default) | Speech-to-text | Pure PyTorch |
| MLX Whisper | Speech-to-text, Apple Silicon | MLX |
| WhisperS2T | Speech-to-text, fast batched inference | Other — CTranslate2 backend |
| Speaker diarization (`resemblyzer`, video only) | Voice-embedding clustering | Pure PyTorch |

## Non-ML utility components

- `ReadingOrderModel` (`docling-ibm-models`) and the list-item normalizer are
  rule-based heuristics, not trained models.
- `HeadingHierarchyModel` is heuristic/regex-based, no ML involved.
- KServe v2 clients (HTTP and gRPC) are shared infrastructure used by the OCR,
  classification, and object-detection engines above.

## Summary by backend

| Backend | Models / components |
|---|---|
| **Pure PyTorch / Transformers** | Layout model (default), TableFormer v1 & v2, GraniteVision table structure & chart extraction, EasyOCR, Nemotron OCR, Document Picture Classifier (default engine), Code/Formula model (default engine), local VLM picture description, VLM-convert models (default engine), extraction models, Native Whisper, `resemblyzer` |
| **ONNX (onnxruntime)** | RapidOCR (default backend), new-path layout object detector (opt-in), image-classification/object-detection ONNX engines (alternate to Transformers default) |
| **MLX (Apple Silicon)** | VLM-convert / picture description / code-formula (`AUTO_INLINE` on macOS ARM), legacy `HuggingFaceMlxModel`, MLX Whisper |
| **vLLM** | VLM-convert / legacy `VllmVlmModel` (opt-in, high-throughput serving) |
| **Other — Tesseract engine** | `tesserocr` binding, Tesseract CLI (`subprocess`) |
| **Other — macOS native OCR** | OcrMac (Apple Vision framework) |
| **Other — CTranslate2** | WhisperS2T |
| **Other — HTTP/remote API** | Picture description API model, legacy `ApiVlmModel`, VLM-convert API engines (Ollama, LM Studio, OpenAI-compatible) |
| **Other — remote inference server (Triton/KServe)** | KServe v2 OCR, KServe v2 image-classification & object-detection engines |

## Confirmed defaults

- `PdfPipelineOptions.layout_options` → `LayoutOptions()` → `LayoutModel` (PyTorch, `docling-ibm-models`).
- `PdfPipelineOptions.table_structure_options` → `TableStructureOptions()` → `TableStructureModel` (PyTorch, TableFormer v1).
- `PdfPipelineOptions.ocr_options` → `OcrAutoOptions()` → `OcrAutoModel` (platform-dependent: OcrMac → Nemotron → RapidOCR+onnxruntime → EasyOCR → RapidOCR+torch).
- Picture classification default → `document_figure_classifier_v2` preset, `TRANSFORMERS` engine (PyTorch); ONNX/API also available.
- Picture description default → `smolvlm` preset, `AUTO_INLINE` (Transformers/MLX auto-select).
- VLM-convert default (VLM pipeline) → `granite_docling` preset, `AUTO_INLINE`.
- Code/formula default → `codeformulav2` preset, `AUTO_INLINE`.
- `docling-ibm-models` depends on `torch`, `torchvision`, `safetensors[torch]`, `transformers`, `accelerate` — it is **not** onnxruntime-based.

## Related pages

- [Model Catalog](model_catalog.md) — models organized by pipeline stage and preset.
- [Vision Models Usage Guide](vision_models.md) — VLM-specific configuration.
- [GPU Support](gpu.md) — GPU acceleration setup.
