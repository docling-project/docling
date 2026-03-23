# Docling Pipelines Reference

Docling has two pipeline families for PDFs: **standard** (parse + OCR + layout/tables)
and **VLM** (page images through a vision-language model). The helper
`scripts/docling-convert.py` exposes **three modes**: `standard`, `vlm-local`, `vlm-api`.
The right choice depends on document type, hardware, and latency budget.

---

## Decision matrix

| Document type | Recommended pipeline | Reason |
|---|---|---|
| Born-digital PDF (text selectable) | Standard | Fast, accurate, no GPU needed |
| Scanned PDF / image-only | Standard + OCR or VLM | Depends on quality |
| Complex layout (multi-column, dense tables) | VLM local | Better structural understanding |
| Handwriting, formulas, figures with embedded text | VLM | Only viable option |
| Air-gapped / no GPU | Standard | Runs on CPU |
| Production scale, GPU server available | VLM API (vLLM) | Best throughput |
| Apple Silicon / local dev | VLM local (MLX) | MPS acceleration |
| Speed-critical, accuracy secondary | Standard, no tables | Fastest path |

---

## Pipeline 1: Standard PDF Pipeline

Uses deterministic PDF parsing (docling-parse) + optional neural OCR + neural
table structure detection.

```python
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

# Minimal — library defaults (standard PDF pipeline)
converter = DocumentConverter()

# Explicit PdfPipelineOptions (docling 2.81+): use InputFormat.PDF + PdfFormatOption.
# Do not use format_options={"pdf": opts}; that raises AttributeError on pipeline options.
opts = PdfPipelineOptions(
    do_ocr=True,                 # False = skip OCR entirely
    do_table_structure=True,     # False = skip table detection (faster)
)
converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=opts),
    }
)
```

### OCR engine options

All engines are plug-and-play via `ocr_options`. Default is EasyOCR.

```python
# EasyOCR (default — no extra install needed)
from docling.datamodel.pipeline_options import PdfPipelineOptions
opts = PdfPipelineOptions(do_ocr=True)  # uses EasyOCR by default

# Tesseract (requires system Tesseract + pip install tesserocr — see Docling install docs)
from docling.datamodel.pipeline_options import TesseractOcrOptions
opts = PdfPipelineOptions(do_ocr=True, ocr_options=TesseractOcrOptions())

# RapidOCR (lightweight, no C deps)
from docling.datamodel.pipeline_options import RapidOcrOptions
opts = PdfPipelineOptions(do_ocr=True, ocr_options=RapidOcrOptions())

# macOS native OCR
from docling.datamodel.pipeline_options import OcrMacOptions
opts = PdfPipelineOptions(do_ocr=True, ocr_options=OcrMacOptions())
```

### Standard pipeline + OCR: CLI vs Python

`scripts/docling-convert.py` (`--pipeline standard`) maps engines like this:

| Engine | CLI | Notes |
|--------|-----|--------|
| EasyOCR | `--ocr-engine easyocr` (default) | No extra pip beyond docling defaults |
| RapidOCR | `--ocr-engine rapidocr` | Lightweight; see Docling notes on read-only FS |
| Tesseract | `--ocr-engine tesseract` | Uses `TesseractOcrOptions` → needs **`pip install tesserocr`** and system Tesseract |
| macOS Vision | `--ocr-engine mac` | `OcrMacOptions` |

**Tesseract without `tesserocr`:** Docling also provides `TesseractCliOcrOptions` (shell out to the `tesseract` binary). This helper CLI does not expose it yet; set `ocr_options=TesseractCliOcrOptions()` in Python if you only have the CLI installed.

**NVIDIA Nemotron OCR:** Not exposed in `docling-convert.py` and not present in docling **2.81.x** `pipeline_options` on PyPI. If a future Docling release adds a Nemotron options class, configure `PdfPipelineOptions(ocr_options=...)` in Python (see [pipeline options](https://docling-project.github.io/docling/reference/pipeline_options/) for your installed version) or extend the CLI.

---

## Pipeline 2: VLM Pipeline — local inference

Processes each page as an image through a vision-language model. Replaces the
standard layout detection + OCR stack entirely.

```python
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import VlmPipelineOptions
from docling.datamodel import vlm_model_specs
from docling.pipeline.vlm_pipeline import VlmPipeline

pipeline_options = VlmPipelineOptions(
    vlm_options=vlm_model_specs.GRANITEDOCLING_TRANSFORMERS,
    generate_page_images=True,
)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_cls=VlmPipeline,
            pipeline_options=pipeline_options,
        )
    }
)
```

### Available model presets (`vlm_model_specs`)

| Preset | Model | Backend | Device | Notes |
|---|---|---|---|---|
| `GRANITEDOCLING_TRANSFORMERS` | granite-docling-258M | HF Transformers | CPU/GPU | Default |
| `SMOLDOCLING_TRANSFORMERS` | smoldocling-256M | HF Transformers | CPU/GPU | Lighter |
| `GRANITEDOCLING_VLLM` | granite-docling-258M | vLLM | GPU | Fast batch |
| `GRANITEDOCLING_MLX` | granite-docling-258M-mlx | MLX | Apple MPS | M-series Macs |

### Hybrid mode: PDF text + VLM for images/tables

Set `force_backend_text=True` to use deterministic text extraction for normal
text regions while routing images and tables through the VLM. Reduces
hallucination risk on text-heavy pages.

```python
pipeline_options = VlmPipelineOptions(
    vlm_options=vlm_model_specs.GRANITEDOCLING_TRANSFORMERS,
    force_backend_text=True,   # <-- hybrid mode
    generate_page_images=True,
)
```

---

## Pipeline 3: VLM Pipeline — remote API

Sends page images to any OpenAI-compatible endpoint. Works with vLLM,
LM Studio, Ollama, or a hosted model API.

```python
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import VlmPipelineOptions
from docling.datamodel.pipeline_options_vlm_model import ApiVlmOptions, ResponseFormat
from docling.pipeline.vlm_pipeline import VlmPipeline

vlm_opts = ApiVlmOptions(
    url="http://localhost:8000/v1/chat/completions",
    params=dict(
        model="ibm-granite/granite-docling-258M",
        max_tokens=4096,
    ),
    headers={"Authorization": "Bearer YOUR_KEY"},  # omit if not needed
    prompt="Convert this page to docling.",
    response_format=ResponseFormat.DOCTAGS,
    timeout=120,
    scale=2.0,
)

pipeline_options = VlmPipelineOptions(
    vlm_options=vlm_opts,
    generate_page_images=True,
    enable_remote_services=True,  # required — gates any HTTP call
)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_cls=VlmPipeline,
            pipeline_options=pipeline_options,
        )
    }
)
```

**`enable_remote_services=True` is mandatory** for API pipelines. Docling
blocks outbound HTTP by default as a safety measure.

### Common API targets

| Server | Default URL | Notes |
|---|---|---|
| vLLM | `http://localhost:8000/v1/chat/completions` | Best throughput |
| LM Studio | `http://localhost:1234/v1/chat/completions` | Local dev |
| Ollama | `http://localhost:11434/v1/chat/completions` | Model: `ibm/granite-docling:258m` |
| OpenAI-compatible cloud | Provider URL | Set Authorization header |

---

## VLM install requirements

Local inference requires PyTorch + Transformers:

```bash
pip install docling[vlm]
# or manually:
pip install torch transformers accelerate
```

MLX (Apple Silicon only):
```bash
pip install mlx mlx-lm
```

vLLM backend (server-side):
```bash
pip install vllm
vllm serve ibm-granite/granite-docling-258M
```
