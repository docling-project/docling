# Model Catalog

This document provides a comprehensive overview of all models and inference engines available in Docling, organized by processing stage.

## Overview

Docling's document processing pipeline consists of multiple stages, each using specialized models and inference engines. This catalog helps you understand:

- What stages are available for document processing
- Which model families power each stage
- What specific models you can use
- Which inference engines support each model

## Stages and Models Overview

The following table shows all processing stages in Docling, their model families, and available models.

<table border="1" cellpadding="6" cellspacing="0">
  <thead>
    <tr>
      <th>Stage</th>
      <th>Model Family</th>
      <th>Models</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="4"><strong>Layout</strong><br/><em>Document structure detection</em></td>
      <td rowspan="4">Object Detection<br/>(RT-DETR based)</td>
      <td>
        <ul>
          <li><code>docling-layout-heron</code> ⭐</li>
          <li><code>docling-layout-heron-101</code></li>
          <li><code>docling-layout-egret-medium</code></li>
          <li><code>docling-layout-egret-large</code></li>
          <li><code>docling-layout-egret-xlarge</code></li>
          <li><code>docling-layout-v2</code> (legacy, no longer supported)</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td colspan="2"><strong>Inference Engine:</strong> Transformers, ONNXRuntime (in progress)</td>
    </tr>
    <tr>
      <td colspan="2"><strong>Purpose:</strong> Detects document elements (paragraphs, tables, figures, headers, etc.)</td>
    </tr>
    <tr>
      <td colspan="2"><strong>Output:</strong> Bounding boxes with element labels (TEXT, TABLE, PICTURE, SECTION_HEADER, etc.)</td>
    </tr>
    <tr>
      <td rowspan="3"><strong>OCR</strong><br/><em>Text recognition</em></td>
      <td rowspan="3">Multiple OCR Engines</td>
      <td>
        <ul>
          <li><strong>Auto</strong> ⭐</li>
          <li><strong>Tesseract</strong> (CLI or Python bindings)</li>
          <li><strong>EasyOCR</strong></li>
          <li><strong>RapidOCR</strong> (ONNX, OpenVINO, PaddlePaddle)</li>
          <li><strong>macOS Vision</strong> (native macOS)</li>
          <li><strong>SuryaOCR</strong></li>
        </ul>
      </td>
    </tr>
    <tr>
      <td colspan="2"><strong>Inference Engines:</strong> Engine-specific</td>
    </tr>
    <tr>
      <td colspan="2"><strong>Purpose:</strong> Extracts text from images and scanned documents</td>
    </tr>
    <tr>
      <td rowspan="3"><strong>Table Structure</strong><br/><em>Table cell recognition</em></td>
      <td rowspan="3">TableFormer</td>
      <td>
        <ul>
          <li><code>TableFormer (accurate mode)</code> ⭐</li>
          <li><code>TableFormer (fast mode)</code></li>
        </ul>
      </td>
    </tr>
    <tr>
      <td colspan="2"><strong>Inference Engine:</strong> docling-ibm-models</td>
    </tr>
    <tr>
      <td colspan="2"><strong>Purpose:</strong> Recognizes table structure (rows, columns, cells) and relationships</td>
    </tr>
    <tr>
      <td rowspan="3"><strong>Table Structure</strong><br/><em>Table cell recognition</em></td>
      <td rowspan="3">Vision-Language Model<br/>(Granite Vision)</td>
      <td>
        <ul>
          <li><code>granite-vision-4.1-4b</code></li>
        </ul>
      </td>
    </tr>
    <tr>
      <td colspan="2"><strong>Inference Engine:</strong> Transformers</td>
    </tr>
    <tr>
      <td colspan="2"><strong>Purpose:</strong> VLM-based table structure recognition using OTSL (Open Table Structure Language) output</td>
    </tr>
    <tr>
      <td rowspan="3"><strong>Table Structure</strong><br/><em>Table cell recognition</em></td>
      <td rowspan="3">Object Detection</td>
      <td>
        <ul>
          <li><em>Work in progress</em></li>
        </ul>
      </td>
    </tr>
    <tr>
      <td colspan="2"><strong>Inference Engine:</strong> TBD</td>
    </tr>
    <tr>
      <td colspan="2"><strong>Purpose:</strong> Alternative approach for table structure recognition using object detection</td>
    </tr>
    <tr>
      <td rowspan="3"><strong>Picture Classifier</strong><br/><em>Image type classification</em></td>
      <td rowspan="3">Image Classifier<br/>(Vision Transformer)</td>
      <td>
        <ul>
          <li><code>DocumentFigureClassifier-v2.5</code> ⭐</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td colspan="2"><strong>Inference Engine:</strong> Transformers</td>
    </tr>
    <tr>
      <td colspan="2"><strong>Purpose:</strong> Classifies pictures into categories (Chart, Diagram, Natural Image, etc.)</td>
    </tr>
    <tr>
      <td rowspan="4"><strong>VLM Convert</strong><br/><em>Full page conversion</em></td>
      <td rowspan="4">Vision-Language Models</td>
      <td>
        <ul>
          <li><strong>Granite-Docling-258M</strong> ⭐ (DocTags)</li>
          <li><strong>SmolDocling-256M</strong> (DocTags)</li>
          <li><strong>DeepSeek-OCR-3B</strong> (Markdown, API-only)</li>
          <li><strong>Granite-Vision-3.3-2B</strong> (Markdown)</li>
          <li><strong>Pixtral-12B</strong> (Markdown)</li>
          <li><strong>GOT-OCR-2.0</strong> (Markdown)</li>
          <li><strong>Phi-4-Multimodal</strong> (Markdown)</li>
          <li><strong>Qwen2.5-VL-3B</strong> (Markdown)</li>
          <li><strong>Nanonets-OCR2-3B</strong> (Markdown)</li>
          <li><strong>Gemma-3-12B/27B</strong> (Markdown, MLX-only)</li>
          <li><strong>Dolphin</strong> (Markdown)</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td colspan="2"><strong>Inference Engines:</strong> Transformers, MLX, API (Ollama, LM Studio, OpenAI), vLLM, AUTO_INLINE</td>
    </tr>
    <tr>
      <td colspan="2"><strong>Purpose:</strong> Converts entire document pages to structured formats (DocTags or Markdown)</td>
    </tr>
    <tr>
      <td colspan="2"><strong>Output Formats:</strong> DocTags (structured), Markdown (human-readable)</td>
    </tr>
    <tr>
      <td rowspan="3"><strong>Picture Description</strong><br/><em>Image captioning</em></td>
      <td rowspan="3">Vision-Language Models</td>
      <td>
        <ul>
          <li><strong>SmolVLM-256M</strong> ⭐</li>
          <li><strong>Granite-Vision-3.3-2B</strong></li>
          <li><strong>Pixtral-12B</strong></li>
          <li><strong>Qwen2.5-VL-3B</strong></li>
        </ul>
      </td>
    </tr>
    <tr>
      <td colspan="2"><strong>Inference Engines:</strong> Transformers, MLX, API (Ollama, LM Studio), vLLM, AUTO_INLINE</td>
    </tr>
    <tr>
      <td colspan="2"><strong>Purpose:</strong> Generates natural language descriptions of images and figures</td>
    </tr>
    <tr>
      <td rowspan="3"><strong>Code & Formula</strong><br/><em>Code/math extraction</em></td>
      <td rowspan="3">Vision-Language Models</td>
      <td>
        <ul>
          <li><strong>CodeFormulaV2</strong> ⭐</li>
          <li><strong>Granite-Docling-258M</strong></li>
        </ul>
      </td>
    </tr>
    <tr>
      <td colspan="2"><strong>Inference Engines:</strong> Transformers, MLX, AUTO_INLINE</td>
    </tr>
    <tr>
      <td colspan="2"><strong>Purpose:</strong> Extracts and recognizes code blocks and mathematical formulas</td>
    </tr>
  </tbody>
</table>

## Inference Engines by Model Family

### Object Detection Models (Layout)

| Model | Inference Engine | Supported Devices |
|-------|------------------|-------------------|
| All layout models | Transformers | CPU, CUDA, MPS, XPU |

**Note:** Layout models run through `AutoModelForObjectDetection`, reading their label map
from the model repository's own `config.json`. Only `docling-layout-heron` also supports ONNXRuntime.

**Deprecated:** `LayoutOptions` and `docling-layout-v2` are superseded by
`LayoutObjectDetectionOptions`. `LayoutOptions` still works — it is translated onto the
object-detection path — but selecting `DOCLING_LAYOUT_V2` warns and falls back to Heron.

### TableFormer Models (Table Structure)

| Model | Inference Engine | Supported Devices |
|-------|------------------|-------------------|
| TableFormer (fast) | docling-ibm-models | CPU, CUDA, XPU |
| TableFormer (accurate) | docling-ibm-models | CPU, CUDA, XPU |

**Note:** MPS is currently disabled for TableFormer due to performance issues.

### Image Classifier (Picture Classifier)

| Model | Inference Engine | Supported Devices |
|-------|------------------|-------------------|
| DocumentFigureClassifier-v2.5 | Transformers (ViT) | CPU, CUDA, MPS, XPU |

### OCR Engines

Languages are given as BCP-47 tags for every engine; see
[OCR engines](../concepts/OCR.md#language-selection).

| OCR Engine | Backend | Language Support | Notes |
|------------|---------|------------------|-------|
| Tesseract | CLI or tesserocr | 100+ languages | Most widely used, good accuracy |
| EasyOCR | PyTorch | 80+ languages | GPU-accelerated, good for Asian languages |
| RapidOCR | ONNX/OpenVINO/Paddle | Multiple | Fast, multiple backend options |
| macOS Vision | Native macOS | 20+ languages | macOS only, excellent quality |
| SuryaOCR | PyTorch | 90+ languages | Modern, good for complex layouts |
| Auto | Automatic | Varies | Automatically selects best available engine |

### Vision-Language Models (VLM)

#### VLM Convert Stage

| Preset ID | Model | Parameters | Transformers | MLX | API (OpenAI-compatible) | vLLM | Output Format |
|-----------|-------|------------|--------------|-----|-------------------------|------|---------------|
| `granite_docling` | Granite-Docling-258M | 258M | ✅ | ✅ | Ollama | ❌ | DocTags |
| `smoldocling` | SmolDocling-256M | 256M | ✅ | ✅ | ❌ | ❌ | DocTags |
| `deepseek_ocr` | DeepSeek-OCR-3B | 3B | ❌ | ❌ | Ollama<br/>LM Studio | ❌ | Markdown |
| `granite_vision` | Granite-Vision-3.3-2B | 2B | ✅ | ❌ | Ollama<br/>LM Studio | ✅ | Markdown |
| `pixtral` | Pixtral-12B | 12B | ✅ | ✅ | ❌ | ❌ | Markdown |
| `got_ocr` | GOT-OCR-2.0 | - | ✅ | ❌ | ❌ | ❌ | Markdown |
| `phi4` | Phi-4-Multimodal | - | ✅ | ❌ | ❌ | ✅ | Markdown |
| `qwen` | Qwen2.5-VL-3B | 3B | ✅ | ✅ | ❌ | ❌ | Markdown |
| `nanonets_ocr2` | Nanonets-OCR2-3B | 3B | ✅ | ✅ | OpenAI-compatible<br/>LM Studio | ✅ | Markdown |
| `nemotron_parse_v2` | NVIDIA Nemotron Parse 2.0 | 0.9B | ✅ | ✅ | ❌ | ✅ | Native layout with Markdown text and LaTeX tables |
| `nemotron_parse_v3_5` | NVIDIA Nemotron 3.5 Parse | 0.6B | ✅ | ❌ | OpenAI-compatible | ❌ | Native layout with Markdown text, LaTeX tables, and chart data |
| `mineru2_pro` | MinerU2.5-Pro-2604 | 1.2B | ✅ | ✅ | OpenAI-compatible<br/>LM Studio | ❌ | Native two-step layout, text, formulas, and OTSL tables |
| `gemma_12b` | Gemma-3-12B | 12B | ❌ | ✅ | ❌ | ❌ | Markdown |
| `gemma_27b` | Gemma-3-27B | 27B | ❌ | ✅ | ❌ | ❌ | Markdown |
| `dolphin` | Dolphin | - | ✅ | ❌ | ❌ | ❌ | Markdown |
| `unlimited_ocr` | Unlimited-OCR | 3.34B (MoE) | ❌ | ❌ | OpenAI-compatible | ✅ | Markdown with layout blocks |
| `unlimited_ocr2` | Unlimited-OCR 2 | 3.34B (MoE) | ❌ | ❌ | OpenAI-compatible | ❌ | Native layout blocks with HTML tables, one page per request |
| `lightonocr3_08b` | LightOnOCR-3-0.8B | 0.8B | ✅ | ❌ | OpenAI-compatible | ✅ | Markdown with layout blocks, HTML tables, and chart data |
| `lightonocr3_4b` | LightOnOCR-3-4B | 4B | ✅ | ❌ | OpenAI-compatible | ✅ | Markdown with layout blocks, HTML tables, and chart data |

`nanonets_ocr2` includes preset API overrides for OpenAI-compatible runtimes and LM Studio, and can also be used with vLLM runtimes.

`mineru2_pro` runs MinerU's native two-step flow: page-level layout detection followed by
type-specific recognition of the detected region crops. Its Transformers and API configurations
use the official `opendatalab/MinerU2.5-Pro-2604-1.2B` repository. The MLX override uses the
community-published `carlesonielfa/MinerU2.5-Pro-2604-1.2B-mlx-bf16` conversion because no official
OpenDataLab MLX checkpoint is currently available. Image/chart analysis and cross-page table merging
are not enabled by this preset.

`nemotron_parse_v3_5` runs code from the `nvidia/Nemotron-3.5-Parse` model repository, pinned to
revision `e7ea79242a87ea0ad3919b672bfedb429a683206`. Docling does not trust that code by default:
review it, then set `trust_remote_code=True` on the VLM engine options (for example
`AutoInlineVlmEngineOptions(trust_remote_code=True)`) or pass `--trust-remote-code` to the CLI.
Without this opt-in, the Transformers engine stops with an error before it downloads the model.
In-process vLLM is not supported because the model needs NVIDIA's out-of-tree vLLM plugin. To
use vLLM, start NVIDIA's server (`vllm serve ... --served-model-name nemotron-parse`, see the
model card) and use `ApiVlmEngineOptions(engine_type=VlmEngineType.API, url=...)`. The API override
sends `top_k=1`, `repetition_penalty=1.1`, and `skip_special_tokens=False`. If you set your own
`params`, the preset parameters are not sent, so include these values yourself. The preset maps
charts to pictures with a chart class and data table, with one picture for each chart panel.

The default is single-pass. To re-parse Table and Chart crops, set
`VlmConvertOptions.nemotron_parse` in the Python SDK:

```python
from docling.datamodel.pipeline_options import (
    NemotronParseOptions,
    VlmConvertOptions,
    VlmPipelineOptions,
)
from docling.datamodel.vlm_engine_options import AutoInlineVlmEngineOptions

pipeline_options = VlmPipelineOptions(
    vlm_options=VlmConvertOptions.from_preset(
        "nemotron_parse_v3_5",
        engine_options=AutoInlineVlmEngineOptions(trust_remote_code=True),
        nemotron_parse=NemotronParseOptions(
            passes="double",  # "single", "double", or "triple"
            triple_regions="union",  # "union" or "detector"; triple-pass only
            crop_batch_size=8,
        ),
    ),
)
```

Double-pass uses the Table and Chart boxes from the page parse. Triple-pass adds a
separate no-text detection request. `triple_regions="detector"` uses detector boxes;
`"union"` also keeps parser boxes that do not overlap detector boxes of the same class.
Both Transformers and API engines use bounded crop batches. Each page has at most
80 crop regions. Crops have a 4% + 12 pixel margin, use JPEG quality 95, and are
skipped if either padded dimension is less than 8 pixels. Crop coordinates map back
to the page frame.

Multi-pass page and detection requests use temperature 0.2 and repetition penalty
1.05. Crop requests use 0.1 and 1.0. All stages use `top_k=50` and `top_p=1.0`.
Detection has a 4096 token limit; page and crop requests have an 8744 token limit.
Explicit API `params` override these request settings. Single-pass keeps the original
preset settings. For every Transformers VLM preset, output that reaches the token
limit without EOS is now reported as incomplete. The VLM pipeline returns
`PARTIAL_SUCCESS` for these pages. Output that reaches EOS at the token limit
remains complete.

Only guarded merging is supported. A crop table must have a valid grid and compatible
column count. A replacement also needs at least 0.5 box intersection over union and
0.65 text similarity. It keeps the page box and reading order. New objects must not
overlap an existing object of the same class. Failed, empty, truncated, and invalid
crop output keeps the page result. Invalid detection output skips the crop stage.
These guards can retain the less detailed page result. NVIDIA's published multi-pass
scores use its alternatives merge policy and do not measure this guarded policy.

Each page prediction has `nemotron_parse` metadata with the mode and model request
counts. The exported document keeps these values at
`body.meta.docling__nemotron_parse.pages`, keyed by source page number. `requests`
counts page, detection, and crop requests, including unsuccessful crop requests;
it is not a stage or engine-batch count. `rejected_crop_requests` counts unsuccessful
or invalid crop generations before merging. Token counts include all stages, and
`usage` keeps each stage response's provider usage when available.

`unlimited_ocr` is served through an OpenAI-compatible endpoint: point `ApiVlmEngineOptions.url` at your own runtime. Its API override sets `skip_special_tokens=False`, without which the layout annotations are stripped from the completion.

`unlimited_ocr2` runs through an OpenAI-compatible server that you start yourself. Docling runs
no model code in-process for this preset. It has no MLX or in-process vLLM configuration. The
Transformers engine cannot run it, because the model needs its own generation loop and
`transformers==4.52.1`. vLLM support needs [vllm-project/vllm#64016](https://github.com/vllm-project/vllm/pull/64016),
which is not merged yet.

Today the server is the customized SGLang wheel from the model repository. It needs an NVIDIA
GPU. The command below is the model card command at the pinned revision, without
`--enable-custom-logit-processor`, because Docling does not send a logit processor. The model card
also lists memory settings for the server.

```shell
hf download baidu/Unlimited-OCR-2 \
    --revision c14071b549f4fcab10909f3c4acc6a2edb366947 --local-dir Unlimited-OCR-2
uv venv --python 3.10 && source .venv/bin/activate
uv pip install Unlimited-OCR-2/wheel/sglang-0.5.20.dev13+g75b9b06b02-py3-none-any.whl \
    fastokens==0.3.2

python -m sglang.launch_server \
    --model-path Unlimited-OCR-2 \
    --served-model-name Unlimited-OCR-2 \
    --chat-template unlimited-ocr \
    --tokenizer-backend fastokens \
    --tokenizer-worker-num 8 \
    --mem-fraction-static 0.7 \
    --max-running-requests 64 \
    --max-prefill-tokens 2048 \
    --enable-streaming-session \
    --allow-auto-truncate \
    --disable-radix-cache \
    --mm-preprocess-cache-size-mb 0 \
    --cuda-graph-backend-decode full --cuda-graph-max-bs-decode 64 \
    --cuda-graph-backend-prefill full \
    --cuda-graph-max-bs-prefill 16384 \
    --cuda-graph-bs-prefill 1024 2048 \
    --cuda-graph-config '{"prefill": {"backend": "full", "full_prefill_max_req": 64}}' \
    --host 127.0.0.1 \
    --port 10000
```

Keep the CUDA Graph flags. The vendor reports that eager mode can repeat box coordinates. Then
point the preset at the server in the Python SDK. The CLI has no option for the API URL, so
`--vlm-model unlimited_ocr2` uses the default OpenAI URL and is not useful without the SDK.

```python
from docling.datamodel.pipeline_options import VlmConvertOptions, VlmPipelineOptions
from docling.datamodel.vlm_engine_options import ApiVlmEngineOptions, VlmEngineType

pipeline_options = VlmPipelineOptions(
    enable_remote_services=True,
    vlm_options=VlmConvertOptions.from_preset(
        "unlimited_ocr2",
        engine_options=ApiVlmEngineOptions(
            engine_type=VlmEngineType.API_OPENAI,
            url="http://127.0.0.1:10000/v1/chat/completions",
            timeout=600,
        ),
    ),
)
```

Limits of this preset:

- **One page per request.** Docling sends each page alone. The main feature of the model is to
  read all pages of a document in one request, and Docling does not use this mode yet. The vendor's
  paper reports only a small gain on single pages: 94.02 against 93.92 for Unlimited-OCR on
  OmniDocBench.
- **No merge across pages.** Docling reads `cross_` and `append_` blocks as normal blocks. A table
  or paragraph that continues on the next page stays split. `<MERGE>` joins text on the same page.
- **No repetition guard.** The vendor client sends a pickled logit processor, and Docling does not.
  A page that repeats ends at `max_tokens` (8000) with the `length` stop reason.
- **Your own `params`.** If you set `params`, the preset parameters are not sent, so include
  `model`, `max_tokens`, and `skip_special_tokens=False`. Without the last value, the server removes
  `<td>` and `<tr>` from tables.

#### Picture Description Stage

| Preset ID | Model | Parameters | Transformers | MLX | API (OpenAI-compatible) | vLLM |
|-----------|-------|------------|--------------|-----|-------------------------|------|
| `smolvlm` | SmolVLM-256M | 256M | ✅ | ✅ | LM Studio | ❌ |
| `granite_vision` | Granite-Vision-3.3-2B | 2B | ✅ | ❌ | Ollama<br/>LM Studio | ✅ |
| `pixtral` | Pixtral-12B | 12B | ✅ | ✅ | ❌ | ❌ |
| `qwen` | Qwen2.5-VL-3B | 3B | ✅ | ✅ | ❌ | ❌ |

#### Code & Formula Stage

| Preset ID | Model | Parameters | Transformers | MLX |
|-----------|-------|------------|--------------|-----|
| `codeformulav2` | CodeFormulaV2 | - | ✅ | ❌ |
| `granite_docling` | Granite-Docling-258M | 258M | ✅ | ✅ |

## Usage Examples

### Layout Detection

```python
from docling.datamodel.pipeline_options import LayoutObjectDetectionOptions

# Use Heron layout model (default)
layout_options = LayoutObjectDetectionOptions.from_preset("layout_heron_default")

# Or a higher-accuracy variant
layout_options = LayoutObjectDetectionOptions.from_preset("layout_egret_large")
```

### Table Structure Recognition

```python
from docling.datamodel.pipeline_options import TableStructureOptions, TableFormerMode

# Use accurate mode for best quality
table_options = TableStructureOptions(
    mode=TableFormerMode.ACCURATE,
    do_cell_matching=True
)
```

### Picture Classification

```python
from docling.models.stages.picture_classifier.document_picture_classifier import (
    DocumentPictureClassifierOptions
)

# Use default picture classifier
classifier_options = DocumentPictureClassifierOptions.from_preset("document_figure_classifier_v2")
```

### OCR

```python
from docling.datamodel.pipeline_options import TesseractOcrOptions

# Use Tesseract with English and German. Languages are BCP-47 tags;
# `eng`/`deu` still work and canonicalize to `en-Latn`/`de-Latn`.
ocr_options = TesseractOcrOptions(lang=["en", "de"])
```

### VLM Convert (Full Page)

```python
from docling.datamodel.pipeline_options import VlmConvertOptions

# Use SmolDocling with auto-selected engine
options = VlmConvertOptions.from_preset("smoldocling")

# Or force specific engine
from docling.datamodel.vlm_engine_options import MlxVlmEngineOptions
options = VlmConvertOptions.from_preset(
    "smoldocling",
    engine_options=MlxVlmEngineOptions()
)
```

### Picture Description

```python
from docling.datamodel.pipeline_options import PictureDescriptionVlmEngineOptions

# Use Granite Vision for detailed descriptions
options = PictureDescriptionVlmEngineOptions.from_preset("granite_vision")
```

### Code & Formula Extraction

```python
from docling.datamodel.pipeline_options import CodeFormulaVlmOptions

# Use specialized CodeFormulaV2 model
options = CodeFormulaVlmOptions.from_preset("codeformulav2")
```

## Additional Resources

- [Vision Models Usage Guide](vision_models.md) - VLM-specific documentation
- [Advanced Options](advanced_options.md) - Advanced configuration
- [GPU Support](gpu.md) - GPU acceleration setup
- [Supported Formats](supported_formats.md) - Input format support

## Notes

- **DocTags Format:** Structured XML-like format optimized for document understanding
- **Markdown Format:** Human-readable format for general-purpose conversion
- **Model Updates:** New models are added regularly. Check the codebase for latest additions
- **Engine Compatibility:** Not all engines work on all platforms. AUTO_INLINE handles this automatically
- **Performance:** Actual performance varies by hardware, document complexity, and model size
