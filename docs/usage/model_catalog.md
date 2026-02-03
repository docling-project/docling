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
          <li><code>docling-layout-v2</code> (legacy)</li>
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
          <li><code>DocumentFigureClassifier-v2.0</code> ⭐</li>
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
| All Layout models | docling-ibm-models | CPU, CUDA, MPS, XPU |

**Note:** Layout models use a specialized RT-DETR-based object detection framework from `docling-ibm-models`.

### TableFormer Models (Table Structure)

| Model | Inference Engine | Supported Devices |
|-------|------------------|-------------------|
| TableFormer (fast) | docling-ibm-models | CPU, CUDA, XPU |
| TableFormer (accurate) | docling-ibm-models | CPU, CUDA, XPU |

**Note:** MPS is currently disabled for TableFormer due to performance issues.

### Image Classifier (Picture Classifier)

| Model | Inference Engine | Supported Devices |
|-------|------------------|-------------------|
| DocumentFigureClassifier-v2.0 | Transformers (ViT) | CPU, CUDA, MPS, XPU |

### OCR Engines

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
| `gemma_12b` | Gemma-3-12B | 12B | ❌ | ✅ | ❌ | ❌ | Markdown |
| `gemma_27b` | Gemma-3-27B | 27B | ❌ | ✅ | ❌ | ❌ | Markdown |
| `dolphin` | Dolphin | - | ✅ | ❌ | ❌ | ❌ | Markdown |

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
from docling.datamodel.pipeline_options import LayoutOptions
from docling.datamodel.layout_model_specs import DOCLING_LAYOUT_HERON

# Use Heron layout model (default)
layout_options = LayoutOptions(model_spec=DOCLING_LAYOUT_HERON)
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
classifier_options = DocumentPictureClassifierOptions()
```

### OCR

```python
from docling.datamodel.pipeline_options import TesseractOcrOptions

# Use Tesseract with English and German
ocr_options = TesseractOcrOptions(lang=["eng", "deu"])
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
from docling.datamodel.pipeline_options import PictureDescriptionVlmOptions

# Use Granite Vision for detailed descriptions
options = PictureDescriptionVlmOptions.from_preset("granite_vision")
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