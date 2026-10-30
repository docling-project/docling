
# Vision Models

The `VlmPipeline` in Docling allows you to convert documents end-to-end using a vision-language model.

Docling supports vision-language models which output:

- DocTags (e.g. [SmolDocling](https://huggingface.co/ds4sd/SmolDocling-256M-preview)), the preferred choice
- Markdown
- HTML

!!! tip "Complete Model Catalog"
    For a comprehensive overview of **all models and stages** in Docling (Layout, Table Structure, OCR, VLM, etc.), see the **[Model Catalog](model_catalog.md)**.

## Quick Start


For running Docling using local models with the `VlmPipeline`:

=== "CLI"

    ```bash
    docling --pipeline vlm FILE
    ```

=== "Python"

    See also the example [minimal_vlm_pipeline.py](./../examples/minimal_vlm_pipeline.py).

    ```python
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.pipeline.vlm_pipeline import VlmPipeline

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=VlmPipeline,
            ),
        }
    )

    doc = converter.convert(source="FILE").document
    ```

To extract chart data after VLM conversion, enable chart enrichment. Docling
classifies pictures before chart extraction and skips fields already present in
the VLM output.

```bash
docling --from image --to dclx IMAGE.png --pipeline vlm \
    --vlm-model mineru2_pro --enrich-chart-extraction
```

The Python SDK uses `VlmPipelineOptions(do_chart_extraction=True)` for the same
behavior. Picture classification is enabled automatically when chart extraction
is requested. Picture description and classification can also be enabled with
`--enrich-picture-description` and `--enrich-picture-classes`.

The default chart model is Granite Vision 4.1 4B. On Apple Silicon, it uses
MLX when `mlx-vlm>=0.7.0` and MPS are available; otherwise it uses Transformers.
To use the older CSV-only
`ibm-granite/granite-vision-3.3-2b-chart2csv-preview` model, add
`--chart-extraction-preset granite_vision_v3_3` to the command. In Python, set
`chart_extraction_options=ChartExtractionVlmEngineOptions.from_preset("granite_vision_v3_3")`
on `VlmPipelineOptions` or `PdfPipelineOptions`.

To require MLX explicitly, select `--chart-extraction-preset granite_vision_v4_mlx`.
This preset loads the same official
`ibm-granite/granite-vision-4.1-4b` repository. In Python, use
`ChartExtractionVlmEngineOptions.from_preset("granite_vision_v4_mlx")`.

## Available local models

By default, the vision-language models are running locally.
Docling allows to choose between the Hugging Face [Transformers](https://github.com/huggingface/transformers) framework and the [MLX](https://github.com/Blaizzy/mlx-vlm) (for Apple devices with MPS acceleration) one.

The following table reports the models currently available out-of-the-box.

| Model instance | Model | Framework | Device | Num pages | Inference time (sec) |
| ---------------|------ | --------- | ------ | --------- | ---------------------|
| `vlm_model_specs.GRANITEDOCLING_TRANSFORMERS` | [ibm-granite/granite-docling-258M](https://huggingface.co/ibm-granite/granite-docling-258M) | `Transformers/AutoModelForVision2Seq` | MPS | 1 |  - |
| `vlm_model_specs.GRANITEDOCLING_MLX` | [ibm-granite/granite-docling-258M-mlx-bf16](https://huggingface.co/ibm-granite/granite-docling-258M-mlx-bf16) | `MLX`| MPS | 1 |    - |
| `vlm_model_specs.SMOLDOCLING_TRANSFORMERS` | [ds4sd/SmolDocling-256M-preview](https://huggingface.co/ds4sd/SmolDocling-256M-preview) | `Transformers/AutoModelForVision2Seq` | MPS | 1 |  102.212 |
| `vlm_model_specs.SMOLDOCLING_MLX` | [ds4sd/SmolDocling-256M-preview-mlx-bf16](https://huggingface.co/ds4sd/SmolDocling-256M-preview-mlx-bf16) | `MLX`| MPS | 1 |    6.15453 |
| `vlm_model_specs.QWEN25_VL_3B_MLX` | [mlx-community/Qwen2.5-VL-3B-Instruct-bf16](https://huggingface.co/mlx-community/Qwen2.5-VL-3B-Instruct-bf16)  |  `MLX`| MPS | 1 |   23.4951 |
| `vlm_model_specs.NANONETS_OCR2_MLX` | [mlx-community/Nanonets-OCR2-3B-bf16](https://huggingface.co/mlx-community/Nanonets-OCR2-3B-bf16) | `MLX` | MPS | 1 | - |
| `vlm_model_specs.PIXTRAL_12B_MLX` | [mlx-community/pixtral-12b-bf16](https://huggingface.co/mlx-community/pixtral-12b-bf16) |  `MLX` | MPS | 1 |  308.856 |
| `vlm_model_specs.GEMMA3_12B_MLX` | [mlx-community/gemma-3-12b-it-bf16](https://huggingface.co/mlx-community/gemma-3-12b-it-bf16) |  `MLX` | MPS | 1 |  378.486 |
| `vlm_model_specs.GRANITE_VISION_TRANSFORMERS` | [ibm-granite/granite-vision-3.2-2b](https://huggingface.co/ibm-granite/granite-vision-3.2-2b) | `Transformers/AutoModelForVision2Seq` | MPS | 1 |  104.75 |
| `vlm_model_specs.NANONETS_OCR2_TRANSFORMERS` | [nanonets/Nanonets-OCR2-3B](https://huggingface.co/nanonets/Nanonets-OCR2-3B) | `Transformers/AutoModelForImageTextToText` | MPS | 1 | - |
| `vlm_model_specs.PHI4_TRANSFORMERS` | [microsoft/Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct) | `Transformers/AutoModelForCasualLM` | CPU | 1 | 1175.67 |
| `vlm_model_specs.PIXTRAL_12B_TRANSFORMERS` | [mistral-community/pixtral-12b](https://huggingface.co/mistral-community/pixtral-12b) | `Transformers/AutoModelForVision2Seq` | CPU | 1 | 1828.21 |

_Inference time is computed on a Macbook M3 Max using the example page `tests/data/pdf/2305.03393v1-pg9.pdf`. The comparison is done with the example [compare_vlm_models.py](./../examples/compare_vlm_models.py)._

For choosing the model, the code snippet above can be extended as follow

```python
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling.datamodel.pipeline_options import (
    VlmPipelineOptions,
)
from docling.datamodel import vlm_model_specs

pipeline_options = VlmPipelineOptions(
    vlm_options=vlm_model_specs.SMOLDOCLING_MLX,  # <-- change the model here
)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_cls=VlmPipeline,
            pipeline_options=pipeline_options,
        ),
    }
)

doc = converter.convert(source="FILE").document
```

### LightOnOCR-3

LightOnOCR-3 has two presets: `lightonocr3_08b` and `lightonocr3_4b`. They use
the official [0.8B](https://huggingface.co/lightonai/LightOnOCR-3-0.8B) and
[4B](https://huggingface.co/lightonai/LightOnOCR-3-4B) models at pinned revisions.
Local inference requires `transformers>=5.5.4` or `vllm>=0.30.0`. The existing
`lightonocr` preset continues to use LightOnOCR-2-1B.

For `docling-slim` installations, the parser also requires the Markdown extra:

```bash
pip install 'docling-slim[format-markdown]'
```

```bash
docling contract.pdf --pipeline vlm --vlm-model lightonocr3_08b
```

```python
from docling.datamodel.pipeline_options import VlmConvertOptions, VlmPipelineOptions

pipeline_options = VlmPipelineOptions(
    vlm_options=VlmConvertOptions.from_preset("lightonocr3_08b"),
)
```

Pass these options to the `PdfFormatOption` in the example above. The presets
use the `grounding` prompt and sample at temperature 0.1, because upstream advises
against greedy decoding, so two runs can give different text. Pages render at
`scale=2.0` with at most 2048 pixels on the long edge. Raise
`VlmConvertOptions.scale` to send more pixels at a higher token cost.

Docling parses the block labels and normalized bounding boxes, Markdown text,
HTML tables, image descriptions, and chart tables. A `chart` block becomes a
picture with the `other_chart` class and its table as chart data. `header_image`
and `footer_image` blocks become furniture pictures, and `page_number` blocks
become page footers. Section headings use the document-wide depth normalization
described below. LaTeX math is preserved.
Coordinates belong to blocks; derived text and table cells do not receive
separate boxes. Continuation blocks keep their base labels without being merged.
Lists keep their structure, without assigning the block box to individual items.
A caption links to a table or picture only when it has exactly one adjacent owner.

The model gives no chart type, so `chart` blocks use the `other_chart` class, and
current docling-core does not fully support that class. JSON, Markdown, and HTML
keep the chart data. DCLX and DocLang exports lose it, and `export_to_doctags()`
raises `ValueError` when a document has a `chart` block.

For plain transcription, set `pipeline_options.vlm_options.model_spec.prompt = ""`.
The parser also accepts that Markdown output, without bounding boxes. Use only
the empty prompt or `grounding`, as specified in the model cards.

Local inference uses Transformers or vLLM. No MLX export is configured for these
presets. An OpenAI-compatible server can also serve the official model
repositories; configure its URL and engine options as described below.

### Other models

Other models can be configured by directly providing the Hugging Face `repo_id`, the prompt and a few more options.

For example:

```python
from docling.datamodel.accelerator_options import AcceleratorDevice
from docling.datamodel.pipeline_options import VlmPipelineOptions
from docling.datamodel.pipeline_options_vlm_model import InlineVlmOptions, InferenceFramework, ResponseFormat, TransformersModelType

pipeline_options = VlmPipelineOptions(
    vlm_options=InlineVlmOptions(
        repo_id="ibm-granite/granite-vision-3.2-2b",
        prompt="Convert this page to markdown. Do not miss any text and only output the bare markdown!",
        response_format=ResponseFormat.MARKDOWN,
        inference_framework=InferenceFramework.TRANSFORMERS,
        transformers_model_type=TransformersModelType.AUTOMODEL_IMAGETEXTTOTEXT,
        supported_devices=[
            AcceleratorDevice.CPU,
            AcceleratorDevice.CUDA,
            AcceleratorDevice.MPS,
            AcceleratorDevice.XPU,
        ],
        scale=2.0,
        temperature=0.0,
    )
)
```


## Remote models

Additionally to local models, the `VlmPipeline` allows to offload the inference to a remote service hosting the models.
Many remote inference services are provided, the key requirement is to offer an OpenAI-compatible API. This includes vLLM, Ollama, etc.

More examples on how to connect with the remote inference services can be found in the following examples:

- [vlm_pipeline_api_model.py](./../examples/vlm_pipeline_api_model.py)

### Request parameters

`params` is merged into every chat-completions request on top of the parameters
Docling adds itself. A key in `params` overrides Docling's value, and a key set to
`None` is left out of the request altogether. What Docling adds depends on the
option class:

- `ApiVlmEngineOptions` (the API inference engine, also used by the engine-based
  picture description): `temperature`, and `max_tokens` when a token limit is
  configured. A `max_completion_tokens` in `params` replaces `max_tokens`. A
  non-empty `params` also replaces the API parameters the model spec would
  otherwise contribute (such as `model`), so name the model in `params` as well.
- `ApiVlmOptions` (the legacy `ApiVlmModel`): only `temperature`, taken from its
  `temperature` option.

This is how to use a model that rejects one of the defaults, for example a GPT-5
deployment on Azure OpenAI, which accepts only its default temperature and
`max_completion_tokens` instead of `max_tokens`:

```python
engine_options = ApiVlmEngineOptions(
    url="https://<resource>.openai.azure.com/openai/deployments/<deployment>/chat/completions?api-version=2025-03-01-preview",
    headers={"api-key": "<key>"},
    params={"model": "<deployment>", "temperature": 1, "max_completion_tokens": 1024},
    # or leave the temperature out entirely: "temperature": None
)
```

## Native dots inline formatting

Native dots.ocr and dots.mocr JSON text can contain Markdown bold and italic
markers. Docling maps this emphasis and inline HTML to styled text spans. Mixed
text uses inline groups. Titles, heading levels, list items, and text block labels
are preserved. Formula blocks, inline math, and code keep their literal content.

Markdown emphasis uses the existing `marko` dependency. For slim installations,
install it with `pip install 'docling-slim[format-markdown]'`.

## Native MinerU and dots captions

MinerU and dots caption blocks are linked when exactly one immediately adjacent
native block maps to a compatible table or picture. MinerU's
`table_caption`, `image_caption`, and `code_caption` labels constrain the owner
type; generic captions from either model consider all three types. Both preceding
and following blocks are considered, without geometry or text-based guesses.
Skipped blocks remain barriers. Captions with no compatible neighbor or two
compatible neighbors remain standalone text in DocLang. Caption text and its
own provenance are preserved.

Code captions and captions containing formatted child runs remain standalone:
the minimum supported Docling Core version cannot faithfully serialize those
associations in DocLang. A container's existing caption takes precedence;
additional captions remain standalone because its DocLang head holds one caption.

## Chandra HTML output

Use `ResponseFormat.CHANDRA_HTML` with `CHANDRA_OCR_LAYOUT_PROMPT` for Chandra's
HTML layout blocks. Docling scales each block's `data-bbox` coordinates from
0–1000 to the source page size. At most one document item representing an
annotated block receives its box. Derived children remain unlocated unless their
source HTML element declares its own `data-bbox`; the model does not provide
separate coordinates for each word or table cell.

The converter preserves paragraphs, heading levels, nested lists, table spans,
inline formatting, links, math, and code whitespace. Tables are recognized by
their markup even inside blocks labeled `Text`, `Form`, or `Figure`. Cells with
structured content use `RichTableCell` references. Form regions retain checkbox
and radio states and fillable text values without inferring key–value links
from visual proximity.

Picture descriptions from `img alt` are stored in `PictureItem.meta.description`.
Chart tables, diagram code, and other picture content remain children of the
picture. Explicit `<chem>` content is stored as SMILES molecule metadata.
Captions and footnotes are linked when nested under their picture or table;
separate layout blocks remain unlinked. Separate table fragments remain separate
tables. CSS layout and styling without corresponding document primitives are
not reconstructed.

To retain picture pixels, enable `generate_picture_images` or
`generate_page_images` on `VlmPipelineOptions`. With page images retained,
`picture.get_image(document)` can crop the picture using its provenance.
When exporting Markdown, use `traverse_pictures=True` to include picture children.
Use JSON to inspect all metadata and rich document structure; individual export
formats may omit some of these details.

Recognizable HTML without layout blocks is recovered with a warning and without
invented coordinates. Invalid bounding boxes also produce a warning while their
content is retained. Nonempty prose or JSON responses without HTML transcription
produce a page-specific inference error and `PARTIAL_SUCCESS`, rather than an
unreported empty result. An explicitly labeled `Blank-Page` may be empty.

## Heading depths in native layout output

For Chandra, dots, Nemotron Parse, DeepSeek-OCR, Unlimited-OCR, Unlimited-OCR 2,
PaddleOCR-VL, and LightOnOCR-3 output, the layout label alone decides title versus
section header.
LightOnOCR-3 has one exception: its `title` label covers all headings, so a `title`
block with `##` or deeper is a section header. Docling
exports a title as `#` and a section header one step below it. If no section
header in the document uses depth 1 (`#` or `<h1>`), each section header exports
with its source depth: `##` stays `##`. Otherwise each one exports one step
deeper, so different source depths never share one section level. A section
header without heading markup counts as `##`. Markdown has no heading deeper than
`######`, so in a document with depth 1, source depths 5 and 6 both export as
`######`, but their section levels stay different.

Unlimited-OCR 2 has one `title` label. A `title` with `<LEVELn>` is a section header
of depth `n`, and a `title` without a level is the title.

The VLM pipeline applies this rule once to the whole document, so one depth gets
the same level on all pages. Each mapper function applies it to its own page when
you call it directly. `parse_paddleocr_vl_result` reads one saved PaddleOCR-VL
page, so it always applies the rule to that page.
