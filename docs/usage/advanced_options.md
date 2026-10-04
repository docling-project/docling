## Model prefetching and offline usage

By default, models are downloaded automatically upon first usage. If you would prefer
to explicitly prefetch them for offline use (e.g. in air-gapped environments) you can do
that as follows:

**Step 1: Prefetch the models**

Use the `docling-tools models download` utility:

```sh
$ docling-tools models download
Downloading layout model...
Downloading tableformer model...
Downloading picture classifier model...
Downloading code formula model...
Downloading rapidocr torch ch models...
Downloading rapidocr onnxruntime ch models...
Models downloaded into $HOME/.cache/docling/models.
```

To prefetch EasyOCR recognition models for specific languages, repeat
`--easyocr-lang` with the same values used by `EasyOcrOptions.lang` -- EasyOCR's own codes, or
BCP-47 tags behind the `iso:` prefix:

```sh
$ docling-tools models download easyocr --easyocr-lang iso:zh-Hans --easyocr-lang ja
```

Alternatively, models can be programmatically downloaded using `docling.utils.model_downloader.download_models()`.

Also, you can use `download-hf-repo` parameter to download arbitrary models from HuggingFace by specifying repo id:

```sh
$ docling-tools models download-hf-repo ds4sd/SmolDocling-256M-preview
Downloading ds4sd/SmolDocling-256M-preview model from HuggingFace...
```

**Step 2: Use the prefetched models**

```python
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import EasyOcrOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

artifacts_path = "/local/path/to/models"

pipeline_options = PdfPipelineOptions(artifacts_path=artifacts_path)
doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)
```

Or using the CLI:

```sh
docling --artifacts-path="/local/path/to/models" FILE
```

Or using the `DOCLING_ARTIFACTS_PATH` environment variable:

```sh
export DOCLING_ARTIFACTS_PATH="/local/path/to/models"
python my_docling_script.py
```

## Using remote services

The main purpose of Docling is to run local models which are not sharing any user data with remote services.
Anyhow, there are valid use cases for processing part of the pipeline using remote services, for example invoking OCR engines from cloud vendors or the usage of hosted LLMs.

In Docling we decided to allow such models, but we require the user to explicitly opt-in in communicating with external services.

```py
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

pipeline_options = PdfPipelineOptions(enable_remote_services=True)
doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)
```

When the value `enable_remote_services=True` is not set, the system will raise an exception `OperationNotAllowed()`.

_Note: This option is only related to the system sending user data to remote services. Control of pulling data (e.g. model weights) follows the logic described in [Model prefetching and offline usage](#model-prefetching-and-offline-usage)._

### List of remote model services

The options in this list require the explicit `enable_remote_services=True` when processing the documents.

- `PictureDescriptionApiOptions`: Using vision models via API calls.
- `KserveV2OcrOptions`: OCR on a KServe v2 inference server (e.g. Triton).
- `ApiKserveV2ObjectDetectionEngineOptions`: Object-detection layout models served by a KServe v2 inference server, set as the layout `engine_options`.
- `ApiKserveV2ImageClassificationEngineOptions`: Picture classification served by a KServe v2 inference server, set as the classifier `engine_options`.
- `ApiVlmEngineOptions`: VLM stages (VLM conversion, code/formula enrichment, picture description) calling an OpenAI-compatible API, set as the stage `engine_options`.
- `ApiVlmOptions`: VLM pipeline models calling an OpenAI-compatible API.


## Adjust pipeline features

The example file [custom_convert.py](../examples/custom_convert.py) contains multiple ways
one can adjust the conversion pipeline and features.

### Image resolution and scale

Page coordinates use 72 points per inch. For image inputs, embedded DPI metadata
determines the physical page size; missing DPI and `(1, 1)` DPI are treated as 72 DPI.
Rendering at scale `n` produces `n` pixels per document point.

### Control PDF table extraction options

You can control if table structure recognition should map the recognized structure back to PDF cells (default) or use text cells from the structure prediction itself.
This can improve output quality if you find that multiple columns in extracted tables are erroneously merged into one.


```python
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions

pipeline_options = PdfPipelineOptions(do_table_structure=True)
pipeline_options.table_structure_options.do_cell_matching = False  # uses text cells predicted from table structure model

doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)
```

Since docling 1.16.0: You can control which TableFormer mode you want to use. Choose between `TableFormerMode.FAST` (faster but less accurate) and `TableFormerMode.ACCURATE` (default) to receive better quality with difficult table structures.

```python
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode

pipeline_options = PdfPipelineOptions(do_table_structure=True)
pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE  # use more accurate TableFormer model

doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)
```


### Use visible PDF rules for reading order

For PDFs whose columns or horizontal bands are separated by visible rules, the
rule-based reading-order stage can use those rules as additional structural
signals. This is enabled by default. Disable it when needed:

```python
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

pipeline_options = PdfPipelineOptions(use_reading_order_separators=False)
doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)
```

The option uses visible vector geometry exposed by the PDF backend. Separator
geometry affects ordering only and is not added to the resulting document.

The same option is available from the CLI. Use `--no-reading-order-separators`
to disable it. `--output-file` selects an exact destination when converting one
input to one output format:

```bash
uv run docling convert --from pdf --to dclx \
  --output-file ./Elsevier-with-separators.dclx \
  ./Elsevier.pdf

uv run docling convert --from pdf --to dclx \
  --no-reading-order-separators \
  --output-file ./Elsevier-without-separators.dclx \
  ./Elsevier.pdf
```


### Extract the native content of a PDF

`NativePdfPipeline` uses docling-parse alone: one text item per native text cell
and one picture per embedded bitmap, without layout, OCR or table models.

```python
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import NativePdfPipelineOptions
from docling.document_converter import DocumentConverter, NativePdfFormatOption

pipeline_options = NativePdfPipelineOptions()
pipeline_options.generate_page_images = True
pipeline_options.images_scale = 2.0

doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: NativePdfFormatOption(pipeline_options=pipeline_options)
    }
)
```

Set `generate_page_images=False` to skip rendering. `parser_threads` configures
docling-parse independently of model-inference `accelerator_options.num_threads`.

```sh
docling --pipeline native --from pdf FILE
docling --pipeline native --from pdf --parser-threads 8 FILE
```


### Recover PDF heading levels

The layout model marks section headers but not how deep they sit, so by default every heading in a
PDF comes out at level 1. Docling can infer the levels from the PDF bookmarks, from outline
numbering and from the heading's font styling:

```python
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    HeadingHierarchyOptions,
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

pipeline_options = PdfPipelineOptions()
pipeline_options.heading_hierarchy_options = HeadingHierarchyOptions(enabled=True)
pipeline_options.generate_parsed_pages = True  # required by the font-style signal

doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)
```

See [PDF heading levels](./heading_levels.md) for the signals, their precedence and all options.

### Apple iWork options

Pages (`.pages`), Numbers (`.numbers`) and Keynote (`.key`) share their
options, since they share their container.

In a Pages document, headers, footers and footnotes go into the `furniture`
content layer and comments into `notes`. In a Keynote presentation, each slide
becomes a chapter group holding what is on it, and the presenter notes and
comments of that slide go into `notes` under it. In a Numbers spreadsheet, each
sheet becomes a page and a sheet group, and the sticky notes on it go into
`notes`. Either way those layers stay out of the reading order by default; to
include them in an export, pass the extra layers explicitly (this applies to any
`DoclingDocument`, not just these):

```python
from docling_core.types.doc import ContentLayer
from docling.document_converter import DocumentConverter

converter = DocumentConverter()

# Pages: headers, footers and footnotes are furniture, comments are notes.
report = converter.convert("report.pages").document
print(report.export_to_markdown(
    included_content_layers={
        ContentLayer.BODY,
        ContentLayer.FURNITURE,
        ContentLayer.NOTES,
    }
))

# Keynote: the presenter notes and comments of each slide are notes.
deck = converter.convert("deck.key").document
print(deck.export_to_markdown(
    included_content_layers={ContentLayer.BODY, ContentLayer.NOTES}
))

# Numbers: the sticky notes on each sheet are notes.
budget = converter.convert("budget.numbers").document
print(budget.export_to_markdown(
    included_content_layers={ContentLayer.BODY, ContentLayer.NOTES}
))
```

A chart on a Keynote slide or a Numbers sheet becomes a picture classified by
its kind, with the data it plots in the picture's `meta.tabular_chart` and its
title as the caption, which is the shape the PowerPoint backend gives a chart. Keynote keeps
no picture of a chart, so the picture itself is empty unless you opt into
`render_chart_images`. That rebuilds each chart from its data as an Office chart
and draws it with LibreOffice, so it needs a LibreOffice installation. The image
has the chart's kind, data and title but not its colours or fonts, and a mixed,
two-axis, bubble or interactive chart gets none:

```python
from docling.datamodel.backend_options import IWorkBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, IWorkKeynoteFormatOption

converter = DocumentConverter(
    format_options={
        InputFormat.IWORK_KEYNOTE: IWorkKeynoteFormatOption(
            backend_options=IWorkBackendOptions(render_chart_images=True)
        )
    }
)
deck = converter.convert("deck.key").document
for picture in deck.pictures:
    if picture.meta is not None and picture.meta.tabular_chart is not None:
        print(picture.caption_text(deck), picture.meta.tabular_chart.chart_data)
```

Charts are read from Keynote 6 and later; a chart in an iWork '09 presentation
is not read. `render_chart_images` draws Keynote charts only — a Numbers chart
carries its data and its classification, but no image.

`sheet_names` converts only the sheets it names, and `page_range` narrows the
selection further, since each sheet is a page:

```python
from docling.datamodel.backend_options import IWorkBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, IWorkNumbersFormatOption

doc_converter = DocumentConverter(
    format_options={
        InputFormat.IWORK_NUMBERS: IWorkNumbersFormatOption(
            backend_options=IWorkBackendOptions(sheet_names=["Summary", "Q1"])
        )
    }
)
```

The container is untrusted input, so size limits apply. They can be tuned with
`IWorkBackendOptions`, which all three formats take:

```python
from docling.datamodel.backend_options import IWorkBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import (
    DocumentConverter,
    IWorkKeynoteFormatOption,
    IWorkNumbersFormatOption,
    IWorkPagesFormatOption,
)

limits = IWorkBackendOptions(max_total_bytes=50 * 1024 * 1024)
doc_converter = DocumentConverter(
    format_options={
        InputFormat.IWORK_PAGES: IWorkPagesFormatOption(backend_options=limits),
        InputFormat.IWORK_NUMBERS: IWorkNumbersFormatOption(backend_options=limits),
        InputFormat.IWORK_KEYNOTE: IWorkKeynoteFormatOption(backend_options=limits),
    }
)
```

### Docling JSON input

A `DoclingDocument` JSON file can be converted again, e.g. to re-export it to
another format. Image references in that JSON which point at local files (bare
paths, relative paths or `file:` URIs, for pictures, tables and page images
alike) are ignored by default and a warning is logged: the images are dropped
from the loaded document, together with their size and resolution. Embedded
`data:` images and `http(s)` URLs are kept.

This also applies to a document saved with `ImageRefMode.REFERENCED`, whose
images are separate files. To load those images again from a JSON file you
trust, enable local fetching on the backend options:

```python
from docling.datamodel.backend_options import DeclarativeBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, DoclingJSONFormatOption

converter = DocumentConverter(
    format_options={
        InputFormat.JSON_DOCLING: DoclingJSONFormatOption(
            backend_options=DeclarativeBackendOptions(enable_local_fetch=True)
        )
    }
)
doc = converter.convert("saved_document.json").document
```

Relative image paths are resolved against the current working directory. The
`docling` CLI has no option for this and always ignores local image references
in JSON input; save the document with `ImageRefMode.EMBEDDED` if it has to go
through the CLI again with its images.

### Fetch HTML images from remote hosts

The HTML backend only downloads images referenced by a page when you opt in with
`fetch_images=True` and `enable_remote_fetch=True` (the CLI equivalent is
`--html-image-fetch remote`). Downloads connect only to public, globally
routable addresses: every address of a host is checked, every redirect is
checked again before it is followed (up to `max_redirects`), and downloads stop
at `max_remote_image_bytes`. When `render_page=True`, the browser requests
remote resources through the same download path, and navigating the page away
from the source document is refused.

`headers` adds HTTP headers, such as credentials, to these downloads. They are
sent only to the origin of the source document and dropped on redirects to
other origins. To send them to other hosts, such as a CDN, list the allowed
origins in `headers_allowed_origins` (this replaces the default, so include the
source origin too if it needs the headers). For a local file or a stream
without a remote `source_uri`, headers are only sent when
`headers_allowed_origins` is set.

```python
from docling.datamodel.backend_options import HTMLBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, HTMLFormatOption

html_options = HTMLBackendOptions(
    fetch_images=True,
    enable_remote_fetch=True,
    headers={"Authorization": "Bearer TOKEN"},
    headers_allowed_origins=["https://example.com", "https://cdn.example.com"],
)
converter = DocumentConverter(
    format_options={
        InputFormat.HTML: HTMLFormatOption(backend_options=html_options)
    }
)
result = converter.convert("https://example.com/page.html")
```

On the CLI, pass `--html-image-headers` with a JSON object and repeat
`--html-image-headers-origin` for each allowed origin.

When a proxy is configured through the `HTTP_PROXY` / `HTTPS_PROXY`
environment variables, downloads go through the proxy and Docling does not check
the destination addresses; the proxy is then responsible for restricting which
destinations it connects to.

## Impose limits on the document size

You can limit the file size and number of pages which should be allowed to process per document:

```python
from pathlib import Path
from docling.document_converter import DocumentConverter

source = "https://arxiv.org/pdf/2408.09869"
converter = DocumentConverter()
result = converter.convert(source, max_num_pages=100, max_file_size=20971520)
```

## Convert from binary PDF streams

You can convert PDFs from a binary stream instead of from the filesystem as follows:

```python
from io import BytesIO
from docling.datamodel.base_models import DocumentStream
from docling.document_converter import DocumentConverter

buf = BytesIO(your_binary_stream)
source = DocumentStream(name="my_doc.pdf", stream=buf)
converter = DocumentConverter()
result = converter.convert(source)
```

## Track conversion progress

To see progress without writing any code, turn on the built-in printer:

```python
from docling.document_converter import DocumentConverter

converter = DocumentConverter(show_progress=True)
result = converter.convert("report.pdf")
```

It prints one line per document to stderr, with the pages done:

```text
[1] Converting report.pdf
  pages 9/9
Finished report.pdf: success
```

When enrichment is enabled (picture classification or description, chart
extraction, code and formulas), each step adds a line such as
`DocumentPictureClassifier 3/3`.

On a terminal the `pages` and enrichment lines update in place; in a log file
only their final value is written. For a batch, give the printer the number
of documents to get `[3/12]` instead of `[3]`:

```python
from docling.utils.progress import ProgressPrinter

printer = ProgressPrinter(total_documents=len(sources))
for result in DocumentConverter(progress_callback=printer).convert_all(sources):
    ...
```

On the command line the same output is on by default when stderr is a
terminal. `--no-progress` or `--quiet` turns it off, and `--progress` turns it
on for pipes and log files.

### Your own progress callback

Pass a `progress_callback` to `DocumentConverter` to drive your own progress
bar or forward the events elsewhere. The callback receives one event object at
a time (see `docling.datamodel.progress`):

| Event | When |
| --- | --- |
| `DocumentStartedProgress` | An input document is picked up, including inputs that are then skipped. |
| `PhaseStartedProgress` | The document enters the `initialize` (pipeline set-up, slow only while the models of a new pipeline load), `build`, `assemble` or `enrich` phase. Every format reports them. |
| `PageCompletedProgress` | A page went through all page-level models, table structure included. PDF and image pipelines only. |
| `EnrichmentProgress` | Item counts of one enrichment step (picture classification, picture description, chart extraction, code and formula), reported per batch. |
| `DocumentCompletedProgress` | The document finished, with its `ConversionStatus`. Always the last event of a document, also when it failed. |

Every event carries `document_index`, the position of the document in the
`convert_all` call, and `document_name`.

```python
from tqdm import tqdm

from docling.datamodel.progress import (
    ConversionProgressEvent,
    DocumentCompletedProgress,
    EnrichmentProgress,
    PageCompletedProgress,
)
from docling.document_converter import DocumentConverter

bars: dict[str, tqdm] = {}


def show_progress(event: ConversionProgressEvent) -> None:
    if isinstance(event, PageCompletedProgress):
        key, done, total = "pages", event.completed_pages, event.total_pages
    elif isinstance(event, EnrichmentProgress):
        key, done, total = event.step, event.completed_items, event.total_items
    elif isinstance(event, DocumentCompletedProgress):
        for bar in bars.values():
            bar.close()
        bars.clear()
        return
    else:
        return
    if key not in bars:
        bars[key] = tqdm(total=total, desc=key)
    bars[key].update(done - bars[key].n)


converter = DocumentConverter(progress_callback=show_progress)
result = converter.convert("https://arxiv.org/pdf/2408.09869")
```

Things to know:

- Pages can finish out of order. Use `completed_pages`, not `page_no`, for
  the progress value. `total_pages` counts the pages selected by `page_range`.
- Enrichment runs after the last page, on the whole document, so the page bar
  reaching 100% does not mean the document is done. Watch the
  `EnrichmentProgress` events, or wait for `DocumentCompletedProgress`.
- Failed pages, and pages cut by `document_timeout`, are reported with
  `success=False`, so `completed_pages` always reaches `total_pages`.
- Exceptions raised by the callback are logged and ignored.
- For a single document all events come from the thread that called
  `convert`. With `settings.perf.doc_batch_concurrency > 1` several documents
  report at once from worker threads, so the callback must be thread-safe.

## Limit resource usage

You can limit the CPU threads used by Docling by setting the environment variable `OMP_NUM_THREADS` accordingly. The default setting is using 4 CPU threads.
