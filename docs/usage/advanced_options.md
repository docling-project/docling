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

Pages (`.pages`), Numbers (`.numbers`) and Keynote (`.key`) share their options,
since they share their container. All three need the `format-iwork` extra.

#### What each one becomes

A **Pages** document converts to a flow of text, like a Word document: its
headers, footers and footnotes go into the `furniture` content layer and its
comments into `notes`.

A **Keynote** presentation converts to one chapter group per slide, holding what
is on that slide, with the slide's presenter notes and comments in `notes`
under it.

A **Numbers** spreadsheet converts to one page and one sheet group per sheet,
the way an Excel or OpenDocument spreadsheet does. The difference is that a
Numbers sheet is a canvas rather than one big grid: it holds separate tables,
charts and sticky notes side by side, each positioned in its own right. So each
table arrives as its own table item — captioned with the name Numbers gives it,
and carrying its own header rows and columns — and the things on a sheet come out
in the order they are laid out down the page rather than the order the file
happens to store them in.

Those `furniture` and `notes` layers stay out of the reading order by default.
To include them in an export, pass the extra layers explicitly (this applies to
any `DoclingDocument`, not just these):

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

# Numbers: the sticky notes on a sheet are notes.
budget = converter.convert("budget.numbers").document
print(budget.export_to_markdown(
    included_content_layers={ContentLayer.BODY, ContentLayer.NOTES}
))
```

#### Spreadsheet cells and charts

A cell is read as the value it holds, not as the text you happen to see:
numbers, dates, durations, checkboxes and the results Numbers last computed for
its formulas all come through. Numbers has stored numbers as decimals since
2017, so a cell you typed `0.1` into reads back as `0.1` rather than as the
binary fraction nearest to it.

A chart becomes a picture item carrying the data it plots — categories down the
first column, one column per series — which is the same shape the Excel backend
gives a chart. Numbers keeps no image of a chart, but it does keep its own copy
of the numbers behind it, so a chart still reads correctly even if the table it
was built from has since been deleted.

#### Choosing sheets

`sheet_names` converts only the sheets you name, and `page_range` narrows the
selection further, since each sheet is a page:
A chart on a Keynote slide becomes a picture classified by its kind, with the
data it plots in the picture's `meta.tabular_chart` and its title as the
caption, which is the shape the PowerPoint backend gives a chart. Keynote keeps
no picture of a chart, so the picture itself is empty unless you opt into
`render_chart_images`. That rebuilds each chart from its data as an Office chart
and draws it with LibreOffice, so it needs a LibreOffice installation. The image
has the chart's kind, data and title but not its colours or fonts, and a mixed,
two-axis, bubble or interactive chart gets none:

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

#### Size limits
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
is not read.

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

!!! note "Not yet extracted from a Numbers spreadsheet"

    Images and shapes are skipped, and so is a comment attached to one cell
    rather than to the sheet. A chart's *kind* — pie, bar, line — is not read
    yet, so every chart is labelled simply as a chart. The number format beside
    a cell is not applied either, so a currency or percentage cell reads as the
    plain number it holds, exactly as it does for Excel and OpenDocument. A cell
    driven by a pop-up menu gives the label it shows in an iWork '09 document,
    but only the menu position in a 2013-or-later one. Password-protected
    documents cannot be read.
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

## Limit resource usage

You can limit the CPU threads used by Docling by setting the environment variable `OMP_NUM_THREADS` accordingly. The default setting is using 4 CPU threads.
