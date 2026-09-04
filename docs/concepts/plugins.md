Docling allows to be extended with third-party plugins which extend the choice of options provided in several steps of the pipeline.

Plugins are loaded via the [pluggy](https://github.com/pytest-dev/pluggy/) system which allows third-party developers to register the new capabilities using the [setuptools entrypoint](https://setuptools.pypa.io/en/latest/userguide/entry_point.html#entry-points-for-plugins).

The actual entrypoint definition might vary, depending on the packaging system you are using. Here are a few examples:

=== "pyproject.toml"

    ```toml
    [project.entry-points."docling"]
    your_plugin_name = "your_package.module"
    ```

=== "poetry v1 pyproject.toml"

    ```toml
    [tool.poetry.plugins."docling"]
    your_plugin_name = "your_package.module"
    ```

=== "setup.cfg"

    ```ini
    [options.entry_points]
    docling =
        your_plugin_name = your_package.module
    ```

=== "setup.py"

    ```py
    from setuptools import setup

    setup(
        # ...,
        entry_points = {
            'docling': [
                'your_plugin_name = "your_package.module"'
            ]
        }
    )
    ```

- `your_plugin_name` is the name you choose for your plugin. This must be unique among the broader Docling ecosystem.
- `your_package.module` is the reference to the module in your package which is responsible for the plugin registration.

## Plugin factories

### OCR factory

The OCR factory allows to provide more OCR engines to the Docling users.

The content of `your_package.module` registers the OCR engines with a code similar to:

```py
# Factory registration
def ocr_engines():
    return {
        "ocr_engines": [
            YourOcrModel,
        ]
    }
```

where `YourOcrModel` must implement the [`BaseOcrModel`](https://github.com/docling-project/docling/blob/main/docling/models/base_ocr_model.py#L40) and provide an options class derived from [`OcrOptions`](https://github.com/docling-project/docling/blob/main/docling/datamodel/pipeline_options.py#L184).

#### OCR languages in an external engine

`OcrOptions.lang` is validated and canonicalized by the base class, so your engine receives
`OcrLanguage` objects rather than raw strings. Each is one of two things:

- a **passthrough**, flagged by `is_passthrough()`: a code of your own engine, carried in `.native`
  and never parsed, which is what any value the user wrote without the `iso:` prefix becomes
- a language: a `(bcp47_language, bcp47_script)` pair such as `en-Latn` or `zh-Hans`, from a value
  the user did write behind `iso:`, never `en` or `chinese`

See [OCR engines](OCR.md#language-selection) for the user-facing contract.

With no further work, `BaseOcrModel.map_ocr_language` refuses a passthrough and hands your engine
the **primary subtag** (`en`, `zh`) of a tag, which is what most ISO-639-based engines want. Two
things still need attention: state your options default in your own codes, and note that the
primary subtag alone loses the Simplified/Traditional distinction.

To participate fully, override three members:

```py
from docling.exceptions import OcrLanguageNotSupportedError
from docling.models.base_ocr_model import BaseOcrModel
from docling.utils.ocr_language import OcrLanguage, OcrLanguageSupport


class YourOcrModel(BaseOcrModel):
    # True if the engine can run several languages at once; when it is
    # False, every language after the first is dropped with a warning.
    multiple_languages = False

    def supported_ocr_languages(self) -> OcrLanguageSupport:
        # What this instance can serve, for error messages. `bcp47` holds the
        # shortest spelling: `de`, not `de-Latn`, because canonicalization puts
        # the script back; `zh-Hant` and `sr-Latn` do have to carry theirs, and
        # the `iso:` prefix is added when the error message renders them.
        # `native` holds your own codes for models no tag can name.
        return OcrLanguageSupport(bcp47=["en", "de"], native=["my_script_model"])

    def map_ocr_language(self, language: OcrLanguage) -> str | list[str]:
        # Map one request onto your engine's native code(s). Handle the
        # passthrough first: `language.bcp47()` is empty for one.
        if language.is_passthrough():
            # `.native` reached you unparsed, so check it against your own
            # vocabulary -- that check is the only one there is.
            if language.native in MY_ENGINE_CODES:
                return language.native
        elif language.short_tag() in self.supported_ocr_languages().bcp47:
            return language.bcp47_language
        raise OcrLanguageNotSupportedError(
            type(self).__name__,
            language.tag(),  # the canonical spelling, `iso:` prefix and all
            supported=self.supported_ocr_languages(),
        )
```

`BaseOcrModel.resolve_ocr_languages()` then drops every language after the first on a
single-language engine, with a warning naming what it kept. It does not touch the error path: an
`OcrLanguageNotSupportedError` your `map_ocr_language` raises propagates unchanged, which is why
the sample above attaches `supported=` itself. Call `resolve_ocr_languages()` once from
`__init__`, inside your `if self.enabled:` block, and use the result in place of `options.lang`.

Build `supported_ocr_languages()` by walking your engine's own vocabulary and sorting each code:

1. Skip it if it is not a recognizer at all, or if it is redundant with a tag you already offer.
2. Canonicalize it, through your reverse deviation table first if you have one.
3. If that succeeds **and the result maps back to this same code**, put the canonical tag in
   `bcp47`, in its `short_tag()` spelling.
4. Otherwise put the code itself in `native`, which is already the spelling a user writes.

Step 4 is the one worth getting right. Dropping a code your engine really serves hides it: a user
who hits a coverage error is never told the model exists. `short_tag()` in step 3 drops the script
whenever canonicalization would infer it anyway, so the list reads `en, de, zh` rather than
`en-Latn, de-Latn, zh-Hans`, and keeps it only where the bare language would name a different
recognizer -- `zh-Hant`, `sr-Latn`, `de-Latf`.

### Layout engine factory

The layout engine factory allows to provide more layout engines to the Docling users.

The content of `your_package.module` registers the layout engines with a code similar to:

```py
# Factory registration
def layout_engines():
    return {
        "layout_engines": [
            YourLayoutModel,
        ]
    }
```

where `YourLayoutModel` must implement the [`BaseLayoutModel`](https://github.com/docling-project/docling/blob/main/docling/models/base_layout_model.py#L35) and provide an options class derived from [`BaseLayoutOptions`](https://github.com/docling-project/docling/blob/main/docling/datamodel/pipeline_options.py#L1454).

### Table structure engine factory

The table structure engine factory allows to provide more table structure recognition engines to the Docling users.

The content of `your_package.module` registers the table structure engines with a code similar to:

```py
# Factory registration
def table_structure_engines():
    return {
        "table_structure_engines": [
            YourTableStructureModel,
        ]
    }
```

where `YourTableStructureModel` must implement the [`BaseTableStructureModel`](https://github.com/docling-project/docling/blob/main/docling/models/base_table_model.py#L13) and provide an options class derived from [`BaseTableStructureOptions`](https://github.com/docling-project/docling/blob/main/docling/datamodel/pipeline_options.py#L130).

If you look for an example, the [default Docling plugins](https://github.com/docling-project/docling/blob/main/docling/models/plugins/defaults.py) is a good starting point.

## Third-party plugins

When the plugin is not provided by the main `docling` package but by a third-party package this have to be enabled explicitly via the `allow_external_plugins` option.

```py
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

pipeline_options = PdfPipelineOptions()
pipeline_options.allow_external_plugins = True  # <-- enable external plugins
pipeline_options.ocr_options = YourOptions  # <-- your OCR options here
pipeline_options.layout_options = YourLayoutOptions  # <-- your layout options here
pipeline_options.table_structure_options = YourTableStructureOptions  # <-- your table structure options here

doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options
        )
    }
)
```

### Using the `docling` CLI

Similarly, when using the `docling` CLI, users have to enable external plugins before selecting the new one.

```sh
# Show the external plugins
docling --show-external-plugins

# Run docling with a custom OCR engine
docling --allow-external-plugins --ocr-engine=NAME

# Run docling with a custom layout engine
docling --allow-external-plugins --layout-engine=NAME

# Run docling with a custom table structure engine
docling --allow-external-plugins --table-structure-engine=NAME
```
