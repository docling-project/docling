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
