Docling can be extended with third-party plugins that add model implementations to several pipeline stages.

Plugins are discovered through Python package [entry points](https://packaging.python.org/en/latest/specifications/entry-points/) in the `docling` group. Entry-point names must be unique across the environment.

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

A plugin module can define one or more of these hooks:

| Hook | Model contract | Options contract |
| --- | --- | --- |
| `ocr_engines` | `BaseOcrModel` | `OcrOptions` |
| `layout_engines` | `BaseLayoutModel` | `BaseLayoutOptions` |
| `table_structure_engines` | `BaseTableStructureModel` | `BaseTableStructureOptions` |
| `picture_description` | `PictureDescriptionBaseModel` | `PictureDescriptionBaseOptions` |

Each hook must be callable and return a mapping whose matching key contains a list of model classes. Every model class must implement the model contract for that hook and return its options class from `get_options_type()`. Option `kind` values must be unique within a factory; Docling rejects conflicting registrations instead of selecting one based on package discovery order.

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

where `YourOcrModel` must implement [`BaseOcrModel`](https://github.com/docling-project/docling/blob/main/docling/models/base_ocr_model.py) and provide an options class derived from [`OcrOptions`](https://github.com/docling-project/docling/blob/main/docling/datamodel/pipeline_options.py).

If you look for an example, the [default Docling plugins](https://github.com/docling-project/docling/blob/main/docling/models/plugins/defaults.py) is a good starting point.

## Third-party plugins

Plugins outside the main `docling` package must be enabled explicitly through the `allow_external_plugins` option. When external plugins are disabled, Docling filters their entry points before importing any third-party plugin code.

```py
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

pipeline_options = PdfPipelineOptions()
pipeline_options.allow_external_plugins = True  # Enable external plugins.
pipeline_options.ocr_options = YourOptions  # <-- your options here

doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options
        )
    }
)
```

### Using the `docling` CLI

The CLI can list external models from all four supported plugin factories. External plugins still have to be enabled before selecting one for conversion.

```sh
# Show the external plugins
docling --show-external-plugins

# Run docling with the new plugin
docling --allow-external-plugins --ocr-engine=NAME
```
