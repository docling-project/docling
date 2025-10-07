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

where `YourOcrModel` must implement the [`BaseOcrModel`](https://github.com/docling-project/docling/blob/main/docling/models/base_ocr_model.py#L23) and provide an options class derived from [`OcrOptions`](https://github.com/docling-project/docling/blob/main/docling/datamodel/pipeline_options.py#L105).

If you look for an example, the [default Docling plugins](https://github.com/docling-project/docling/blob/main/docling/models/plugins/defaults.py) is a good starting point.

## Third-party plugins

When the plugin is not provided by the main `docling` package but by a third-party package this have to be enabled explicitly via the `allow_external_plugins` option.

```py
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

pipeline_options = PdfPipelineOptions()
pipeline_options.allow_external_plugins = True  # <-- enabled the external plugins
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

Similarly, when using the `docling` users have to enable external plugins before selecting the new one.

```sh
# Show the external plugins
docling --show-external-plugins

# Run docling with the new plugin
docling --allow-external-plugins --ocr-engine=NAME
```

---

## Example plugin: API-backed picture description with token usage

A complete, installable plugin is provided in this repository under:

- `docs/examples/third_party_plugins/` — installable example package
- Install in editable mode: `pip install -e docs/examples/third_party_plugins`

File structure:
```
docs/examples/third_party_plugins/
  pyproject.toml
  main.py
  api_usage/
    api_usage_plugin.py
    datamodel/
      pipeline_options/
        picture_description_api_model_with_usage.py
      utils/
        api_image_request_with_usage.py
    models/
      picture_description_api_model.py
```

### Packaging: entry point and metadata

- `docs/examples/third_party_plugins/pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "docling-plugin-api-usage-example"
version = "0.1.0"
description = "Example Docling third-party plugin: picture description via OpenAI-compatible API with token usage."
readme = "README.md"
requires-python = ">=3.10"
authors = [
  { name = "Docling Examples", email = "examples@docling.dev" }
]
dependencies = [
  "docling>=0.1.0",  # pin to a compatible version for your environment
  "pydantic>=2.0.0",
  "Pillow>=10.0.0",
  "requests>=2.31.0",
]

[project.entry-points."docling"]
api_usage_plugin = "api_usage.api_usage_plugin"

[tool.setuptools]
packages = ["api_usage"]
```

### Plugin registration

- `docs/examples/third_party_plugins/api_usage/api_usage_plugin.py`

```py
from api_usage.models.picture_description_api_model import PictureDescriptionApiModelWithUsage


def picture_description():
    return {
        "picture_description": [
            PictureDescriptionApiModelWithUsage
        ]
    }
```

This exposes the `picture_description` factory attribute returning the model class to register.

### Options: configuring the model

- `docs/examples/third_party_plugins/api_usage/datamodel/pipeline_options/picture_description_api_model_with_usage.py`

```py
from typing import Any, ClassVar, Dict, List, Literal, Optional, Union
from pydantic import (
    AnyUrl,
    BaseModel,
    ConfigDict,
    Field,
)

from docling.datamodel.pipeline_options import PictureDescriptionBaseOptions


class PictureDescriptionApiOptionsWithUsage(PictureDescriptionBaseOptions):
    """DescriptionAnnotation."""

    kind: ClassVar[Literal["api_token"]] = "api_token"

    url: AnyUrl = AnyUrl("http://localhost:8000/v1/chat/completions")
    headers: Dict[str, str] = {}
    params: Dict[str, Any] = {}
    timeout: float = 20
    concurrency: int = 1

    prompt: str = "Describe this image in a few sentences."
    provenance: str = ""
    # Key inside the response 'usage' (or similar) which will be used to extract
    # the token/response text. Example: 'content' or 'text'. If None, no
    # token extraction will be performed by default.
    token_extract_key: Optional[str] = Field(
        None,
        description=(
            "Key in the response usage dict whose value contains the token/"
            "response to extract. For example 'content' or 'text'."
        ),
    )
```

Key points:
- The options class provides a unique `kind` string that identifies the model variant.
- It carries endpoint URL, headers, params, and concurrency, plus a `token_extract_key` for usage extraction.

### Utility: API request with usage extraction

- `docs/examples/third_party_plugins/api_usage/datamodel/utils/api_image_request_with_usage.py`

```py
import base64
import json
import logging
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import requests
from PIL import Image
from pydantic import AnyUrl

from docling.datamodel.base_models import OpenAiApiResponse
from docling.models.utils.generation_utils import GenerationStopper

_log = logging.getLogger(__name__)


def api_image_request(
    image: Image.Image,
    prompt: str,
    url: AnyUrl,
    timeout: float = 20,
    headers: Optional[Dict[str, str]] = None,
    token_extract_key: Optional[str] = None,
    **params,
) -> Tuple[str, Optional[dict]]:
    """Send an image+prompt to an OpenAI-compatible API and return (text, usage).

    If no usage data is available, the second tuple element will be None.
    """
    img_io = BytesIO()
    image.save(img_io, "PNG")
    image_base64 = base64.b64encode(img_io.getvalue()).decode("utf-8")
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    payload = {"messages": messages, **params}
    headers = headers or {}

    r = requests.post(str(url), headers=headers, json=payload, timeout=timeout)
    if not r.ok:
        _log.error(f"Error calling the API. Response was {r.text}")
    r.raise_for_status()

    # Try to parse JSON body
    try:
        resp_json = r.json()
    except Exception:
        api_resp = OpenAiApiResponse.model_validate_json(r.text)
        generated_text = api_resp.choices[0].message.content.strip()
        return generated_text, None

    usage = None
    if isinstance(resp_json, dict):
        usage = resp_json.get("usage")

    # Extract generated text using common OpenAI shapes
    generated_text = ""
    try:
        generated_text = resp_json["choices"][0]["message"]["content"].strip()
    except Exception:
        try:
            generated_text = resp_json["choices"][0].get("text", "")
            if isinstance(generated_text, str):
                generated_text = generated_text.strip()
        except Exception:
            try:
                api_resp = OpenAiApiResponse.model_validate_json(r.text)
                generated_text = api_resp.choices[0].message.content.strip()
            except Exception:
                generated_text = ""

    # If an explicit token_extract_key is provided and found in usage, use it
    if token_extract_key and isinstance(usage, dict) and token_extract_key in usage:
        extracted = usage.get(token_extract_key)
        generated_text = str(extracted).strip() if extracted is not None else generated_text

    return generated_text, usage
```

This helper handles typical OpenAI‑compatible shapes and optional extraction from a response `usage` dict.

### Model: picture description with concurrency and telemetry

- `docs/examples/third_party_plugins/api_usage/models/picture_description_api_model.py`

```py
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Literal, Optional, Type, Union

from docling_core.types.doc import DoclingDocument, NodeItem, PictureItem
from docling_core.types.doc.document import \
    BaseAnnotation  # TODO: move import to docling_core.types.doc
from PIL import Image

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.pipeline_options import PictureDescriptionBaseOptions
from docling.exceptions import OperationNotAllowed
from docling.models.base_model import ItemAndImageEnrichmentElement
from docling.models.picture_description_base_model import \
    PictureDescriptionBaseModel
from api_usage.datamodel.utils.api_image_request_with_usage import api_image_request
from api_usage.datamodel.pipeline_options.picture_description_api_model_with_usage import \
    PictureDescriptionApiOptionsWithUsage


class DescriptionAnnotationWithUsage(BaseAnnotation):
    """DescriptionAnnotation."""

    kind: Literal["description"] = "description"
    text: str
    provenance: str
    token_usage: Optional[dict] = None


class PictureDescriptionApiModelWithUsage(PictureDescriptionBaseModel):
    # elements_batch_size = 4

    @classmethod
    def get_options_type(cls) -> Type[PictureDescriptionBaseOptions]:
        return PictureDescriptionApiOptionsWithUsage

    def __init__(
        self,
        enabled: bool,
        enable_remote_services: bool,
        artifacts_path: Optional[Union[Path, str]],
        options: PictureDescriptionApiOptionsWithUsage,
        accelerator_options: AcceleratorOptions,
    ):
        super().__init__(
            enabled=enabled,
            enable_remote_services=enable_remote_services,
            artifacts_path=artifacts_path,
            options=options,
            accelerator_options=accelerator_options,
        )
        self.options: PictureDescriptionApiOptionsWithUsage
        self.concurrency = self.options.concurrency

        if self.enabled:
            if not enable_remote_services:
                raise OperationNotAllowed(
                    "Connections to remote services is only allowed when set explicitly. "
                    "pipeline_options.enable_remote_services=True."
                )

    def _annotate_images(self, images: Iterable[Image.Image]) -> Iterable[str]:
        # Note: technically we could make a batch request here,
        # but not all APIs will allow for it. For example, vllm won't allow more than 1.
        def _api_request(image):
            # Pass token_extract_key so api_image_request can return token usage
            return api_image_request(
                image=image,
                prompt=self.options.prompt,
                url=self.options.url,
                timeout=self.options.timeout,
                headers=self.options.headers,
                token_extract_key=getattr(self.options, "token_extract_key", None),
                **self.options.params,
            )

        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            yield from executor.map(_api_request, images)


    def __call__(
        self,
        doc: DoclingDocument,
        element_batch: Iterable[ItemAndImageEnrichmentElement],
    ) -> Iterable[NodeItem]:
        if not self.enabled:
            for element in element_batch:
                yield element.item
            return

        images: List[Image.Image] = []
        elements: List[PictureItem] = []
        for el in element_batch:
            assert isinstance(el.item, PictureItem)
            describe_image = True
            # Don't describe the image if it's smaller than the threshold
            if len(el.item.prov) > 0:
                prov = el.item.prov[0]  # PictureItems have at most a single provenance
                page = doc.pages.get(prov.page_no)
                if page is not None:
                    page_area = page.size.width * page.size.height
                    if page_area > 0:
                        area_fraction = prov.bbox.area() / page_area
                        if area_fraction < self.options.picture_area_threshold:
                            describe_image = False
            if describe_image:
                elements.append(el.item)
                images.append(el.image)

        outputs = self._annotate_images(images)

        for item, output in zip(elements, outputs):
            # api_image_request now may return (text, usage) or plain text;
            # normalize to tuple
            if isinstance(output, tuple):
                text, usage = output
            else:
                text, usage = output, None

            item.annotations.append(
                DescriptionAnnotationWithUsage(
                    text=text, provenance=self.provenance, token_usage=usage
                )
            )
            yield item
```

Highlights:
- Enforces `enable_remote_services=True` when the model is enabled.
- Filters small pictures via `picture_area_threshold`.
- Uses a thread pool to annotate images concurrently.
- Stores per‑annotation `token_usage` data when available.

### Running the example

- `docs/examples/third_party_plugins/main.py`

```py
"""
Example: Docling pipeline using the third-party picture description plugin
located in docs/examples/third_party_plugins/api_usage.

Prerequisites:
- Ensure you have Docling installed in the same Python environment
- Install this example plugin in editable mode:
    pip install -e docs/examples/third_party_plugins
- Optionally set environment variables for the API backend:
    OPENAI_COMPATIBLE_API_URL, OPENAI_COMPATIBLE_API_KEY, OPENAI_COMPATIBLE_API_HEADER_NAME
    (or provide url/headers directly below)

Run:
    python docs/examples/third_party_plugins/main.py
"""

import os
from typing import Dict

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat

# Import the options class from the installed example plugin package
from api_usage.datamodel.pipeline_options.picture_description_api_model_with_usage import (
    PictureDescriptionApiOptionsWithUsage,
)


def main():
    # Resolve a simple OpenAI-compatible backend from environment variables
    url = os.getenv("OPENAI_COMPATIBLE_API_URL", "http://localhost:8000/v1/chat/completions")
    key = os.getenv("OPENAI_COMPATIBLE_API_KEY")
    header_name = os.getenv("OPENAI_COMPATIBLE_API_HEADER_NAME", "api-key")
    headers: Dict[str, str] = {header_name: key} if key else {}

    # Configure pipeline options
    pipeline_options = PdfPipelineOptions()
    pipeline_options.allow_external_plugins = True

    # Enable image processing for paginated PDF processing
    pipeline_options.generate_picture_images = True
    pipeline_options.images_scale = 2  # higher resolution thumbnails
    pipeline_options.do_picture_description = True

    # Enable remote services (required for external API calls)
    pipeline_options.enable_remote_services = True

    # Configure picture description via the example plugin options
    pipeline_options.picture_description_options = PictureDescriptionApiOptionsWithUsage(
        url=url,
        headers=headers,
        params={"model": "gpt-4o-mini", "temperature": 0},
        prompt="Describe the image clearly and concisely in a few sentences.",
        timeout=45.0,
        concurrency=2,
        # If your server returns token usage in a dict under 'usage', you can
        # extract a specific field and make it the generated text:
        # token_extract_key="content",
    )

    # Create converter with the configured options
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    # Convert the document (local path or URL)
    source = os.getenv("SOURCE_DOCUMENT", "https://arxiv.org/pdf/2408.09869")
    print(f"\nConverting source: {source}\n")

    result = converter.convert(source)
    doc = result.document

    # Print the markdown result
    print(doc.export_to_markdown())

    # Print token usage for each picture annotation (if provided by backend)
    for idx, pic in enumerate(doc.pictures):
        print(f"\nPicture #{idx}:")
        if not getattr(pic, "annotations", None):
            print("  (no annotations)")
            continue

        for ann_idx, ann in enumerate(pic.annotations):
            token_usage = getattr(ann, "token_usage", None)
            ann_text = getattr(ann, "text", None)
            print(f"  Annotation {ann_idx}: text={repr(ann_text)} token_usage={repr(token_usage)}")


if __name__ == "__main__":
    main()
```

Steps to run:

1. Install Docling in your environment (e.g., `pip install docling`).
2. From the repo root, install the example plugin:
   ```sh
   pip install -e docs/examples/third_party_plugins
   ```
3. Ensure your API is reachable (e.g., local vLLM, OpenAI‑compatible gateway).
4. Run:
   ```sh
   python docs/examples/third_party_plugins/main.py
   ```

You should see console output for conversion, plus per‑picture annotation text and any token usage returned by your backend.

---


## Best practices for plugin authors

- Unique kind: Define a unique `kind` on your options class; factories use this to select the implementation.
- Minimal options: Keep options typed and minimal; for complex objects (stores/callables), use `pydantic.ConfigDict(arbitrary_types_allowed=True)` if needed.

---

## Troubleshooting

- Plugin not visible?
  - Confirm the package is importable (`pip install -e .`), the entry point group is `docling`, and the module exposes a callable `ocr_engines()` or `picture_description()` returning the correct dict structure.
  - Programmatically: `PdfPipelineOptions.allow_external_plugins=True`.
  - CLI: pass `--allow-external-plugins`.

- HTTP errors or empty text:
  - Check `url`, `headers`, and your API server logs.
  - Verify payload shape and model name in `params` (`{"model": "...", ...}`) matches your backend.

- No `usage` data:
  - Your backend may not return token information; `token_usage` will be `None`.
  - If your backend returns a nested dict inside `usage`, set `token_extract_key` to extract and use that as generated text.


---

## Example code index (quick links)

- Packaging and entry point:
  - `docs/examples/third_party_plugins/pyproject.toml`
  - `docs/examples/third_party_plugins/api_usage/api_usage_plugin.py`

- Options and utilities:
  - `docs/examples/third_party_plugins/api_usage/datamodel/pipeline_options/picture_description_api_model_with_usage.py`
  - `docs/examples/third_party_plugins/api_usage/datamodel/utils/api_image_request_with_usage.py`

- Model:
  - `docs/examples/third_party_plugins/api_usage/models/picture_description_api_model.py`

- Runner:
  - `docs/examples/third_party_plugins/main.py`

If you need a more advanced backend resolver (e.g., auto‑switch between Azure OpenAI and OpenAI‑compatible endpoints), see the root `main.py` in this repository, which demonstrates environment‑driven configuration and printing per‑image usage. That implementation informed this example and can be adapted to your needs.
