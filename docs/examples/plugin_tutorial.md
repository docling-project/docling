[![Run locally](https://img.shields.io/badge/Run-Local-blue)](#running-the-example)
# Third‑party Plugin Tutorial — API‑backed Picture Description with Usage


## At a glance

| Step | Tech | Execution |
| --- | --- | --- |
| Packaging | Setuptools (entry points) | 🖥️ Local |
| Registration | pluggy `docling` group | 🖥️ Local |
| Options | Docling options (inherits core) | 🖥️ Local |
| Utility | requests + PIL (OpenAI‑compatible payload) | 🖥️ Local + 🌐 Remote |
| Model | Docling model class with concurrency | 🖥️ Local |
| Gen AI | OpenAI‑compatible API (e.g. vLLM, gateways) | 🌐 Remote |
| Runner | Docling `DocumentConverter` | 🖥️ Local |

## Overview

This tutorial shows how to build and use a third‑party Docling plugin that describes pictures via an OpenAI‑compatible API and captures usage telemetry per annotation. You will:

- Package a plugin with a `docling` entry point
- Register a picture description model
- Reuse Docling’s core options by subclassing
- Call a remote API using a utility function
- Implement a model that annotates images concurrently
- Run a sample pipeline and print per‑picture usage

The example plugin is included in this repo at [`docs/examples/third_party_plugins`](https://github.com/docling-project/docling/blob/main/docs/examples/third_party_plugins). You can install it in editable mode and run the sample driver.

## Setting up your environment

Install Docling and the example plugin (plus optional `.env` support):

```sh
# From the repository root
pip install "docling>=0.1.0"
pip install -e docs/examples/third_party_plugins
pip install python-dotenv  # optional, used by main.py to load .env
```

Environment variables (optional, used by the runner):

- OPENAI_COMPATIBLE_API_URL (e.g. http://localhost:8000/v1/chat/completions)
- OPENAI_COMPATIBLE_API_KEY
- OPENAI_COMPATIBLE_API_HEADER_NAME (default: api-key)

---

## File structure

```
your-project-root/
  pyproject.toml
  main.py
  api_usage/
    api_usage_plugin.py
    datamodel/
      pipeline_options/
        picture_description_api_options_with_usage.py
      utils/
        api_image_request_with_usage.py
    models/
      picture_description_api_model.py
```

---

## Step-by-step guide

### Part 1: Packaging - entry point and metadata

- [`docs/examples/third_party_plugins/pyproject.toml`](https://github.com/docling-project/docling/blob/main/docs/examples/third_party_plugins/pyproject.toml)

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "docling-plugin-api-usage-example"
version = "0.1.0"
description = "Example Docling third-party plugin: picture description via OpenAI-compatible API with usage telemetry."
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
  "python-dotenv>=1.0.0",  # used by main.py to load .env (optional)
]

[project.entry-points."docling"]
api_usage_plugin = "api_usage.api_usage_plugin"

[tool.setuptools]
packages = ["api_usage"]
```

This exposes a `docling` entry point group with a callable in `api_usage.api_usage_plugin`.

---

### Part 2: Registration - picture description factory

- [`docs/examples/third_party_plugins/api_usage/api_usage_plugin.py`](https://github.com/docling-project/docling/blob/main/docs/examples/third_party_plugins/api_usage/api_usage_plugin.py)

```py
from api_usage.models.picture_description_api_model import (
    PictureDescriptionApiModelWithUsage,
)


def picture_description():
    return {"picture_description": [PictureDescriptionApiModelWithUsage]}
```

The function name `picture_description` matches the factory Docling looks for; it returns the model class to register.

---

### Part 3: Options - reuse core options, add a unique kind

- [`docs/examples/third_party_plugins/api_usage/datamodel/pipeline_options/picture_description_api_options_with_usage.py`](https://github.com/docling-project/docling/blob/main/docs/examples/third_party_plugins/api_usage/datamodel/pipeline_options/picture_description_api_options_with_usage.py)

```py
from typing import Any, ClassVar, Dict, List, Literal, Optional, Union

from pydantic import (
    AnyUrl,
    Field,
)

from docling.datamodel.pipeline_options import PictureDescriptionApiOptions


class PictureDescriptionApiOptionsWithUsage(PictureDescriptionApiOptions):
    """DescriptionAnnotation."""

    kind: ClassVar[Literal["api_usage"]] = "api_usage"
```

Key points:

- Provide a unique `kind` (here `api_usage`) to identify the variant.
- Inherit runtime fields (url, headers, params, timeout, concurrency, prompts, thresholds) from `PictureDescriptionApiOptions`.

---

### Part 4: Utility - API request with usage extraction

- [`docs/examples/third_party_plugins/api_usage/datamodel/utils/api_image_request_with_usage.py`](https://github.com/docling-project/docling/blob/main/docs/examples/third_party_plugins/api_usage/datamodel/utils/api_image_request_with_usage.py)

```py
import base64
import json
import logging
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import requests
from PIL import Image
from pydantic import AnyUrl

from docling.datamodel.base_models import OpenAiApiResponse, OpenAiResponseUsage
from docling.models.utils.generation_utils import GenerationStopper

_log = logging.getLogger(__name__)


def api_image_request_with_usage(
    image: Image.Image,
    prompt: str,
    url: AnyUrl,
    timeout: float = 20,
    headers: Optional[Dict[str, str]] = None,
    **params,
) -> Tuple[str, Optional[OpenAiResponseUsage]]:
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
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]

    payload = {
        "messages": messages,
        **params,
    }

    headers = headers or {}

    r = requests.post(
        str(url),
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    if not r.ok:
        _log.error(f"Error calling the API. Response was {r.text}")
    r.raise_for_status()

    api_resp = OpenAiApiResponse.model_validate_json(r.text)
    generated_text = api_resp.choices[0].message.content.strip()

    usage = api_resp.usage if hasattr(api_resp, "usage") else None

    return generated_text, usage
```

Parses typical OpenAI‑compatible responses via `OpenAiApiResponse` and returns typed `OpenAiResponseUsage` when available.

---

### Part 5: Model - concurrency and usage telemetry

- [`docs/examples/third_party_plugins/api_usage/models/picture_description_api_model.py`](https://github.com/docling-project/docling/blob/main/docs/examples/third_party_plugins/api_usage/models/picture_description_api_model.py)

```py
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Type, Union

from api_usage.datamodel.pipeline_options.picture_description_api_options_with_usage import (
    PictureDescriptionApiOptionsWithUsage,
)
from api_usage.datamodel.utils.api_image_request_with_usage import (
    api_image_request_with_usage,
)
from docling_core.types.doc import DoclingDocument, NodeItem, PictureItem
from docling_core.types.doc.document import (
    DescriptionAnnotation,
)  # TODO: move import to docling_core.types.doc
from PIL import Image

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import OpenAiResponseUsage
from docling.datamodel.pipeline_options import PictureDescriptionBaseOptions
from docling.exceptions import OperationNotAllowed
from docling.models.base_model import ItemAndImageEnrichmentElement
from docling.models.picture_description_api_model import PictureDescriptionApiModel


class DescriptionAnnotationWithUsage(DescriptionAnnotation):
    """DescriptionAnnotation."""

    usage: Optional[OpenAiResponseUsage] = None


class PictureDescriptionApiModelWithUsage(PictureDescriptionApiModel):
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
            return api_image_request_with_usage(
                image=image,
                prompt=self.options.prompt,
                url=self.options.url,
                timeout=self.options.timeout,
                headers=self.options.headers,
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
                    text=text, provenance=self.provenance, usage=usage
                )
            )
            yield item
```

Highlights:

- Stores per‑annotation `usage` (`OpenAiResponseUsage`) when provided by your backend

---

### Part 6: Runner - run the pipeline

- [`docs/examples/third_party_plugins/main.py`](https://github.com/docling-project/docling/blob/main/docs/examples/third_party_plugins/main.py)

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

from api_usage.datamodel.pipeline_options.picture_description_api_options_with_usage import (
    PictureDescriptionApiOptionsWithUsage,
)
from dotenv import load_dotenv

# Import the options class from the installed example plugin package
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

load_dotenv()


def main():
    # Resolve a simple OpenAI-compatible backend from environment variables
    url = os.getenv(
        "OPENAI_COMPATIBLE_API_URL", "http://localhost:8000/v1/chat/completions"
    )
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
    pipeline_options.picture_description_options = (
        PictureDescriptionApiOptionsWithUsage(
            url=url,
            headers=headers,
            params={"model": "gpt-5-mini", "temperature": 1},
            prompt="Describe the image clearly and concisely in a few sentences.",
            timeout=45.0,
            concurrency=2,
        )
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

    # Print usage for each picture annotation (if provided by backend)
    for idx, pic in enumerate(doc.pictures):
        print(f"\nPicture #{idx}:")
        if not getattr(pic, "annotations", None):
            print("  (no annotations)")
            continue

        for ann_idx, ann in enumerate(pic.annotations):
            usage = getattr(ann, "usage", None)
            ann_text = getattr(ann, "text", None)
            print(f"  Annotation {ann_idx}: text={ann_text!r} usage={usage!r}")


if __name__ == "__main__":
    main()
```

---

## Running the example

1) Ensure your API is reachable (e.g., local vLLM, OpenAI‑compatible gateway).  
2) Install the example plugin (editable mode):

```sh
pip install -e docs/examples/third_party_plugins
```

3) Run the driver:

```sh
python docs/examples/third_party_plugins/main.py
```

You should see console output for conversion, plus per‑picture annotation text and any usage returned by your backend.

---

## Troubleshooting

- Plugin not visible?
  - Ensure the package is importable (`pip install -e ...`), the entry point group is `docling`, and the module exposes a callable `picture_description()` returning the correct dict structure.
  - Programmatically: set `PdfPipelineOptions.allow_external_plugins=True`.
  - CLI: pass `--allow-external-plugins`.

- HTTP errors or empty text:
  - Check `url`, `headers`, and your API server logs.
  - Verify payload shape and model name in `params` (e.g., `{"model": "...", ...}`) matches your backend.

- No usage data:
  - Your backend may not return token information; `usage` will be `None`.
  - If your backend omits usage, ensure the response shape is compatible with `OpenAiApiResponse`; otherwise adjust the utility to your response format.

---

## Best practices

- Unique kind: Define a unique `kind` on your options class; factories use this to select the implementation.
- Minimal options: Keep options typed and minimal; prefer inheriting from Docling core option classes. For complex objects, consider `pydantic.ConfigDict(arbitrary_types_allowed=True)`.

---

## Example code index (quick links)

- Packaging and entry point:
  - [`docs/examples/third_party_plugins/pyproject.toml`](https://github.com/docling-project/docling/blob/main/docs/examples/third_party_plugins/pyproject.toml)
  - [`docs/examples/third_party_plugins/api_usage/api_usage_plugin.py`](https://github.com/docling-project/docling/blob/main/docs/examples/third_party_plugins/api_usage/api_usage_plugin.py)
- Options and utilities:
  - [`docs/examples/third_party_plugins/api_usage/datamodel/pipeline_options/picture_description_api_options_with_usage.py`](https://github.com/docling-project/docling/blob/main/docs/examples/third_party_plugins/api_usage/datamodel/pipeline_options/picture_description_api_options_with_usage.py)
  - [`docs/examples/third_party_plugins/api_usage/datamodel/utils/api_image_request_with_usage.py`](https://github.com/docling-project/docling/blob/main/docs/examples/third_party_plugins/api_usage/datamodel/utils/api_image_request_with_usage.py)
- Model:
  - [`docs/examples/third_party_plugins/api_usage/models/picture_description_api_model.py`](https://github.com/docling-project/docling/blob/main/docs/examples/third_party_plugins/api_usage/models/picture_description_api_model.py)
- Runner:
  - [`docs/examples/third_party_plugins/main.py`](https://github.com/docling-project/docling/blob/main/docs/examples/third_party_plugins/main.py)
