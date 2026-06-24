# %% [markdown]
# # Capture API usage payloads from picture descriptions
#
# This example converts a PDF, describes its pictures through an OpenAI-compatible
# chat-completions endpoint, and prints the raw usage payload preserved on each
# picture description metadata field.
#
# Run from the repository root. The PDF argument is optional; when omitted, the
# bundled test PDF at `tests/data/pdf/regression/2206.01062.pdf` is used. The example exits
# early without contacting any endpoint when neither
# `PICTURE_DESCRIPTION_API_URL` nor `AZURE_API_BASE` is set, which keeps it safe
# to run in environments without a configured VLM provider (for example CI).
#
# ```sh
# python docs/examples/picture_description_api_usage.py path/to/input.pdf
# ```
#
# Or use the companion shell wrapper:
#
# ```sh
# docs/examples/run_picture_description_api_usage.sh path/to/input.pdf
# ```
#
# Optional environment variables:
#
# - `AZURE_API_KEY`: Azure OpenAI API key. Used as the `api-key` header when
#   `AZURE_API_BASE` is configured. Do not commit this value.
# - `AZURE_API_BASE`: Azure OpenAI resource base URL, for example
#   `https://my-resource.openai.azure.com`.
# - `AZURE_OPENAI_DEPLOYMENT`: Azure deployment name. Defaults to `gpt-4.1` in
#   the shell wrapper.
# - `AZURE_OPENAI_API_VERSION`: Azure OpenAI API version used in the request URL.
# - `PICTURE_DESCRIPTION_API_URL`: Chat-completions endpoint. If set, this
#   overrides Azure URL construction.
# - `PICTURE_DESCRIPTION_API_KEY`: Bearer token added as the `Authorization`
#   header when set.
# - `PICTURE_DESCRIPTION_MODEL`: Model parameter sent in the request body when
#   set for non-Azure endpoints.
# - `PICTURE_DESCRIPTION_USAGE_RESPONSE_KEY`: Response JSON key or dotted path
#   to preserve as usage metadata. Defaults to `usage`, which matches
#   OpenAI-compatible responses.
# - `PICTURE_DESCRIPTION_PARAMS_JSON`: Extra JSON object merged into the request
#   body.
# - `PICTURE_DESCRIPTION_AREA_THRESHOLD`: Minimum picture area fraction to
#   describe. Defaults to `0.0` in this example so small figures are not
#   silently skipped.
#
# The usage payload is stored as custom metadata on each picture description:
#
# ```py
# picture.meta.description.get_custom_part()["docling__usage"]
# ```
#
# Clients can then validate that raw provider payload with their own Pydantic
# model, because token accounting differs across providers.

# %%

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

from docling_core.types.doc import PictureItem

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    PictureDescriptionApiOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

_USAGE_META_KEY = "docling__usage"

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)


def _build_azure_openai_url() -> str | None:
    api_base = os.environ.get("AZURE_API_BASE") or None
    if api_base is None:
        return None

    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT") or os.environ.get(
        "PICTURE_DESCRIPTION_MODEL",
        "gpt-4.1",
    )
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
    return (
        f"{api_base.rstrip('/')}/openai/deployments/{deployment}"
        f"/chat/completions?api-version={api_version}"
    )


def _get_explicit_api_url() -> str | None:
    explicit_api_url = os.environ.get("PICTURE_DESCRIPTION_API_URL") or None
    azure_api_base = os.environ.get("AZURE_API_BASE") or None
    if (
        explicit_api_url is not None
        and azure_api_base is not None
        and explicit_api_url.rstrip("/") == azure_api_base.rstrip("/")
    ):
        _log.warning(
            "Ignoring PICTURE_DESCRIPTION_API_URL because it matches AZURE_API_BASE. "
            "The example will build the Azure chat-completions deployment URL."
        )
        return None
    return explicit_api_url


def _load_extra_params() -> dict[str, Any]:
    params_json = os.environ.get("PICTURE_DESCRIPTION_PARAMS_JSON")
    if params_json is None:
        return {}

    parsed = json.loads(params_json)
    if not isinstance(parsed, dict):
        raise ValueError("PICTURE_DESCRIPTION_PARAMS_JSON must be a JSON object.")
    return parsed


def _build_picture_description_options() -> PictureDescriptionApiOptions:
    headers: dict[str, str] = {}
    explicit_api_url = _get_explicit_api_url()
    uses_azure_openai = (
        explicit_api_url is None
        and (os.environ.get("AZURE_API_BASE") or None) is not None
    )
    api_url = explicit_api_url or _build_azure_openai_url()

    if uses_azure_openai and (azure_api_key := os.environ.get("AZURE_API_KEY")):
        headers["api-key"] = azure_api_key
    elif uses_azure_openai:
        raise ValueError("Set AZURE_API_KEY before using the Azure OpenAI example.")
    elif api_key := os.environ.get("PICTURE_DESCRIPTION_API_KEY"):
        headers["Authorization"] = f"Bearer {api_key}"

    params = _load_extra_params()
    model = os.environ.get("PICTURE_DESCRIPTION_MODEL")
    if model is not None and not uses_azure_openai:
        params["model"] = model

    return PictureDescriptionApiOptions(
        url=api_url or "http://localhost:8000/v1/chat/completions",
        headers=headers,
        params=params,
        prompt="Describe this picture in a few concise sentences.",
        picture_area_threshold=float(
            os.environ.get("PICTURE_DESCRIPTION_AREA_THRESHOLD", "0.0")
        ),
        usage_response_key=os.environ.get(
            "PICTURE_DESCRIPTION_USAGE_RESPONSE_KEY",
            "usage",
        ),
    )


def _extract_usage_payload(item: PictureItem) -> Any | None:
    if item.meta is None or item.meta.description is None:
        return None

    return item.meta.description.get_custom_part().get(_USAGE_META_KEY)


def _has_api_endpoint_configured() -> bool:
    return bool(
        os.environ.get("PICTURE_DESCRIPTION_API_URL")
        or os.environ.get("AZURE_API_BASE")
    )


def run(input_pdf: Path) -> None:
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_picture_description = True
    pipeline_options.picture_description_options = _build_picture_description_options()
    pipeline_options.enable_remote_services = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )
    result = converter.convert(input_pdf)

    pictures_with_usage = 0
    for item, _level in result.document.iterate_items():
        if not isinstance(item, PictureItem):
            continue

        usage = _extract_usage_payload(item)
        description = None
        has_description_meta = False
        if item.meta is not None and item.meta.description is not None:
            has_description_meta = True
            description = item.meta.description.text

        print(f"Picture: {item.self_ref}")
        if not has_description_meta:
            print("Description: <not generated>")
        elif not description:
            print("Description: <empty API result>")
        else:
            print(f"Description: {description}")
        if usage is None:
            print("Usage: <not captured>")
        else:
            pictures_with_usage += 1
            print("Usage:")
            print(json.dumps(usage, indent=2, sort_keys=True))
        print()

    print(f"Pictures with usage payloads: {pictures_with_usage}")


_DEFAULT_PDF = Path(__file__).resolve().parents[2] / "tests/data/pdf/regression/2206.01062.pdf"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a PDF with API picture descriptions and print raw usage "
            "payloads stored in DoclingDocument metadata."
        )
    )
    parser.add_argument(
        "pdf",
        type=Path,
        nargs="?",
        default=_DEFAULT_PDF,
        help=(
            "Path to the input PDF. Defaults to the bundled test PDF at "
            f"{_DEFAULT_PDF}."
        ),
    )
    args = parser.parse_args()

    if not _has_api_endpoint_configured():
        _log.warning(
            "Skipping: no picture description API endpoint configured. Set "
            "PICTURE_DESCRIPTION_API_URL or AZURE_API_BASE (with the matching "
            "credentials) to actually run this example."
        )
        return

    run(input_pdf=args.pdf)


if __name__ == "__main__":
    main()
