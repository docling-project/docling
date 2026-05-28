# %% [markdown]
# Batch convert multiple PDF files and export results in several formats with Docling Serve.

# What this example does
# - Loads a small set of sample PDFs.
# - Runs the Docling Serve PDF pipeline once per file.
# - Writes outputs to `scratch/` in multiple formats (JSON, HTML, Markdown, text, doctags, YAML).

# Prerequisites
# - Install Docling, Docling Serve and dependencies as described in the repository README.
# - Ensure you can import `docling.service_client` from your Python environment.
# <!-- YAML export requires `PyYAML` (`pip install pyyaml`). -->

# Input documents
# - Default set of sample PDFs.
# - Update `input_doc_paths` to a desired list of file URLs.
# - All input sources must start with `http://` or `https://`.

# Output formats (controlled by flags)
# - `USE_V2 = True` enables the current Docling document exports (recommended).
# - `USE_LEGACY = False` keeps legacy Deep Search exports disabled.
#   You can set it to `True` if you need legacy formats for compatibility tests.

# Notes
# - Set `pipeline_options.generate_page_images = True` to include page images in HTML.
# - The script logs conversion progress and raises if any documents fail.
# <!-- This example shows both helper methods like `save_as_*` and lower-level
#   `export_to_*` + manual file writes; outputs may overlap intentionally. -->
# %%

import json
import logging
import time
from collections.abc import Iterable
from pathlib import Path

import yaml
from docling_core.types.doc.base import ImageRefMode

from docling.datamodel.base_models import ConversionStatus
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.service.options import ConvertDocumentsOptions
from docling.service_client import DoclingServiceClient

_log = logging.getLogger(__name__)

# Export toggles:
# - USE_V2 controls modern Docling document exports.
# - USE_LEGACY enables legacy Deep Search exports for comparison or migration.
USE_V2 = True
USE_LEGACY = False


def export_documents(
    conv_results: Iterable[ConversionResult],
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    failure_count = 0
    partial_success_count = 0

    for conv_res in conv_results:
        if conv_res.status == ConversionStatus.SUCCESS:
            success_count += 1
            doc_filename = conv_res.input.file.stem

            if USE_V2:
                # Recommended modern Docling exports. These helpers mirror the
                # lower-level "export_to_*" methods used below, but handle
                # common details like image handling.
                conv_res.document.save_as_json(
                    output_dir / f"{doc_filename}.json",
                    image_mode=ImageRefMode.PLACEHOLDER,
                )
                conv_res.document.save_as_html(
                    output_dir / f"{doc_filename}.html",
                    image_mode=ImageRefMode.EMBEDDED,
                )
                conv_res.document.save_as_doctags(
                    output_dir / f"{doc_filename}.doctags.txt"
                )
                conv_res.document.save_as_markdown(
                    output_dir / f"{doc_filename}.md",
                    image_mode=ImageRefMode.PLACEHOLDER,
                )
                conv_res.document.save_as_markdown(
                    output_dir / f"{doc_filename}.txt",
                    image_mode=ImageRefMode.PLACEHOLDER,
                    strict_text=True,
                )

                # Export Docling document format to YAML:
                with (output_dir / f"{doc_filename}.yaml").open("w") as fp:
                    fp.write(yaml.safe_dump(conv_res.document.export_to_dict()))

                # Export Docling document format to doctags:
                with (output_dir / f"{doc_filename}.doctags.txt").open("w") as fp:
                    fp.write(conv_res.document.export_to_doctags())

                # Export Docling document format to markdown:
                with (output_dir / f"{doc_filename}.md").open("w") as fp:
                    fp.write(conv_res.document.export_to_markdown())

                # Export Docling document format to text:
                with (output_dir / f"{doc_filename}.txt").open("w") as fp:
                    fp.write(conv_res.document.export_to_markdown(strict_text=True))

            if USE_LEGACY:
                # Export Deep Search document JSON format:
                with (output_dir / f"{doc_filename}.legacy.json").open(
                    "w", encoding="utf-8"
                ) as fp:
                    fp.write(json.dumps(conv_res.legacy_document.export_to_dict()))

                # Export Text format:
                with (output_dir / f"{doc_filename}.legacy.txt").open(
                    "w", encoding="utf-8"
                ) as fp:
                    fp.write(
                        conv_res.legacy_document.export_to_markdown(strict_text=True)
                    )

                # Export Markdown format:
                with (output_dir / f"{doc_filename}.legacy.md").open(
                    "w", encoding="utf-8"
                ) as fp:
                    fp.write(conv_res.legacy_document.export_to_markdown())

                # Export Document Tags format:
                with (output_dir / f"{doc_filename}.legacy.doctags.txt").open(
                    "w", encoding="utf-8"
                ) as fp:
                    fp.write(conv_res.legacy_document.export_to_document_tokens())

        elif conv_res.status == ConversionStatus.PARTIAL_SUCCESS:
            _log.info(
                f"Document {conv_res.input.file} was partially converted with the following errors:"
            )
            for item in conv_res.errors:
                _log.info(f"\t{item.error_message}")
            partial_success_count += 1
        else:
            _log.info(f"Document {conv_res.input.file} failed to convert.")
            failure_count += 1

    _log.info(
        f"Processed {success_count + partial_success_count + failure_count} docs, "
        f"of which {failure_count} failed "
        f"and {partial_success_count} were partially converted."
    )
    return success_count, partial_success_count, failure_count


def main():
    logging.basicConfig(level=logging.INFO)

    # Add your HTTP or HTTPS URLs here
    input_doc_urls = [
        "https://arxiv.org/pdf/1706.03762",
        "https://arxiv.org/pdf/1103.0398",
        "https://arxiv.org/pdf/2501.17887",
        "https://arxiv.org/pdf/2408.09869",
    ]

    # Configure the PDF pipeline. Enabling page image generation improves HTML
    # previews (embedded images) but adds processing time.
    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_page_images = True

    SERVE_URL = "http://localhost:5001"

    options = ConvertDocumentsOptions(
        do_ocr=pipeline_options.do_ocr,
        ocr_lang=pipeline_options.ocr_options.lang if pipeline_options.do_ocr else None,
    )

    start_time = time.time()

    # Convert all inputs. Set `raises_on_error=False` to keep processing other
    # files even if one fails; errors are summarized after the run.
    conv_results = []
    failure_count = 0
    with DoclingServiceClient(url=SERVE_URL) as client:
        for input_doc_url in input_doc_urls:
            try:
                conv_result = client.convert(source=input_doc_url, options=options)
                conv_results.append(conv_result)
            except Exception as e:
                _log.error(f"Failed to convert {input_doc_url}: {e}")
                failure_count += 1

    # Write outputs to ./scratch and log a summary.
    _success_count, _partial_success_count, export_failure_count = export_documents(
        conv_results, output_dir=Path("scratch")
    )

    # Add conversion failures to the total failure count
    total_failures = failure_count + export_failure_count

    end_time = time.time() - start_time

    _log.info(f"Document conversion complete in {end_time:.2f} seconds.")

    if total_failures > 0:
        raise RuntimeError(
            f"The example failed converting {total_failures} on {len(input_doc_urls)}."
        )


if __name__ == "__main__":
    main()
