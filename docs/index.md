<p align="center">
  <img loading="lazy" alt="Docling" src="assets/docling_processing.png" width="100%" />
  <a href="https://trendshift.io/repositories/12132" target="_blank"><img src="https://trendshift.io/api/badge/repositories/12132" alt="DS4SD%2Fdocling | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

[![arXiv](https://img.shields.io/badge/arXiv-2408.09869-b31b1b.svg)](https://arxiv.org/abs/2408.09869)
[![PyPI version](https://img.shields.io/badge/PyPI%20version-2.48.0-blue.svg)](https://pypi.org/project/docling/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/docling)](https://pypi.org/project/docling/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pydantic v2](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pydantic/pydantic/main/docs/badge/v2.json)](https://pydantic.dev)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![License MIT](https://img.shields.io/github/license/docling-project/docling)](https://opensource.org/licenses/MIT)
[![PyPI Downloads](https://static.pepy.tech/badge/docling/month)](https://pepy.tech/projects/docling)
[![Docling Actor](https://apify.com/actor-badge?actor=vancura/docling?fpr=docling)](https://apify.com/actor/vancura/docling)
[![Chat with Dosu](https://dosu.dev/dosu-chat-badge.svg)](https://app.dosu.dev/097760a8-135e-4789-8234-90c8837d7f1c/ask?utm_source=github)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/10101/badge)](https://www.bestpractices.dev/projects/10101)
[![LF AI & Data](https://img.shields.io/badge/LF%20AI%20%26%20Data-003778?logo=linuxfoundation&logoColor=fff&color=0094ff&labelColor=003778)](https://lfaidata.foundation/projects/)

Docling simplifies document processing, parsing diverse formats — including advanced PDF understanding — and providing seamless integrations with the gen AI ecosystem.

## Features

I've organized the features to show what Docling can actually do for you:

* **Document Parsing** - Parsing of [multiple document formats][supported_formats] incl. PDF, DOCX, PPTX, XLSX, HTML, WAV, MP3, images (PNG, TIFF, JPEG, ...), and more
* **Advanced PDF Understanding** - Advanced PDF understanding incl. page layout, reading order, table structure, code, formulas, image classification, and more
* **Unified Document Model** - Unified, expressive [DoclingDocument][docling_document] representation format
* **Multiple Export Formats** - Various [export formats][supported_formats] and options, including Markdown, HTML, [DocTags](https://arxiv.org/abs/2503.11576) and lossless JSON
* **Local Execution** - Local execution capabilities for sensitive data and air-gapped environments
* **AI Framework Integrations** - Plug-and-play [integrations][integrations] incl. LangChain, LlamaIndex, Crew AI & Haystack for agentic AI
* **OCR Support** - Extensive OCR support for scanned PDFs and images
* **Visual Language Models** - Support of several Visual Language Models ([SmolDocling](https://huggingface.co/ds4sd/SmolDocling-256M-preview))
* **Audio Processing** - Support for Audio with Automatic Speech Recognition (ASR) models
* **Command Line Interface** - Simple and convenient CLI

### Coming soon

* **Metadata Extraction** - Metadata extraction, including title, authors, references & language
* **Chart Understanding** - Chart understanding (Barchart, Piechart, LinePlot, etc)
* **Chemistry Understanding** - Complex chemistry understanding (Molecular structures)

## Get started

Check out our [getting started](./getting_started/index.md) page to get the ball rolling!

## Live assistant

Do you want to leverage the power of AI and get live support on Docling?
Try out the [Chat with Dosu](https://app.dosu.dev/097760a8-135e-4789-8234-90c8837d7f1c/ask?utm_source=github) functionalities provided by our friends at [Dosu](https://dosu.dev/).

[![Chat with Dosu](https://dosu.dev/dosu-chat-badge.svg)](https://app.dosu.dev/097760a8-135e-4789-8234-90c8837d7f1c/ask?utm_source=github)

## LF AI & Data

Docling is hosted as a project in the [LF AI & Data Foundation](https://lfaidata.foundation/projects/).

### IBM ❤️ Open Source AI

The project was started by the AI for knowledge team at IBM Research Zurich.

[supported_formats]: ./usage/supported_formats.md
[docling_document]: ./concepts/docling_document.md
[integrations]: ./integrations/index.md
