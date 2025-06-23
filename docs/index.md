<p align="center">
  <img loading="lazy" alt="Docling" src="assets/docling_processing.png" width="100%" />
  <a href="https://trendshift.io/repositories/12132" target="_blank"><img src="https://trendshift.io/api/badge/repositories/12132" alt="DS4SD%2Fdocling | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

[![arXiv](https://img.shields.io/badge/arXiv-2408.09869-b31b1b.svg)](https://arxiv.org/abs/2408.09869)
[![PyPI version](https://img.shields.io/pypi/v/docling)](https://pypi.org/project/docling/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/docling)](https://pypi.org/project/docling/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pydantic v2](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pydantic/pydantic/main/docs/badge/v2.json)](https://pydantic.dev)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![License MIT](https://img.shields.io/github/license/docling-project/docling)](https://opensource.org/licenses/MIT)
[![PyPI Downloads](https://static.pepy.tech/badge/docling/month)](https://pepy.tech/projects/docling)
[![Docling Actor](https://apify.com/actor-badge?actor=vancura/docling?fpr=docling)](https://apify.com/vancura/docling)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/10101/badge)](https://www.bestpractices.dev/projects/10101)
[![LF AI & Data](https://img.shields.io/badge/LF%20AI%20%26%20Data-003778?logo=linuxfoundation&logoColor=fff&color=0094ff&labelColor=003778)](https://lfaidata.foundation/projects/)

Docling simplifies document processing, parsing diverse formats — including advanced PDF understanding — and providing seamless integrations with the gen AI ecosystem.

## Features

* 🗂️  Parsing of [multiple document formats][supported_formats] incl. PDF, DOCX, PPTX, XLSX, HTML, WAV, MP3, images (PNG, TIFF, JPEG, ...), and more
* 📑 Advanced PDF understanding incl. page layout, reading order, table structure, code, formulas, image classification, and more
* 🧬 Unified, expressive [DoclingDocument][docling_document] representation format
* ↪️  Various [export formats][supported_formats] and options, including Markdown, HTML, [DocTags](https://arxiv.org/abs/2503.11576) and lossless JSON
* 🔒 Local execution capabilities for sensitive data and air-gapped environments
* 🤖 Plug-and-play [integrations][integrations] incl. LangChain, LlamaIndex, Crew AI & Haystack for agentic AI
* 🔍 Extensive OCR support for scanned PDFs and images
* 👓 Support of several Visual Language Models ([SmolDocling](https://huggingface.co/ds4sd/SmolDocling-256M-preview))
* 🎙️  Support for Audio with Automatic Speech Recognition (ASR) models
* 💻 Simple and convenient CLI

### Coming soon

* 📝 Metadata extraction, including title, authors, references & language
* 📝 Chart understanding (Barchart, Piechart, LinePlot, etc)
* 📝 Complex chemistry understanding (Molecular structures)

## Get started

<div class="grid">
  <a href="concepts/" class="card"><b>Concepts</b><br />Learn Docling fundamentals</a>
  <a href="examples/" class="card"><b>Examples</b><br />Try out recipes for various use cases, including conversion, RAG, and more</a>
  <a href="integrations/" class="card"><b>Integrations</b><br />Check out integrations with popular frameworks and tools</a>
  <a href="reference/document_converter/" class="card"><b>Reference</b><br />See more API details</a>
</div>

## LF AI & Data

Docling is hosted as a project in the [LF AI & Data Foundation](https://lfaidata.foundation/projects/).

### IBM ❤️ Open Source AI

The project was started by the AI for knowledge team at IBM Research Zurich.

[supported_formats]: ./usage/supported_formats.md
[docling_document]: ./concepts/docling_document.md
[integrations]: ./integrations/index.md
