#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
docling-convert.py — Parse a document and emit Markdown, JSON, or RAG chunks.

Requires: docling>=2.81.0, docling-core>=2.67.1, packaging
Install:  pip install -r scripts/requirements.txt   (from the bundle root directory)

Usage (from bundle root, i.e. docling-document-intelligence/):
  python3 scripts/docling-convert.py <source> [options]

Arguments:
  source           Local file path or URL (required)

Pipeline selection:
  --pipeline       standard | vlm-local | vlm-api  (default: standard)

Standard pipeline options:
  --ocr-engine     easyocr | tesseract | rapidocr | mac  (default: easyocr)
  --no-ocr         Disable OCR entirely
  --no-tables      Skip table structure parsing (faster)

VLM local pipeline options (--pipeline vlm-local):
  --vlm-model      granitedocling | smoldocling | granitedocling-vllm | granitedocling-mlx
                   (default: granitedocling)
  --force-backend-text
                   Hybrid mode: use PDF text extraction for text, VLM for images/tables

VLM API pipeline options (--pipeline vlm-api):
  --vlm-api-url    OpenAI-compatible endpoint (e.g. http://localhost:8000/v1/chat/completions)
  --vlm-api-model  Model name on the server (e.g. ibm-granite/granite-docling-258M)
  --vlm-api-key    API key if required (default: none)

Output options:
  --format         markdown | json | chunks  (default: markdown)
  --max-tokens     Max tokens per chunk (default: 512)
  --tokenizer      HuggingFace model id for chunking
  --openai-model   Use OpenAI tiktoken tokenizer for chunking
                   Requires: pip install 'docling-core[chunking-openai]'
  --out            Write output to file instead of stdout
"""

import argparse
import json
import sys
from pathlib import Path

MIN_DOCLING_VERSION = "2.81.0"
MIN_DOCLING_CORE_VERSION = "2.67.1"


def parse_args():
    p = argparse.ArgumentParser(description="Docling document converter")
    p.add_argument("source", help="File path or URL")

    p.add_argument(
        "--pipeline", choices=["standard", "vlm-local", "vlm-api"], default="standard"
    )

    p.add_argument(
        "--ocr-engine",
        choices=["easyocr", "tesseract", "rapidocr", "mac"],
        default="easyocr",
    )
    p.add_argument("--no-ocr", action="store_true")
    p.add_argument("--no-tables", action="store_true")

    p.add_argument(
        "--vlm-model",
        choices=[
            "granitedocling",
            "smoldocling",
            "granitedocling-vllm",
            "granitedocling-mlx",
        ],
        default="granitedocling",
    )
    p.add_argument(
        "--force-backend-text",
        action="store_true",
        help="Hybrid: PDF text for text regions, VLM for images/tables",
    )

    p.add_argument("--vlm-api-url", default="http://localhost:8000/v1/chat/completions")
    p.add_argument("--vlm-api-model", default="ibm-granite/granite-docling-258M")
    p.add_argument("--vlm-api-key", default=None)

    p.add_argument(
        "--format", choices=["markdown", "json", "chunks"], default="markdown"
    )
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--tokenizer", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--openai-model", default=None)
    p.add_argument("--out", default=None)
    return p.parse_args()


def check_dependencies():
    from importlib.metadata import PackageNotFoundError, version as dist_version

    from packaging.version import Version

    missing: list[str] = []
    checks = [
        ("docling", "docling", MIN_DOCLING_VERSION),
        ("docling_core", "docling-core", MIN_DOCLING_CORE_VERSION),
    ]
    for import_name, dist_name, min_ver in checks:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(dist_name)
            continue
        try:
            ver = dist_version(dist_name)
        except PackageNotFoundError:
            ver = "0.0.0"
        if Version(ver) < Version(min_ver):
            print(
                f"WARNING: {dist_name}>={min_ver} recommended, found {ver}. "
                f"Run: pip install --upgrade {dist_name}",
                file=sys.stderr,
            )
    if missing:
        print(
            f"ERROR: missing packages: {' '.join(missing)}\n"
            f"Run: pip install -r scripts/requirements.txt  (from the bundle root directory)",
            file=sys.stderr,
        )
        sys.exit(1)


def build_standard_converter(args):
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    ocr_opts = None
    if not args.no_ocr:
        engine = args.ocr_engine
        if engine == "tesseract":
            from docling.datamodel.pipeline_options import TesseractOcrOptions

            ocr_opts = TesseractOcrOptions()
        elif engine == "rapidocr":
            from docling.datamodel.pipeline_options import RapidOcrOptions

            ocr_opts = RapidOcrOptions()
        elif engine == "mac":
            from docling.datamodel.pipeline_options import OcrMacOptions

            ocr_opts = OcrMacOptions()

    kwargs = dict(
        do_ocr=not args.no_ocr,
        do_table_structure=not args.no_tables,
    )
    if ocr_opts is not None:
        kwargs["ocr_options"] = ocr_opts

    pipeline_options = PdfPipelineOptions(**kwargs)
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )


def build_vlm_local_converter(args):
    from docling.datamodel import vlm_model_specs
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import VlmPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.pipeline.vlm_pipeline import VlmPipeline

    model_map = {
        "granitedocling": vlm_model_specs.GRANITEDOCLING_TRANSFORMERS,
        "smoldocling": vlm_model_specs.SMOLDOCLING_TRANSFORMERS,
        "granitedocling-vllm": vlm_model_specs.GRANITEDOCLING_VLLM,
        "granitedocling-mlx": vlm_model_specs.GRANITEDOCLING_MLX,
    }
    vlm_opts = model_map[args.vlm_model]

    pipeline_options = VlmPipelineOptions(
        vlm_options=vlm_opts,
        generate_page_images=True,
        force_backend_text=args.force_backend_text,
    )

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=VlmPipeline,
                pipeline_options=pipeline_options,
            )
        }
    )


def build_vlm_api_converter(args):
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import VlmPipelineOptions
    from docling.datamodel.pipeline_options_vlm_model import (
        ApiVlmOptions,
        ResponseFormat,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.pipeline.vlm_pipeline import VlmPipeline

    headers = {}
    if args.vlm_api_key:
        headers["Authorization"] = f"Bearer {args.vlm_api_key}"

    vlm_opts = ApiVlmOptions(
        url=args.vlm_api_url,
        params=dict(
            model=args.vlm_api_model,
            max_tokens=4096,
        ),
        headers=headers if headers else None,
        prompt="Convert this page to docling.",
        response_format=ResponseFormat.DOCTAGS,
        timeout=120,
        scale=2.0,
    )

    pipeline_options = VlmPipelineOptions(
        vlm_options=vlm_opts,
        generate_page_images=True,
        force_backend_text=args.force_backend_text,
        enable_remote_services=True,
    )

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=VlmPipeline,
                pipeline_options=pipeline_options,
            )
        }
    )


def build_tokenizer(hf_model_id: str, openai_model, max_tokens: int):
    if openai_model:
        try:
            import tiktoken
            from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
        except ImportError:
            print(
                "ERROR: OpenAI tokenizer requires:\n"
                "  pip install 'docling-core[chunking-openai]'",
                file=sys.stderr,
            )
            sys.exit(1)
        return OpenAITokenizer(
            tokenizer=tiktoken.encoding_for_model(openai_model),
            max_tokens=max_tokens,
        )
    from docling_core.transforms.chunker.tokenizer.huggingface import (
        HuggingFaceTokenizer,
    )

    return HuggingFaceTokenizer.from_pretrained(
        model_name=hf_model_id,
        max_tokens=max_tokens,
    )


def output_markdown(doc) -> str:
    return doc.export_to_markdown()


def output_json(doc) -> str:
    return json.dumps(doc.export_to_dict(), indent=2, ensure_ascii=False)


def output_chunks(doc, tokenizer) -> str:
    from docling.chunking import HybridChunker

    chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True)
    chunks = list(chunker.chunk(doc))
    texts = [chunker.contextualize(c) for c in chunks]

    try:
        counts = [tokenizer.count_tokens(t) for t in texts]
        stats = (
            f"chunks={len(chunks)}  "
            f"min={min(counts)}  max={max(counts)}  "
            f"avg={sum(counts) // len(counts)}"
        )
    except Exception:
        stats = f"chunks={len(chunks)}"

    lines = [f"# Chunks ({stats})", ""]
    for i, (chunk, text) in enumerate(zip(chunks, texts)):
        headings = (
            " > ".join(chunk.meta.headings) if chunk.meta.headings else "(no heading)"
        )
        lines += [f"## Chunk {i + 1}  |  {headings}", "", text, "", "---", ""]
    return "\n".join(lines)


def page_count(doc) -> int:
    pages = set()
    for item, _ in doc.iterate_items():
        for prov in getattr(item, "prov", []):
            pages.add(prov.page_no)
    return len(pages)


def main():
    args = parse_args()
    check_dependencies()

    tokenizer = None
    if args.format == "chunks":
        tokenizer = build_tokenizer(args.tokenizer, args.openai_model, args.max_tokens)

    if args.pipeline == "standard":
        converter = build_standard_converter(args)
        print(
            f"Pipeline: standard (ocr={not args.no_ocr}, engine={args.ocr_engine})",
            file=sys.stderr,
        )
    elif args.pipeline == "vlm-local":
        converter = build_vlm_local_converter(args)
        print(
            f"Pipeline: vlm-local (model={args.vlm_model}, "
            f"force_backend_text={args.force_backend_text})",
            file=sys.stderr,
        )
    elif args.pipeline == "vlm-api":
        converter = build_vlm_api_converter(args)
        print(
            f"Pipeline: vlm-api (url={args.vlm_api_url}, model={args.vlm_api_model})",
            file=sys.stderr,
        )

    print(f"Converting: {args.source}", file=sys.stderr)
    result = converter.convert(args.source)
    doc = result.document
    print(f"Pages processed: {page_count(doc)}", file=sys.stderr)

    if args.format == "markdown":
        output = output_markdown(doc)
    elif args.format == "json":
        output = output_json(doc)
    else:
        output = output_chunks(doc, tokenizer)

    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
        print(f"Written to {args.out}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
