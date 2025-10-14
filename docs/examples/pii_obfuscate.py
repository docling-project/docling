# %% [markdown]
# Detect and obfuscate PII using a Hugging Face NER model.
#
# What this example does
# - Converts a PDF and saves original Markdown with embedded images.
# - Runs a HF token-classification pipeline (NER) to detect PII-like entities.
# - Obfuscates occurrences in TextItem and TableItem by stable, type-based IDs.
#
# Prerequisites
# - Install Docling. Install Transformers: `pip install transformers`.
# - Optional (advanced): Install GLiNER for richer PII labels:
#     pip install gliner
#     # If needed for CPU-only envs:
#     pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
# - Optionally, set `HF_MODEL` to a different NER/PII model.
#
# How to run
# - From the repo root: `python docs/examples/pii_obfuscate.py`.
# - To use GLiNER instead of HF pipeline:
#     python docs/examples/pii_obfuscate.py --engine gliner
#   or set env var `PII_ENGINE=gliner`.
# - The script writes original and obfuscated Markdown to `scratch/`.
#
# Notes
# - This is a simple demonstration. For production PII detection, consider
#   specialized models/pipelines and thorough evaluation.
# %%

import logging
import re
import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

from tabulate import tabulate

from docling_core.types.doc import ImageRefMode, TableItem, TextItem

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

_log = logging.getLogger(__name__)

IMAGE_RESOLUTION_SCALE = 2.0
HF_MODEL = "dslim/bert-base-NER"  # Swap with another HF NER/PII model if desired, eg https://huggingface.co/urchade/gliner_multi_pii-v1 looks very promising too!
GLINER_MODEL = "urchade/gliner_multi_pii-v1"


def _build_simple_ner_pipeline():
    """Create a Hugging Face token-classification pipeline for NER.

    Returns a callable like: ner(text) -> List[dict]
    """
    try:
        from transformers import (
            AutoModelForTokenClassification,
            AutoTokenizer,
            pipeline,
        )
    except Exception:
        _log.error("Transformers not installed. Please run: pip install transformers")
        raise

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
    model = AutoModelForTokenClassification.from_pretrained(HF_MODEL)
    ner = pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",  # groups subwords into complete entities
        # Note: modern Transformers returns `start`/`end` when possible with aggregation
    )
    return ner


class SimplePiiObfuscator:
    """Tracks PII strings and replaces them with stable IDs per entity type."""

    def __init__(self, ner_callable):
        self.ner = ner_callable
        self.entity_map: Dict[str, str] = {}
        self.counters: Dict[str, int] = {
            "person": 0,
            "org": 0,
            "location": 0,
            "misc": 0,
        }
        # Map model labels to our coarse types
        self.label_map = {
            "PER": "person",
            "PERSON": "person",
            "ORG": "org",
            "ORGANIZATION": "org",
            "LOC": "location",
            "LOCATION": "location",
            "GPE": "location",
            # Fallbacks
            "MISC": "misc",
            "O": "misc",
        }
        # Only obfuscate these by default. Adjust as needed.
        self.allowed_types = {"person", "org", "location"}

    def _next_id(self, typ: str) -> str:
        self.counters[typ] += 1
        return f"{typ}-{self.counters[typ]}"

    def _normalize(self, s: str) -> str:
        return re.sub(r"\s+", " ", s).strip()

    def _extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """Run NER and return a list of (surface_text, type) to obfuscate."""
        if not text:
            return []
        results = self.ner(text)
        # Collect normalized items with optional span info
        items = []
        for r in results:
            raw_label = r.get("entity_group") or r.get("entity") or "MISC"
            label = self.label_map.get(raw_label, "misc")
            if label not in self.allowed_types:
                continue
            start = r.get("start")
            end = r.get("end")
            word = self._normalize(r.get("word") or r.get("text") or "")
            items.append({"label": label, "start": start, "end": end, "word": word})

        found: List[Tuple[str, str]] = []
        # If the pipeline provides character spans, merge consecutive/overlapping
        # entities of the same type into a single span, then take the substring
        # from the original text. This handles cases like subword tokenization
        # where multiple adjacent pieces belong to the same named entity.
        have_spans = any(i["start"] is not None and i["end"] is not None for i in items)
        if have_spans:
            spans = [i for i in items if i["start"] is not None and i["end"] is not None]
            # Ensure processing order by start (then end)
            spans.sort(key=lambda x: (x["start"], x["end"]))

            merged = []
            for s in spans:
                if not merged:
                    merged.append(dict(s))
                    continue
                last = merged[-1]
                if s["label"] == last["label"] and s["start"] <= last["end"]:
                    # Merge identical, overlapping, or touching spans of same type
                    last["start"] = min(last["start"], s["start"])
                    last["end"] = max(last["end"], s["end"])
                else:
                    merged.append(dict(s))

            for m in merged:
                surface = self._normalize(text[m["start"] : m["end"]])
                if surface:
                    found.append((surface, m["label"]))

            # Include any items lacking spans as-is (fallback)
            for i in items:
                if i["start"] is None or i["end"] is None:
                    if i["word"]:
                        found.append((i["word"], i["label"]))
        else:
            # Fallback when spans aren't provided: return normalized words
            for i in items:
                if i["word"]:
                    found.append((i["word"], i["label"]))
        return found

    def obfuscate_text(self, text: str) -> str:
        if not text:
            return text

        entities = self._extract_entities(text)
        if not entities:
            return text

        # Deduplicate per text, keep stable global mapping
        unique_words: Dict[str, str] = {}
        for word, label in entities:
            if word not in self.entity_map:
                replacement = self._next_id(label)
                self.entity_map[word] = replacement
            unique_words[word] = self.entity_map[word]

        # Replace longer matches first to avoid partial overlaps
        sorted_pairs = sorted(
            unique_words.items(), key=lambda x: len(x[0]), reverse=True
        )

        def replace_once(s: str, old: str, new: str) -> str:
            # Use simple substring replacement; for stricter matching, use word boundaries
            # when appropriate (e.g., names). This is a demo, keep it simple.
            pattern = re.escape(old)
            return re.sub(pattern, new, s)

        obfuscated = text
        for old, new in sorted_pairs:
            obfuscated = replace_once(obfuscated, old, new)
        return obfuscated


def _build_gliner_model():
    """Create a GLiNER model for PII-like entity extraction.

    Returns a tuple (model, labels) where model.predict_entities(text, labels)
    yields entities with "text" and "label" fields.
    """
    try:
        from gliner import GLiNER  # type: ignore
    except Exception:
        _log.error(
            "GLiNER not installed. Please run: pip install gliner torch --extra-index-url https://download.pytorch.org/whl/cpu"
        )
        raise

    model = GLiNER.from_pretrained(GLINER_MODEL)
    # Curated set of labels for PII detection. Adjust as needed.
    labels = [
        "work",
        "booking number",
        "personally identifiable information",
        "driver licence",
        "person",
        "full address",
        "company",
        "actor",
        "character",
        "email",
        "passport number",
        "Social Security Number",
        "phone number",
    ]
    return model, labels


class AdvancedPIIObfuscator:
    """PII obfuscator powered by GLiNER with fine-grained labels.

    - Uses GLiNER's `predict_entities(text, labels)` to detect entities.
    - Obfuscates with stable IDs per fine-grained label, e.g. `email-1`.
    """

    def __init__(self, gliner_model, labels: List[str]):
        self.model = gliner_model
        self.labels = labels
        self.entity_map: Dict[str, str] = {}
        self.counters: Dict[str, int] = {}

    def _normalize(self, s: str) -> str:
        return re.sub(r"\s+", " ", s).strip()

    def _norm_label(self, label: str) -> str:
        return re.sub(r"[^a-z0-9_]+", "_", label.lower().replace(" ", "_").replace("-", "_")).strip("_") or "pii"

    def _next_id(self, typ: str) -> str:
        self.cc(typ)
        self.counters[typ] += 1
        return f"{typ}-{self.counters[typ]}"

    def cc(self, typ: str) -> None:
        if typ not in self.counters:
            self.counters[typ] = 0

    def _extract_entities(self, text: str) -> List[Tuple[str, str]]:
        if not text:
            return []
        results = self.model.predict_entities(text, self.labels)  # expects dicts with text/label
        found: List[Tuple[str, str]] = []
        for r in results:
            label = self._norm_label(str(r.get("label", "pii")))
            surface = self._normalize(str(r.get("text", "")))
            if surface:
                found.append((surface, label))
        return found

    def obfuscate_text(self, text: str) -> str:
        if not text:
            return text
        entities = self._extract_entities(text)
        if not entities:
            return text

        unique_words: Dict[str, str] = {}
        for word, label in entities:
            if word not in self.entity_map:
                replacement = self._next_id(label)
                self.entity_map[word] = replacement
            unique_words[word] = self.entity_map[word]

        sorted_pairs = sorted(unique_words.items(), key=lambda x: len(x[0]), reverse=True)

        def replace_once(s: str, old: str, new: str) -> str:
            pattern = re.escape(old)
            return re.sub(pattern, new, s)

        obfuscated = text
        for old, new in sorted_pairs:
            obfuscated = replace_once(obfuscated, old, new)
        return obfuscated


def main():
    logging.basicConfig(level=logging.INFO)

    data_folder = Path(__file__).parent / "../../tests/data"
    input_doc_path = data_folder / "pdf/2206.01062.pdf"
    output_dir = Path("scratch")  # ensure this directory exists before saving

    # Choose engine via CLI flag or env var (default: hf)
    parser = argparse.ArgumentParser(description="PII obfuscation example")
    parser.add_argument(
        "--engine",
        choices=["hf", "gliner"],
        default=os.getenv("PII_ENGINE", "hf"),
        help="NER engine: 'hf' (Transformers) or 'gliner' (GLiNER)",
    )
    args = parser.parse_args()

    # Ensure output dir exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Keep and generate images so Markdown can embed them
    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    conv_res = doc_converter.convert(input_doc_path)
    conv_doc = conv_res.document
    doc_filename = conv_res.input.file.name

    # Save markdown with embedded pictures in original text
    md_filename = output_dir / f"{doc_filename}-with-images-orig.md"
    conv_doc.save_as_markdown(md_filename, image_mode=ImageRefMode.EMBEDDED)

    # Build NER pipeline and obfuscator
    if args.engine == "gliner":
        _log.info("Using GLiNER-based AdvancedPIIObfuscator")
        gliner_model, gliner_labels = _build_gliner_model()
        obfuscator = AdvancedPIIObfuscator(gliner_model, gliner_labels)
    else:
        _log.info("Using HF Transformers-based SimplePiiObfuscator")
        ner = _build_simple_ner_pipeline()
        obfuscator = SimplePiiObfuscator(ner)

    for element, _level in conv_res.document.iterate_items():
        if isinstance(element, TextItem):
            element.orig = element.text
            element.text = obfuscator.obfuscate_text(element.text)
            # print(element.orig, " => ", element.text)

        elif isinstance(element, TableItem):
            for cell in element.data.table_cells:
                cell.text = obfuscator.obfuscate_text(cell.text)

    # Save markdown with embedded pictures and obfuscated text
    md_filename = output_dir / f"{doc_filename}-with-images-pii-obfuscated.md"
    conv_doc.save_as_markdown(md_filename, image_mode=ImageRefMode.EMBEDDED)

    # Optional: log mapping summary
    if obfuscator.entity_map:
        data = []
        for key, val in obfuscator.entity_map.items():
            data.append([key, val])
        
        _log.info(
            f"Obfuscated entities:\n\n{tabulate(data)}", 
        )


if __name__ == "__main__":
    main()
