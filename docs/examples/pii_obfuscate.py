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
# - Optionally, set `HF_MODEL` to a different NER/PII model.
#
# How to run
# - From the repo root: `python docs/examples/pii_obfuscate.py`.
# - The script writes original and obfuscated Markdown to `scratch/`.
#
# Notes
# - This is a simple demonstration. For production PII detection, consider
#   specialized models/pipelines and thorough evaluation.
# %%

import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

from docling_core.types.doc import ImageRefMode, TableItem, TextItem

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

_log = logging.getLogger(__name__)

IMAGE_RESOLUTION_SCALE = 2.0
HF_MODEL = "dslim/bert-base-NER"  # Swap with another HF NER/PII model if desired, eg https://huggingface.co/urchade/gliner_multi_pii-v1 looks very promising too!


def _build_ner_pipeline():
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


class PiiObfuscator:
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
        found: List[Tuple[str, str]] = []
        for r in results:
            print(r)
            # Handle different key names across models/pipelines
            label = r.get("entity_group") or r.get("entity") or "MISC"
            label = self.label_map.get(label, "misc")
            if label not in self.allowed_types:
                continue
            word = r.get("word") or r.get("text") or ""
            word = self._normalize(word)
            if not word:
                continue
            found.append((word, label))
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


def main():
    logging.basicConfig(level=logging.INFO)

    data_folder = Path(__file__).parent / "../../tests/data"
    input_doc_path = data_folder / "pdf/2206.01062.pdf"
    output_dir = Path("scratch")  # ensure this directory exists before saving

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
    ner = _build_ner_pipeline()
    obfuscator = PiiObfuscator(ner)

    for element, _level in conv_res.document.iterate_items():
        if isinstance(element, TextItem):
            element.orig = element.text
            element.text = obfuscator.obfuscate_text(element.text)

            print(element.orig, " => ", element.text)

        elif isinstance(element, TableItem):
            for cell in element.data.table_cells:
                cell.text = obfuscator.obfuscate_text(cell.text)

    # Save markdown with embedded pictures and obfuscated text
    md_filename = output_dir / f"{doc_filename}-with-images-pii-obfuscated.md"
    conv_doc.save_as_markdown(md_filename, image_mode=ImageRefMode.EMBEDDED)

    # Optional: log mapping summary
    if obfuscator.entity_map:
        _log.info(
            "Obfuscated entities (sample): %s", list(obfuscator.entity_map.items())[:10]
        )


if __name__ == "__main__":
    main()
