# 🧠 Semantica

Docling is available as a native integration in [Semantica](https://github.com/Hawksight-AI/semantica), an open-source framework for building **semantic layers** and **knowledge graphs** from unstructured data.

By combining Docling's high-fidelity structural parsing with Semantica's knowledge engineering, you can transform complex documents into AI-ready, structured knowledge for GraphRAG, AI agents, and multi-agent systems.

- 📖 [Semantica Documentation](https://hawksight-ai.github.io/semantica/)
- 💻 [Semantica GitHub](https://github.com/Hawksight-AI/semantica)
- 🧑🏽‍🍳 [Earnings Call Analysis Example](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/use_cases/finance/03_Earnings_Call_Analysis.ipynb)
- 📦 [Semantica PyPI](https://pypi.org/project/semantica/)

## Why Semantica + Docling?

While Docling excels at extracting structural elements (like tables and nested headers), Semantica bridges the **semantic gap** by converting that structure into a queryable knowledge base.

| Feature | Docling | Semantica |
|:---|:---|:---|
| **Parsing** | 💎 High-fidelity layout & table extraction | Native `DoclingParser` integration |
| **Structuring** | Markdown, JSON, HTML export | Knowledge Graph & RDF Triplet construction |
| **Refining** | - | Entity normalization & deduplication |
| **Intelligence** | - | Automated ontology generation & GraphRAG |

## Components

### Docling Parser
The `DoclingParser` is a specialized module within Semantica that uses Docling's `DocumentConverter` to extract high-fidelity Markdown and structured tables. It serves as the entry point for turning raw documents into semantic data.

- 💻 [Docling Parser Implementation](https://github.com/Hawksight-AI/semantica/blob/main/semantica/parse/docling_parser.py)

### Knowledge Graph Builder
Semantica uses the output from the `DoclingParser` to extract entities and relations, which are then stored in a property graph (Neo4j, FalkorDB) or a triplet store (RDF).

## Installation

Install Semantica with Docling support:

```bash
pip install "semantica[all]" docling
```

## Usage: The Semantic Pipeline

The following example demonstrates the full pipeline: parsing a document with Docling, normalizing the text, and extracting semantic triplets for a Knowledge Graph.

```python
from semantica.parse import DoclingParser
from semantica.normalize import TextNormalizer
from semantica.split import TextSplitter
from semantica.semantic_extract import TripletExtractor

# 1. Structural Parsing with Docling
# Docling handles the complex layout and table extraction
parser = DoclingParser(enable_ocr=True)
result = parser.parse("earnings_call.pdf")

# 2. Semantic Normalization
# Standardizes text (Unicode, whitespace) to improve LLM extraction accuracy
normalizer = TextNormalizer()
clean_text = normalizer.normalize(result["full_text"])

# 3. Knowledge Extraction
# Semantica extracts semantic triplets (Subject-Predicate-Object) from the parsed structure
extractor = TripletExtractor()
triplets = extractor.extract_triplets(clean_text)

for triplet in triplets[:3]:
    print(f"Extracted: {triplet.subject} --({triplet.predicate})--> {triplet.object}")
```

!!! tip "Real-World Finance Use Case"
    For a complete end-to-end example showing how to build a Knowledge Graph from Finance Earnings Calls using Docling and Semantica, see the [Earnings Call Analysis notebook](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/use_cases/finance/03_Earnings_Call_Analysis.ipynb).

---

*Transform chaotic data into intelligent knowledge with Semantica and Docling.*
