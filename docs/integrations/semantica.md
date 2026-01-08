# 🧠 Semantica

Docling is integrated into [Semantica](https://github.com/Hawksight-AI/semantica) as a high-fidelity document parsing engine.

## What is Semantica?

Semantica is an open-source framework for building **Semantic Layers and Knowledge Engineering** systems. It transforms unstructured data into AI-ready knowledge graphs that power:

- **GraphRAG**: Knowledge Graph-Powered Retrieval-Augmented Generation.
- **AI Agents**: Providing structured semantic memory for autonomous agents.
- **Multi-Agent Systems**: Enabling consistent shared knowledge across agent swarms.

By bridging the gap between raw data and AI engineering, Semantica enables developers to build more reliable and context-aware AI applications.

## Installation

To use Docling with Semantica, install the `semantica` package (version 0.1.1 or higher) and `docling`:

```bash
pip install "semantica>=0.1.1" docling
```

## Components

### Docling Parser

Semantica provides a dedicated `DoclingParser` that leverages Docling's advanced PDF understanding, layout analysis, and table extraction capabilities. It seamlessly converts various document formats into Semantica's unified internal representation for downstream RAG and agentic workflows.

## Usage

You can use the `DoclingParser` to extract high-fidelity Markdown and structured tables from your documents:

```python
from semantica.parse import DoclingParser

# Initialize the parser
parser = DoclingParser()

# Parse a document (PDF, DOCX, etc.)
result = parser.parse("path/to/document.pdf")

# Access the extracted content
print(f"Markdown Content:\n{result.markdown[:500]}...")
print(f"Tables found: {len(result.tables)}")

# Or extract tables directly as DataFrames
tables = parser.extract_tables("path/to/document.pdf")
for i, df in enumerate(tables):
    print(f"Table {i+1}:\n{df.head()}")
```

- 💻 [Semantica GitHub][github]
- 📖 [Semantica Docs][docs]
- 📦 [Semantica PyPI][pypi]

[github]: https://github.com/Hawksight-AI/semantica
[docs]: https://semantica.ai/docs
[pypi]: https://pypi.org/project/semantica/
