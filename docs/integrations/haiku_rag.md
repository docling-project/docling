Docling is the default document converter and chunker in [haiku.rag](https://ggozad.github.io/haiku.rag/), an agentic RAG library. haiku.rag stores the full `DoclingDocument` and uses it for structure-aware context expansion and visual grounding, and mounts its structure as a virtual filesystem so agents can navigate a document's outline and items directly when analyzing it. Both `docling` and [Docling Serve](https://github.com/docling-project/docling-serve) backends are supported.

- 💻 [haiku.rag GitHub][github]
- 📖 [haiku.rag docs][docs]
- 📦 [haiku.rag PyPI][pypi]
- ⚙️ [Docling setup in haiku.rag][setup]
- 🌐 [Docling Serve setup in haiku.rag][serve]

[github]: https://github.com/ggozad/haiku.rag
[docs]: https://ggozad.github.io/haiku.rag/
[pypi]: https://pypi.org/project/haiku.rag/
[setup]: https://ggozad.github.io/haiku.rag/configuration/processing/
[serve]: https://ggozad.github.io/haiku.rag/remote-processing/
