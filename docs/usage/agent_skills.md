Agent [skills](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills) are self-contained instructions that teach an AI coding agent how to use a tool. Docling ships a **usage skill** inside its Python package, so that once Docling is installed, agents can discover it and learn — from the docs authored by the Docling project — how to convert, extract, and chunk documents correctly.

## The Docling usage skill

The skill lives inside the installed package at:

```text
docling/.agents/skills/docling/
├── SKILL.md                     # entry point: what Docling is + when to use each path
└── references/
    ├── cli.md                   # convert any format from the command line
    ├── python-sdk.md            # DocumentConverter + PipelineOptions, batch, ASR, image/table export
    ├── extraction.md            # DocumentExtractor — pull typed fields out of a document (beta)
    ├── rag.md                   # chunking + LangChain / LlamaIndex / Haystack loaders
    ├── service-client.md        # remote conversion via docling-serve (self-hosted or managed)
    └── slim-packaging.md        # docling-slim install extras
```

`SKILL.md` is a short router; the `references/*.md` files are loaded on demand,
so an agent reads only what the current task needs.

## Installing and discovering the skill

The skill is packaged into the `docling` wheel and source distribution — there
is nothing extra to install. Any of the following makes it available on disk:

```bash
pip install docling
# or, to run without a persistent install:
uvx --from docling docling --help
```

Agent runtimes that follow the emerging **library-skills** convention look for
skills under a package's `.agents/skills/` directory and expose them
automatically once the package is importable. Because the exact discovery
mechanism differs between tools and is still evolving, consult your agent
framework's documentation for how it loads packaged skills. The `SKILL.md`
frontmatter (`name`, `description`) is what those runtimes use to decide when the
skill is relevant.

You can always point an agent at the files directly — locate them with:

```bash
python -c "import importlib.util, pathlib; \
print(pathlib.Path(importlib.util.find_spec('docling').origin).parent / '.agents/skills/docling')"
```

## What the skill covers

The skill routes an agent to the right Docling entry point for the task:

| Task | Skill reference |
|---|---|
| Read/convert a file from the shell | `cli.md` |
| Convert in code and tune the pipeline (`PipelineOptions`) | `python-sdk.md` |
| Extract specific typed fields from a document | `extraction.md` |
| Chunk documents for RAG | `rag.md` |
| Offload conversion to a remote service | `service-client.md` |
| Install only the dependencies you need | `slim-packaging.md` |

## Usage vs. development skills

The skill described here is a **usage** skill — it helps agents *use* Docling.
Contributors working *on* Docling use separate **development** skills kept in the
repository's own `.agents/skills/` directory; those are not shipped in the
package.
