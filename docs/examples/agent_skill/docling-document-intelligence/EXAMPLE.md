# Using the Docling agent skill

[Agent Skills](https://agentskills.io/specification) are folders of instructions that AI coding agents (Cursor, Claude Code, GitHub Copilot, etc.) can load when relevant.

## Where this bundle lives

- **Cursor (local):** `~/.cursor/skills/docling-document-intelligence/` (or copy this folder there).
- **Docling repository (docs + PRs):** `docs/examples/agent_skill/docling-document-intelligence/` in [github.com/docling-project/docling](https://github.com/docling-project/docling).

The two trees are kept in sync; use either source.

## Install (copy into your agent's skills directory)

```bash
# From a checkout of the Docling repo
cp -r docs/examples/agent_skill/docling-document-intelligence ~/.cursor/skills/

# Or copy from another machine / archive into e.g. ~/.claude/skills/
```

No extra config is required beyond installing Python dependencies (below).

## Usage

Open your agent-enabled IDE and ask, for example:

```
Parse report.pdf and give me a structural outline
```

```
Convert https://arxiv.org/pdf/2408.09869 to markdown
```

```
Chunk invoice.pdf for RAG ingestion with 512 token chunks
```

```
Process scanned.pdf using the VLM pipeline
```

The agent should read `SKILL.md`, match the task, and run the appropriate
`docling` CLI command or Python API call.

## Running the docling CLI directly

```bash
pip install docling docling-core

# Basic conversion to Markdown
docling report.pdf --output /tmp/

# JSON output
docling report.pdf --to json --output /tmp/

# Custom OCR engine
docling report.pdf --ocr-engine rapidocr --output /tmp/

# VLM pipeline
docling scanned.pdf --pipeline vlm --output /tmp/

# VLM with specific model
docling scanned.pdf --pipeline vlm --vlm-model granite_docling --output /tmp/

# Remote VLM services
docling doc.pdf --pipeline vlm --enable-remote-services --output /tmp/
```

## Evaluate and refine

```bash
docling report.pdf --to json --output /tmp/
docling report.pdf --to md --output /tmp/
python3 scripts/docling-evaluate.py /tmp/report.json --markdown /tmp/report.md
```

If the report shows `warn` or `fail`, follow `recommended_actions`, re-convert
with `docling` using the suggested flags, and optionally append a note to
`improvement-log.md` (see `SKILL.md` section 7).

## What the skill covers

| Task | How to ask |
|---|---|
| Parse PDF / DOCX / PPTX / HTML / image | "parse this file" |
| Convert to Markdown | "convert to markdown" |
| Export as structured JSON | "export as JSON" |
| Chunk for RAG | "chunk for RAG", "prepare for ingestion" |
| Analyze structure | "show me the headings and tables" |
| Use VLM pipeline | "use the VLM pipeline", "process scanned PDF" |
| Use remote inference | "use vLLM", "call the API pipeline" |

## Further reading

- [Agent Skills specification](https://agentskills.io/specification)
- [Docling documentation](https://docling-project.github.io/docling/)
- [Docling CLI reference](https://docling-project.github.io/docling/reference/cli/)
- [Docling GitHub](https://github.com/docling-project/docling)
