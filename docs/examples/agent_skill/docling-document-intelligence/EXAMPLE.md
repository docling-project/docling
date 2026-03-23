# Using the Docling agent skill

[Agent Skills](https://agentskills.io/specification) are folders of instructions that AI coding agents (Cursor, Claude Code, GitHub Copilot, etc.) can load when relevant. This bundle lives in the Docling repo at:

`docs/examples/agent_skill/docling-document-intelligence/`

## Install (copy into your agent’s skills directory)

```bash
# From a checkout of github.com/docling-project/docling
cp -r docs/examples/agent_skill/docling-document-intelligence ~/.cursor/skills/
# or e.g. ~/.claude/skills/ depending on your tool
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

The agent should read `SKILL.md`, match the task, and run the appropriate pipeline.

## Running the helper scripts directly

From the **bundle root** (the `docling-document-intelligence` directory):

```bash
pip install -r scripts/requirements.txt

python3 scripts/docling-convert.py report.pdf

python3 scripts/docling-convert.py report.pdf --ocr-engine rapidocr

python3 scripts/docling-convert.py report.pdf --format chunks --max-tokens 512

python3 scripts/docling-convert.py scanned.pdf --pipeline vlm-local

python3 scripts/docling-convert.py doc.pdf \
    --pipeline vlm-api \
    --vlm-api-url http://localhost:8000/v1/chat/completions \
    --vlm-api-model ibm-granite/granite-docling-258M
```

## Evaluate and refine

```bash
python3 scripts/docling-convert.py report.pdf --format json --out /tmp/doc.json
python3 scripts/docling-convert.py report.pdf --format markdown --out /tmp/doc.md
python3 scripts/docling-evaluate.py /tmp/doc.json --markdown /tmp/doc.md
```

If the report shows `warn` or `fail`, follow `recommended_actions`, re-convert,
and optionally append a note to `improvement-log.md` (see `SKILL.md` section 6).

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
- [Docling GitHub](https://github.com/docling-project/docling)
