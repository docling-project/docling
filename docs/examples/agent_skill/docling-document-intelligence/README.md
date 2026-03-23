# Docling agent skill (Cursor & compatible assistants)

This folder is an **[Agent Skill](https://agentskills.io/specification)**-style bundle for AI coding assistants: structured instructions (`SKILL.md`), a pipeline reference (`pipelines.md`), helper scripts under `scripts/`, and an evaluator for conversion quality.

It complements the official [Docling documentation](https://docling-project.github.io/docling/) and the [`docling` CLI](https://docling-project.github.io/docling/reference/cli/); use it when you want agents to follow a consistent **convert → export JSON → evaluate → refine** workflow.

The same layout is published in the Docling repo at `docs/examples/agent_skill/docling-document-intelligence/` (for docs and PRs).

## Contents

| Path | Purpose |
|------|---------|
| [`SKILL.md`](SKILL.md) | Full skill instructions (pipelines, chunking, evaluation loop) |
| [`pipelines.md`](pipelines.md) | Standard vs VLM pipelines, OCR engines, API notes |
| [`EXAMPLE.md`](EXAMPLE.md) | Installing into `~/.cursor/skills/`; running scripts |
| [`improvement-log.md`](improvement-log.md) | Optional template for local “what worked” notes |
| [`scripts/docling-convert.py`](scripts/docling-convert.py) | CLI: Markdown / JSON / RAG chunks |
| [`scripts/docling-evaluate.py`](scripts/docling-evaluate.py) | Heuristic quality report on JSON (+ optional Markdown) |
| [`scripts/requirements.txt`](scripts/requirements.txt) | Minimal pip deps for the scripts |

## Quick start (from this directory)

```bash
pip install -r scripts/requirements.txt
python3 scripts/docling-convert.py https://arxiv.org/pdf/2408.09869 --out /tmp/out.md
python3 scripts/docling-convert.py https://arxiv.org/pdf/2408.09869 --format json --out /tmp/out.json
python3 scripts/docling-evaluate.py /tmp/out.json --markdown /tmp/out.md
```

Use `--pipeline vlm-local` or `--pipeline vlm-api` for vision-model pipelines; see `SKILL.md` and `pipelines.md`.

## License

MIT (aligned with [Docling](https://github.com/docling-project/docling)).
