Agent [skills](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills) are self-contained instruction files that teach an AI coding agent how to use a tool. Docling ships a **usage skill** inside its Python package so that once Docling is installed, agents can discover it automatically and learn — from documentation authored by the Docling project — how to convert, extract, and chunk documents correctly.

## Installing and discovering the skill

Use [`library-skills`](https://library-skills.io) — a small CLI tool that scans
your project's installed dependencies, finds any bundled skills, and installs
them as **symbolic links** into your agent's skills directory. Because symlinks
are used, the skill content updates automatically whenever you upgrade Docling.

### If you already use Docling in your project (recommended)

If `docling` is already a dependency in your project, run:

=== "Most agents (Codex, Cursor, Copilot, …)"

    ```bash
    uvx library-skills
    ```

=== "Claude Code"

    Claude Code uses `.claude/` instead of `.agents/`. Pass `--claude` so the
    skill is installed into `.claude/skills`:

    ```bash
    uvx library-skills --claude
    ```

`library-skills` reads your `pyproject.toml`, scans the project environment for
installed packages, and creates a symlink to Docling's bundled skill in
`.agents/skills/docling` (or `.claude/skills/docling`). That's all there is to it.

#### If Docling is not yet in your project

If Docling is not yet a dependency, `library-skills` won't find it automatically.
Install Docling first (`uv add docling` / `pip install docling`), then run
`uvx library-skills` as above.

### Manual path registration

If your agent runtime does not yet support the `.agents/` directory convention,
locate the skill directory and register it manually:

```bash
python -c "import importlib.util, pathlib; \
print(pathlib.Path(importlib.util.find_spec('docling').origin).parent / '.agents/skills/docling')"
```

Then point your agent at the printed path (or at `SKILL.md` inside it).

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
