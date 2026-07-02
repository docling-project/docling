"""Pre-render Jupyter notebooks and jupytext .py scripts under docs/examples.

Zensical does not provide an mkdocs-jupyter equivalent, so the example sources
are converted to markdown at docs build time and emitted to
``docs/_generated/examples`` preserving relative structure. The nav in
``mkdocs.yml`` references the generated markdown paths rather than the .ipynb
or .py sources.

Supported sources:
- ``*.ipynb`` — converted directly via ``nbconvert.MarkdownExporter``.
- ``*.py`` — only files in jupytext percent format (containing ``# %%`` cell
  markers) are converted; plain Python scripts are skipped.

Idempotent: a source is skipped if the corresponding output is newer. Pass
``--force`` to re-render unconditionally.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jupytext
from nbconvert import MarkdownExporter
from nbconvert.writers import FilesWriter

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "docs" / "examples"
OUT_DIR = REPO_ROOT / "docs" / "_generated" / "examples"

# Jupytext percent-format marker. Files that don't contain this near the top
# are treated as plain Python and skipped (they're consumed as code only).
PERCENT_MARKER = "# %%"


def _is_up_to_date(src: Path, dest: Path) -> bool:
    return dest.exists() and dest.stat().st_mtime >= src.stat().st_mtime


def _is_percent_format(path: Path) -> bool:
    with path.open("r", encoding="utf-8") as fp:
        head = fp.read(4096)
    return PERCENT_MARKER in head


def _write_markdown(body: str, resources: dict, dest_dir: Path, stem: str) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    writer = FilesWriter(build_directory=str(dest_dir))
    writer.write(body, resources, notebook_name=stem)


def _convert_ipynb(src: Path) -> Path:
    rel = src.relative_to(SRC_DIR).with_suffix("")
    dest_dir = OUT_DIR / rel.parent
    body, resources = MarkdownExporter().from_filename(str(src))
    _write_markdown(body, resources, dest_dir, rel.name)
    return dest_dir / f"{rel.name}.md"


def _convert_py_percent(src: Path) -> Path:
    rel = src.relative_to(SRC_DIR).with_suffix("")
    dest_dir = OUT_DIR / rel.parent
    nb = jupytext.read(src, fmt="py:percent")
    # Jupytext percent-format .py files have no kernel metadata. Without
    # language_info.name set, nbconvert emits bare ``` fences for code cells,
    # which means no syntax highlighting in the rendered docs site. Stamp the
    # notebook as Python so fences come out as ```python.
    nb.metadata.setdefault("language_info", {})
    nb.metadata["language_info"].setdefault("name", "python")
    body, resources = MarkdownExporter().from_notebook_node(nb)
    _write_markdown(body, resources, dest_dir, rel.name)
    return dest_dir / f"{rel.name}.md"


def _iter_sources() -> list[Path]:
    sources: list[Path] = []
    for path in sorted(SRC_DIR.rglob("*")):
        if ".ipynb_checkpoints" in path.parts:
            continue
        if path.suffix == ".ipynb":
            sources.append(path)
        elif path.suffix == ".py" and _is_percent_format(path):
            sources.append(path)
    return sources


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-render even if the output is newer than the source.",
    )
    args = parser.parse_args()

    if not SRC_DIR.is_dir():
        print(f"error: source directory not found: {SRC_DIR}", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sources = _iter_sources()
    rendered = 0
    skipped = 0
    for src in sources:
        rel = src.relative_to(SRC_DIR).with_suffix("")
        dest = OUT_DIR / rel.parent / f"{rel.name}.md"
        if not args.force and _is_up_to_date(src, dest):
            skipped += 1
            continue

        rel_in = src.relative_to(REPO_ROOT)
        rel_out = dest.relative_to(REPO_ROOT)
        print(f"render: {rel_in} -> {rel_out}")

        if src.suffix == ".ipynb":
            _convert_ipynb(src)
        else:
            _convert_py_percent(src)
        rendered += 1

    print(f"\n{rendered} rendered, {skipped} up-to-date, {len(sources)} total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
