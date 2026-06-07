"""Generate the CLI reference page for the docs.

Replaces mkdocs-click, which is not on the Zensical compatibility roadmap.
Introspects docling's Typer-derived Click command tree and emits a markdown
file with:

- one section per (sub)command,
- a usage line,
- the description / help text,
- a markdown table of positional arguments (if any),
- a markdown table of options (if any),
- for groups, a table of subcommands with short descriptions.

This is style-equivalent to mkdocs-click's ``style: table`` output. Writes to
``docs/reference/cli.md``; the file is gitignored.
"""

from __future__ import annotations

import argparse
import enum
import inspect
import re
import sys
from pathlib import Path, PurePath

import click
import typer

from docling.cli.main import click_app as docling_click_app
from docling.cli.tools import app as docling_tools_typer_app

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_FILE = REPO_ROOT / "docs" / "reference" / "cli.md"

_TOOLS_CLICK_APP = typer.main.get_command(docling_tools_typer_app)


# ---------------------------------------------------------------------------
# Markdown formatting helpers
# ---------------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"\s+")


def _md_cell(text: str | None) -> str:
    """Sanitize text so it can live inside a markdown table cell."""
    if not text:
        return ""
    # Collapse whitespace (including newlines) to single spaces so multi-line
    # help text fits on one row. Escape pipe characters so they don't break
    # the column boundary.
    flat = _WHITESPACE_RE.sub(" ", text).strip()
    return flat.replace("|", "\\|")


def _code(value: str) -> str:
    return f"`{value}`"


def _format_choices(choices: tuple[str, ...] | list[str]) -> str:
    return ", ".join(_code(c) for c in choices)


def _format_type(param: click.Parameter) -> str:
    """Human-readable type for the Type column."""
    t = param.type
    if isinstance(t, click.Choice):
        rendered = _format_choices(t.choices)
    elif isinstance(t, click.Tuple):
        parts = [getattr(sub, "name", "value") for sub in t.types]
        rendered = " ".join(_code(p) for p in parts)
    elif getattr(param, "is_flag", False):
        rendered = "flag"
    else:
        rendered = _code(getattr(t, "name", str(t)))

    if param.multiple:
        rendered = f"{rendered} (repeatable)"
    return rendered


def _format_default(param: click.Parameter) -> str:
    default = param.default
    if default is None:
        return ""
    if callable(default):
        # Click uses callables for dynamic defaults (e.g. functions). Showing
        # the repr of a function isn't useful, so omit.
        return ""
    if isinstance(default, bool):
        return _code(str(default).lower())
    if default == () or default == []:
        return ""
    # Enum.value is the underlying primitive (str / int) used by the CLI.
    # ``isinstance(default, enum.Enum)`` covers StrEnum / IntEnum subclasses too.
    if isinstance(default, enum.Enum):
        return _code(str(default.value))
    if isinstance(default, PurePath):
        return _code(str(default))
    if isinstance(default, str):
        return _code(default)
    return _code(repr(default))


def _format_option_names(param: click.Option) -> str:
    """Render option names like ``--foo / --no-foo`` as code."""
    names = list(param.opts) + list(param.secondary_opts)
    return " / ".join(_code(n) for n in names)


def _format_argument_name(param: click.Argument) -> str:
    return _code(param.human_readable_name)


def _required_marker(param: click.Parameter) -> str:
    return "yes" if param.required else "no"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _usage_line(cmd: click.Command, prog: str) -> str:
    ctx = click.Context(cmd, info_name=prog)
    pieces = cmd.collect_usage_pieces(ctx)
    return f"{prog} {' '.join(pieces)}".strip()


def _description(cmd: click.Command) -> str:
    text = cmd.help or cmd.short_help or ""
    return inspect.cleandoc(text).strip()


def _render_arguments_table(args: list[click.Argument], lines: list[str]) -> None:
    if not args:
        return
    lines.append("**Arguments**")
    lines.append("")
    lines.append("| Name | Type | Required | Description |")
    lines.append("| --- | --- | --- | --- |")
    for p in args:
        lines.append(
            "| {name} | {ty} | {req} | {desc} |".format(
                name=_md_cell(_format_argument_name(p)),
                ty=_md_cell(_format_type(p)),
                req=_required_marker(p),
                # Click Arguments don't carry a help string; Typer adds one via
                # the metavar / via the underlying parameter info. Fall back to
                # an empty cell when missing.
                desc=_md_cell(getattr(p, "help", "") or ""),
            )
        )
    lines.append("")


def _render_options_table(opts: list[click.Option], lines: list[str]) -> None:
    if not opts:
        return
    lines.append("**Options**")
    lines.append("")
    lines.append("| Name | Type | Default | Description |")
    lines.append("| --- | --- | --- | --- |")
    for p in opts:
        lines.append(
            "| {name} | {ty} | {default} | {desc} |".format(
                name=_md_cell(_format_option_names(p)),
                ty=_md_cell(_format_type(p)),
                default=_md_cell(_format_default(p)),
                desc=_md_cell(p.help or ""),
            )
        )
    lines.append("")


def _render_subcommands_table(
    group: click.Group, prog_path: list[str], lines: list[str]
) -> None:
    ctx = click.Context(group, info_name=prog_path[-1])
    rows: list[tuple[str, str]] = []
    for name in sorted(group.list_commands(ctx)):
        sub = group.get_command(ctx, name)
        if sub is None or sub.hidden:
            continue
        full = " ".join([*prog_path, name])
        desc = sub.short_help or sub.help or ""
        # Take only the first sentence/line for the subcommand table.
        first_line = desc.strip().splitlines()[0] if desc.strip() else ""
        rows.append((_code(full), _md_cell(first_line)))
    if not rows:
        return
    lines.append("**Subcommands**")
    lines.append("")
    lines.append("| Command | Description |")
    lines.append("| --- | --- |")
    for cmd_name, desc in rows:
        lines.append(f"| {cmd_name} | {desc} |")
    lines.append("")


def _render_command(
    cmd: click.Command, prog_path: list[str], lines: list[str], depth: int
) -> None:
    full_name = " ".join(prog_path)
    heading = "#" * min(2 + depth, 6)
    lines.append(f"{heading} `{full_name}`")
    lines.append("")

    description = _description(cmd)
    if description:
        lines.append(description)
        lines.append("")

    lines.append("**Usage**")
    lines.append("")
    lines.append("```text")
    lines.append(_usage_line(cmd, full_name))
    lines.append("```")
    lines.append("")

    args = [p for p in cmd.params if isinstance(p, click.Argument)]
    opts = [p for p in cmd.params if isinstance(p, click.Option) and not p.hidden]
    _render_arguments_table(args, lines)
    _render_options_table(opts, lines)

    if isinstance(cmd, click.Group):
        _render_subcommands_table(cmd, prog_path, lines)
        ctx = click.Context(cmd, info_name=prog_path[-1])
        for sub_name in sorted(cmd.list_commands(ctx)):
            sub = cmd.get_command(ctx, sub_name)
            if sub is None or sub.hidden:
                continue
            _render_command(sub, [*prog_path, sub_name], lines, depth + 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUT_FILE,
        help="Path to write the generated markdown (default: %(default)s).",
    )
    args = parser.parse_args()

    lines: list[str] = [
        "# CLI reference",
        "",
        "This page documents Docling's command line tools. It is generated by",
        "`scripts/render_cli_reference.py` from the live Typer apps — do not",
        "edit by hand.",
        "",
    ]

    _render_command(docling_click_app, ["docling"], lines, depth=0)
    _render_command(_TOOLS_CLICK_APP, ["docling-tools"], lines, depth=0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.output.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
