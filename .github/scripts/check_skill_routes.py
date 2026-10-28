# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def check_routes(repo_root: Path) -> list[str]:
    skills_root = repo_root / ".agents/skills"
    router = skills_root / "skill-router.json"
    try:
        routes = json.loads(router.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return [f"Cannot read {router}: {error}"]

    if not isinstance(routes, dict) or not routes:
        return ["Skill routes must be a non-empty object of skill names and tasks."]
    if any(
        not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name)
        or not isinstance(task, str)
        or not task.strip()
        for name, task in routes.items()
    ):
        return ["Each route needs a skill directory name and a non-empty task."]

    errors: list[str] = []
    discovered = {path.name for path in skills_root.iterdir() if path.is_dir()}
    for name in sorted(discovered - routes.keys()):
        errors.append(f"Contributor skill has no task route: {name}")
    for name in sorted(routes):
        skill = skills_root / name
        if not (skill / "SKILL.md").is_file():
            errors.append(f"Route has no SKILL.md: {name}")
        for agent in ("codex", "claude"):
            link = repo_root / f".{agent}/skills" / name
            if not link.is_symlink() or link.resolve() != skill.resolve():
                errors.append(f".{agent}/skills/{name} must link to {skill}")

    for agent in ("codex", "claude"):
        links = repo_root / f".{agent}/skills"
        if links.is_dir():
            for link in sorted(links.iterdir()):
                if link.name not in routes:
                    errors.append(
                        f"Agent skill has no task route: .{agent}/skills/{link.name}"
                    )
    return errors


def main() -> int:
    errors = check_routes(Path(__file__).resolve().parents[2])
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("Contributor skill routes and agent links are valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
