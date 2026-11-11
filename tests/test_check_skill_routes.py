# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[1] / ".github/scripts/check_skill_routes.py"
)
SPEC = importlib.util.spec_from_file_location("check_skill_routes", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
skill_routes = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(skill_routes)


def make_skill(repo_root: Path, name: str) -> Path:
    skill = repo_root / ".agents/skills" / name
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# Review\n", encoding="utf-8")
    for agent in ("codex", "claude"):
        link = repo_root / f".{agent}/skills" / name
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(f"../../.agents/skills/{name}", target_is_directory=True)
    return skill


def test_routing_rejects_missing_skills_and_broken_discovery(tmp_path: Path) -> None:
    skill = make_skill(tmp_path, "review")
    (skill.parent / "skill-router.json").write_text(
        '{"review": "Review a PR"}', encoding="utf-8"
    )
    assert skill_routes.check_routes(tmp_path) == []

    (skill / "SKILL.md").unlink()
    assert "Route has no SKILL.md: review" in skill_routes.check_routes(tmp_path)
    (skill / "SKILL.md").write_text("# Review\n", encoding="utf-8")
    link = tmp_path / ".codex/skills/review"
    link.unlink()
    link.symlink_to("../../.agents/skills/missing", target_is_directory=True)
    assert any(
        ".codex/skills/review must link" in error
        for error in skill_routes.check_routes(tmp_path)
    )


def test_routing_rejects_unrouted_contributor_and_agent_skills(tmp_path: Path) -> None:
    skill = make_skill(tmp_path, "review")
    (skill.parent / "skill-router.json").write_text(
        '{"review": "Review a PR"}', encoding="utf-8"
    )
    make_skill(tmp_path, "new-skill")
    errors = skill_routes.check_routes(tmp_path)
    assert "Contributor skill has no task route: new-skill" in errors
    assert "Agent skill has no task route: .claude/skills/new-skill" in errors


@pytest.mark.parametrize(
    "content", ["{", "[]", "{}", '{"../escape": "Task"}', '{"review": null}']
)
def test_routing_rejects_invalid_router(tmp_path: Path, content: str) -> None:
    skills = tmp_path / ".agents/skills"
    skills.mkdir(parents=True)
    (skills / "skill-router.json").write_text(content, encoding="utf-8")
    assert skill_routes.check_routes(tmp_path)
