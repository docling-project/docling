# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Shared helpers of the maintainer tools: gh calls, progress output, Chrome."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any

from tqdm import tqdm

DEFAULT_REPO = "docling-project/docling"
MAINTAINER_ROLES = {"OWNER", "MEMBER", "COLLABORATOR"}
GRAPHQL_ATTEMPTS = 3
CHROME_PROGRAMS = (
    "google-chrome",
    "google-chrome-stable",
    "chromium",
    "chromium-browser",
    "chrome",
)


class Progress:
    """Status lines and progress bars on stderr. ``--quiet`` turns them off.

    The report goes to stdout, so ``> report.md`` gives a clean file.
    """

    quiet = False

    @classmethod
    def info(cls, message: str) -> None:
        if not cls.quiet:
            tqdm.write(
                f"\N{BLACK RIGHT-POINTING SMALL TRIANGLE} {message}", file=sys.stderr
            )

    @classmethod
    def bar(cls, total: int, desc: str) -> tqdm:
        return tqdm(
            total=total,
            desc=desc,
            unit="PR",
            file=sys.stderr,
            disable=cls.quiet,
            dynamic_ncols=True,
        )


class HelpFormatter(
    argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
):
    """Keep the docstring layout and show the default of each option."""


def run(command: list[str], stdin: str | None = None) -> str:
    result = subprocess.run(
        command, input=stdin, check=True, capture_output=True, text=True
    )
    return result.stdout


def gh_json(args: list[str]) -> Any:
    return json.loads(run(["gh", *args]))


def gh_api_list(path: str, key: str | None = None) -> list[dict[str, Any]]:
    pages = gh_json(["api", "--paginate", "--slurp", path])
    items: list[dict[str, Any]] = []
    for page in pages:
        items.extend(page[key] if key else page)
    return items


def gh_login() -> str:
    """Return the login of the gh user."""
    return run(["gh", "api", "user", "--jq", ".login"]).strip()


def graphql(query: str, **variables: str | int | list[str]) -> dict[str, Any]:
    command = ["gh", "api", "graphql", "-f", f"query={query}"]
    for key, value in variables.items():
        if isinstance(value, list):
            # gh sends key[]=item as a list. An empty list sends null: no filter.
            for item in value:
                command += ["-f", f"{key}[]={item}"]
            continue
        flag = "-F" if isinstance(value, int) else "-f"
        command += [flag, f"{key}={value}"]
    for attempt in range(1, GRAPHQL_ATTEMPTS + 1):
        try:
            return gh_json(command[1:])["data"]
        except subprocess.CalledProcessError as exc:
            # Heavy queries sometimes time out on GitHub (HTTP 502).
            if attempt == GRAPHQL_ATTEMPTS or "HTTP 50" not in exc.stderr:
                raise
            error = exc.stderr.strip().splitlines()[-1]
            Progress.info(
                f"GitHub returned '{error}'. Try again"
                f" ({attempt + 1}/{GRAPHQL_ATTEMPTS}) in {2 * attempt} s."
            )
            time.sleep(2 * attempt)
    raise AssertionError("unreachable")


def graphql_prs(
    repo: str,
    numbers: list[int],
    fragment: str,
    fragment_name: str,
    batch_size: int,
    desc: str,
) -> dict[int, dict[str, Any]]:
    """Read PRs by number, ``batch_size`` PRs per query (one alias per PR).

    ``fragment`` is the GraphQL text of ``fragment <fragment_name> on
    PullRequest``. Return the data of each PR by its number.
    """
    owner, name = repo.split("/")
    result: dict[int, dict[str, Any]] = {}
    with Progress.bar(len(numbers), desc) as bar:
        for start in range(0, len(numbers), batch_size):
            batch = numbers[start : start + batch_size]
            fields = " ".join(
                f"pr{n}: pullRequest(number: {n}) {{ ...{fragment_name} }}"
                for n in batch
            )
            query = (
                fragment + "query($owner: String!, $name: String!) {"
                f" repository(owner: $owner, name: $name) {{ {fields} }} }}"
            )
            bar.set_postfix_str(f"#{batch[0]} ... #{batch[-1]}")
            data = graphql(query, owner=owner, name=name)["repository"]
            for number in batch:
                result[number] = data[f"pr{number}"]
            bar.update(len(batch))
    return result


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def age(since: datetime) -> str:
    hours = int((datetime.now(timezone.utc) - since).total_seconds() // 3600)
    if hours < 1:
        return "less than 1 h"
    return f"{hours // 24} d" if hours >= 48 else f"{hours} h"


def describe_labels(labels: list[str]) -> str:
    return f" with label {' or '.join(labels)}" if labels else ""


def open_in_chrome(urls: list[str]) -> None:
    """Open a new Google Chrome window with one tab for each URL."""
    if sys.platform == "darwin":
        # -n passes --args also when Chrome already runs.
        run(["open", "-na", "Google Chrome", "--args", "--new-window", *urls])
        return
    for name in CHROME_PROGRAMS:
        program = shutil.which(name)
        if program is None:
            continue
        # Chrome keeps running when it was not running before; do not wait.
        subprocess.Popen(
            [program, "--new-window", *urls],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return
    raise RuntimeError(
        "Google Chrome or Chromium not found on PATH: " + ", ".join(CHROME_PROGRAMS)
    )
