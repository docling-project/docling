from __future__ import annotations

import argparse
import json
from typing import Any

SUCCESS = "success"
SKIPPED = "skipped"


def parse_allowed_skips(raw_allowed_skips: str) -> set[str]:
    return {job for job in raw_allowed_skips.split() if job}


def parse_needs(raw_needs: str) -> dict[str, Any]:
    loaded = json.loads(raw_needs)
    if not isinstance(loaded, dict):
        msg = "--needs-json must decode to a JSON object."
        raise ValueError(msg)
    return loaded


def result_for_job(job: str, value: Any) -> str:
    if not isinstance(value, dict):
        msg = f"{job}: needs entry must be a JSON object."
        raise ValueError(msg)

    result = value.get("result")
    if not isinstance(result, str):
        msg = f"{job}: needs entry must contain a string result."
        raise ValueError(msg)

    return result


def collect_failures(needs: dict[str, Any], allowed_skips: set[str]) -> list[str]:
    failures: list[str] = []
    for job, value in needs.items():
        result = result_for_job(job, value)
        if result == SUCCESS:
            continue
        if result == SKIPPED and job in allowed_skips:
            print(f"::notice title=Allowed skipped job::{job}")
            continue
        failures.append(f"{job}={result}")
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail unless all required GitHub Actions needs succeeded."
    )
    parser.add_argument("--needs-json", required=True)
    parser.add_argument("--allowed-skips", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    needs = parse_needs(args.needs_json)
    allowed_skips = parse_allowed_skips(args.allowed_skips)
    failures = collect_failures(needs, allowed_skips)
    if failures:
        print(f"::error title=Required jobs failed::{', '.join(failures)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
