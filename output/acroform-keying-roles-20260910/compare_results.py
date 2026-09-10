"""Verify replay invariants and enumerate changes against accepted coverage replay."""

import hashlib
import json
from pathlib import Path

folder = Path(__file__).parent
baseline = Path("output/acroform-keying-coverage-20260910/report")
results = {}
for variant in ("report", "uncertainty-report"):
    changes, groups = [], []
    for path in sorted(baseline.glob("*.json")):
        if path.name == "summary.json":
            continue
        old = json.loads(path.read_text())
        new = json.loads((folder / variant / path.name).read_text())
        assert new["solver_status"] == "optimal"
        assert [v["native"] for v in old["ordered_values"]] == [
            v["native"] for v in new["ordered_values"]
        ]
        assert [
            (r["widget_index"], r["reference_disposition"]) for r in old["reviews"]
        ] == [(r["widget_index"], r["reference_disposition"]) for r in new["reviews"]]
        assert [
            r["widget_index"]
            for r in old["reviews"]
            if r["status"] == "excluded by table rule"
        ] == [
            r["widget_index"]
            for r in new["reviews"]
            if r["status"] == "excluded by table rule"
        ]
        for before, after in zip(old["reviews"], new["reviews"]):
            if (
                before["status"] != after["status"]
                or before["predicted"] != after["predicted"]
            ):
                changes.append(
                    {
                        "page": path.stem,
                        "widget": after["widget_index"],
                        "before": before["status"],
                        "after": after["status"],
                        "expected": after["expected"],
                        "old_text": before["predicted"],
                        "new_text": after["predicted"],
                    }
                )
        groups.append(
            {
                "page": path.stem,
                "before": [g for g in old["fields"] if g["kind"] == "choice_group"],
                "after": [g for g in new["fields"] if g["kind"] == "choice_group"],
            }
        )
    results[variant] = {"changes": changes, "groups": groups}
    print(variant)
    for row in changes:
        if row["before"] != row["after"]:
            print(row["page"], row["widget"], row["before"], "->", row["after"])
assert (
    Path("scripts/acroform_keying.py").read_bytes()
    == (
        folder.parent / "acroform-keying-grouping-20260910/baseline_algorithm.py"
    ).read_bytes()
)
checks = json.loads(
    Path("output/acroform-keying-failure-audit/input-hashes.json").read_text()
)
for name, sha in checks.items():
    if name == "scripts/acroform_keying.py":
        continue
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == sha, name
(folder / "comparison.json").write_text(
    json.dumps(results, ensure_ascii=False, indent=2)
)
print(
    "All 19 pages preserve native values/order and table exclusions; accepted code and reference inputs unchanged."
)
