"""Check text independence and retain a reproducible source fingerprint."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

from run_experiment import module

folder = Path(__file__).parent
checks = []
for path in sorted(
    Path("output/acroform-keying-review-20260908/snapshots").glob("*/*.json")
):
    page = module.Snapshot.model_validate_json(path.read_text())
    values, labels, h = module.inputs(page)
    proposals = module.choice_configurations(values, labels, h)
    renamed_values = [
        replace(
            v,
            native=v.native.model_copy(
                update={"widget_field_name": "?", "widget_description": "?"}
            ),
        )
        for v in values
    ]
    renamed_labels = [replace(label, text="?") for label in labels]
    assert proposals == module.choice_configurations(renamed_values, renamed_labels, h)
    checks.append(
        {"page": f"{path.parent.name}-p{page.page}", "groups": len(proposals)}
    )
assert len(checks) == 19
(folder / "geometry-checks.json").write_text(json.dumps(checks, indent=2))
(folder / "source-hashes.json").write_text(
    json.dumps(
        {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.glob("*.py"))
        },
        indent=2,
    )
)
print(
    "All 19 group proposal sets unchanged by replacing visible text and native names/descriptions."
)
