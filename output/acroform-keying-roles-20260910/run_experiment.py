"""Replay the isolated grouping experiment through the existing evaluator/report."""

import html
import importlib.util
import json
import sys
from pathlib import Path

from scripts import acroform_keying, replay_acroform_keying

folder = Path(__file__).parent
spec = importlib.util.spec_from_file_location(
    "grouping_experiment", folder / "experimental_algorithm.py"
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
# The evaluator compares Scope values; use the shared identity across modules.
module.Scope = acroform_keying.Scope
replay_acroform_keying.assign = module.assign
original_page_report = replay_acroform_keying.page_report


def page_report(path, image, snapshot, assignment, reviews):
    original_page_report(path, image, snapshot, assignment, reviews)
    record_path = path.with_suffix(".json")
    record = json.loads(record_path.read_text())
    refs = []
    for field in record["fields"]:
        candidate = assignment.candidates[field["id"]]
        if candidate.parent is None:
            continue
        parent_ref = next(
            j
            for j in assignment.selected
            if assignment.candidates[j].kind != "choice_group"
            and candidate.parent in assignment.candidates[j].members
            and assignment.candidates[j].label == candidate.label
        )
        field["parent_widget_index"] = assignment.values[candidate.parent].native.index
        field["parent_field_ref"] = parent_ref
        refs.append(
            f"Child group {field['widget_indices']} references widget {field['parent_widget_index']}'s caption (field {parent_ref})."
        )
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
    if refs:
        path.write_text(
            path.read_text()
            + "<section><h2>Parent option references</h2><p>"
            + html.escape(" ".join(refs))
            + "</p></section>"
        )


replay_acroform_keying.page_report = page_report
if __name__ == "__main__":
    sys.argv[1:] = ["--out", str(folder / "report"), *sys.argv[1:]]
    replay_acroform_keying.main()
