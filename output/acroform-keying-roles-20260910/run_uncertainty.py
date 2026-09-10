"""Render experimental decisions with close alternatives explicitly withheld."""

import html
import json
import sys
from pathlib import Path

from run_experiment import replay_acroform_keying
from uncertainty import audit_assignment

records = {}
original_page_report = replay_acroform_keying.page_report


def assign(snapshot, **kwargs):
    result, diagnostic = audit_assignment(snapshot, **kwargs)
    records[id(result)] = diagnostic
    return result


def page_report(path, image, snapshot, assignment, reviews):
    original_page_report(path, image, snapshot, assignment, reviews)
    diagnostic = records.pop(id(assignment))
    evidence = path.with_suffix(".group-audit.json")
    evidence.write_text(json.dumps(diagnostic, ensure_ascii=False, indent=2))
    summaries = "".join(
        "<li>"
        + html.escape(str(g["selected"]["widgets"]))
        + " — "
        + ("uncertain; withheld" if g["uncertain"] else "retained")
        + "; alternative objective gap "
        + str(None if g["objective_gap"] is None else round(g["objective_gap"], 3))
        + "</li>"
        for g in diagnostic["groups"]
    )
    note = (
        '<section style="padding:16px;background:#fff3cd"><strong>Grouping experiment with uncertainty filtering.</strong><p>Solver status/objective describe the raw optimization before withholding. Close alternatives within 0.25 cost units are not treated as a resolved group; disputed local captions are also withheld. This margin is experimental, not a probability.</p><ul>'
        + summaries
        + '</ul><a href="'
        + evidence.name
        + '">Full competing configurations</a></section>'
    )
    path.write_text(path.read_text() + note)


replay_acroform_keying.assign = assign
replay_acroform_keying.page_report = page_report
if __name__ == "__main__":
    sys.argv[1:] = [
        "--out",
        str(Path(__file__).parent / "uncertainty-report"),
        *sys.argv[1:],
    ]
    replay_acroform_keying.main()
