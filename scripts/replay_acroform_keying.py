# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Replay saved AcroForm pages and write an inspectable HTML/JSON report.

Run from the repository root:
    uv run --no-sync python -m scripts.replay_acroform_keying
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Literal

import click
from docling_core.types.doc import BoundingBox
from pydantic import BaseModel, Field, model_validator

from scripts.acroform_keying import (
    Assignment,
    NativeWidget,
    Snapshot,
    anchors,
    assign,
    inputs,
    overlap,
    regions,
    scope_of,
)


class ExpectedLabel(BaseModel):
    bbox: tuple[float, float, float, float]
    text_hint: str

    def box(self) -> BoundingBox:
        left, top, right, bottom = self.bbox
        return BoundingBox(l=left, t=top, r=right, b=bottom)


class Annotation(BaseModel):
    fixture: str
    page: int
    kind: str
    widget_index: int | None = None
    widget_indices: list[int] = Field(default_factory=list)
    relation: str = ""
    group_type: str = ""
    expected_label: ExpectedLabel | None = None


class FieldReview(BaseModel):
    fixture: str
    page: int
    widget_index: int
    disposition: Literal["label", "no_visible_label", "ambiguous", "excluded_table"]
    expected_label: ExpectedLabel | None = None
    reason: str
    detected_form_ids: list[int]
    table: int | None
    cell: int | None

    @model_validator(mode="after")
    def check_label(self) -> FieldReview:
        if (self.disposition == "label") != (self.expected_label is not None):
            raise ValueError("Only a positive label review must have expected_label")
        if not self.reason.strip():
            raise ValueError("A review requires a reason")
        return self


def validate_field_reviews(snapshot: Snapshot, reviews: list[FieldReview]) -> None:
    """Reject stale/incomplete references instead of silently dropping cases."""
    values, _, _ = inputs(snapshot)
    if [r.widget_index for r in reviews] != [v.native.index for v in values]:
        raise ValueError(
            "Field reviews must cover every retained widget in native order"
        )
    for value, review in zip(values, reviews):
        forms = [
            r.id
            for r in regions(snapshot)
            if r.label == "form"
            and overlap(r.bbox, value.bbox) > 0.8 * value.bbox.area()
        ]
        if (
            review.table != value.scope.table
            or review.cell != value.scope.cell
            or review.detected_form_ids != forms
            or (review.disposition == "excluded_table") == value.scope.eligible
        ):
            raise ValueError(f"Stale detection scope for widget #{review.widget_index}")
        if review.expected_label is not None:
            box = review.expected_label.box()
            if not (
                0 <= box.l < box.r <= snapshot.size.width
                and 0 <= box.t < box.b <= snapshot.size.height
            ):
                raise ValueError(f"Invalid label box for widget #{review.widget_index}")
            if scope_of(box, snapshot) != value.scope:
                raise ValueError(
                    f"Reference crosses table boundary: #{review.widget_index}"
                )


class Review(BaseModel):
    widget_index: int
    status: str
    expected: str
    predicted: str
    expected_bbox: BoundingBox | None = None
    predicted_bbox: BoundingBox | None = None
    features: dict[str, float] = Field(default_factory=dict)
    reference_disposition: str = "unreviewed"
    reason: str = ""
    detected_form_ids: list[int] = Field(default_factory=list)


class FieldAssociation(BaseModel):
    id: int
    kind: str
    widget_indices: list[int]
    text: str
    bbox: BoundingBox
    features: dict[str, float]


class OrderedValue(BaseModel):
    native: NativeWidget
    bbox: BoundingBox
    table: int | None
    cell: int | None
    field_refs: list[int]


class PageResult(BaseModel):
    schema_version: str = "1.0"
    fixture: str
    page: int
    solver_status: str
    seconds: float
    null_cost: float
    objective: float
    ordered_values: list[OrderedValue]
    fields: list[FieldAssociation]
    reviews: list[Review]
    counts: dict[str, int]
    groups: dict[str, int]


def associations(
    assignment: Assignment,
) -> tuple[list[OrderedValue], list[FieldAssociation]]:
    fields = [
        FieldAssociation(
            id=c,
            kind=assignment.candidates[c].kind,
            widget_indices=[
                assignment.values[i].native.index
                for i in assignment.candidates[c].members
            ],
            text=assignment.labels[assignment.candidates[c].label].text,
            bbox=assignment.labels[assignment.candidates[c].label].bbox,
            features=assignment.candidates[c].features,
        )
        for c in assignment.selected
    ]
    values = [
        OrderedValue(
            native=v.native,
            bbox=v.bbox,
            table=v.scope.table,
            cell=v.scope.cell,
            field_refs=[f.id for f in fields if v.native.index in f.widget_indices],
        )
        for v in assignment.values
    ]
    return values, fields


def label_matches(expected: BoundingBox, actual: BoundingBox) -> bool:
    """Spatial association only; text hints are not exact-text truth.

    Require coverage in both directions so a page-sized label cannot pass.
    Complete wording and semantic ambiguity still need human review.
    """
    area = overlap(expected, actual)
    return area >= 0.5 * expected.area() and area >= 0.4 * actual.area()


def evaluate(
    snapshot: Snapshot,
    assignment: Assignment,
    annotations: list[Annotation],
    field_reviews: list[FieldReview] | None = None,
) -> tuple[list[Review], Counter]:
    if field_reviews is not None:
        validate_field_reviews(snapshot, field_reviews)
    references = {r.widget_index: r for r in field_reviews or []}
    primary = {
        i: assignment.candidates[c]
        for c in assignment.selected
        if assignment.candidates[c].kind != "choice_group"
        for i in assignment.candidates[c].members
    }
    links = {a.widget_index: a for a in annotations if a.kind == "label_link"}
    reviews = []
    for i, value in enumerate(assignment.values):
        ann = links.get(value.native.index)
        candidate = primary.get(i)
        label = None if candidate is None else assignment.labels[candidate.label]
        expected = None if ann is None else ann.expected_label
        reference = references.get(value.native.index)
        if reference is not None:
            expected = reference.expected_label
        status = "unreviewed"
        if not value.scope.eligible:
            status = "excluded by table rule"
        elif reference is not None and reference.disposition == "ambiguous":
            status = "ambiguous"
        elif reference is not None and reference.disposition == "no_visible_label":
            status = (
                "correct abstention" if label is None else "wrong: expected no label"
            )
        elif expected is not None:
            expected_scope = scope_of(expected.box(), snapshot)
            if (
                not value.scope.eligible
                or not expected_scope.eligible
                or expected_scope != value.scope
            ):
                status = "excluded by table rule"
            elif label is None:
                status = "unassigned"
            elif label_matches(expected.box(), label.bbox):
                status = "correct"
            else:
                status = "wrong"
        reviews.append(
            Review(
                widget_index=value.native.index,
                status=status,
                expected=expected.text_hint
                if expected is not None
                else {
                    "excluded by table rule": "Excluded: detected table",
                    "ambiguous": "Ambiguous — no unique reference pairing",
                    "correct abstention": "No visible label",
                    "wrong: expected no label": "No visible label",
                }.get(status, "Not annotated"),
                predicted=label.text if label is not None else "No pairing",
                expected_bbox=expected.box() if expected is not None else None,
                predicted_bbox=label.bbox if label is not None else None,
                features=candidate.features if candidate is not None else {},
                reference_disposition=reference.disposition
                if reference is not None
                else "unreviewed",
                reason=reference.reason if reference is not None else "",
                detected_form_ids=reference.detected_form_ids
                if reference is not None
                else [],
            )
        )
    group_scores: Counter = Counter()
    for ann in annotations:
        if (
            ann.kind != "group"
            or ann.group_type == "form_scope"
            or ann.expected_label is None
        ):
            continue
        target = tuple(ann.widget_indices)
        scope = scope_of(ann.expected_label.box(), snapshot)
        members = [v for v in assignment.values if v.native.index in target]
        if not scope.eligible or any(v.scope != scope for v in members):
            group_scores["excluded by table rule"] += 1
            continue
        found = [
            assignment.candidates[c]
            for c in assignment.selected
            if assignment.candidates[c].kind == ann.group_type
            and tuple(
                assignment.values[i].native.index
                for i in assignment.candidates[c].members
            )
            == target
        ]
        correct = any(
            label_matches(ann.expected_label.box(), assignment.labels[c.label].bbox)
            for c in found
        )
        group_scores["correct" if correct else "not recovered"] += 1
    return reviews, group_scores


def rect(box: BoundingBox, color: str, *, dashed: bool = False) -> str:
    dash = ' stroke-dasharray="4 3"' if dashed else ""
    return f'<rect x="{box.l}" y="{box.t}" width="{box.width}" height="{box.height}" fill="none" stroke="{color}" stroke-width="1.5"{dash}/>'


def page_report(
    path: Path,
    image: Path,
    snapshot: Snapshot,
    assignment: Assignment,
    reviews: list[Review],
) -> None:
    encoded = base64.b64encode(image.read_bytes()).decode("ascii")
    colors = {
        "correct": "#16814b",
        "wrong": "#c32932",
        "unassigned": "#a26700",
        "unreviewed": "#316bbc",
        "excluded by table rule": "#737373",
        "ambiguous": "#8a5a99",
        "correct abstention": "#16814b",
        "wrong: expected no label": "#c32932",
    }
    overlays, options, descriptions = [], [], []
    for value, review in zip(assignment.values, reviews):
        index = value.native.index
        shapes = rect(value.bbox, colors[review.status])
        if review.expected_bbox is not None:
            shapes += rect(review.expected_bbox, "#16814b", dashed=True)
        if review.predicted_bbox is not None:
            shapes += rect(review.predicted_bbox, "#316bbc")
            a, b = anchors(review.predicted_bbox, value.bbox)
            shapes += f'<path d="M{a[0]},{a[1]} L{b[0]},{b[1]}" stroke="#316bbc" stroke-width="1.2"/>'
        overlays.append(f'<g data-index="{index}" hidden>{shapes}</g>')
        options.append(
            f'<option value="{index}">#{index}: {html.escape(review.status)} — {html.escape(review.predicted[:80])}</option>'
        )
        related = [
            assignment.candidates[c]
            for c in assignment.selected
            if index
            in [
                assignment.values[i].native.index
                for i in assignment.candidates[c].members
            ]
        ]
        group_text = "<br>".join(
            html.escape(
                f"{c.kind}: {assignment.labels[c.label].text} → {[assignment.values[i].native.index for i in c.members]}"
            )
            for c in related
        )
        feature_text = ", ".join(f"{k}: {v:.2f}" for k, v in review.features.items())
        if value.scope.table is not None:
            location = f"Detected table #{value.scope.table}"
            if value.scope.cell is not None:
                location += f", cell #{value.scope.cell} (cell-local pairing only)"
        elif review.detected_form_ids:
            location = (
                f"Detected form {review.detected_form_ids}, outside its detected tables"
            )
        else:
            location = (
                "Outside detected forms and tables; native widget remains eligible"
            )
        descriptions.append(
            f'<section data-index="{index}" hidden><p><b>{html.escape(review.status)}</b></p><p>{html.escape(location)}</p><p>Reference: {html.escape(review.expected)}</p><p>{html.escape(review.reason)}</p><p>Chosen: {html.escape(review.predicted)}</p><p>{group_text}</p><details><summary>Score contributions</summary>{html.escape(feature_text) or "No candidate selected"}</details></section>'
        )
    first = next(
        (r.widget_index for r in reviews if r.status == "wrong"),
        reviews[0].widget_index if reviews else -1,
    )
    detected_tables = [
        r for r in regions(snapshot) if r.label in {"table", "document_index"}
    ]
    table_holes = "".join(
        rect(r.bbox, "black").replace('fill="none"', 'fill="black"')
        for r in detected_tables
    )
    boundaries = f'<defs><mask id="outside-tables"><rect width="100%" height="100%" fill="white"/>{table_holes}</mask></defs>'
    for region in regions(snapshot):
        if region.label not in {"form", "table", "document_index"}:
            continue
        is_form = region.label == "form"
        color = "#0089a8" if is_form else "#737373"
        shape = rect(region.bbox, color, dashed=True)
        shape = shape.replace('fill="none"', f'fill="{color}" fill-opacity="0.08"')
        if is_form:
            shape = f'<g mask="url(#outside-tables)">{shape}</g>'
        boundaries += shape
        boundaries += f'<text x="{region.bbox.l + 2}" y="{region.bbox.t + 8}" fill="{color}" font-size="7">{region.label} #{region.id}</text>'
    content = f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(path.stem)}</title>
<style>body{{font:16px system-ui;margin:24px;color:#202020;background:#fff}}a{{color:#2459a6}} main{{display:grid;grid-template-columns:minmax(0,2fr) minmax(260px,1fr);gap:24px}}svg{{width:100%;height:auto}}select{{max-width:100%;padding:8px;font:inherit}}button{{padding:8px;font:inherit}}section{{overflow-wrap:anywhere}}[hidden]{{display:none}}@media(max-width:750px){{main{{display:flex;flex-direction:column-reverse}}}}</style>
<a href="index.html">All pages</a><h1>{html.escape(path.stem)}</h1>
<main><svg viewBox="0 0 {snapshot.size.width} {snapshot.size.height}" role="img" aria-label="Source PDF with detected forms, nested table exclusions, and selected widget and label overlays"><image href="data:image/png;base64,{encoded}" width="{snapshot.size.width}" height="{snapshot.size.height}"/>{boundaries}{"".join(overlays)}</svg>
<aside><label for="widget">Native widget</label><br><select id="widget">{"".join(options)}</select><p><button id="prev">Previous</button> <button id="next">Next</button></p><div aria-live="polite">{"".join(descriptions)}</div><p>Green dashed: reviewed label.<br>Blue: proposed label and connection.<br>Cyan: detected form, with detected tables cut out.<br>Grey: detected table.</p><p>Every eligible widget has a reference judgment. Ambiguous cases are shown but not scored. Table exclusions are separate. Label matching checks spatial association, not complete semantic keys or exact wording.</p></aside></main>
<script>const pick=document.getElementById('widget');function show(){{document.querySelectorAll('[data-index]').forEach(e=>e.toggleAttribute('hidden',e.dataset.index!==pick.value));}}pick.value='{first}';pick.onchange=show;document.getElementById('prev').onclick=()=>{{pick.selectedIndex=Math.max(0,pick.selectedIndex-1);show();}};document.getElementById('next').onclick=()=>{{pick.selectedIndex=Math.min(pick.options.length-1,pick.selectedIndex+1);show();}};show();</script></html>"""
    path.write_text(content, encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence", type=Path, default=root / "output/acroform-keying-review-20260908"
    )
    parser.add_argument(
        "--out", type=Path, default=root / "output/acroform-keying-prototype"
    )
    parser.add_argument("--fixtures", type=Path)
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument(
        "--null-cost",
        type=float,
        default=3.0,
        help="Experimental cost of leaving one eligible widget unpaired",
    )
    args = parser.parse_args()
    truth = root / "tests/data/groundtruth/acroform_keying"
    manifest = json.loads((truth / "manifest.json").read_text(encoding="utf-8"))
    annotations = [
        Annotation.model_validate_json(line)
        for line in (truth / "annotations.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    field_reviews = [
        FieldReview.model_validate_json(line)
        for line in (truth / manifest["files"]["field_reviews"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    review_keys = [(r.fixture, r.page, r.widget_index) for r in field_reviews]
    if len(review_keys) != len(set(review_keys)):
        raise ValueError("Duplicate field review identity")
    review_pages = {
        (p["fixture"], p["page"]): p for p in manifest["review_run"]["pages"]
    }
    snapshots = sorted((args.evidence / "snapshots").glob("*/*.json"))
    if args.only:
        snapshots = [
            p for p in snapshots if any(part in p.parent.name for part in args.only)
        ]
    if not snapshots:
        parser.error("No saved snapshots matched; supply --evidence or check --only")
    if (
        args.out.resolve() == args.evidence.resolve()
        or args.out.resolve().is_relative_to(args.evidence.resolve())
    ):
        parser.error("Output must be separate from the frozen evidence directory")
    fixtures = args.fixtures or Path(manifest["fixture_root_hint"])
    manifests = {f["id"]: f for f in manifest["fixtures"]}
    references = {
        (ref["fixture"], ref["page"], ref["widget_index"]): ref
        for ref in (
            json.loads(line)
            for line in (truth / "widgets.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    }
    supplied = {(p.parent.name, int(p.stem)) for p in snapshots}
    required = {
        (name, page["number"])
        for name in {p.parent.name for p in snapshots}
        for page in manifests[name]["pages"]
    }
    if supplied != required:
        raise ValueError(
            f"Incomplete snapshot set: missing {sorted(required - supplied)}"
        )
    for name in {p.parent.name for p in snapshots}:
        record = manifests[name]
        digest = hashlib.sha256(
            (fixtures / record["filename"]).read_bytes()
        ).hexdigest()
        if digest != record["sha256"]:
            raise ValueError(f"Fixture hash mismatch: {name}")
    args.out.mkdir(parents=True, exist_ok=True)
    totals: Counter = Counter()
    group_totals: Counter = Counter()
    pages = []
    for path in snapshots:
        snapshot = Snapshot.model_validate_json(path.read_text(encoding="utf-8"))
        review_page = review_pages[path.parent.name, snapshot.page]
        source_image = (
            args.evidence / "pages" / path.parent.name / f"{snapshot.page}.png"
        )
        if (
            hashlib.sha256(path.read_bytes()).hexdigest()
            != review_page["snapshot_sha256"]
            or hashlib.sha256(source_image.read_bytes()).hexdigest()
            != review_page["image_sha256"]
        ):
            raise ValueError(f"Reviewed input changed: {path}")
        expected = next(
            p
            for p in manifests[path.parent.name]["pages"]
            if p["number"] == snapshot.page
        )
        if (
            [w.index for w in snapshot.widgets] != expected["native_widget_order"]
            or abs(snapshot.size.width - expected["width"]) > 0.01
            or abs(snapshot.size.height - expected["height"]) > 0.01
        ):
            raise ValueError(f"Snapshot native sequence/dimensions mismatch: {path}")
        for native in snapshot.widgets:
            reference = references[path.parent.name, snapshot.page, native.index]
            bbox = native.rect.to_bounding_box().to_top_left_origin(
                snapshot.size.height
            )
            actual = (bbox.l, bbox.t, bbox.r, bbox.b)
            if (
                any(abs(a - b) > 0.01 for a, b in zip(actual, reference["bbox"]))
                or native.widget_field_type != reference["field_type"]
                or native.widget_field_name != reference["field_name"]
            ):
                raise ValueError(
                    f"Snapshot widget identity/geometry mismatch: {path}, #{native.index}"
                )
        started = perf_counter()
        assignment = assign(snapshot, null_cost=args.null_cost)
        elapsed = perf_counter() - started
        local_annotations = [
            a
            for a in annotations
            if a.fixture == path.parent.name and a.page == snapshot.page
        ]
        local_reviews = [
            r
            for r in field_reviews
            if r.fixture == path.parent.name and r.page == snapshot.page
        ]
        reviews, group_counts = evaluate(
            snapshot, assignment, local_annotations, local_reviews
        )
        counts = Counter(r.status for r in reviews)
        totals.update(counts)
        group_totals.update(group_counts)
        slug = f"{path.parent.name}-p{snapshot.page}"
        ordered_values, fields = associations(assignment)
        record = PageResult(
            fixture=path.parent.name,
            page=snapshot.page,
            solver_status=assignment.solver_status,
            seconds=elapsed,
            null_cost=args.null_cost,
            objective=assignment.objective,
            ordered_values=ordered_values,
            fields=fields,
            reviews=reviews,
            counts=counts,
            groups=group_counts,
        )
        (args.out / f"{slug}.json").write_text(
            record.model_dump_json(indent=2), encoding="utf-8"
        )
        page_report(
            args.out / f"{slug}.html",
            source_image,
            snapshot,
            assignment,
            reviews,
        )
        pages.append(
            {
                "name": slug,
                "counts": counts,
                "seconds": elapsed,
                "solver_status": assignment.solver_status,
            }
        )
        click.echo(
            f"{slug}: {counts['correct']} correct, {counts['wrong']} wrong, {counts['unassigned']} unassigned ({elapsed:.2f}s, {assignment.solver_status})"
        )
    summary = {
        "links": totals,
        "groups": group_totals,
        "pages": pages,
        "reference_coverage": Counter(
            r.disposition for r in field_reviews if (r.fixture, r.page) in supplied
        ),
        "note": "Complete local-label review on pinned development inputs. Ambiguous cases are not scored; table exclusions are separate. Group annotations remain partial. No table recovery or cross-cell/header association.",
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    rows = "".join(
        f'<tr><td><a href="{html.escape(p["name"])}.html">{html.escape(p["name"])}</a></td><td>{p["counts"]["correct"]}</td><td>{p["counts"]["wrong"]}</td><td>{p["counts"]["unassigned"]}</td><td>{p["counts"]["correct abstention"]}</td><td>{p["counts"]["wrong: expected no label"]}</td><td>{p["counts"]["ambiguous"]}</td><td>{p["counts"]["excluded by table rule"]}</td></tr>'
        for p in pages
    )
    (args.out / "index.html").write_text(
        f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AcroForm pairing prototype</title><style>body{{font:16px system-ui;margin:24px}}table{{border-collapse:collapse}}td,th{{padding:8px;text-align:left;border-bottom:1px solid #ddd}}a{{color:#2459a6}}</style><h1>AcroForm pairing prototype</h1><p>Every eligible field has been visually reviewed. {totals["excluded by table rule"]} widgets are excluded by detected tables; {totals["ambiguous"]} ambiguous cases are shown separately and not scored.</p><p>Visible labels: {totals["correct"]} spatial matches · {totals["wrong"]} wrong pairings · {totals["unassigned"]} missed labels.<br>No visible label: {totals["correct abstention"]} correctly left unpaired · {totals["wrong: expected no label"]} wrongly assigned a label.</p><p>Open a page to inspect references, predictions, cyan form areas and grey table exclusions. These are frozen Docling detections. A visual grid without a TABLE detection remains eligible. Local-label matching does not certify complete semantic keys; group annotations remain partial. This is a development set, not a held-out test.</p><table><thead><tr><th>Page</th><th>Matched label</th><th>Wrong label</th><th>Missed label</th><th>Correctly no label</th><th>Invented label</th><th>Ambiguous</th><th>Table excluded</th></tr></thead><tbody>{rows}</tbody></table></html>',
        encoding="utf-8",
    )
    click.echo(f"Report: {args.out / 'index.html'}")
    if any(page["solver_status"] != "optimal" for page in pages):
        raise SystemExit(
            "At least one page did not solve; inspect the report for details"
        )


if __name__ == "__main__":
    main()
