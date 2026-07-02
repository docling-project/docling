from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StructureReconciliationCandidate:
    name: str
    table_cells: list[object]
    num_rows: int
    num_cols: int
    otsl_seq: list[str]
    diagnostics: object
    changed: bool
    notes: tuple[str, ...]


@dataclass(frozen=True)
class StructureCandidateDecisionReport:
    accepted: bool
    reason: str
    score: float
    candidate_name: str


@dataclass(frozen=True)
class StructureCandidateSelection:
    selected: StructureReconciliationCandidate
    report: StructureCandidateDecisionReport


def _structure_diagnostics_valid(diagnostics: object) -> bool:
    return bool(getattr(diagnostics, "valid", True))


def _structure_issue_count(diagnostics: object) -> int:
    total = 0

    for attr in (
        "overlaps",
        "overlapping_cells",
        "out_of_bounds",
        "out_of_bounds_cells",
        "invalid_spans",
        "conflicts",
    ):
        value = getattr(diagnostics, attr, None)
        if value is None:
            continue

        try:
            total += len(value)
        except TypeError:
            if value:
                total += 1

    return total


def _structure_coverage_ratio(diagnostics: object) -> float:
    value = getattr(diagnostics, "coverage_ratio", None)
    if value is None:
        return 0.0

    return float(value)


def _score_structure_candidate(
    *,
    baseline: StructureReconciliationCandidate,
    candidate: StructureReconciliationCandidate,
) -> float:
    baseline_issues = _structure_issue_count(baseline.diagnostics)
    candidate_issues = _structure_issue_count(candidate.diagnostics)

    score = 0.0

    if _structure_diagnostics_valid(candidate.diagnostics):
        score += 1000.0

    score += (baseline_issues - candidate_issues) * 100.0
    score += _structure_coverage_ratio(candidate.diagnostics) * 10.0

    if candidate.num_rows > baseline.num_rows:
        score += 5.0

    if candidate.num_cols > baseline.num_cols:
        score += 5.0

    # Prefer later reconciliation stages only after safety gates have accepted
    # them; this makes the selector deterministic without hardcoding a PDF.
    score += len(candidate.notes) * 0.1

    return score


def select_reconciled_structure_candidate(
    *,
    baseline: StructureReconciliationCandidate,
    candidates: list[tuple[StructureReconciliationCandidate, object]],
) -> StructureCandidateSelection:
    accepted: list[
        tuple[
            StructureReconciliationCandidate,
            StructureCandidateDecisionReport,
        ]
    ] = []

    for candidate, acceptance_report in candidates:
        if not bool(getattr(acceptance_report, "accepted", False)):
            continue

        score = _score_structure_candidate(
            baseline=baseline,
            candidate=candidate,
        )
        accepted.append(
            (
                candidate,
                StructureCandidateDecisionReport(
                    accepted=True,
                    reason=str(getattr(acceptance_report, "reason", "accepted")),
                    score=score,
                    candidate_name=candidate.name,
                ),
            )
        )

    if not accepted:
        return StructureCandidateSelection(
            selected=baseline,
            report=StructureCandidateDecisionReport(
                accepted=False,
                reason="incumbent_preserved",
                score=_score_structure_candidate(
                    baseline=baseline,
                    candidate=baseline,
                ),
                candidate_name=baseline.name,
            ),
        )

    accepted.sort(key=lambda item: item[1].score, reverse=True)
    selected, report = accepted[0]
    return StructureCandidateSelection(selected=selected, report=report)
