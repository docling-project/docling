"""Compare complete assignments, retaining close alternatives as uncertainty."""

from dataclasses import replace

from run_experiment import module


def group_description(result, candidate):
    return {
        "widgets": [result.values[i].native.index for i in candidate.members],
        "prompt": result.labels[candidate.label].text,
        "prompt_bbox": result.labels[candidate.label].bbox.model_dump(mode="json"),
        "captions": [result.labels[i].text for i in candidate.option_labels],
    }


def primary_atoms(result):
    return {
        i: result.labels[result.candidates[c].label].atoms
        for c in result.selected
        if result.candidates[c].kind != "choice_group"
        for i in result.candidates[c].members
    }


def audit_assignment(snapshot, *, null_cost=3.0, time_limit=10.0, margin=0.25):
    result = module.assign(snapshot, null_cost=null_cost, time_limit=time_limit)
    original_primary = primary_atoms(result)
    records, withheld_groups, disputed_values = [], set(), set()
    for index in result.selected:
        candidate = result.candidates[index]
        if candidate.kind != "choice_group":
            continue
        alternative = module.assign(
            snapshot,
            null_cost=null_cost,
            time_limit=time_limit,
            forbidden_groups=frozenset({index}),
        )
        solved = alternative.solver_status == "optimal"
        delta = alternative.objective - result.objective if solved else None
        uncertain = not solved or delta <= margin + 1e-8
        record = {
            "selected": group_description(result, candidate),
            "alternative_solved": solved,
            "objective_gap": delta,
            "uncertain": uncertain,
            "alternative_groups": [],
        }
        if uncertain:
            withheld_groups.add(index)
            if solved:
                alt_groups = {
                    j
                    for j in alternative.selected
                    if alternative.candidates[j].kind == "choice_group"
                }
                for j in result.selected:
                    if (
                        result.candidates[j].kind == "choice_group"
                        and j not in alt_groups
                    ):
                        withheld_groups.add(j)
                alternate_primary = primary_atoms(alternative)
                disputed_values.update(
                    i
                    for i in range(len(result.values))
                    if original_primary.get(i) != alternate_primary.get(i)
                )
                record["alternative_groups"] = [
                    group_description(alternative, alternative.candidates[j])
                    for j in alt_groups
                    if j not in result.selected
                ]
        records.append(record)
    # Withhold disputed local captions too; removing only the question could
    # otherwise leave its arbitrary option-caption side effect visible as fact.
    keep = [
        j
        for j in result.selected
        if j not in withheld_groups
        and not disputed_values.intersection(result.candidates[j].members)
    ]
    diagnostic = {
        "raw_objective": result.objective,
        "margin": margin,
        "groups": records,
        "withheld_widgets": [
            result.values[i].native.index for i in sorted(disputed_values)
        ],
        "withheld_group_count": len(withheld_groups),
    }
    return replace(result, selected=keep), diagnostic
