"""Compare association alternatives, including captions and parent references."""

from dataclasses import replace

from run_experiment import module


def primary_atoms(result):
    return {
        i: result.labels[result.candidates[c].label].atoms
        for c in result.selected
        if result.candidates[c].kind != "choice_group"
        for i in result.candidates[c].members
    }


def group_key(result, c):
    return c.members, result.labels[c.label].atoms, c.parent


def group_description(result, candidate):
    primary = {
        i: result.labels[result.candidates[c].label].text
        for c in result.selected
        if result.candidates[c].kind != "choice_group"
        for i in result.candidates[c].members
    }
    return {
        "widgets": [result.values[i].native.index for i in candidate.members],
        "prompt": result.labels[candidate.label].text,
        "prompt_bbox": result.labels[candidate.label].bbox.model_dump(mode="json"),
        "parent_widget": None
        if candidate.parent is None
        else result.values[candidate.parent].native.index,
        "captions": [primary.get(i) for i in candidate.members],
    }


def audit_assignment(snapshot, *, null_cost=3.0, time_limit=10.0, margin=0.25):
    result = module.assign(snapshot, null_cost=null_cost, time_limit=time_limit)
    original_primary = primary_atoms(result)
    records, caption_records, withheld_groups, disputed = [], [], set(), set()
    selected_groups = [
        j for j in result.selected if result.candidates[j].kind == "choice_group"
    ]

    def compare(forbidden):
        alternative = module.assign(
            snapshot,
            null_cost=null_cost,
            time_limit=time_limit,
            forbidden_groups=frozenset(forbidden),
            _candidates=result.candidates,
        )
        solved = alternative.solver_status == "optimal"
        delta = alternative.objective - result.objective if solved else None
        uncertain = not solved or delta <= margin + 1e-8
        if uncertain and solved:
            alternate_primary = primary_atoms(alternative)
            disputed.update(
                i
                for i in range(len(result.values))
                if original_primary.get(i) != alternate_primary.get(i)
            )
            alt_groups = {
                group_key(alternative, alternative.candidates[j])
                for j in alternative.selected
                if alternative.candidates[j].kind == "choice_group"
            }
            withheld_groups.update(
                j
                for j in selected_groups
                if group_key(result, result.candidates[j]) not in alt_groups
            )
        return alternative, solved, delta, uncertain

    for index in selected_groups:
        candidate = result.candidates[index]
        key = group_key(result, candidate)
        forbidden = [
            j
            for j, c in enumerate(result.candidates)
            if c.kind == "choice_group" and group_key(result, c) == key
        ]
        alternative, solved, delta, uncertain = compare(forbidden)
        if uncertain:
            withheld_groups.add(index)
        records.append(
            {
                "selected": group_description(result, candidate),
                "alternative_solved": solved,
                "objective_gap": delta,
                "uncertain": uncertain,
                "alternative_groups": [
                    group_description(alternative, alternative.candidates[j])
                    for j in alternative.selected
                    if alternative.candidates[j].kind == "choice_group"
                ]
                if uncertain and solved
                else [],
            }
        )

    members = {i for j in selected_groups for i in result.candidates[j].members}
    members.update(
        result.candidates[j].parent
        for j in selected_groups
        if result.candidates[j].parent is not None
    )
    # Test each selected primary association once, even for an inline clause.
    for index in result.selected:
        c = result.candidates[index]
        if c.kind == "choice_group" or not members.intersection(c.members):
            continue
        atoms = result.labels[c.label].atoms
        forbidden = [
            j
            for j, other in enumerate(result.candidates)
            if other.kind != "choice_group"
            and set(c.members).intersection(other.members)
            and result.labels[other.label].atoms == atoms
        ]
        alternative, solved, delta, uncertain = compare(forbidden)
        if uncertain and not solved:
            disputed.update(c.members)
        caption_records.append(
            {
                "widgets": [result.values[i].native.index for i in c.members],
                "selected_caption": result.labels[c.label].text,
                "alternative_solved": solved,
                "objective_gap": delta,
                "uncertain": uncertain,
                "alternative_captions": [
                    group_description(alternative, alternative.candidates[j])
                    for j in alternative.selected
                    if alternative.candidates[j].kind == "choice_group"
                ]
                if uncertain and solved
                else [],
            }
        )

    # Removing one disputed member also removes its shared primary association.
    for j in result.selected:
        c = result.candidates[j]
        if c.kind != "choice_group" and disputed.intersection(c.members):
            disputed.update(c.members)
    keep = [
        j
        for j in result.selected
        if j not in withheld_groups
        and not disputed.intersection(result.candidates[j].members)
        and result.candidates[j].parent not in disputed
    ]
    return replace(result, selected=keep), {
        "raw_objective": result.objective,
        "margin": margin,
        "groups": records,
        "caption_checks": caption_records,
        "withheld_widgets": [result.values[i].native.index for i in sorted(disputed)],
        "withheld_group_count": sum(j not in keep for j in selected_groups),
    }
