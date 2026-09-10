"""Build an isolated optimizer with conditional captions and parent references."""

from pathlib import Path

folder = Path(__file__).parent
source = (
    folder.parent / "acroform-keying-grouping-20260910/baseline_algorithm.py"
).read_text()
source = source.replace(
    "from itertools import combinations", "from itertools import combinations, pairwise"
)
source = source.replace(
    "    features: dict[str, float]\n",
    "    features: dict[str, float]\n    option_choices: tuple[tuple[int, ...], ...] = ()\n    parent: int | None = None\n",
)
start = source.index("    # Shared PDF names are corroboration")
end = source.index("\n\ndef crossing(", start)
source = (
    source[:start]
    + "    candidates.extend(choice_configurations(values, labels, h))\n    return candidates\n\n"
    + (folder / "configurations.py").read_text()
    + source[end:]
)
source = source.replace(
    "snapshot: Snapshot, *, null_cost: float = 3.0, time_limit: float = 10.0",
    "snapshot: Snapshot, *, null_cost: float = 3.0, time_limit: float = 10.0, forbidden_groups: frozenset[int] = frozenset(), _candidates: list[Candidate] | None = None",
)
source = source.replace(
    'if c.cost < (0 if c.kind == "choice_group" else null_cost * len(c.members))',
    'if i not in forbidden_groups and c.cost < (0 if c.kind == "choice_group" else null_cost * len(c.members))',
)
# Conditional rewards can make an otherwise expensive primary association useful.
start_active = source.index("    # A candidate worse than leaving")
end_active = source.index(
    "    costs = [proposed[i].cost for i in active]", start_active
)
source = (
    source[:start_active]
    + """    group_support = set()
    for i, c in enumerate(proposed):
        if i in forbidden_groups or c.kind != "choice_group" or c.cost >= 0:
            continue
        group_support.update((member, caption) for member, choices in zip(c.members, c.option_choices) for caption in choices)
        if c.parent is not None:
            group_support.add((c.parent, c.label))
    active = [i for i, c in enumerate(proposed) if i not in forbidden_groups and (
        c.cost < (0 if c.kind == "choice_group" else null_cost*len(c.members))
        or (c.kind != "choice_group" and any((member, c.label) in group_support for member in c.members))
    )]
"""
    + source[end_active:]
)
source = source.replace(
    "        for atom in labels[candidate.label].atoms:\n            spans[atom][column] = 1.0",
    "        if candidate.parent is None:\n            for atom in labels[candidate.label].atoms:\n                spans[atom][column] = 1.0",
)
source = source.replace(
    "    proposed = candidates_for(values, labels, h)",
    "    proposed = candidates_for(values, labels, h) if _candidates is None else _candidates",
)
constraints = """    # A child question references the parent's selected caption; it does not
    # consume the source text again. Other text ownership stays exclusive.
    for column, index in enumerate(active):
        candidate = proposed[index]
        requirements = list(zip(candidate.members, candidate.option_choices))
        if candidate.parent is not None:
            requirements.append((candidate.parent, (candidate.label,)))
        for member, captions in requirements:
            compatible = {
                other_column: -1.0
                for other_column, other_index in enumerate(active)
                if proposed[other_index].kind != "choice_group"
                and member in proposed[other_index].members
                and proposed[other_index].label in captions
            }
            constraint({column: 1.0, **compatible}, -math.inf, 0)
"""
needle = "    # ponytail: quadratic candidate scan"
assert source.count(needle) == 1
source = source.replace(needle, constraints + needle)
(folder / "experimental_algorithm.py").write_text(source)
print("Built", folder / "experimental_algorithm.py")
