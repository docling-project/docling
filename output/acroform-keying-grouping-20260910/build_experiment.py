"""Build an isolated source copy; the accepted prototype is untouched."""

from pathlib import Path

folder = Path(__file__).parent
source = (folder / "baseline_algorithm.py").read_text()
source = source.replace(
    "    features: dict[str, float]\n",
    "    features: dict[str, float]\n    option_labels: tuple[int, ...] = ()\n",
)
start = source.index("    # Shared PDF names are corroboration")
end = source.index("\n\ndef crossing(", start)
source = (
    source[:start]
    + "    candidates.extend(choice_configurations(values, labels, h))\n    return candidates\n\n"
    + (folder / "configurations.py").read_text()
    + source[end:]
)
needle = "    # ponytail: quadratic candidate scan"
assert source.count(needle) == 1
constraints = """    # Selecting a question configuration requires its specified local captions.
    # All ordinary primary assignments and abstention remain available without it.
    for column, index in enumerate(active):
        candidate = proposed[index]
        if not candidate.option_labels:
            continue
        for member, caption in zip(candidate.members, candidate.option_labels):
            compatible = {
                other_column: -1.0
                for other_column, other_index in enumerate(active)
                if proposed[other_index].kind != "choice_group"
                and member in proposed[other_index].members
                and proposed[other_index].label == caption
            }
            constraint({column: 1.0, **compatible}, -math.inf, 0)
"""
source = source.replace(needle, constraints + needle)
source = source.replace(
    "snapshot: Snapshot, *, null_cost: float = 3.0, time_limit: float = 10.0",
    "snapshot: Snapshot, *, null_cost: float = 3.0, time_limit: float = 10.0, forbidden_groups: frozenset[int] = frozenset()",
)
source = source.replace(
    'if c.cost < (0 if c.kind == "choice_group" else null_cost * len(c.members))',
    'if i not in forbidden_groups and c.cost < (0 if c.kind == "choice_group" else null_cost * len(c.members))',
)
(folder / "experimental_algorithm.py").write_text(source)
print("Built", folder / "experimental_algorithm.py")
