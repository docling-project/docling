# ruff: noqa: F821
# Inserted into the frozen prototype by build_experiment.py.
"""Caption alternatives and geometric parent/sibling hypotheses; no lexical signals."""


def edge_position(box, edge):
    return box.l if edge == "l" else box.r


def caption_context(values, labels, h):
    buttons = [i for i, v in enumerate(values) if v.checkbox and v.scope.eligible]
    captions = {}
    for i in buttons:
        w = values[i].bbox
        for side in ("left", "right"):
            choices = []
            for li, label in enumerate(labels):
                r = label.bbox
                if label.scope != values[i].scope or abs(r.t - w.t) > h:
                    continue
                inline = w.intersection_over_self(r) >= INLINE_WIDGET_COVERAGE
                if (
                    inline
                    and sum(
                        values[j].bbox.intersection_over_self(r)
                        >= INLINE_WIDGET_COVERAGE
                        for j in buttons
                    )
                    != 1
                ):
                    continue
                fits = (
                    (r.r <= w.l + 0.25 * h or (inline and r.l < w.l - h))
                    if side == "left"
                    else (r.l >= w.r - 0.25 * h or (inline and r.r > w.r + h))
                )
                if not fits:
                    continue
                a, b = anchors(w, r)
                if any(
                    j != i
                    and values[j].scope == values[i].scope
                    and obstructs(a, b, values[j].bbox, 0.05 * h)
                    for j in buttons
                ):
                    continue
                choices.append(li)
            captions[i, side] = tuple(choices)

    # Immediate geometric neighbours supply row context, never an output order.
    neighbours = {}
    for i in buttons:
        w = values[i].bbox
        nearby = []
        for axis, direction in (("column", -1), ("column", 1), ("row", -1), ("row", 1)):
            matches = []
            for j in buttons:
                z = values[j].bbox
                if i == j or values[j].scope != values[i].scope:
                    continue
                aligned = (
                    abs(z.l - w.l) <= h if axis == "column" else abs(z.t - w.t) <= h
                )
                delta = z.t - w.t if axis == "column" else z.l - w.l
                if aligned and delta * direction > h:
                    matches.append((abs(delta), j))
            if matches:
                nearby.append(min(matches)[1])
        neighbours[i] = nearby

    row_owners = [set() for _ in labels]
    for i in buttons:
        w = values[i].bbox
        for side in ("left", "right"):
            for li in captions[i, side]:
                label = labels[li]
                r = label.bbox
                # Touching source spans may be fragments of one caption row.
                touching = any(
                    other != li
                    and labels[other].atoms.isdisjoint(label.atoms)
                    and abs(labels[other].bbox.t - r.t) <= 0.5 * h
                    and gap(labels[other].bbox, r) <= h
                    for other in captions[i, side]
                )
                repeated = any(
                    abs(
                        (edge_position(labels[other].bbox, edge) - values[j].bbox.l)
                        - (edge_position(r, edge) - w.l)
                    )
                    <= h
                    for j in neighbours[i]
                    for other in captions[j, side]
                    for edge in ("l", "r")
                )
                if (
                    touching
                    or repeated
                    or w.intersection_over_self(r) >= INLINE_WIDGET_COVERAGE
                ):
                    row_owners[li].add(i)

    return buttons, captions, row_owners


def boundary_between(other, question, prev, curr, owned_row, side, h):
    r, q = other.bbox, question.bbox
    if (
        other.scope != question.scope
        or not other.atoms.isdisjoint(question.atoms)
        or r.t < prev.b
    ):
        return False
    if owned_row:
        # A new inline question can share the section margin while containing
        # its own checkbox. Keep a cut here even though it is also a caption.
        margin = (
            (abs(r.l - q.l) <= 0.25 * h and r.l < curr.l - h)
            if side == "right"
            else (abs(r.r - q.r) <= 0.25 * h and r.r > curr.r + h)
        )
        return margin and curr.intersection_over_self(r) >= INLINE_WIDGET_COVERAGE
    above = r.b <= curr.t and r.l <= curr.l + 0.25 * h and r.r >= curr.r
    beside = (
        r.t < curr.b
        and r.b > curr.t
        and r.width >= 2 * h
        and ((side == "right" and r.r <= curr.l) or (side == "left" and r.l >= curr.r))
    )
    return above or beside


def choice_configurations(values, labels, h):
    buttons, captions, row_owners = caption_context(values, labels, h)
    configurations = []
    for side in ("right", "left"):
        indent = 1 if side == "right" else -1
        for first in buttons:
            first_box = values[first].bbox
            eligible = [
                i
                for i in buttons
                if values[i].scope == values[first].scope and i >= first
            ]
            # A family aligns either starts or ends, allowing ragged captions.
            families = {
                (edge, edge_position(labels[li].bbox, edge) - first_box.l)
                for li in captions[first, side]
                for edge in ("l", "r")
            }
            for edge, offset in sorted(families):
                options = {
                    i: tuple(
                        li
                        for li in captions[i, side]
                        if abs(
                            edge_position(labels[li].bbox, edge)
                            - values[i].bbox.l
                            - offset
                        )
                        <= h
                    )
                    for i in eligible
                }
                if not options[first]:
                    continue
                members = []
                for i in eligible:
                    w = values[i].bbox
                    if members:
                        prev = values[members[-1]].bbox
                        column = abs(w.l - prev.l) <= h and w.t >= prev.b
                        row = abs(w.t - prev.t) <= h and w.l >= prev.r
                        reset = (
                            w.l > prev.l + h
                            and abs(w.t - first_box.t) <= h
                            and w.t < prev.t - h
                        )
                        if not (column or row or reset):
                            # An indented branch may intervene before the next sibling.
                            if indent * (w.l - prev.l) > h and w.t >= prev.b:
                                continue
                            break
                    if not options[i]:
                        break
                    members.append(i)
                    if len(members) < 2:
                        continue
                    for qi, q in enumerate(labels):
                        if q.scope != values[first].scope or q.role in {
                            "page_header",
                            "page_footer",
                            "footnote",
                        }:
                            continue
                        above = q.bbox.b <= first_box.t
                        beside = (
                            q.bbox.r <= first_box.l
                            and q.bbox.t <= first_box.b
                            and q.bbox.b >= first_box.t
                        )
                        if not (above or beside):
                            continue
                        parents = [None]
                        if row_owners[qi]:
                            parents = [
                                j
                                for j in row_owners[qi]
                                if j < first
                                and j not in members
                                and indent * (first_box.l - values[j].bbox.l) > h
                                and values[j].bbox.b <= first_box.t
                                and first_box.t - values[j].bbox.b <= 3 * h
                            ]
                        for parent in parents:
                            compatible = tuple(
                                tuple(
                                    li
                                    for li in options[j]
                                    if labels[li].atoms.isdisjoint(q.atoms)
                                )
                                for j in members
                            )
                            if any(not choices for choices in compatible):
                                continue
                            envelope = BoundingBox.enclosing_bbox(
                                [values[j].bbox for j in members]
                                + [
                                    labels[li].bbox
                                    for choices in compatible
                                    for li in choices
                                ]
                            )
                            if above and min(q.bbox.r, envelope.r) <= max(
                                q.bbox.l, envelope.l
                            ):
                                continue
                            boundaries = 0
                            for a, b in pairwise(members):
                                prev, curr = values[a].bbox, values[b].bbox
                                if abs(curr.l - prev.l) > h:
                                    continue
                                boundaries += any(
                                    k != qi
                                    and boundary_between(
                                        other,
                                        q,
                                        prev,
                                        curr,
                                        bool(row_owners[k]),
                                        side,
                                        h,
                                    )
                                    for k, other in enumerate(labels)
                                )
                            distance = (
                                (first_box.t - q.bbox.b) / h
                                if above
                                else (first_box.l - q.bbox.r)
                                / max(h, first_box.l - q.bbox.l + envelope.width)
                            )
                            configurations.append(
                                Candidate(
                                    tuple(members),
                                    qi,
                                    "choice_group",
                                    {
                                        "question_gap": math.log1p(max(0, distance)),
                                        "supported_repetition": -2.0
                                        * (len(members) - 1)
                                        / len(members),
                                        "intervening_blocks": 0.25 * boundaries,
                                        "prompt_fragment": 0.5 * q.fragment,
                                    },
                                    compatible,
                                    parent,
                                )
                            )
    # Remove duplicate alignment families and unmotivated cuts within a block.
    unique = {}
    for c in configurations:
        unique[c.members, c.label, c.option_choices, c.parent] = c
    proposals = list(unique.values())
    return [
        c
        for c in proposals
        if not any(
            other.label == c.label
            and other.parent == c.parent
            and other.features["intervening_blocks"] == c.features["intervening_blocks"]
            and len(other.members) > len(c.members)
            and all(
                i in other.members
                and set(choices).issubset(other.option_choices[other.members.index(i)])
                for i, choices in zip(c.members, c.option_choices)
            )
            for other in proposals
        )
    ]
