# ruff: noqa: F821
# Names are supplied by insertion into the frozen prototype; see build_experiment.py.
"""Geometric question-and-option hypotheses for an isolated experiment.

This file is inserted into a frozen prototype by build_experiment.py. It reuses
that prototype's dataclasses and geometry; it never reads annotation text.
"""


def choice_configurations(values, labels, h):
    configurations = []
    buttons = [i for i, v in enumerate(values) if v.checkbox and v.scope.eligible]
    for qi, question in enumerate(labels):
        if question.role in {"page_header", "page_footer", "footnote"}:
            continue
        eligible = [i for i in buttons if values[i].scope == question.scope]
        for side in ("right", "left"):
            captions = {}
            offsets = {}
            for i in eligible:
                widget = values[i].bbox
                alternatives = []
                for li, caption in enumerate(labels):
                    if caption.scope != question.scope or not caption.atoms.isdisjoint(
                        question.atoms
                    ):
                        continue
                    rect = caption.bbox
                    if abs(rect.t - widget.t) > h:
                        continue
                    inlined = (
                        widget.intersection_over_self(rect) >= INLINE_WIDGET_COVERAGE
                    )
                    if (
                        inlined
                        and sum(
                            values[j].bbox.intersection_over_self(rect)
                            >= INLINE_WIDGET_COVERAGE
                            for j in eligible
                        )
                        != 1
                    ):
                        continue
                    if side == "right":
                        fits = rect.l >= widget.r - 0.25 * h or (
                            inlined and rect.r > widget.r + h
                        )
                        offset = max(0, rect.l - widget.r)
                    else:
                        fits = rect.r <= widget.l + 0.25 * h or (
                            inlined and rect.l < widget.l - h
                        )
                        offset = max(0, widget.l - rect.r)
                    if not fits or offset > 4 * h:
                        continue
                    # Geometry and existing span boundaries only: no word count,
                    # character class, token identity, or native field name.
                    score = (
                        math.log1p(offset / h)
                        + abs(rect.t - widget.t) / h
                        + 0.5 * caption.fragment
                    )
                    alternatives.append((score, li, offset / h))
                if alternatives:
                    _, captions[i], offsets[i] = min(alternatives)
            for start, first in enumerate(eligible):
                if first not in captions:
                    continue
                first_box = values[first].bbox
                q = question.bbox
                above = q.b <= first_box.t and first_box.t - q.b <= 3 * h
                beside = (
                    q.r <= first_box.l and q.t <= first_box.b and q.b >= first_box.t
                )
                if not (above or beside):
                    continue
                members, option_labels, used_atoms = [], [], set(question.atoms)
                boundaries = 0
                for i in eligible[start:]:
                    if i not in captions:
                        break
                    widget = values[i].bbox
                    caption = labels[captions[i]]
                    if used_atoms.intersection(caption.atoms):
                        break
                    if members:
                        prev = values[members[-1]].bbox
                        column = abs(widget.l - prev.l) <= h and widget.t >= prev.b
                        row = abs(widget.t - prev.t) <= h and widget.l >= prev.r
                        reset = (
                            widget.l > prev.l + h
                            and abs(widget.t - first_box.t) <= h
                            and widget.t < prev.t - h
                        )
                        if not (column or row or reset):
                            break
                        # Keep both a boundary and a continuation hypothesis
                        # at intervening blocks, including side-by-side prompts.
                        bottom = max(prev.b, labels[option_labels[-1]].bbox.b)
                        barrier = column and any(
                            other.atoms.isdisjoint(used_atoms | set(caption.atoms))
                            and other.scope == question.scope
                            and other.bbox.t >= bottom
                            and (
                                (
                                    other.bbox.b <= widget.t
                                    and other.bbox.l <= widget.l + 0.25 * h
                                    and other.bbox.r >= widget.r
                                )
                                or (
                                    other.bbox.t < widget.b
                                    and other.bbox.b > widget.t
                                    and other.bbox.width >= 2 * h
                                    and (
                                        (side == "right" and other.bbox.r <= widget.l)
                                        or (side == "left" and other.bbox.l >= widget.r)
                                    )
                                )
                            )
                            for other in labels
                        )
                        boundaries += bool(barrier)
                    members.append(i)
                    option_labels.append(captions[i])
                    used_atoms.update(caption.atoms)
                    if len(members) < 2:
                        continue
                    envelope = BoundingBox.enclosing_bbox(
                        [values[j].bbox for j in members]
                        + [labels[j].bbox for j in option_labels]
                    )
                    if above and min(q.r, envelope.r) <= max(q.l, envelope.l):
                        continue
                    # Reward supported repetition, not arbitrary nearby headings.
                    # An ungrouped solution pays neither this reward nor any
                    # within-group consistency cost.
                    distance = (
                        (first_box.t - q.b) / h
                        if above
                        else (first_box.l - q.r)
                        / max(h, first_box.l - q.l + envelope.width)
                    )
                    consistency = sum(
                        abs(offsets[j] - offsets[first]) for j in members
                    ) / len(members)
                    # A block directly captioning a different checkbox is not
                    # a free-standing prompt for this group. Nested questions
                    # sharing their option caption need a separate hierarchy.
                    if any(
                        j not in members
                        and abs(values[j].bbox.t - q.t) <= h
                        and (
                            (q.l >= values[j].bbox.r and q.l - values[j].bbox.r <= h)
                            or (q.r <= values[j].bbox.l and values[j].bbox.l - q.r <= h)
                        )
                        for j in eligible
                    ):
                        continue
                    features = {
                        "question_gap": math.log1p(max(0, distance)),
                        "caption_consistency": consistency,
                        "supported_repetition": -2.0
                        * (len(members) - 1)
                        / len(members),
                        "intervening_blocks": 0.25 * boundaries,
                        "prompt_fragment": 0.5 * question.fragment,
                    }
                    configurations.append(
                        Candidate(
                            tuple(members),
                            qi,
                            "choice_group",
                            features,
                            tuple(option_labels),
                        )
                    )
    # Within one uninterrupted block, do not invent a cut between otherwise
    # compatible options. Keep cuts at intervening blocks as alternatives.
    return [
        c
        for c in configurations
        if not any(
            other.label == c.label
            and other.features["intervening_blocks"] == c.features["intervening_blocks"]
            and len(other.members) > len(c.members)
            and set(zip(c.members, c.option_labels)).issubset(
                zip(other.members, other.option_labels)
            )
            for other in configurations
        )
    ]
