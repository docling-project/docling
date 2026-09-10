# Caption alternatives and nested groups — September 10

Implemented as an isolated follow-up to the [question-role investigation](../acroform-keying-grouping-20260910/role-investigation.md). **Keep this experimental; the accepted prototype and production code are unchanged.**

[Review the uncertainty-filtered results](uncertainty-report/index.html) · [Inspect raw choices](report/index.html) · [Accepted baseline](../acroform-keying-coverage-20260910/report/index.html) · [Every changed association](comparison.json)

## What changed

- Caption alternatives are proposed before choosing a question. They can sit on either side; there is no fixed maximum caption-to-checkbox gap. A repeated arrangement can align the starts or ends of its captions, accommodating different text widths. Selecting a group requires the solver to choose a compatible caption for every member; it no longer preselects one caption per member.
- Existing spans gain geometric option-row context from adjacent source spans, neighbouring checkboxes and repeated text positions. Text in such a row cannot freely become a question for unrelated following options or a neighbouring column. This uses existing spans; it does not assemble caption fragments.
- A run can skip an indented child branch when proposing its next sibling, in either indentation direction. A child question references its parent option's selected caption instead of consuming that text a second time. The JSON and HTML reports identify these parent references. Native values remain in their original page-wide order.
- A block at the current question's section margin can propose a boundary even if it contains its own checkbox. Both stopping and continuing remain candidates. This corrects the earlier candidate cleanup that removed the valid Yes/No boundary.
- Alternative-assignment checks now compare group identities across equivalent alignment proposals and also forbid each selected group-related primary association in turn. Consequently, a question can be stable while its caption choice remains uncertain. Withholding a disputed parent caption also withholds its dependent child group.
- A primary association that costs more than abstention remains available when a conditional group reward could make the complete assignment preferable. The former pruning assumption was invalid after introducing conditional rewards.

The repetition reward, question-distance cost, fragment cost and boundary weight remain those of the previous grouping experiment. The old caption-offset consistency cost is replaced by competing start/end alignment families. Thus this is not a pure single-variable ablation, but it does not tune the repetition reward against the new results.

## What the actual cases show

| Case | Result |
| --- | --- |
| Language-preference form | One full 21-option question group survives uncertainty filtering. Arabic caption text and the closing parenthesis no longer seed false groups. All local captions remain correct. |
| RC7190 | Application type and Type of rebate remain separate. The first option keeps Type 1A as its caption. |
| F1120 nested Yes options | Yes keeps its local caption and introduces suboptions (i)/(ii) through a parent reference. The correct outer Yes/No candidate now exists. |
| F1120 outer boundary | Raw scoring still prefers including the next independent checkbox with Yes/No. A close alternative exists, so the outer group is withheld. The Yes/No captions and the valid nested child group remain. |
| Sámi RF-1125 | Both left-caption and right-explanation arrangements are available; the real shared question is selected. **The scorer still chooses the wrong right-hand explanations**, and their alternatives are outside the current uncertainty margin. This is not solved. |
| GST111 | The first option's caption is no longer selected as a question for the remaining options. The distant real question can be proposed, but its score does not justify selection; the shared question remains unassigned. |
| Chinese F14446 | The note-versus-question ambiguity remains explicit. Page 1's disputed local association is withheld; page 3 preserves the local Yes/No captions while withholding the disputed question. |

The nested representation is:

```text
Question 6: possible outer group [Yes, No]
    [ ] Yes  <── text owned by this option
        [ ] (i)   ┐ child group references Yes
        [ ] (ii)  ┘
    [ ] No

[ ] Next independent question
    ↑ raw grouping still crosses this boundary; filtered output abstains
```

The filter also withholds F1120 question 5's previously correct shared group because a competing group is close. Its local captions remain correct. Group uncertainty should therefore be reviewed separately from local-caption accuracy.

## Replay totals

All 19 frozen pages were replayed. Against the accepted coverage prototype:

| Local labels | Accepted | Raw new experiment | With uncertainty filtering |
| --- | ---: | ---: | ---: |
| Correct | 276 | 277 | 277 |
| Wrong | 85 | 84 | 83 |
| Unassigned | 73 | 73 | 74 |

The one newly correct local association is RC7190 widget 1. No previously correct local association becomes wrong or unassigned. Filtering additionally withholds the previously wrong Chinese page 1 widget 6 association. These are development-set results, not evidence of generalization.

Separate counts remain unchanged: 503 detected-table exclusions, 7 correct no-label abstentions, 1 wrong association on a no-label reference, and 1 ambiguous reference. Partial group references score 12 correct for the accepted baseline, 13 for the raw experiment, and 12 after filtering. The group references are incomplete and do not count every false-positive group.

## What remains before adoption

The new candidate representation addresses the concrete feasibility defects. The unresolved cases now expose limitations in scoring and geometric evidence:

- Competing caption columns can both have a coherent arrangement. Distance still favors explanations when they are closer than the reviewed captions. Increasing a confidence margin until these examples disappear would not establish that the scoring is reliable.
- A group's reward can still outweigh evidence for stopping at an independent inline question. The boundary alternatives now exist, but the preferred boundary is not always correct.
- Uncertainty is a comparison of the best alternatives the experiment generated, using a provisional 0.25 cost margin. It is not a probability or a guarantee that all interpretations were considered.
- Caption alignment, row ownership and indentation still use provisional geometric tolerances. Their development-set successes need independent layouts, scale/position variations and further counterexamples before integration.

There is no new keyword, language, text-length, field-name meaning or alphabetic-content signal in proposal construction. Replacing visible label text and native names/descriptions with `?` produces identical group proposals on all 19 pages. The inherited primary scorer still has its existing alphabetic/short-text heuristics; this experiment does not remove or endorse those.

Detected-table handling, existing widget coverage and native order are unchanged. No missed-table recovery, caption reconstruction, reference edits, commits or production integration are included.

## Verification and reproduction

Thirty tests pass: the accepted prototype's 13 checks, the previous experiment's 10 structural checks, and 7 added cases covering retained caption alternatives, nested ownership, mirrored indentation, fragment questions, equal caption-side interpretations and conditional-cost pruning. All 19 replay pages solve optimally; native values/order, table exclusions and all frozen reference/input hashes match the accepted run. Scoped `make validate` is run in the isolated validation checkout to preserve unrelated workspace files.

From the repository root:

```bash
.venv/bin/python output/acroform-keying-roles-20260910/build_experiment.py
DOCLING_DEVICE=cpu PYTHONPATH=. .venv/bin/python output/acroform-keying-roles-20260910/run_experiment.py
DOCLING_DEVICE=cpu PYTHONPATH=. .venv/bin/python output/acroform-keying-roles-20260910/run_uncertainty.py
DOCLING_DEVICE=cpu DOCLING_GEN_TEST_DATA=0 PYTHONPATH=.:output/acroform-keying-roles-20260910 .venv/bin/python -m pytest output/acroform-keying-roles-20260910/test_configurations.py tests/test_acroform_keying_prototype.py -q
.venv/bin/python output/acroform-keying-roles-20260910/compare_results.py
DOCLING_DEVICE=cpu PYTHONPATH=.:output/acroform-keying-roles-20260910 .venv/bin/python output/acroform-keying-roles-20260910/check_geometry.py
```

The builder uses the unchanged accepted source frozen in the previous experiment. Generated solver code stays in this directory. `source-hashes.json` fingerprints the runnable files; `geometry-checks.json` records the text-independence checks. Per-page `.group-audit.json` files record the actual competing assignments and caption checks. Raw replay took roughly 14 seconds total; the extra alternative solves took roughly 70 seconds on this machine. Most of that work was the language form and F1120; this is an offline experiment rather than a production performance result.
