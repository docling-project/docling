# AcroForm association failure audit

Reviewed 2026-09-09. This is an analysis of the frozen 19-page prototype, not a change to the optimizer, annotations, or production pipeline. [Browse every case](index.html). Native widget numbers below are the original zero-based indices used by the prototype report.

## What the evidence says

There are 179 flagged local-label cases: 105 wrong bounding-box matches, 73 missing labels, and one label assigned to a visually unlabeled operand. I classified every one in [failure-cases.json](failure-cases.json), with its expected text, actual text, explanation and candidate diagnostics. All selected non-singleton associations are separately recorded in [selected-groups.json](selected-groups.json); local-label scores do not certify those groups.

77 flagged cases fall in clearly missed or truncated tables: GST111 page 2 lower grid (24), Manitoba page 1 tax-bracket grid (21), Manitoba page 2 contribution grid (19), RF-1125 lower vehicle rows (12), RF-1177 greenhouse-area row (1). This is a visual audit classification, not an estimate of the effect of retraining the layout model. The Italian C-section strips could also be reviewed for table detection, but their caption-construction defects are recorded independently: the same failures occur in ordinary form areas. These 77 cases are not evidence that the association algorithm should infer missing grids.

Three flagged cases need reference-policy clarification before treating them as semantic failures: Manitoba page 1 #24 predicts exactly “Line 1” but differs in whitespace extent; #27 chooses the specific Schedule 11 component caption while the reference includes its parent caption; RC7190 page 2 #8 chooses an instruction explicitly addressing the target value rather than the canonical field heading. Several other predictions identify part of the right caption or include extra text. These remain defects in caption extent, but are different from associating a value with an unrelated field.

## 1. Construct the right caption before scoring associations

Italian page 2 #79, #83 and #87 select “NATO” from the four printed lines “INDETER- / MINATO/ / DETERMI- / NATO”. The complete caption is not a candidate. The current code offers whole individual layout regions and their individual text cells; it cannot join a caption split across regions. Raising the reward for the correct answer cannot help when that answer is absent.

The inverse also occurs. Italian page 1 #25 needs MINORE, but the source cell combines it with a neighboring caption. RF-1177 #76 gets the correct word Vuvdon plus a separate instruction because they share a source cell. Joining more text indiscriminately would worsen these cases.

**Change:** build candidate captions from positioned text lines/cells within the permitted scope. Join aligned neighboring lines with compatible spacing; preserve alternatives when a boundary is uncertain. Use finer positioned text when one source cell merges separate captions; never fabricate coordinates by dividing a string equally. Keep source-text identities so alternative spans still compete for the same evidence. Start with simple geometric joining and supported subspans, not language-specific phrase rules.

**What should count in its favor:** compact aligned lines forming one caption, and a consistent relationship to a field. **What should count against it:** crossing another caption/field lane, swallowing instructions or neighboring captions, or selecting a fragment when the complete local caption is supported. Do not universally reward either short or long text. Detected table/cell ownership remains a hard boundary.

## 2. Judge a checkbox question together with its options

On RC7190 page 1 the first checkbox gets “Application type…” instead of “Type 1A”. The remaining four option captions are correct. The shared question then becomes “Fill out Section 1, 2, or 3 only”. The first box has consumed the real question as though it were an option caption.

The Chinese page 1 case is the same mechanism with a different layout: option A contains a long explanation. The gap-based group builder separates it from B–E, A takes the prompt, and B–E receive an intervening note. On Chinese page 3, the Yes/No captions are all locally correct, but the second pair receives the long preceding legal paragraph instead of the short consent prompt printed on their row. That prompt is too far away for the current ten-line-height cutoff.

**Change:** propose a question with its candidate option captions and members as one configuration. Support a prompt above the options or beside their row, long wrapped options, and intervening text controls. Evaluate member captions and the common prompt together; include a no-common-question alternative. A nearby heading should not receive a fixed reward merely because a checkbox run exists. Group membership may reference noncontiguous native indices; exported widget traversal must remain unchanged.

**Reward:** a prompt spanning the option block, coherent option-caption placement, and consistent membership. **Penalize:** using a prompt as one option's caption, taking another option's explanation as the question, or crossing a new question/section. These are relationships between positioned elements, not tests for “Yes”, “Type 1A”, a particular language, or fixture names. Long horizontal prompts need row/block geometry rather than the ordinary short label-distance cutoff.

## 3. Represent shared clauses and genuine components, without merging neighbors

The F1120 page has checkbox clauses containing date blanks. Several correct clause candidates are omitted because a widget extends less than one PDF point outside the text's rectangular envelope. For example, the “Adopt an apportionment plan…” text ends at y=275.93855 while date widgets end at y=276. Exact containment excludes the dates. The same model successfully shares another, taller three-line clause with its checkbox and dates.

**Verified experiment:** expanding only the inline-clause containment test by 0.1 text-line height corrected 20 flagged local labels on F1120, with no previously correct local label lost across all 19 pages. Expansion by 0.2 line height produced the same local scores. Both left all 503 table exclusions unchanged. This includes indirect corrections to neighboring boxes freed from bad assignments. It establishes the mechanism on this development set; it does not validate a general grouping policy.

Do not relax the table-containment helper. The experiment changes only eligibility for an inline-clause proposal. Before implementation, require actual text-line/blank relationships as well as tolerant geometry; a large paragraph envelope alone must not collect unrelated fields.

The “From [date], until [date]” clause is a further case. Sharing the full clause fixes its checkbox but still loses the identities of the two dates. It needs a clause containing two labeled date components, each of which can contain two native controls. This is a small hierarchy, not one flat label for every blank.

There are also false composites: RF-1177 merges business name and organization number under Org.nr.; Italian #94–95 merges two separately captioned benefit fields. **Disabling all composites fixed three local labels but broke two previously correct ones, including a legitimate split business number.** Keep composites, but require evidence of a shared caption and reject proposals that swallow independently captioned neighbors. Do not allow unrestricted label reuse; permit sharing only through a supported clause/component/group.

## 4. Score the field's row and lane, not just its nearest rectangle

RF-1084 has a left caption, a long blank, then a second caption and blank on the same row. The long first blank touches the second caption, so the algorithm chooses the second caption and displaces its actual field. Manitoba has the complementary problem: a correct caption across a long calculation row is farther away than an unrelated nearby paragraph, or is rejected in favor of abstention.

Current numbers make this understandable. RF-1084 #1's correct caption costs about 2.54, while the next column's caption costs 0.09. Manitoba's first taxable-income field has a correct candidate costing 3.85, but leaving it blank costs only 3. The optimizer is solving the stated objective; the stated evidence is inadequate. Increasing the blank price everywhere would encourage additional bad associations.

**Change:** compare candidate relationships within repeated local rows/lanes. Measure baseline/vertical alignment in text-line heights, but evaluate horizontal separation relative to the local field lane and row. Use the actual text lines and plausible label-to-blank edges; overlap with a large text envelope should not mean zero semantic distance. Use detected form/section context where available, while retaining eligible fields outside forms. The present outside-table scope is effectively page-wide.

**Reward:** the same caption-to-blank pattern repeated nearby and a strong row relationship. **Penalize:** crossing another field's caption/lane, jumping into a neighboring section, and treating an explanatory paragraph or reference code as the primary caption. Prefer local consistency rather than a universal “labels are left/right/above” rule: RF-1125 has the short checkbox caption on the left and its explanation on the right. Existing broad form regions alone will not solve within-form columns or question boundaries.

The overlapping ambiguous GST111 widget 7 also demonstrates why this matters: it takes Postal code and causes two displaced assignments. Give weakly supported controls an abstention alternative instead of forcing them to own a neighbor's strongly supported caption. Compare competing complete assignments after improving the evidence; a score margin can expose uncertainty, but is not a calibrated probability or a cure for a systematically wrong cost.

## 5. Make the report distinguish what failed

Keep separate judgments for the primary field/caption relationship, caption completeness/extra text, and question/component membership. Preserve table exclusions and ambiguous/no-visible-label decisions. The current green local label can hide a wrong common question; a red bounding-box match can merely indicate whitespace or a partial caption.

For the first revision, use the existing explicit reviews and add diagnostic distinctions. Review genuine reference-policy exceptions explicitly; do not silently rewrite annotations to match predictions, or equate identical text at different positions. Group references remain partial, so general group accuracy and false-positive rates are not yet established. Expand explicit group reviews before claiming a grouping improvement from an aggregate score.

## Implementation order and protection against overfitting

1. Fix inline-clause tolerance with focused containment and neighboring-clause checks. Keep native order and detected-table boundaries unchanged. Replay the complete set and show individual changes.
2. Construct complete captions and meaningful subspans. Check whether the reference caption can be proposed before tuning selection costs; document cases that require finer source text geometry.
3. Replace loose checkbox/composite proposals with supported question/option and clause/component configurations. Reuse the current global assignment solver; this evidence does not justify replacing it. Remove the unconditional common-question reward in favor of supported structural evidence, not simply all question candidates.
4. Improve row/lane evidence and reconsider abstention after the missing alternatives are representable. Keep the number of cost terms small and interpretable.

For every change, replay all pages and inspect regressions as well as improvements. Add a few structural counterexamples: wrapped caption versus two adjacent captions; long option A with an inserted date; two independent neighboring fields versus one split identifier; slightly perturbed clause bounds; a long row with an unrelated nearby paragraph; labels/values on opposite sides of a detected cell boundary. Test equivalent layouts with changed text, scale and small coordinate shifts. Do not add country names, field-name regexes, phrase lists, or fixture IDs to candidate logic.

These 11 PDFs are now a development set. Leave an entire form family out when choosing any parameters, and validate the final choices on newly annotated forms with different layouts/languages. The 20-field experiment is a regression result on known evidence, not a claim of held-out robustness. Missed tables stay a separate layout-model error bucket and do not get an optimizer recovery heuristic.

## Reproducibility

- [Candidate diagnostics](candidate-diagnostics.json), [case classifications](failure-cases.json), [selected groups](selected-groups.json), [experimental changes/results](ablations.json).
- `DOCLING_DEVICE=cpu PYTHONPATH=. .venv/bin/python output/acroform-keying-failure-audit/ablation_replay.py` runs the four isolated in-memory experiments from the repository root. It changes no optimizer, reference, or original report files.
- Experimental counts use the existing spatial evaluator, with the reference-policy limits described above. Removing all question candidates corrected no flagged local labels and lost three previously correct ones; this is additional evidence for changing group construction rather than dropping grouping.
