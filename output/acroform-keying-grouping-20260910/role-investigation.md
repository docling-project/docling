# Question roles and group boundaries: investigation

The current failures come from **missing or incompatible interpretations before solving**, as well as a reward that favors splitting. Increasing the search distance or strengthening the grouping reward cannot fix all of them. Conditional question–caption selection and complete-assignment comparisons remain useful, but the interpretations being compared need to improve first.

This investigation leaves the accepted optimizer, the previous experiment, its reports, and all references unchanged. It adds a [runnable diagnostic](investigate_roles.py) and its [recorded evidence](role-investigation.json). All widget numbers below are native indices. Diagnostic label IDs refer only to this frozen experimental input; they are not proposed algorithm rules.

## 1. Unused caption text becomes an unrelated question

On the [language-preference form](report/usa_cluster011_partial_page_prefilled__f1040lep-p1.html), the intended structure is:

```text
Question about preferred language
    [ ] language 1                 [ ] language 12
    [ ] language 2                 [ ] language 13
    ...                           ...
    [ ] Arabic (Arabic text)       ...
    ...                           ...
```

The model correctly labels the Arabic checkbox with an existing caption fragment. The separate Arabic-text region then remains unused. It is on the same line as that option, but the question guard looks only one text-height away from a checkbox. This fragment starts 6.53 text-heights beyond its own checkbox, so it passes as a question for five subsequent options.

Worse, the closing parenthesis from that same left-column option becomes the question for four options in the **right column**. The code permits a question anywhere to the left of the first member on the same row. It does not establish that the text belongs to the target column. It also counts left-column option captions as intervening questions for some right-column group proposals. These are two effects of the same missing context: which existing option row a text block belongs to.

This does not require joining the fragmented caption. Its existing fragment can remain the caption. The other source regions need geometric role context so that being unused does not itself make them suitable questions.

There is also a scoring failure. The correct complete 21-option configuration exists. Requiring it and resolving the entire original objective costs 0.906 more than the chosen false groups. It is not merely absent from the nearest-alternative report: the present model actually prefers the wrong segmentation.

The repetition reward helps explain why. A five-option group gets a reward of 1.6, so two such groups get 3.2; one ten-option group gets only 1.8. Before evaluating whether the new question is credible, starting another group can already be advantageous. The earlier linear-reward control reversed the problem: it restored the language group but merged RC7190's two questions. Neither reward is an adequate substitute for boundary evidence.

## 2. Correct caption alternatives are removed before global comparison

On [RF-1125 page 1](report/norway__rf-1125s-p1.html), three reviewed captions sit to the left of three checkboxes, with explanatory text to the right:

```text
Shared prompt     Caption A       [ ]    Explanation A
                  Longer caption [ ]    Explanation B
                  Caption C       [ ]    Explanation C
```

The caption proposal uses the gap between the end of a left-hand caption and the checkbox. It permits at most four text-heights. The three actual gaps are 5.72, 3.34 and 4.79: two correct captions are excluded. Their common left alignment is visible, but their varying widths produce varying end gaps.

For each remaining question/side/widget combination the code then chooses **one** caption before invoking the solver. Conditional selection therefore does not yet mean a joint choice among all plausible captions. Here it leaves only the right-hand explanation arrangement for the complete group.

The model chooses the first actual caption, “Vearrodieđáhusa,” as the shared question. The real left-hand prompt is also available, but even forcing that prompt leaves the wrong right-hand captions. Changing the question alone is insufficient. The existing visual local references explicitly identify the left-hand captions; no translation was needed to diagnose this.

The next experiment needs to retain competing caption associations and recognize a repeated column of text without requiring the ends of differently sized captions to align. It should not merely raise four to another fixed number.

## 3. A caption may legitimately introduce suboptions

On [F1120 Schedule O, question 6](report/usa_cluster019_singlecolumn__f1120so-p1.html):

```text
Question 6
    [ ] Yes
        [ ] (i)  ...
        [ ] (ii) ...
    [ ] No
```

Two separate defects occur:

- There is **no candidate for the outer Yes/No group**, native widgets 28 and 37. The native-sequence walk reaches the indented widget 29 after Yes and stops. It cannot skip a nested branch to find the next sibling. Keeping native output order does not require treating every next checkbox as the next sibling.
- There is a child group for widgets 29 and 32 under “a Yes.” But its question text shares source atom 94 with the Yes checkbox's caption. The global one-owner-per-text constraint makes those two legitimate uses incompatible. Once the child group takes Yes, question 6 becomes the Yes checkbox's local caption.

A controlled solve removing the child questions that claim that Yes atom restores “a Yes.” as the checkbox caption. That directly confirms the ownership conflict.

The representation should let the child group refer to the already-labelled parent option. It should not allow arbitrary duplicate ownership of text. Indentation can propose that parent/child relationship; where multiple parents fit, the alternatives should remain explicit.

## 4. Explanatory paragraphs put the real question outside the search window

On [GST111 page 1](report/ca_cluster001_forms__gst111-fill-08e-p1.html):

```text
Question about institution type
    Explanatory paragraph
    Explanatory paragraph
    Explanatory paragraph
Caption A [ ]       Caption B [ ]       Caption C [ ]
```

The real question is 14.58 text-heights above the first checkbox. The above-question window is three. There is no proposal using that question.

The first option's caption is instead used as the question for the other two options. The guard meant to prevent that allows only a one-height caption gap, whereas this legitimate caption is 1.62 heights from its own checkbox.

Thus one threshold hides the actual question and another fails to recognize a caption. Increasing both would still leave a semantic ambiguity between an instruction, a note, and a question above a group. Group proposals should first establish the option arrangement, then consider nearby existing blocks around that arrangement, retaining uncertain prompt choices. Correct local captions must remain usable when the shared prompt cannot be established.

## 5. Some boundaries remain genuinely ambiguous from these inputs

The [Chinese A–E example](uncertainty-report/usa_cluster000_chinese__f14446cn-p1.html) has a full-width note after A and before B. Its left edge is aligned with the question and checkbox column; it is not conveniently indented like a caption continuation. Two plausible geometric structures remain:

```text
Question                    Question 1
    A                           A
    Note                    Question 2 / note
    B ... E                     B ... E
```

“Stop at an intervening block” and “continue across all intervening blocks” each choose one interpretation without sufficient evidence. The uncertainty comparison appropriately withholds the disputed association. On page 3 it similarly exposes the short consent prompt versus the preceding legal paragraph. These cases should remain controls where abstention is acceptable; they should not drive keyword rules.

In contrast, [RC7190](report/ca_other__rc7190-ws-p1.html) has two successive side prompts accompanying two repeated option lists. That successful separation is a required regression control. A rule that indiscriminately rejects text beside an option would destroy the correct “Application type” question.

## Recommended next experiment

Work on the candidate set and compatible roles before another objective sweep:

1. **Propose option arrangements independently of a selected question.** Reuse existing caption candidates, retaining plausible alternatives on either side. Use the repeated positions of widgets and text columns to propose siblings. Allow native subsequences to skip a proposed indented child branch; export still follows the exact native sequence.
2. **Evaluate question and boundary roles in that context.** A block inside an existing option row is evidence for that row, not a free new question merely because some of its text is unused. A candidate in another column needs a supported connection to that group. Above and beside prompts are both valid. Ambiguous text-row ownership remains an alternative, not a universal hard ban.
3. **Represent a child group as referring to its parent option.** This permits the legitimate Yes example while retaining exclusive ownership of the source text itself.

Then evaluate the objective on complete competing arrangements: one group versus two, continued group versus new question, and flat versus nested options. Starting another group must have supporting question/boundary evidence; a fresh size-based bonus alone is insufficient. Do not tune coefficients against the language form while ignoring RC7190.

This is a recommended design direction, **not an implemented or validated replacement**. The smallest useful implementation gate is: retain the missing correct alternatives, exclude neither left nor right prompts categorically, and make the valid nested Yes interpretation feasible. Then replay all 19 pages and inspect group changes separately from local-caption scores. Correct local captions must not regress just to improve shared-question coverage.

Keep the frozen table exclusions and native order. Missed/truncated tables and caption reconstruction remain outside this work. Do not add new lexical heuristics. The group proposal sets were identical on all 19 pages after replacing every label's text with “?” while keeping geometry and atom identity. This verifies the current group builder's text independence; it does **not** remove or validate the inherited singleton scorer's alphabetic/length heuristics.

## Reproduce the new checks

From the repository root:

```bash
DOCLING_DEVICE=cpu PYTHONPATH=.:output/acroform-keying-grouping-20260910 \
  .venv/bin/python output/acroform-keying-grouping-20260910/investigate_roles.py
```

The diagnostic checks proposal text invariance on all 19 snapshots; confirms the two absent question/group candidates; forces existing group alternatives with the unchanged objective; and demonstrates the Yes text-ownership conflict. It asserts optimal solves and verifies that the experimental source and accepted optimizer are unchanged. It reads frozen inputs and writes only `role-investigation.json`.
