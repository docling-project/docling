# Source fidelity and targeted verification

Use this guidance when exact values, formulas, table relationships,
quotations, or conversion anomalies affect the requested result. Ordinary
reading does not require a full-document manual audit; scale checks to the
consequences of an error and the requested scope.

## Completion and correctness

Conversion status/errors and page coverage describe processing, not accuracy
against the source. For a selected-page conversion, report the selected range
and retain the mapping to physical source pages; printed page labels may
differ. Checking a few regions does not verify the whole document. Warning
counts are not error counts, and absence of warnings does not prove accuracy.

Report consequential findings as unchecked, verified, corrected with evidence,
or unresolved. These are reporting conventions, not Docling API fields. Keep
an unresolved critical value qualified, while still using independent verified
content.

## Triage by symptom

| Symptom | Check against the source | Avoid |
|---|---|---|
| Suspicious scientific notation or units | Exponent sign, superscript placement, multiplication sign, denominator and unit power | Globally rewriting digit sequences into exponents |
| Ambiguous decimal, range, sign or inequality | Original glyph, separator, bounds and nearby label | Treating plausible values or expected totals as proof |
| Formula incomplete or flattened | Fraction grouping, scripts, operators and whether formula enrichment produced output | Treating enabled enrichment as verification |
| Table value lacks a clear meaning | Row/column headers, merged spans, units, caption and applicable footnote | Treating the Markdown grid as the full table structure |
| Words interrupted by another column, caption or citation | Item tree, source layout, bounding boxes and line order | Joining nonadjacent fragments into a verbatim quotation |
| Statement ends at a page/chunk boundary | Adjacent clause, heading, table header, condition, negation and footnote | Dropping a condition to fit the chunk budget |
| Figure labels have unclear partners | Spatial pairing, legend, arrows and caption | Assigning numbers from flattened OCR order |
| Figure and body disagree | Whether the disagreement also exists in the original | Harmonizing a source contradiction as an OCR correction |
| Empty, duplicate or garbled output | Page image, native text, OCR, raw JSON and final export | Assuming every anomaly is an OCR problem |

For example, `7×10 12` or a flattened unit power can indicate lost layout, but
ordinary integers can look equally suspicious. Regexes can nominate regions
for review; they cannot restore an unknown exponent or sign.

## Locate the first stage that differs

Compare the affected source page/image (and native text where relevant) with
Docling's raw item text, labels, provenance and tables. Then compare the export
or custom wrapper, chunks, extracted claims and final quotations.

If raw JSON is correct but an export loses a relationship, investigate the
exporter/wrapper. If parsed text is present but a chunk drops a condition,
investigate chunk context. OCR run directly on Office images is a different
path from Docling's PDF pipeline. Record the observed stage separately from
the suspected cause: a glyph missing in raw JSON still needs investigation to
distinguish PDF decoding, OCR, layout and model behavior.

## Structure and chunks

For evidence-sensitive work, preserve the raw structured result alongside the
readable export, including:

- Source identity/hash, conversion options and actual package/model versions.
- Item `self_ref`, label, parent/child relationships and document body order.
- Available provenance: physical page, bounding box, coordinate origin and
  character spans. Mark absent provenance as unavailable rather than guessing.
- Table cells with row/column offsets, spans and header flags; retain linked
  captions/footnotes where present. Review associations inferred by a wrapper.
- A source page or readable crop sufficient to inspect the finding, with page
  mapping and render settings. A crop may omit an applicable footnote.

When wrapping an OCR engine directly, retain available line/polygon positions
and scores rather than only concatenating text. Engine confidence is not proof
of numerical correctness.

Give each chunk the adjacent context needed to retain conditions, negations,
table headers and applicable footnotes. Keep context references distinct from
primary content so overlap is not counted as independent evidence. Do not
attach every nearby footnote indiscriminately.

Bind quotations to continuous source spans. If reading order interleaves
unrelated text, use separately located spans or a labelled paraphrase. A
visually verified transcription must be distinguished from a verbatim quote
of the raw parser output.

## Targeted retries

Keep the original result and inspect one affected page/region first. Record a
hypothesis and the option being changed. Check the installed version's API or
CLI help rather than assuming a flag from another release exists.

- For damaged native text or missing scanned content, compare appropriate OCR
  modes or a supported local engine on selected pages. Full-page OCR can also
  replace good native text with worse recognition.
- For table boundaries/matching, compare relevant structure options such as
  accuracy mode or cell matching. Neither is a universal fix; accurate mode
  may already be the default.
- For undecoded formulas, evaluate formula enrichment on a small sample and
  compare with the source. It does not guarantee recovery of every inline
  exponent or equation.
- A local VLM is another candidate for difficult layouts. Verify newly
  generated content, including values previously correct.

Increasing an exported image's scale alone does not prove an OCR/model stage
used a higher resolution. Avoid repeated identical retries. If a targeted
alternative does not improve the critical region, retain the uncertainty and
choose further diagnosis or an explicit limitation. A retry limit does not
make content verified. Expand to a batch only after the sample supports it.

Keep sensitive samples local unless the user has authorized the chosen remote
service. Disabling remote model services is not an offline guarantee: URL
inputs and model downloads have separate network behavior.

## Corrections and regression checks

Preserve the source and raw extraction. Record a correction separately with
the source hash, parse/configuration identity, item reference and physical
page/region, exact raw text or fingerprint, corrected text, reference image
hash/page/render settings, reason, reviewer and review time.

Apply it only while the source, location and raw text still match. On mismatch,
leave it unapplied pending re-verification. A similar string elsewhere needs
its own evidence. A corrected exponent does not resolve unrelated header or
scientific-validity concerns; retain target/measured/simulated context.

For reusable parser/wrapper or configuration changes, use a small regression
set that exercises the failure mechanism: signed exponents, unit powers,
merged headers, footnotes, cross-page conditions, interrupted reading order
and deliberate source contradictions. Include an ordinary integer/plain-text
control so a repair does not manufacture exponents or alter correct content.

Prefer synthetic or publicly licensed minimal samples for upstream work.
Check expected values/relationships, provenance, appropriate unresolved
findings and unintended changes to neighboring content. Report the scope
checked and whether output came from the raw parser, an evidence-backed
correction or an alternative conversion.
