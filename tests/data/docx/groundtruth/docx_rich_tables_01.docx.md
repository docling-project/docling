**Changes to the DOCLANG Core Spec**

Last Updated: 2026-06-18

This document identifies the edits needed for a PAS submission to ISO via ISO/IEC JTC/1.

The transformation of this technical draft for ISO compliance involves two phases:

- **Phase 1** (Macro-Level Structural Realignment) fixes the document's main layout. It forces the strict standard clause order, removes forbidden company and author names, cuts out critiques of competing formats, and changes all casual requirement language or uppercase tokens into standard lowercase ISO verbal forms (shall, should, may).
- **Phase 2** (Micro-Level Editorial Optimization) cleans up the small details required by the ISO House Style. This fixes layout bugs like ha nging paragraphs, adds spaces to large numbers (e.g., 65 535), writes out printable web links for physical paper copies, uses proper math subscript formatting, and limits all typography to standard fonts so it passes the final publishing system.

**Phase 1**

| **Feature** | **Action Needed** | **Comment/Links** |
| - | - | - |
| Document Clause Ordering | Reorganize into the mandatory sequence:  • Foreword (Mandatory, unnumbered)  • Introduction (Optional, unnumbered)  • 1 Scope (Mandatory, Clause 1)  • 2 Normative references (Mandatory, Clause 2)  • 3 Terms and definitions (Mandatory, Clause 3)  • 4 Symbols and abbreviated terms (Mandatory, Clause 4)  • 5 Conformance (Mandatory, Clause 5)  • 6 Technical Clauses...  • Annex A (normative)  • Annex B/C (informative)  • Bibliography (Mandatory, absolute end) | The current draft improperly mixes introductory narrative, design principles, and examples before standard layout clauses are established. See ISO/IEC Directives, Part 2, Clause 6. |
| Author & Corporate Profiles | Completely remove the table listing individual names and company affiliations. | Personal names, corporate branding, logos, and affiliations are prohibited in the normative body of an ISO publication. See ISO/IEC Directives, Part 2, Clauses 4 and 12.5.2. |
| Foreword Boilerplate | Replace the original foreword with the mandatory ISO/IEC JTC 1 PAS foreword text and retain only a neutral statement identifying the originating workshop. | ISO forewords follow prescribed wording and structure. See ISO/IEC Directives, Part 2, Clause 12. |
| Market / Design Motivation | Remove the Motivation clause and all comparisons criticizing Markdown, HTML, LaTeX, PageXML, ALTO XML, hOCR, or similar technologies. | ISO standards shall remain technologically neutral and avoid comparative or competitive claims. See ISO/IEC Directives, Part 2, Clause 4. |
| Evolutionary Versioning Prose | Remove historical development narratives, beta-version discussions, and Version 0.x compatibility explanations. | ISO standards describe current requirements, not product-development history. See ISO/IEC Directives, Part 2, Clauses 11 and 12. |
| Clause Header Casing | Convert all clause and subclause headings to sentence case. | Title-case headings are not permitted by ISO house style. See ISO/IEC Directives, Part 2, Clause 22.2. |
| Self-References | Replace all occurrences of “this specification”, “this standard”, and “this format” with “this document”. | ISO publications refer to themselves exclusively as “this document”. See ISO/IEC Directives, Part 2, Clause 10.6. |
| Scope Clause Compliance | Rewrite Clause 1 to describe the subject matter only and remove normative statements, implementation obligations, and references to conformance requirements. | The Scope clause shall define the extent and field of application of the document and shall not contain requirements. See ISO/IEC Directives, Part 2, Clause 14. |
| Normative References Clause | Create Clause 2 immediately after Scope. If no references are required, insert: “There are no normative references in this document.” | Clause 2 is mandatory. See ISO/IEC Directives, Part 2, Clause 15. |
| External Dependency Classification | Review references to XML, XSD, Unicode, URI specifications, and RFCs, classifying each as normative (Clause 2) or informative (Bibliography). | Any referenced document required to implement this document shall appear in Clause 2. See ISO/IEC Directives, Part 2, Clause 15. |
| Terms and Definitions Clause | Convert “Terminology” into “Terms and definitions” and format all entries according to ISO terminology conventions. | Definitions shall follow the prescribed ISO structure. See ISO/IEC Directives, Part 2, Clause 16. |
| Entry Notes in Definitions | Convert embedded explanatory remarks into formal “Note X to entry:” format. | Informal notes embedded in definitions are not permitted. See ISO/IEC Directives, Part 2, Clause 16.5.9. |
| Terminological Source Attribution | Add formal [SOURCE: ...] citations to imported definitions. | Borrowed definitions require source attribution. See ISO/IEC Directives, Part 2, Clause 16.5.10. |
| Terminology Consistency Audit | Establish a single preferred term for each concept and remove synonyms throughout the document. | ISO drafting follows a one-concept–one-term principle. See ISO/IEC Directives, Part 2, Clause 16. |
| Symbols and Abbreviated Terms Clause | Create Clause 4 or merge abbreviations into Clause 3. | Acronyms such as OTSL, XML, VLM, RAG, PII, GDPR, and XSD require centralized treatment. See ISO/IEC Directives, Part 2, Clause 17. |
| Conformance Definition | Create Clause 5 Conformance defining the implementation classes eligible to claim compliance. | Requirements cannot exist without identifying the responsible implementation target. See ISO/IEC Directives, Part 2, Clause 33. |
| Conformance Classes | Define separate conformance classes where applicable (e.g., document producers, validators, parsers, processors, renderers). | Distinct implementation categories may require different obligations and test criteria. See ISO/IEC Directives, Part 2, Clause 33. |
| Requirement Traceability | Ensure every “shall” statement identifies a responsible subject and can be objectively tested. | Requirements must be measurable and verifiable. See ISO/IEC Directives, Part 2, Clause 33. |
| RFC Verbal Compliance | Replace RFC 2119 uppercase keywords (MUST, SHOULD, MAY) with ISO verbal forms (shall, should, may). | ISO does not recognize RFC keyword conventions. See ISO/IEC Directives, Part 2, Clause 7. |
| Introduction Compliance | Remove all requirements, permissions, recommendations, and implementation rules from the Introduction. | The Introduction is informative only. See ISO/IEC Directives, Part 2, Clause 13. |
| Informative Annex Compliance | Remove normative language from all informative annexes. | Informative annexes shall not contain requirements. See ISO/IEC Directives, Part 2, Clause 20.2. |
| Annex Naming | Rename Appendix A/B/C to Annex A (normative), Annex B (informative), and Annex C (informative). | ISO does not use the term “Appendix”. See ISO/IEC Directives, Part 2, Clause 20. |
| Annex Reference Audit | Ensure Annexes A, B, and C are explicitly referenced from the body text. | Unreferenced annexes are commonly flagged during editorial review. See ISO/IEC Directives, Part 2, Clause 20. |

**Phase 2**

| **Feature** | **Action Needed** | **Comment/Links** |
| - | - | - |
| Cross-Reference Normalization | Replace Markdown anchors and informal references with ISO clause references. | References should use forms such as “see 7.3.2” or “see Annex A”. See ISO/IEC Directives, Part 2, Clause 10. |
| Hanging Paragraph Subclauses | Insert “General” subclauses beneath major clauses before introducing subordinate subclauses. | Prevents hanging paragraphs and ambiguous clause structures. See ISO/IEC Directives, Part 2, Clause 22.3.3. |
| Orphan Subclause Remediation | Ensure any clause containing a subclause .1 also contains a .2 or merge the subdivision back into the parent clause. | ISO numbering rules prohibit single-child subdivisions. See ISO/IEC Directives, Part 2, Clause 22.3.2. |
| XML Tag Typography | Apply consistent literal formatting to element names, attributes, markup fragments, and grammar symbols. | Distinguishes identifiers from prose and reduces ambiguity. See ISO/IEC Directives, Part 2, Clauses 9 and 24. |
| Formal Language Notation | Introduce a dedicated clause describing any normative grammar notation, schema language, or syntax conventions. | Formal languages should be specified using recognized notation rather than examples alone. See ISO/IEC Directives, Part 2, Clause 9.2. |
| Attribute Value Enumerations | Consolidate controlled vocabularies and attribute values into structured tables or formal rules. | Improves implementability, validation, and conformance testing. See ISO/IEC Directives, Part 2, Clause 5.6 & Clause 29. |
| Number Formatting | Replace comma-separated thousands with ISO spacing conventions (e.g., 65 535). | See ISO/IEC Directives, Part 2, Clause 9.1. |
| Printable URI / URL Strings | Display explicit URI strings for externally referenced resources. | Documents must remain usable when printed. See ISO/IEC Directives, Part 2, Clause 10.3. |
| Bi-directional Reference Auditing | Verify that all normative references are cited normatively and all bibliography entries are cited informatively. | Orphan references are not permitted. See ISO/IEC Directives, Part 2, Clauses 10.1 and 15.1. |
| Font and Style Normalization | Remove non-standard formatting, decorative boxes, custom colors, and visual styling. | ISO publishing systems normalize typography and layout automatically. See ISO/IEC Directives, Part 2, Clause 1. |
| Informative Code / Example Marking | Label all examples explicitly as EXAMPLE. | Examples must be clearly distinguished from requirements. See ISO/IEC Directives, Part 2, Clause 25. |
| Example Separation | Audit explanatory text surrounding examples to ensure examples cannot be interpreted as normative requirements. | Examples are informative only. See ISO/IEC Directives, Part 2, Clause 25. |
| Tabular Formats | Convert pipe-delimited text tables into proper table structures. | Tables shall be structurally defined. See ISO/IEC Directives, Part 2, Clause 29. |
| Non-Normative Implementation Narrative | Move implementation guidance and authoring-process commentary into notes or informative annexes. | Standards define technical outcomes, not internal author workflows. See ISO/IEC Directives, Part 2, Clause 24. |
| Mermaid Diagram Transmutation | Convert Mermaid source code into static figures. | Text-based diagram source code is completely unsuitable for ISO publication systems. See ISO/IEC Directives, Part 2, Clause 28.6.4. |
| Graphic Text Normalization | Standardize figure typography and remove branding from graphics. | Figures shall be neutral, completely clear, legible, and publication-ready. See ISO/IEC Directives, Part 2, Clause 28.5.2. |
| Bibliography Construction | Relocate all informative references into a final unnumbered Bibliography. | See ISO/IEC Directives, Part 2, Clause 21. |
| Asset Caption Formatting | Convert captions to ISO figure/table caption style and sentence case. | See ISO/IEC Directives, Part 2, Clause 28.2 & Clause 29.2. |
| Commercial Tools Footnotes | Add non-endorsement wording for references to trademarked products or technologies. | See ISO/IEC Directives, Part 2, Clause 31. |
| Mathematical Interval & Unit Formatting | Convert interval notation and enforce spacing between values and units. | See ISO/IEC Directives, Part 2, Clauses 9.1 and 9.4.1. |
| Percentage Symbol Clean-up | Replace % in prose with “per cent”. | The symbol % is restricted entirely to tabular matrices and literal code blocks. See ISO/IEC Directives, Part 2, Clause 9.4.1 & Annex B. |
| Schema Placeholder Optimization | Remove placeholder values, ellipses, and drafting artifacts from examples. | Draft remnants are flagged as incomplete specification leaks by automated ingestion systems. See ISO/IEC Directives, Part 2, Clause 4.1. |
| Subscript Coordinate Notation | Convert snake_case mathematical variables into proper mathematical notation with subscripts. | See ISO/IEC Directives, Part 2, Clause 9.3.1. |