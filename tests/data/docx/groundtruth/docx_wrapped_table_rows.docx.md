## Wrapped table rows

Word processors and document management tools can wrap a table row in a custom XML element or in a content control. They can also wrap a single cell. Every row and every cell below must appear in the converted table.

Review log, one wrapper kind per row:

| Entry | Wrapper | Reviewer |
| - | - | - |
| 1 | none | A. Reviewer |
| 2 | customXml around the row | B. Editor |
| 3 | content control around the row | C. Approver |
| 4 | none | D. Auditor |
| 5 | nested wrappers around the row | E. Owner |

Release plan, a merged cell that continues into a wrapped row:

| Phase | Task |
| - | - |
| Preparation | Freeze the scope |
| Preparation | Collect the sign-offs |
| Preparation | Tag the release |
| Publication | Publish the notes |

End of document.