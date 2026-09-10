# AcroForm replay inputs

`f1040lep.json` contains the native widgets, layout text, and detected tables
from the frozen September 8 review snapshot for page 1 of
`usa_cluster011_partial_page_prefilled__f1040lep.pdf`.

The source PDF SHA-256 is
`3239ffa65938d73021792bd6793538fe2a164391644910db6f98c0bc90c18fdd`.
The snapshot was parsed with `scripts.acroform_keying.Snapshot` and serialized
with defaults omitted. Existing field predictions and timing were discarded.
No expected associations were inserted into the input.

Expected spatial labels and group membership remain in the separately reviewed
`tests/data/groundtruth/acroform_keying/annotations.jsonl`. This input supports a
fast regression without external PDFs, images, OCR, or native parser bindings.
