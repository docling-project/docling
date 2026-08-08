"""Character-level normalization applied to text extracted from documents."""

import re
import unicodedata

# Arabic Presentation Forms-A (U+FB50-U+FDFF) and Forms-B (U+FE70-U+FEFF).
# Some PDF fonts encode Arabic with these compatibility codepoints, which carry
# one codepoint per contextual letter shape (initial/medial/final/isolated)
# instead of the canonical letter. Text stored that way is not searchable or
# comparable against normally-encoded Arabic, since a shaped codepoint never
# matches its U+0600 block equivalent.
_ARABIC_PRESENTATION_RE = re.compile(r"[ﭐ-﷿ﹰ-﻿]+")


def normalize_arabic_presentation_forms(text: str) -> str:
    """Fold Arabic presentation forms back to canonical Arabic letters.

    NFKC is applied per matched run rather than to the whole string, so it can
    never rewrite other scripts: a global NFKC would also fold superscripts,
    subscripts and full-width forms (e.g. "x²" -> "x2").
    """
    return _ARABIC_PRESENTATION_RE.sub(
        lambda m: unicodedata.normalize("NFKC", m.group(0)), text
    )
