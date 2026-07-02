## Issue 2: Text-mode escapes in math context

The equation below uses U+2013 EN DASH as minus and U+005E as caret.
Expected LaTeX: x - y^2
Docling produces: x \text{ \textendash } y\text{ \textasciicircum }2

Both characters appear inside &lt;m:r&gt;&lt;m:t&gt; math runs but are escaped as text-mode LaTeX macros instead of kept as math operators.

$$x - y^2$$