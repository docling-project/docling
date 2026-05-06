## Issue 4: Missing \log command

The equation below uses &lt;m:func&gt; with fName='log'.
Expected LaTeX: y = \log(x)
Docling produces: y = l o g(x)  (letters rendered as italic variables)
and emits a warning about an unrecognized function name.

The &lt;m:fName&gt; element contains a plain-text run with style 'p' (upright).

$$y = \log(\left(x\right))$$