## Issue 1: Fraction as superscript base not grouped

The equation below raises a fraction to the power 2.
Expected LaTeX: {\frac{(x-c)}{v}}^2
Docling produces: \frac{(x-c)}{v}^2

When &lt;m:sSup&gt; has an &lt;m:f&gt; (fraction) as its base &lt;m:e&gt;, docling emits
\frac{num}{den}^2 without grouping braces. In LaTeX this is ambiguous:
some renderers apply ^2 only to the last token, not the whole fraction.
The output should be {\frac{(x-c)}{v}}^2 or \left(\frac{(x-c)}{v}\right)^2.

$${\frac{(x-c)}{v}}^{2}$$