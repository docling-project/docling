## Issue 3: Concatenated equation blocks

The paragraph below contains three separate &lt;m:oMath&gt; elements.
Expected: three separate $$ blocks ($$a = b$$, $$c = d$$, $$e = f$$)
Docling produces: one $$ block with all equations concatenated.

All three &lt;m:oMath&gt; elements are siblings inside a single &lt;w:p&gt;.

$$a=b$$

$$c=d$$

$$e=f$$