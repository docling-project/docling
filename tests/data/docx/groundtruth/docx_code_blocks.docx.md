## Code-detection fixture

### Case A - normal prose

This is an ordinary paragraph of prose describing how the parser works. It uses the default body font and must remain a plain text item.

### Case B - prose with one inline monospaced word

Call the printf function to print formatted output to standard out.

### Case C - paragraph in 'Source Code' style

```
import sys
print(sys.argv)
```

### Case D - fully monospaced paragraph, no code style

```
SELECT * FROM users WHERE active = 1;
```

### Case E - multi-line monospaced block

```
def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
```

### Case F - 'Source Reference' style (substring trap)

See the original source for details.

### Case G - 'Listing Number' style, no code chars (substring trap)

Listing 3.2

### Case H - pure-Courier prose, no code-indicative characters

This memo is set in a typewriter face for a vintage look and feel throughout.