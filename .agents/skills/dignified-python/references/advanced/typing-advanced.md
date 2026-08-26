---
description: Advanced typing patterns including cast() with assertions, Literal types for programmatic strings.
---

# Advanced Typing Reference

**Read when**: Using typing.cast(), creating Literal type aliases, narrowing types

---

## Using `typing.cast()`

### Core Rule

**ALWAYS verify `cast()` with a runtime assertion, unless there's a documented reason not to.**

`typing.cast()` is a compile-time only construct—it tells the type checker to trust you but performs
no runtime verification. If your assumption is wrong, you'll get silent misbehavior instead of a
clear error.

### Required Pattern

```python
from collections.abc import MutableMapping
from typing import Any, cast

# CORRECT: Runtime assertion before cast
assert isinstance(doc, MutableMapping), f"Expected MutableMapping, got {type(doc)}"
cast(dict[str, Any], doc)["key"] = value

# CORRECT: Alternative with hasattr for duck typing
assert hasattr(obj, '__setitem__'), f"Expected subscriptable, got {type(obj)}"
cast(dict[str, Any], obj)["key"] = value
```

### Anti-Pattern

```python
# WRONG: Cast without runtime verification
cast(dict[str, Any], doc)["key"] = value  # If doc isn't a dict-like, silent failure
```

### When to Skip Runtime Verification

**Default: Always add the assertion when cost is trivial (O(1) checks like `in`, `isinstance`).**

Skip the assertion only in these narrow cases:

1. **Immediately after a type guard**: The check was just performed and would be redundant

   ```python
   if isinstance(value, str):
       # No assertion needed - we just checked
       result = cast(str, value).upper()
   ```

2. **Performance-critical hot path**: Add a comment explaining the measured overhead
   ```python
   # Skip assertion: called 10M times/sec, isinstance adds 15% overhead
   # Type invariant maintained by _validate_input() at entry point
   cast(int, cached_value)
   ```

**What is NOT a valid reason to skip:**

- "Click validates the choice set" - Add assertion anyway; cost is trivial
- "The library guarantees the type" - Add assertion anyway; defense in depth
- "It's obvious from context" - Add assertion anyway; future readers benefit

### Why This Matters

- **Silent bugs are worse than loud bugs**: An assertion failure gives you a stack trace and clear
  error message
- **Documentation**: The assertion documents your assumption for future readers
- **Defense in depth**: Third-party libraries can change behavior between versions

---

## Programmatically Significant Strings

**Use `Literal` types for strings that have programmatic meaning.**

When strings represent a fixed set of valid values (error codes, status values, command types),
model them in the type system using `Literal`.

### Why This Matters

1. **Type safety** - Typos caught at type-check time, not runtime
2. **IDE support** - Autocomplete shows valid options
3. **Documentation** - Valid values are explicit in the code
4. **Refactoring** - Rename operations work correctly

### Naming Convention

**Use kebab-case for all internal Literal string values:**

```python
# CORRECT: kebab-case for internal values
IssueCode = Literal["orphan-state", "orphan-dir", "missing-branch"]
ErrorType = Literal["not-found", "invalid-format", "timeout-exceeded"]
```

**Exception: When modeling external systems, match the external API's convention:**

```python
# CORRECT: Match GitHub API's UPPER_CASE
PRState = Literal["OPEN", "MERGED", "CLOSED"]

# CORRECT: Match GitHub Actions API's lowercase
WorkflowStatus = Literal["completed", "in_progress", "queued"]
```

The rule is: kebab-case by default, external convention when modeling external APIs.

### Pattern

```python
from dataclasses import dataclass
from typing import Literal

# CORRECT: Define a type alias for the valid values
IssueCode = Literal["orphan-state", "orphan-dir", "missing-branch"]

@dataclass(frozen=True)
class Issue:
    code: IssueCode
    message: str

def check_state() -> list[Issue]:
    issues: list[Issue] = []
    if problem_detected:
        issues.append(Issue(code="orphan-state", message="description"))  # Type-checked!
    return issues

# WRONG: Bare strings without type constraint
def check_state() -> list[tuple[str, str]]:
    issues: list[tuple[str, str]] = []
    issues.append(("orphen-state", "desc"))  # Typo goes unnoticed!
    return issues
```

### When to Use Literal

- Error/issue codes
- Status values (pending, complete, failed)
- Command types or action names
- Configuration keys with fixed valid values
- Any string that is compared programmatically

### Decision Checklist

Before using a bare `str` type, ask:

- Is this string compared with `==` or `in` anywhere?
- Is there a fixed set of valid values?
- Would a typo in this string cause a bug?

If any answer is "yes", use `Literal` instead.
