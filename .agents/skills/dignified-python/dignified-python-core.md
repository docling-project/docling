---
---

# Dignified Python - Core Standards

This document contains the core Python coding standards that apply to 80%+ of Python code. These
principles are loaded with every skill invocation.

For conditional loading of specialized patterns:

- CLI development -> Load `cli-patterns.md`
- Subprocess operations -> Load `subprocess.md`

For detailed reference material, see the "When to Read Each Reference" section in SKILL.md.

---

## Default Stance: Prefer Explicit Preconditions

This skill leans LBYL when a cheap, precise precondition keeps intent clearer than a `try/except`.

```python
# CORRECT: Check first
if key in mapping:
    value = mapping[key]
    process(value)

# WRONG: Exception as control flow
try:
    value = mapping[key]
    process(value)
except KeyError:
    pass
```

---

## Exception Handling Basics

### Core Principle

**Prefer LBYL for routine branching when the precondition is cheap and precise.**

LBYL means checking conditions before acting. EAFP (Easier to Ask for Forgiveness than Permission)
means trying operations and catching exceptions. In this skill, default to LBYL for ordinary
branching, but use EAFP when the operation itself is the authoritative test or when you're
translating failures at a boundary.

### Dictionary Access Patterns

```python
# CORRECT: Membership testing
if key in mapping:
    value = mapping[key]
    process(value)
else:
    handle_missing()

# ALSO CORRECT: .get() with default
value = mapping.get(key, default_value)
process(value)

# CORRECT: Check before nested access
if "config" in data and "timeout" in data["config"]:
    timeout = data["config"]["timeout"]

# WRONG: KeyError as control flow
try:
    value = mapping[key]
except KeyError:
    handle_missing()
```

### When Exceptions ARE Acceptable

Exceptions are a good fit at:

1. **Error boundaries** (CLI/API level)
2. **Operations where the action itself is the authoritative test**
3. **Adding context before re-raising**

**Default: Let exceptions bubble up**

For detailed exception handling patterns including B904 chaining, third-party API examples, and
anti-patterns, see `references/advanced/exception-handling.md`.

---

## Path Operations

### The Golden Rule

Use `.exists()` when filesystem presence is part of your requirement, not as a blanket precondition
for `.resolve()` or `.is_relative_to()`.

### Why This Matters

- `Path.resolve()` on Python 3.11 resolves non-existent paths unless you pass `strict=True`
- `Path.is_relative_to()` returns `bool`; it does not raise `ValueError` when a path is not under
  another path
- Broad exception wrappers around these APIs usually hide intent instead of clarifying it

### Correct Patterns

```python
from pathlib import Path

# CORRECT: Check existence only when you need a real filesystem entry
for wt_path in worktree_paths:
    wt_path_resolved = wt_path.resolve()
    if not wt_path_resolved.exists():
        continue
    if current_dir.is_relative_to(wt_path_resolved):
        current_worktree = wt_path_resolved
        break

# ALSO CORRECT: Ask resolve() to fail when absence is an error
config_dir = config_path.resolve(strict=True)

# WRONG: Broad exception handling around APIs that already communicate the result directly
for wt_path in worktree_paths:
    try:
        wt_path_resolved = wt_path.resolve()
        if current_dir.is_relative_to(wt_path_resolved):
            current_worktree = wt_path_resolved
            break
    except OSError:
        continue
```

### Pathlib Best Practices

**Always Use Pathlib (Never os.path)**

```python
# CORRECT: Use pathlib.Path
from pathlib import Path

config_file = Path.home() / ".config" / "app.yml"
if config_file.exists():
    content = config_file.read_text(encoding="utf-8")

# WRONG: Use os.path
import os.path
config_file = os.path.join(os.path.expanduser("~"), ".config", "app.yml")
```

**Always Specify Encoding**

```python
# CORRECT: Always specify encoding
content = path.read_text(encoding="utf-8")
path.write_text(data, encoding="utf-8")

# WRONG: Default encoding
content = path.read_text()  # Platform-dependent!
```

---

## Import Organization

### Core Rules

1. **Default: ALWAYS place imports at module level**
2. **Use absolute imports only** (no relative imports)
3. **Inline imports only for specific exceptions** (circular deps, TYPE_CHECKING, conditional
   features)

```python
# CORRECT: Module-level imports
import json
import click
from pathlib import Path
from myapp.config import load_config

def my_function() -> None:
    data = json.loads(content)

# CORRECT: Absolute import
from myapp.config import load_config

# WRONG: Relative import
from .config import load_config

# WRONG: Inline imports without justification
def my_function() -> None:
    import json  # NEVER do this
```

For detailed inline import patterns and when they're legitimate, see `references/module-design.md`.

---

## Performance Guidelines

### Properties Must Be O(1)

```python
# WRONG: Property doing I/O
@property
def size(self) -> int:
    return self._fetch_from_db()

# CORRECT: Explicit method name
def fetch_size_from_db(self) -> int:
    return self._fetch_from_db()

# CORRECT: O(1) property
@property
def size(self) -> int:
    return self._cached_size
```

### Magic Methods Must Be O(1)

```python
# WRONG: __len__ doing iteration
def __len__(self) -> int:
    return sum(1 for _ in self._items)

# CORRECT: O(1) __len__
def __len__(self) -> int:
    return self._count
```

---

## Anti-Patterns

### No Backwards Compatibility Preservation (Default)

```python
# WRONG: Keeping old API unnecessarily
def process_data(data: dict, legacy_format: bool = False) -> Result:
    if legacy_format:
        return legacy_process(data)
    return new_process(data)

# CORRECT: Break and migrate immediately
def process_data(data: dict) -> Result:
    return new_process(data)
```

### No Re-Exports: One Canonical Import Path

**Core Principle:** Every symbol has exactly one import path. Never re-export.

```python
# WRONG: __all__ exports create duplicate import paths
# myapp/__init__.py
from myapp.core import Process
__all__ = ["Process"]

# CORRECT: Empty __init__.py, import from canonical location
# from myapp.core import Process
```

**When re-exports ARE required** (plugin entry points): Use explicit `import X as X` syntax:

```python
# CORRECT: Explicit re-export syntax for required entry points
from myapp.core.feature import my_function as my_function
```

### Declare Variables Close to Use

```python
# WRONG: Variable declared far from use
def process_data(ctx, items):
    result_path = compute_result_path(ctx)  # Declared here...
    # 20+ lines of other logic...
    save_to_path(transformed, result_path)  # ...used here

# CORRECT: Inline at use site
def process_data(ctx, items):
    validate_items(items)
    transformed = transform_items(items)
    save_to_path(transformed, compute_result_path(ctx))
```

### Don't Destructure Objects Into Single-Use Locals

```python
# WRONG: Unnecessary field extraction
result = fetch_user(user_id)
name = result.name      # only used once below
email = result.email    # only used once below
send_notification(name, email, role)

# CORRECT: Access fields directly
user = fetch_user(user_id)
send_notification(user.name, user.email, user.role)
```

### Indentation Depth Limit

**Maximum indentation: 4 levels**

```python
# WRONG: Too deeply nested (5 levels)
def process_items(items):
    for item in items:
        if item.valid:
            for child in item.children:
                if child.enabled:
                    for grandchild in child.descendants:
                        pass  # 5 levels deep!

# CORRECT: Extract helper functions
def process_items(items):
    for item in items:
        if item.valid:
            process_children(item.children)

def process_children(children):
    for child in children:
        if child.enabled:
            process_descendants(child.descendants)
```

### Keep Context Managers Inline in `with` Statements

```python
# CORRECT: Context manager stays in with statement
with (cm_a if condition else nullcontext()):
    do_work()

# CORRECT: Multiple conditional context managers
with (lock if thread_safe else nullcontext()):
    process(data)

# WRONG: Extracting to intermediate variable obscures lifecycle
cm = cm_a if condition else nullcontext()
with cm:
    do_work()
```

Context managers belong in `with` statements where the `__enter__`/`__exit__` lifecycle is explicit.
Do not extract them to intermediate variables. If the inline expression is genuinely overwhelming,
extract the logic into a helper function that returns the context manager.

---

## Backwards Compatibility Philosophy

**Default stance: NO backwards compatibility preservation**

Only preserve backwards compatibility when:

- Code is clearly part of public API
- User explicitly requests it
- Migration cost is prohibitively high (rare)

Benefits:

- Cleaner, maintainable codebase
- Faster iteration
- No legacy code accumulation
- Simpler mental models

---

## See Also

For detailed guidance on specialized topics:

- **Exception chaining (B904)**: `references/advanced/exception-handling.md`
- **ABC vs Protocol**: `references/advanced/interfaces.md`
- **typing.cast() assertions**: `references/advanced/typing-advanced.md`
- **Import-time side effects, @cache**: `references/module-design.md`
- **Default parameters, keyword-only args**: `references/advanced/api-design.md`
- **All decision checklists**: `references/checklists.md`
