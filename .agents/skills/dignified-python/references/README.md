# Dignified Python Reference Documentation

Opinionated Python standards for writing clean, maintainable, modern Python code.

## Table of Contents

### Core Standards

- **[dignified-python-core.md](../dignified-python-core.md)** - Essential Python standards (always
  apply)
  - Modern type syntax (list[str], str | None)
  - LBYL exception handling patterns
  - Pathlib operations
  - Absolute imports
  - Error boundaries

- **[cli-patterns.md](../cli-patterns.md)** - Command-line interface patterns
  - Click usage patterns
  - Argparse best practices
  - CLI error handling
  - Configuration management

### Version-Specific Features (`versions/`)

- **[python-3.10.md](../versions/python-3.10.md)** - Python 3.10+ features
  - Structural pattern matching (match/case)
  - Parenthesized context managers
  - Better error messages
  - Type union operator (X | Y)

- **[python-3.11.md](../versions/python-3.11.md)** - Python 3.11+ features
  - Exception groups (ExceptionGroup)
  - except\* syntax
  - Self type
  - Variadic generics

- **[python-3.12.md](../versions/python-3.12.md)** - Python 3.12+ features
  - Type parameter syntax (Generic[T])
  - Override decorator
  - Per-interpreter GIL
  - f-string improvements

- **[python-3.13.md](../versions/python-3.13.md)** - Python 3.13+ features
  - Experimental free-threading
  - JIT compilation
  - Improved error messages
  - Performance improvements

### Advanced Topics (`advanced/`)

- **[exception-handling.md](./advanced/exception-handling.md)** - Exception patterns
  - LBYL (Look Before You Leap) patterns
  - Error boundaries
  - Exception chaining
  - Custom exceptions

- **[interfaces.md](./advanced/interfaces.md)** - Interface design
  - ABC (Abstract Base Class) patterns
  - Protocol types
  - Gateway layer interfaces
  - Type narrowing

- **[typing-advanced.md](./advanced/typing-advanced.md)** - Advanced typing
  - Generic types
  - Type narrowing
  - Literal types
  - TypedDict and dataclasses

- **[api-design.md](./advanced/api-design.md)** - API design principles
  - Function signatures
  - Parameter complexity
  - Code organization
  - Production application code examples

## Philosophy

### Core Principles

**Modern Type Syntax**: Use Python 3.10+ type syntax everywhere

```python
# Good (modern)
def process(items: list[str]) -> str | None:
    pass

# Avoid (legacy)
from typing import List, Optional
def process(items: List[str]) -> Optional[str]:
    pass
```

**Prefer LBYL When It Is Cheap and Precise**: This skill leans toward explicit precondition
checks for routine branching, while still using targeted `try/except` blocks when parsing or API
calls are the authoritative test.

```python
# Good (LBYL)
if path.exists():
    content = path.read_text()

# Avoid (EAFP)
try:
    content = path.read_text()
except FileNotFoundError:
    pass
```

**Pathlib Over os.path**: Use pathlib for all file operations

```python
# Good
from pathlib import Path
config_path = Path("config.yaml")
if config_path.exists():
    content = config_path.read_text()

# Avoid
import os
if os.path.exists("config.yaml"):
    with open("config.yaml") as f:
        content = f.read()
```

**Absolute Imports**: Never use relative imports

```python
# Good
from myproject.utils import helper

# Avoid
from .utils import helper
from ..shared import helper
```

**Error Boundaries at CLI Level**: Handle errors at the CLI entry point, not deep in the stack

## Quick Reference

### Type Annotations

```python
# Basic types
def greet(name: str) -> str:
    return f"Hello, {name}"

# Collections (modern syntax)
def process(items: list[str], mapping: dict[str, int]) -> tuple[str, int]:
    pass

# Optional/Union (modern syntax)
def find(query: str) -> str | None:
    pass

# Multiple types
def parse(value: str | int | float) -> float:
    pass
```

### LBYL Patterns

```python
# File operations
if path.exists():
    content = path.read_text()

# Dictionary access
if "key" in data:
    value = data["key"]

# Attribute access
if hasattr(obj, "method"):
    obj.method()

# Type checking
if isinstance(value, str):
    result = value.upper()
```

### Pathlib Operations

```python
from pathlib import Path

# Create path
config = Path("config.yaml")
data_dir = Path("/data")

# Check existence
if config.exists():
    pass

# Read/write
content = config.read_text()
config.write_text("data")

# Directory operations
for file in data_dir.glob("*.txt"):
    print(file.name)

# Path manipulation
full_path = data_dir / "subdir" / "file.txt"
parent = full_path.parent
name = full_path.name
```

### CLI Patterns (Click)

```python
import click

@click.command()
@click.option("--name", required=True, help="User name")
@click.option("--count", default=1, help="Number of times")
def greet(name: str, count: int) -> None:
    """Greet a user multiple times."""
    for _ in range(count):
        click.echo(f"Hello, {name}!")

if __name__ == "__main__":
    greet()
```

## Version Detection

**Automatic version detection** determines which Python version features are available:

1. Check `pyproject.toml` for `requires-python` field
2. Check `setup.py`/`setup.cfg` for `python_requires`
3. Check `.python-version` file
4. Default to Python 3.12 if not specified

Based on detected version, appropriate version-specific features are recommended.

## Navigation Tips

- **Start with dignified-python-core.md** for essential patterns that apply to all code
- **Check version-specific docs** based on your project's Python version
- **Reference advanced topics** when dealing with specialized patterns
- **Use cli-patterns.md** when building command-line tools

## When to Read Each Reference

| Situation                  | Reference                      |
| -------------------------- | ------------------------------ |
| Writing any Python code    | dignified-python-core.md       |
| Building a CLI tool        | cli-patterns.md                |
| Using Python 3.10 features | versions/python-3.10.md        |
| Using Python 3.11 features | versions/python-3.11.md        |
| Using Python 3.12 features | versions/python-3.12.md        |
| Using Python 3.13 features | versions/python-3.13.md        |
| Handling exceptions        | advanced/exception-handling.md |
| Designing interfaces       | advanced/interfaces.md         |
| Complex type hints         | advanced/typing-advanced.md    |
| API design decisions       | advanced/api-design.md         |

## Related Skills

- **`/dagster-best-practices`** - Dagster-specific patterns (not general Python)
- **`/dg`** - Dagster CLI operations
- **`/dagster-expert`** - Dagster expertise including integrations

**Important**: `/dignified-python` is for **general-purpose Python style guidance**, not
Dagster-specific patterns. It is intentionally opinionated rather than universal. For Dagster
patterns, use `/dagster-best-practices`.

## Cross-Skill Usage

Users invoke `/dignified-python` when they need Python code quality guidance, regardless of whether
it's for a Dagster project or any other Python project.

**Workflow:**

```
User: "Is this good Python code?"
→ /dignified-python (check dignified-python-core.md)
→ Apply modern type syntax, LBYL, pathlib patterns
→ Check version-specific features based on project

User: "How should I structure my Dagster assets?"
→ /dagster-best-practices (NOT dignified-python)
→ Learn asset patterns, dependencies, partitions
```

## Production Patterns

These standards reflect production-tested conventions:

- ✅ **Modern type syntax** - Improves IDE support and type checking
- ✅ **LBYL patterns** - More explicit and easier to debug than EAFP
- ✅ **Pathlib** - More readable and cross-platform than os.path
- ✅ **Absolute imports** - Avoid import confusion and relative import issues
- ✅ **Error boundaries at CLI** - Clean error messages for end users

## Documentation Structure

Each reference document follows a consistent structure:

1. **Overview** - High-level concepts
2. **Patterns** - Common code patterns with examples
3. **Best Practices** - Recommended approaches
4. **Anti-Patterns** - What to avoid
5. **Real-World Examples** - Production code samples
6. **Related Topics** - Cross-references

## Self-Selecting Usage

Users only invoke `/dignified-python` when they want Python standards guidance. The skill
description makes it clear it's for general Python quality, not Dagster-specific patterns, so users
naturally select it when appropriate.

**Users will invoke this when they want:**

- Code review and quality improvements
- Modern Python patterns
- Type annotation guidance
- Exception handling best practices
- CLI implementation patterns

**Users will NOT invoke this when they want:**

- Dagster-specific patterns (they'll use `/dagster-best-practices`)
- Creating Dagster projects (they'll use `/dg`)
- Finding integrations (they'll use `/dagster-expert`)
