---
---

# Type Annotations - Python 3.11

This document captures type annotation guidance for Python 3.11.

## Overview

Python 3.11 builds on 3.10's type syntax with the addition of the `Self` type (PEP 673), making
method chaining and builder patterns significantly cleaner. All modern syntax from 3.10 continues to
work.

**What's new in 3.11:**

- `Self` type for self-returning methods (PEP 673)
- Variadic generics with TypeVarTuple (PEP 646)
- Significantly improved error messages

**Available from 3.10:**

- Built-in generic types: `list[T]`, `dict[K, V]`, etc. (PEP 585)
- Union types with `|` operator (PEP 604)
- Optional with `X | None`

**What you need from typing module:**

- `Self` for self-returning methods (NEW)
- `TypeVar` for generic functions/classes
- `Generic` for generic classes
- `Protocol` for structural typing (rare - prefer ABC)
- `TYPE_CHECKING` for conditional imports
- `Any` (use sparingly)

## Complete Type Annotation Syntax for Python 3.11

### Basic Collection Types

✅ **PREFERRED** - Use built-in generic types:

```python
names: list[str] = []
mapping: dict[str, int] = {}
unique_ids: set[str] = set()
coordinates: tuple[int, int] = (0, 0)
```

❌ **WRONG** - Don't use typing module equivalents:

```python
from typing import List, Dict, Set, Tuple  # Don't do this
names: List[str] = []
```

### Union Types

✅ **PREFERRED** - Use `|` operator:

```python
def process(value: str | int) -> str:
    return str(value)

def find_config(name: str) -> dict[str, str] | dict[str, int]:
    ...

# Multiple unions
def parse(input: str | int | float) -> str:
    return str(input)
```

❌ **WRONG** - Don't use `typing.Union`:

```python
from typing import Union
def process(value: Union[str, int]) -> str:  # Don't do this
    ...
```

### Optional Types

✅ **PREFERRED** - Use `X | None`:

```python
def find_user(id: str) -> User | None:
    """Returns user or None if not found."""
    if id in users:
        return users[id]
    return None
```

❌ **WRONG** - Don't use `typing.Optional`:

```python
from typing import Optional
def find_user(id: str) -> Optional[User]:  # Don't do this
    ...
```

### Self Type for Self-Returning Methods (NEW in 3.11)

✅ **PREFERRED** - Use Self for methods that return the instance:

```python
from typing import Self

class Builder:
    def set_name(self, name: str) -> Self:
        self.name = name
        return self

    def set_value(self, value: int) -> Self:
        self.value = value
        return self

# Usage with type safety
builder = Builder().set_name("app").set_value(42)
```

❌ **WRONG** - Don't use bound TypeVar anymore:

```python
from typing import TypeVar

T = TypeVar("T", bound="Builder")

class Builder:
    def set_name(self: T, name: str) -> T:  # Don't do this
        ...
```

**When to use Self:**

- Methods that return `self`
- Builder pattern methods
- Fluent interfaces with method chaining
- Factory classmethods

**Self in classmethod:**

```python
from typing import Self

class Config:
    def __init__(self, data: dict[str, str]) -> None:
        self.data = data

    @classmethod
    def from_file(cls, path: str) -> Self:
        """Load config from file."""
        import json
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(data)
```

### Generic Functions with TypeVar

✅ **PREFERRED** - Use TypeVar for generic functions:

```python
from typing import TypeVar

T = TypeVar("T")

def first(items: list[T]) -> T | None:
    """Return first item or None if empty."""
    if not items:
        return None
    return items[0]

def identity(value: T) -> T:
    return value
```

**Note**: Python 3.12 introduces better syntax (PEP 695) for this pattern.

### Generic Classes

✅ **PREFERRED** - Use Generic with TypeVar:

```python
from typing import Generic, TypeVar

T = TypeVar("T")

class Stack(Generic[T]):
    """A generic stack data structure."""

    def __init__(self) -> None:
        self._items: list[T] = []

    def push(self, item: T) -> Self:  # Can combine with Self!
        self._items.append(item)
        return self

    def pop(self) -> T | None:
        if not self._items:
            return None
        return self._items.pop()

# Usage
int_stack = Stack[int]()
int_stack.push(42).push(43)  # Method chaining works!
```

**Note**: Python 3.12 introduces cleaner syntax for generic classes.

### Constrained and Bounded TypeVars

✅ **Use TypeVar constraints when needed**:

```python
from typing import TypeVar

# Constrained to specific types
Numeric = TypeVar("Numeric", int, float)

def add(a: Numeric, b: Numeric) -> Numeric:
    return a + b

# Bounded to base class
T = TypeVar("T", bound=BaseClass)

def process(obj: T) -> T:
    return obj
```

### Callable Types

✅ **PREFERRED** - Use `collections.abc.Callable`:

```python
from collections.abc import Callable

# Function that takes int, returns str
processor: Callable[[int], str] = str

# Function with no args, returns None
callback: Callable[[], None] = lambda: None

# Function with multiple args
validator: Callable[[str, int], bool] = lambda s, i: len(s) > i
```

### Type Aliases

✅ **Use simple assignment for type aliases**:

```python
# Simple alias
UserId = str
Config = dict[str, str | int | bool]

# Complex nested type
JsonValue = dict[str, "JsonValue"] | list["JsonValue"] | str | int | float | bool | None

def load_config() -> Config:
    return {"host": "localhost", "port": 8080}
```

**Note**: Python 3.12 introduces `type` statement for better alias support.

### When from **future** import annotations is Needed

Use `from __future__ import annotations` when you encounter:

**Forward references** (class referencing itself):

```python
from __future__ import annotations

class Node:
    def __init__(self, value: int, parent: Node | None = None):
        self.value = value
        self.parent = parent
```

**Circular type imports**:

```python
# a.py
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from b import B

class A:
    def method(self) -> B:
        ...
```

**Complex recursive types**:

```python
from __future__ import annotations

JsonValue = dict[str, JsonValue] | list[JsonValue] | str | int | float | bool | None
```

### Interfaces: ABC vs Protocol

✅ **PREFERRED** - Use ABC for interfaces:

```python
from abc import ABC, abstractmethod

class Repository(ABC):
    @abstractmethod
    def get(self, id: str) -> User | None:
        """Get user by ID."""

    @abstractmethod
    def save(self, user: User) -> None:
        """Save user."""
```

🟡 **VALID** - Use Protocol only for structural typing:

```python
from typing import Protocol

class Drawable(Protocol):
    def draw(self) -> None: ...

# Any object with draw() method matches
def render(obj: Drawable) -> None:
    obj.draw()
```

**Dignified Python prefers ABC** because it makes inheritance and intent explicit.

## Complete Examples

### Builder Pattern with Self

```python
from typing import Self

class QueryBuilder:
    """SQL query builder with fluent interface."""

    def __init__(self) -> None:
        self._select: list[str] = ["*"]
        self._from: str | None = None
        self._where: list[str] = []
        self._limit: int | None = None

    def select(self, *columns: str) -> Self:
        """Specify columns to select."""
        self._select = list(columns)
        return self

    def from_table(self, table: str) -> Self:
        """Specify table to query."""
        self._from = table
        return self

    def where(self, condition: str) -> Self:
        """Add WHERE condition."""
        self._where.append(condition)
        return self

    def limit(self, n: int) -> Self:
        """Set LIMIT."""
        self._limit = n
        return self

    def build(self) -> str:
        """Build final SQL query."""
        if not self._from:
            raise ValueError("FROM table not specified")

        parts = [f"SELECT {', '.join(self._select)}"]
        parts.append(f"FROM {self._from}")

        if self._where:
            parts.append(f"WHERE {' AND '.join(self._where)}")

        if self._limit:
            parts.append(f"LIMIT {self._limit}")

        return " ".join(parts)

# Usage with type-safe method chaining
query = (
    QueryBuilder()
    .select("id", "name", "email")
    .from_table("users")
    .where("active = true")
    .where("age > 18")
    .limit(10)
    .build()
)
```

### Factory Methods with Self

```python
from typing import Self
from pathlib import Path
import json

class Config:
    """Application configuration with multiple factory methods."""

    def __init__(self, data: dict[str, str | int]) -> None:
        self.data = data

    @classmethod
    def from_json(cls, path: Path) -> Self:
        """Load configuration from JSON file."""
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")

        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        return cls(data)

    @classmethod
    def from_env(cls) -> Self:
        """Load configuration from environment variables."""
        import os
        data = {
            k.lower(): v
            for k, v in os.environ.items()
            if k.startswith("APP_")
        }
        return cls(data)

    @classmethod
    def default(cls) -> Self:
        """Create default configuration."""
        return cls({"host": "localhost", "port": 8080})

    def with_override(self, key: str, value: str | int) -> Self:
        """Return new config with overridden value."""
        new_data = self.data.copy()
        new_data[key] = value
        return type(self)(new_data)

# All factory methods return correct type
config = Config.from_json(Path("config.json"))
dev_config = config.with_override("debug", True)
```

## Type Checking Rules

### What to Type

✅ **MUST type**:

- All public function parameters (except `self`, `cls`)
- All public function return values
- All class attributes (public and private)
- Module-level constants

🟡 **SHOULD type**:

- Internal function signatures
- Complex local variables

🟢 **MAY skip**:

- Simple local variables where type is obvious (`count = 0`)
- Lambda parameters in short inline lambdas
- Loop variables in short comprehensions

### Running Type Checker

```bash
uv run ty check
```

All code should pass type checking without errors.

### Type Checking Configuration

Configure ty in `pyproject.toml`:

```toml
[tool.ty.environment]
python-version = "3.11"
```

## Common Patterns

### Checking for None

✅ **CORRECT** - Check before use:

```python
def process_user(user: User | None) -> str:
    if user is None:
        return "No user"
    return user.name
```

### Dict.get() with Type Safety

✅ **CORRECT** - Handle None case:

```python
def get_port(config: dict[str, int]) -> int:
    port = config.get("port")
    if port is None:
        return 8080
    return port
```

### List Operations

✅ **CORRECT** - Check before accessing:

```python
def first_or_default(items: list[str], default: str) -> str:
    if not items:
        return default
    return items[0]
```

## Migration from Python 3.10

If upgrading from Python 3.10:

1. **Replace bound TypeVar with Self** for self-returning methods:
   - Old: `T = TypeVar("T", bound="ClassName")`
   - New: `from typing import Self` and use `-> Self`

2. **Enjoy improved error messages** (no code changes needed)

3. **All existing 3.10 syntax continues to work**

## References

- [PEP 673: Self Type](https://peps.python.org/pep-0673/)
- [PEP 646: Variadic Generics](https://peps.python.org/pep-0646/)
- [Python 3.11 What's New](https://docs.python.org/3.11/whatsnew/3.11.html)
