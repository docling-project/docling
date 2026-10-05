---
---

# Type Annotations - Python 3.10

This document captures type annotation guidance for Python 3.10.
This is the baseline for modern Python type syntax.

## Overview

Python 3.10 introduced major improvements to type annotation syntax through PEP 604 (union types via
`|`) and PEP 585 (generic types in standard collections). These features eliminated the need for
most `typing` module imports and made type annotations more concise and readable.

**What's new in 3.10:**

- Union types with `|` operator (PEP 604)
- Built-in generic types: `list[T]`, `dict[K, V]`, etc. (PEP 585)
- No more need for `List`, `Dict`, `Union`, `Optional` from typing

**What you need from typing module:**

- `TypeVar` for generic functions/classes
- `Protocol` for structural typing (rare - prefer ABC)
- `TYPE_CHECKING` for conditional imports
- `Any` (use sparingly)

## Complete Type Annotation Syntax for Python 3.10

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
mapping: Dict[str, int] = {}
```

**Why**: Built-in types are more concise, don't require imports, and are the modern Python standard.

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

def get_config(key: str) -> str | None:
    return config.get(key)
```

❌ **WRONG** - Don't use `typing.Optional`:

```python
from typing import Optional
def find_user(id: str) -> Optional[User]:  # Don't do this
    ...
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
    """Return the value unchanged."""
    return value
```

**Note**: This is the standard way in Python 3.10. Python 3.12 introduces better syntax (PEP 695).

### Generic Classes

✅ **PREFERRED** - Use Generic with TypeVar:

```python
from typing import Generic, TypeVar

T = TypeVar("T")

class Stack(Generic[T]):
    """A generic stack data structure."""

    def __init__(self) -> None:
        self._items: list[T] = []

    def push(self, item: T) -> None:
        self._items.append(item)

    def pop(self) -> T | None:
        if not self._items:
            return None
        return self._items.pop()

# Usage
int_stack = Stack[int]()
int_stack.push(42)
```

**Note**: Python 3.12 introduces cleaner syntax for this pattern.

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

### when from **future** import annotations is Needed

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

### Repository Pattern

```python
from abc import ABC, abstractmethod

class Repository(ABC):
    """Abstract base class for data repositories."""

    @abstractmethod
    def get(self, id: str) -> dict[str, str] | None:
        """Get entity by ID."""

    @abstractmethod
    def save(self, entity: dict[str, str]) -> None:
        """Save entity."""

    @abstractmethod
    def delete(self, id: str) -> bool:
        """Delete entity, return success."""

class UserRepository(Repository):
    def __init__(self) -> None:
        self._users: dict[str, dict[str, str]] = {}

    def get(self, id: str) -> dict[str, str] | None:
        return self._users.get(id)

    def save(self, entity: dict[str, str]) -> None:
        if "id" not in entity:
            raise ValueError("Entity must have id")
        self._users[entity["id"]] = entity

    def delete(self, id: str) -> bool:
        if id in self._users:
            del self._users[id]
            return True
        return False
```

### Generic Data Structures

```python
from typing import Generic, TypeVar

T = TypeVar("T")

class Node(Generic[T]):
    """A node in a tree structure."""

    def __init__(self, value: T, children: list[Node[T]] | None = None) -> None:
        self.value = value
        self.children = children or []

    def add_child(self, child: Node[T]) -> None:
        self.children.append(child)

    def find(self, predicate: Callable[[T], bool]) -> Node[T] | None:
        """Find first node matching predicate."""
        if predicate(self.value):
            return self
        for child in self.children:
            result = child.find(predicate)
            if result:
                return result
        return None

# Usage
from collections.abc import Callable

root = Node[int](1)
root.add_child(Node[int](2))
root.add_child(Node[int](3))
```

### Configuration Management

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class DatabaseConfig:
    host: str
    port: int
    username: str
    password: str | None = None
    ssl_enabled: bool = False

@dataclass(frozen=True)
class AppConfig:
    app_name: str
    debug_mode: bool
    database: DatabaseConfig
    feature_flags: dict[str, bool]

def load_config(path: str) -> AppConfig:
    """Load application configuration from file."""
    import json
    from pathlib import Path

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    data: dict[str, str | int | bool | dict[str, str | int | bool]] = json.loads(
        config_path.read_text(encoding="utf-8")
    )

    # Parse and validate...
    return AppConfig(...)
```

### API Client with Error Handling

```python
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

class ApiResponse(Generic[T]):
    """Container for API response with data or error."""

    def __init__(self, data: T | None = None, error: str | None = None) -> None:
        self.data = data
        self.error = error

    def is_success(self) -> bool:
        return self.error is None

    def map(self, func: Callable[[T], U]) -> ApiResponse[U]:
        """Transform successful response data."""
        if self.is_success() and self.data is not None:
            return ApiResponse(data=func(self.data))
        return ApiResponse(error=self.error)

U = TypeVar("U")

def fetch_user(id: str) -> ApiResponse[dict[str, str]]:
    """Fetch user from API."""
    # Implementation...
    return ApiResponse(data={"id": id, "name": "Alice"})
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
python-version = "3.10"
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

## Migration from Python 3.9

If upgrading from Python 3.9, apply these changes:

1. **Replace typing module types**:
   - `List[X]` → `list[X]`
   - `Dict[K, V]` → `dict[K, V]`
   - `Set[X]` → `set[X]`
   - `Tuple[X, Y]` → `tuple[X, Y]`
   - `Union[X, Y]` → `X | Y`
   - `Optional[X]` → `X | None`

2. **Add future annotations if needed**:
   - Add `from __future__ import annotations` for forward references
   - Add for circular imports with `TYPE_CHECKING`

3. **Remove unnecessary imports**:
   - Remove `from typing import List, Dict, Optional, Union`
   - Keep only `TypeVar`, `Generic`, `Protocol`, `TYPE_CHECKING`, `Any`

## References

- [PEP 604: Union Types](https://peps.python.org/pep-0604/)
- [PEP 585: Type Hinting Generics In Standard Collections](https://peps.python.org/pep-0585/)
- [PEP 563: Postponed Evaluation of Annotations](https://peps.python.org/pep-0563/)
- [Python 3.10 What's New - Type Hints](https://docs.python.org/3.10/whatsnew/3.10.html)
