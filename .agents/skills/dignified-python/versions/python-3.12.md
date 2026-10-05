---
---

# Type Annotations - Python 3.12

This document captures type annotation guidance for Python 3.12.

## Overview

Python 3.12 introduces PEP 695, a major syntactic improvement for generic types. The new type
parameter syntax makes generic functions and classes significantly more readable. All syntax from
3.10 and 3.11 continues to work.

**What's new in 3.12:**

- PEP 695 type parameter syntax: `def func[T](x: T) -> T`
- `type` statement for better type aliases
- Cleaner generic class syntax

**Available from 3.11:**

- `Self` type for self-returning methods

**Available from 3.10:**

- Built-in generic types: `list[T]`, `dict[K, V]`, etc.
- Union types with `|` operator
- Optional with `X | None`

**What you need from typing module:**

- `Self` for self-returning methods
- `TypeVar` only for constrained/bounded generics
- `Protocol` for structural typing (rare - prefer ABC)
- `TYPE_CHECKING` for conditional imports
- `Any` (use sparingly)

## Complete Type Annotation Syntax for Python 3.12

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

### Self Type for Self-Returning Methods

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
```

### Generic Functions with PEP 695 (NEW in 3.12)

✅ **PREFERRED** - Use PEP 695 type parameter syntax:

```python
def first[T](items: list[T]) -> T | None:
    """Return first item or None if empty."""
    if not items:
        return None
    return items[0]

def identity[T](value: T) -> T:
    """Return value unchanged."""
    return value

# Multiple type parameters
def zip_dicts[K, V](keys: list[K], values: list[V]) -> dict[K, V]:
    """Create dict from separate key and value lists."""
    return dict(zip(keys, values))
```

🟡 **VALID** - TypeVar still works:

```python
from typing import TypeVar

T = TypeVar("T")

def first(items: list[T]) -> T | None:
    if not items:
        return None
    return items[0]
```

**Note**: Prefer PEP 695 syntax for simple generics. TypeVar is still needed for constraints/bounds.

### Generic Classes with PEP 695 (NEW in 3.12)

✅ **PREFERRED** - Use PEP 695 class syntax:

```python
class Stack[T]:
    """A generic stack data structure."""

    def __init__(self) -> None:
        self._items: list[T] = []

    def push(self, item: T) -> Self:
        self._items.append(item)
        return self

    def pop(self) -> T | None:
        if not self._items:
            return None
        return self._items.pop()

# Usage
int_stack = Stack[int]()
int_stack.push(42).push(43)
```

🟡 **VALID** - Generic with TypeVar still works:

```python
from typing import Generic, TypeVar

T = TypeVar("T")

class Stack(Generic[T]):
    def __init__(self) -> None:
        self._items: list[T] = []
    # ... rest of implementation
```

**Note**: PEP 695 is cleaner - no imports needed, type parameter scope is local to class.

### Type Parameter Bounds

✅ **Use bounds with PEP 695**:

```python
class Comparable:
    def compare(self, other: object) -> int:
        ...

def max_value[T: Comparable](items: list[T]) -> T:
    """Get maximum value from comparable items."""
    return max(items, key=lambda x: x)
```

### Constrained TypeVars (Still Use TypeVar)

✅ **Use TypeVar for specific type constraints**:

```python
from typing import TypeVar

# Constrained to specific types - must use TypeVar
Numeric = TypeVar("Numeric", int, float)

def add(a: Numeric, b: Numeric) -> Numeric:
    return a + b
```

❌ **WRONG** - PEP 695 doesn't support constraints:

```python
# This doesn't constrain to int|float
def add[Numeric](a: Numeric, b: Numeric) -> Numeric:
    return a + b
```

### Type Aliases with type Statement (NEW in 3.12)

✅ **PREFERRED** - Use `type` statement:

```python
# Simple alias
type UserId = str
type Config = dict[str, str | int | bool]

# Generic type alias
type Result[T] = tuple[T, str | None]

def process(value: str) -> Result[int]:
    try:
        return (int(value), None)
    except ValueError as e:
        return (0, str(e))
```

🟡 **VALID** - Simple assignment still works:

```python
UserId = str  # Still valid
Config = dict[str, str | int | bool]  # Still valid
```

**Note**: `type` statement is more explicit and works better with generics.

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

type JsonValue = dict[str, JsonValue] | list[JsonValue] | str | int | float | bool | None
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

def render(obj: Drawable) -> None:
    obj.draw()
```

**Dignified Python prefers ABC** because it makes inheritance and intent explicit.

## Complete Examples

### Generic Stack with PEP 695

```python
from typing import Self

class Stack[T]:
    """Type-safe stack with PEP 695 syntax."""

    def __init__(self) -> None:
        self._items: list[T] = []

    def push(self, item: T) -> Self:
        """Push item and return self for chaining."""
        self._items.append(item)
        return self

    def pop(self) -> T | None:
        """Pop item or return None if empty."""
        if not self._items:
            return None
        return self._items.pop()

    def peek(self) -> T | None:
        """Peek at top item without removing."""
        if not self._items:
            return None
        return self._items[-1]

    def is_empty(self) -> bool:
        """Check if stack is empty."""
        return len(self._items) == 0

# Usage
numbers = Stack[int]()
numbers.push(1).push(2).push(3)
top = numbers.pop()  # Type checker knows this is int | None
```

### Generic Repository with PEP 695

```python
from abc import ABC, abstractmethod
from typing import Self

class Repository[T]:
    """Abstract repository with generic type parameter."""

    @abstractmethod
    def get(self, id: str) -> T | None:
        """Get entity by ID."""

    @abstractmethod
    def save(self, entity: T) -> Self:
        """Save entity, return self for chaining."""

    @abstractmethod
    def delete(self, id: str) -> bool:
        """Delete entity, return success."""

    def get_or_fail(self, id: str) -> T:
        """Get entity or raise error."""
        entity = self.get(id)
        if entity is None:
            raise ValueError(f"Entity not found: {id}")
        return entity

class InMemoryRepository[T](Repository[T]):
    """In-memory repository implementation."""

    def __init__(self) -> None:
        self._storage: dict[str, T] = {}

    def get(self, id: str) -> T | None:
        return self._storage.get(id)

    def save(self, entity: T) -> Self:
        # Assume entity has 'id' attribute
        entity_id = str(getattr(entity, "id", id(entity)))
        self._storage[entity_id] = entity
        return self

    def delete(self, id: str) -> bool:
        if id in self._storage:
            del self._storage[id]
            return True
        return False

# Usage
from dataclasses import dataclass

@dataclass
class User:
    id: str
    name: str

repo = InMemoryRepository[User]()
repo.save(User("1", "Alice")).save(User("2", "Bob"))
user = repo.get("1")  # Type: User | None
```

### Type Aliases with type Statement

```python
# Simple aliases
type UserId = str
type ErrorMessage = str

# Complex nested types
type JsonValue = dict[str, JsonValue] | list[JsonValue] | str | int | float | bool | None

# Generic type aliases
type Result[T] = tuple[T, ErrorMessage | None]
type AsyncResult[T] = tuple[T | None, ErrorMessage | None]

def parse_int(value: str) -> Result[int]:
    """Parse string to int, return result with optional error."""
    try:
        return (int(value), None)
    except ValueError as e:
        return (0, str(e))

def fetch_user(id: UserId) -> AsyncResult[dict[str, str]]:
    """Fetch user data asynchronously."""
    # Implementation...
    return ({"id": id, "name": "Alice"}, None)
```

### Builder Pattern with Self and PEP 695

```python
from typing import Self

class QueryBuilder[T]:
    """Generic query builder with fluent interface."""

    def __init__(self, result_type: type[T]) -> None:
        self._result_type = result_type
        self._filters: list[str] = []
        self._limit: int | None = None

    def filter(self, condition: str) -> Self:
        """Add filter condition."""
        self._filters.append(condition)
        return self

    def limit(self, n: int) -> Self:
        """Set result limit."""
        self._limit = n
        return self

    def build(self) -> str:
        """Build query string."""
        query = " AND ".join(self._filters)
        if self._limit:
            query += f" LIMIT {self._limit}"
        return query

# Usage
@dataclass
class User:
    name: str
    age: int

builder = QueryBuilder[User](User)
query = (
    builder
    .filter("active = true")
    .filter("age > 18")
    .limit(10)
    .build()
)
```

### Generic Function Utilities

```python
def map_list[T, U](items: list[T], func: Callable[[T], U]) -> list[U]:
    """Map function over list items."""
    from collections.abc import Callable
    return [func(item) for item in items]

def filter_list[T](items: list[T], predicate: Callable[[T], bool]) -> list[T]:
    """Filter list by predicate."""
    from collections.abc import Callable
    return [item for item in items if predicate(item)]

def reduce_list[T, U](
    items: list[T],
    func: Callable[[U, T], U],
    initial: U,
) -> U:
    """Reduce list to single value."""
    from collections.abc import Callable
    result = initial
    for item in items:
        result = func(result, item)
    return result

# Usage
numbers = [1, 2, 3, 4, 5]
doubled = map_list(numbers, lambda x: x * 2)  # list[int]
evens = filter_list(numbers, lambda x: x % 2 == 0)  # list[int]
sum_val = reduce_list(numbers, lambda acc, x: acc + x, 0)  # int
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
python-version = "3.12"
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
def first_or_default[T](items: list[T], default: T) -> T:
    if not items:
        return default
    return items[0]
```

## When to Use PEP 695 vs TypeVar

**Use PEP 695 for**:

- Simple generic functions (no constraints/bounds)
- Simple generic classes
- Most common generic use cases
- New code

**Still use TypeVar for**:

- Constrained type variables: `TypeVar("T", str, bytes)`
- Bound type variables with complex bounds
- Covariant/contravariant type variables
- Reusing same TypeVar across multiple functions

## Migration from Python 3.11

If upgrading from Python 3.11:

1. **Consider migrating to PEP 695 syntax**:
   - `TypeVar` + `def func(x: T) -> T` → `def func[T](x: T) -> T`
   - `Generic[T]` + `class C(Generic[T])` → `class C[T]`

2. **Consider using `type` statement for aliases**:
   - `Config = dict[str, str]` → `type Config = dict[str, str]`

3. **Keep TypeVar for constraints**:
   - `TypeVar` with constraints still needed

4. **All existing 3.11 syntax continues to work**:
   - `Self` type still preferred
   - Union with `|` still preferred

## References

- [PEP 695: Type Parameter Syntax](https://peps.python.org/pep-0695/)
- [Python 3.12 What's New - Type Hints](https://docs.python.org/3.12/whatsnew/3.12.html)
