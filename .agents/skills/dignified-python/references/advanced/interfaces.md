---
description: ABC vs Protocol decision guide, dependency injection patterns, and complete DI examples.
---

# Interface Design Reference

**Read when**: Creating ABC/Protocol classes, writing @abstractmethod, designing gateway interfaces

---

## ABC vs Protocol: Choosing the Right Interface

**ABCs (nominal typing)** and **Protocols (structural typing)** serve different purposes. Choose
based on ownership and coupling needs.

| Use Case                                  | Recommended | Why                                                  |
| ----------------------------------------- | ----------- | ---------------------------------------------------- |
| Internal interfaces you control           | ABC         | Explicit enforcement, runtime validation, code reuse |
| Third-party library boundaries            | Protocol    | No inheritance required, loose coupling              |
| Plugin systems with isinstance checks     | ABC         | Reliable runtime type validation                     |
| Minimal interface contracts (1-2 methods) | Protocol    | Less boilerplate, focused contracts                  |

**Default for internal application code you own: ABC. Default for external library facades:
Protocol.**

---

## ABC Interface Pattern

```python
# CORRECT: Use ABC for interfaces
from abc import ABC, abstractmethod

class Repository(ABC):
    @abstractmethod
    def save(self, entity: Entity) -> None:
        """Save entity to storage."""
        ...

    @abstractmethod
    def load(self, id: str) -> Entity:
        """Load entity by ID."""
        ...

class PostgresRepository(Repository):
    def save(self, entity: Entity) -> None:
        # Implementation
        pass

    def load(self, id: str) -> Entity:
        # Implementation
        pass
```

---

## Benefits of ABC (Internal Interfaces)

1. **Explicit inheritance** - Clear class hierarchy, explicit opt-in
2. **Runtime validation** - Errors at instantiation if abstract methods missing
3. **Code reuse** - Can include concrete methods and shared logic
4. **Reliable isinstance()** - Full signature checking at runtime

---

## Benefits of Protocol (External Boundaries)

1. **No inheritance required** - Works with code you don't control
2. **Loose coupling** - Implementations don't know about the protocol
3. **Minimal contracts** - Define only the methods you need
4. **Duck typing** - Aligns with Python's philosophy

---

## Complete DI Example

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

# Define the interface
class DataStore(ABC):
    @abstractmethod
    def get(self, key: str) -> str | None:
        """Retrieve value by key."""
        ...

    @abstractmethod
    def set(self, key: str, value: str) -> None:
        """Store value with key."""
        ...

# Real implementation
class RedisStore(DataStore):
    def get(self, key: str) -> str | None:
        return self.client.get(key)

    def set(self, key: str, value: str) -> None:
        self.client.set(key, value)

# Fake for testing
class FakeStore(DataStore):
    def __init__(self) -> None:
        self._data: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        if key not in self._data:
            return None
        return self._data[key]

    def set(self, key: str, value: str) -> None:
        self._data[key] = value

# Business logic accepts interface
@dataclass
class Service:
    store: DataStore  # Depends on abstraction

    def process(self, item: str) -> None:
        cached = self.store.get(item)
        if cached is None:
            result = expensive_computation(item)
            self.store.set(item, result)
        else:
            result = cached
        use_result(result)
```

---

## When to Use Protocol

**Protocols excel at defining interfaces for code you don't control:**

```python
# CORRECT: Protocol for third-party library facade
from typing import Protocol

class HttpClient(Protocol):
    """Interface for HTTP operations - decouples from requests/httpx/aiohttp."""
    def get(self, url: str) -> Response: ...
    def post(self, url: str, data: dict) -> Response: ...

# Any HTTP library that has these methods works - no inheritance needed
def fetch_data(client: HttpClient, endpoint: str) -> dict:
    response = client.get(endpoint)
    return response.json()
```

**Protocols are also appropriate for minimal, focused interfaces:**

```python
# CORRECT: Protocol for structural typing with minimal interface
from typing import Protocol

class Closeable(Protocol):
    def close(self) -> None: ...

def cleanup_resources(resources: list[Closeable]) -> None:
    for r in resources:
        r.close()
```

---

## Protocol Limitations

1. **No runtime validation** - `@runtime_checkable` only checks method existence, not signatures
2. **No code reuse** - Protocols shouldn't have method implementations
3. **Weaker isinstance() checks** - ABCs provide more reliable runtime type checking

---

## Decision Checklist

Before defining an interface (ABC or Protocol):

- [ ] Do I own all implementations? -> Prefer ABC
- [ ] Am I wrapping a third-party library? -> Prefer Protocol
- [ ] Do I need runtime isinstance() validation? -> Use ABC
- [ ] Is this a minimal interface (1-2 methods)? -> Protocol may be simpler
- [ ] Do I need shared method implementations? -> Use ABC

**Default for internal application code you own: ABC. Default for external library facades:
Protocol.**
