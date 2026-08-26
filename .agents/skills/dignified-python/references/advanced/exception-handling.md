---
description:
  Detailed exception handling patterns including B904 chaining, third-party API compatibility, and
  anti-patterns.
---

# Exception Handling Reference

**Read when**: Writing try/except blocks, wrapping third-party APIs, seeing `from e` or `from None`

---

## When Exceptions Are a Good Fit

This skill prefers explicit condition checks when they are cheap and precise, but exceptions are
the right tool in a few common situations:

1. **Error boundaries** (CLI/API level)
2. **Operations where the call itself is the authoritative test**
3. **Adding context before re-raising**

### 1. Error Boundaries

```python
# ACCEPTABLE: CLI command error boundary
@click.command("create")
@click.pass_obj
def create(ctx: AppContext, name: str) -> None:
    """Create a resource."""
    try:
        create_resource(ctx, name)
    except subprocess.CalledProcessError as e:
        click.echo(f"Error: Git command failed: {e.stderr}", err=True)
        raise SystemExit(1) from e
```

### 2. Third-Party API Compatibility

```python
# ACCEPTABLE: Third-party API forces exception handling
def _get_bigquery_sample(sql_client, table_name):
    """
    BigQuery's TABLESAMPLE doesn't work on views.
    There's no reliable way to determine a priori whether
    a table supports TABLESAMPLE.
    """
    try:
        return sql_client.run_query(f"SELECT * FROM {table_name} TABLESAMPLE...")
    except Exception:
        return sql_client.run_query(f"SELECT * FROM {table_name} ORDER BY RAND()...")
```

> **The test for "use LBYL first"**: Can you validate the condition with a cheap, precise check
> before calling the API? If yes, prefer that. If the operation itself is the authoritative
> validator, a small `try/except` is often clearer.

### Prefer Real Parsers Over Brittle Pre-Checks

Do not replace parser calls with incomplete string-shape checks such as `str.isdigit()` or
hand-rolled ISO date heuristics. Those checks often reject valid inputs and accept invalid ones.

When the same try/parse/default pattern recurs, extract a generic helper:

```python
from typing import TypeVar, Callable

T = TypeVar("T")

def try_parse(parse: Callable[[str], T], value: str, default: T) -> T:
    """Parse *value* with *parse*, returning *default* on ValueError."""
    try:
        return parse(value)
    except ValueError:
        return default
```

Usage:

```python
from datetime import datetime

port = try_parse(int, user_input, 80)
ts = try_parse(datetime.fromisoformat, timestamp_str, None)
```

Use a separate pre-check only when you intentionally accept a narrower format than the parser and
can state that rule precisely.

### 3. Adding Context Before Re-raising

```python
# ACCEPTABLE: Adding context before re-raising
try:
    process_file(config_file)
except yaml.YAMLError as e:
    raise ValueError(f"Failed to parse config file {config_file}: {e}") from e
```

---

## Exception Chaining (B904 Lint Compliance)

**Ruff rule B904** requires explicit exception chaining when raising inside an `except` block. This
prevents losing the original traceback.

```python
# CORRECT: Chain to preserve context
try:
    parse_config(path)
except ValueError as e:
    click.echo(json.dumps({"success": False, "error": str(e)}))
    raise SystemExit(1) from e  # Preserves traceback

# CORRECT: Explicitly break chain when intentional
try:
    fetch_from_cache(key)
except KeyError:
    # Original exception is not relevant to caller
    raise ValueError(f"Unknown key: {key}") from None

# WRONG: Missing exception chain (B904 violation)
try:
    parse_config(path)
except ValueError:
    raise SystemExit(1)  # Lint error: missing 'from e' or 'from None'

# CORRECT: CLI error boundary with JSON output
try:
    result = some_operation()
except RuntimeError as e:
    click.echo(json.dumps({"success": False, "error": str(e)}))
    raise SystemExit(0) from None  # Exception is in JSON, traceback irrelevant to CLI user
```

**When to use each:**

- `from e` - Preserve original exception for debugging
- `from None` - Intentionally suppress original (e.g., transforming exception type, CLI JSON output)

---

## Exception Anti-Patterns

**Never swallow exceptions silently**

Even at error boundaries, you must at least log/warn so issues can be diagnosed:

```python
# WRONG: Silent exception swallowing
try:
    risky_operation()
except:
    pass

# WRONG: Silent swallowing even at error boundary
try:
    optional_feature()
except Exception:
    pass  # Silent - impossible to diagnose issues

# CORRECT: Let exceptions bubble up (default)
risky_operation()

# CORRECT: At error boundaries, log the exception
try:
    optional_feature()
except Exception as e:
    logging.warning("Optional feature failed: %s", e)  # Diagnosable
```

**Never use silent fallback behavior**

```python
# WRONG: Silent fallback masks failure
def process_text(text: str) -> dict:
    try:
        return llm_client.process(text)
    except Exception:
        return regex_parse_fallback(text)

# CORRECT: Let error bubble to boundary
def process_text(text: str) -> dict:
    return llm_client.process(text)
```
