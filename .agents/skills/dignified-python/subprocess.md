---
---

# Subprocess Handling - Safe Execution

## Core Rule

**ALWAYS set `check` explicitly on `subprocess.run()`** — either `check=True` (raise on non-zero exit) or `check=False` (handle the return code yourself). Never rely on the default.

## Basic Subprocess Pattern

```python
import subprocess
from pathlib import Path

# ✅ CORRECT: check=True to raise on error
result = subprocess.run(
    ["git", "status"],
    check=True,
    capture_output=True,
    text=True
)
print(result.stdout)

# ✅ ALSO CORRECT: check=False when you intend to inspect returncode yourself
result = subprocess.run(["git", "status"], check=False, capture_output=True, text=True)
if result.returncode != 0:
    ...

# ❌ WRONG: check unset - intent is ambiguous
result = subprocess.run(["git", "status"])
```

## Complete Subprocess Example

```python
def run_git_command(args: list[str], cwd: Path | None = None) -> str:
    """Run a git command and return output."""
    try:
        result = subprocess.run(
            ["git"] + args,
            check=True,          # Raise on non-zero exit
            capture_output=True, # Capture stdout/stderr
            text=True,          # Return strings, not bytes
            cwd=cwd            # Working directory
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        # Error boundary - add context
        raise RuntimeError(f"Git command failed: {e.stderr}") from e
```

## Error Handling

```python
try:
    result = subprocess.run(
        ["make", "test"],
        check=True,
        capture_output=True,
        text=True
    )
except subprocess.CalledProcessError as e:
    # Access error details
    print(f"Command: {e.cmd}")
    print(f"Exit code: {e.returncode}")
    print(f"Stdout: {e.stdout}")
    print(f"Stderr: {e.stderr}")
    raise
```

## Common Patterns

```python
# Silent execution (no output)
subprocess.run(["git", "fetch"], check=True, capture_output=True)

# Stream output in real-time
process = subprocess.Popen(
    ["pytest", "-v"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)
for line in process.stdout:
    print(line, end="")
process.wait()
if process.returncode != 0:
    raise subprocess.CalledProcessError(process.returncode, process.args)

# With timeout
try:
    subprocess.run(["long-command"], check=True, timeout=30)
except subprocess.TimeoutExpired:
    print("Command timed out")
```

## Key Takeaways

1. **Always set `check` explicitly**: Use `check=True` to raise, or `check=False` when you'll handle `returncode` yourself — never leave it unset
2. **Capture output**: Use `capture_output=True` for stdout/stderr
3. **Text mode**: Use `text=True` for string output
4. **Error context**: Wrap in try/except at boundaries
5. **Timeout safety**: Set timeout for long-running commands
