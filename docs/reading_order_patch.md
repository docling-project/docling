# Reading Order Patch Documentation

## Overview

This document explains the monkey patch applied to fix a KeyError in the `docling_ibm_models.reading_order.reading_order_rb` module.

## Problem Description

Users encountered a `KeyError` when converting certain PDF files using docling. The error occurred in the reading order prediction phase:

```
KeyError: 22
  File "docling_ibm_models/reading_order/reading_order_rb.py", line 366, in _init_ud_maps
    self.dn_map[i].append(j)
```

The error number (22 in this example) varied depending on the PDF being processed.

## Root Cause

The `_init_ud_maps` method in `docling_ibm_models.reading_order.reading_order_rb` performs the following operations:

1. Initializes `dn_map` and `up_map` dictionaries for all page elements (indices 0 to N-1)
2. Iterates through elements to build spatial relationships
3. When processing element j, it may follow a left-to-right mapping chain:
   ```python
   while i in state.l2r_map:
       i = state.l2r_map[i]
   ```
4. After following the chain, it accesses `state.dn_map[i]`

The KeyError occurs when:
- The final value of `i` after following the `l2r_map` chain doesn't exist in `dn_map`
- This can happen when maps are reinitialized with different element lists
- Or when invalid mappings exist in `r2l_map`

## Solution

A monkey patch was created in `/docling/models/_reading_order_patch.py` that:

1. **Defensive checks for map access**: Before accessing `dn_map[i]` or `up_map[j]`, the patch verifies the key exists
2. **Infinite loop prevention**: Tracks the original value of `i` and breaks if a circular reference is detected
3. **Graceful degradation**: Silently skips invalid mappings instead of crashing

### Patch Application

The patch is automatically applied when the `readingorder_model` module is imported:

```python
from docling.models import _reading_order_patch
_reading_order_patch.apply_patch()
```

This ensures the fix is transparent to users and doesn't require code changes.

## Testing

Comprehensive tests were added in `tests/test_reading_order_patch.py` to verify:

1. **Patch application**: Confirms the monkey patch is correctly applied
2. **Basic functionality**: Verifies reading order model can be initialized
3. **Defensive checks**: Tests that edge cases don't raise KeyError
4. **l2r_map chains**: Validates proper handling of left-to-right mapping chains
5. **Invalid mappings**: Ensures invalid `r2l_map` references are handled gracefully

All tests pass successfully.

## Impact

- **Minimal changes**: Only adds a monkey patch, no modifications to core docling code
- **Backward compatible**: Existing functionality is preserved
- **Transparent**: Applied automatically, no user action required
- **Safe**: Adds defensive checks without modifying core logic

## Future Work

This is a temporary workaround. The proper fix should be submitted to the upstream `docling-ibm-models` repository. Once the upstream fix is released and the dependency is updated, this monkey patch can be removed.

## References

- Issue: [Link to GitHub issue]
- External package: `docling-ibm-models` (version 3.10.2)
- Affected file: `docling_ibm_models/reading_order/reading_order_rb.py`
