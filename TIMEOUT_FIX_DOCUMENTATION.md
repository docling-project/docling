# Fix for ReadingOrderModel AssertionError with document_timeout

## Problem Description

When `pipeline_options.document_timeout` was set in the latest version of docling (v2.24.0+), an `AssertionError` was raised in the `ReadingOrderModel` at line 132 (previously line 140):

```python
assert size is not None, "Page size is not initialized."
```

This error occurred in `ReadingOrderModel._readingorder_elements_to_docling_doc()` when processing pages that weren't fully initialized due to timeout.

## Root Cause

When a document processing timeout occurs:
1. The pipeline stops processing pages mid-way through the document
2. Some pages remain uninitialized with `page.size = None`
3. These uninitialized pages are passed to the `ReadingOrderModel`
4. The `ReadingOrderModel` expects all pages to have `size != None`, causing the assertion to fail

## Solution

The fix was implemented in `docling/pipeline/base_pipeline.py` (lines 196-206):

```python
# Filter out uninitialized pages (those with size=None) that may remain
# after timeout or processing failures to prevent assertion errors downstream
initial_page_count = len(conv_res.pages)
conv_res.pages = [page for page in conv_res.pages if page.size is not None]

if len(conv_res.pages) < initial_page_count:
    _log.info(
        f"Filtered out {initial_page_count - len(conv_res.pages)} uninitialized pages "
        f"due to timeout or processing failures"
    )
```

This fix:
1. **Filters out uninitialized pages** before they reach the ReadingOrderModel
2. **Prevents the AssertionError** by ensuring all pages have `size != None`
3. **Maintains partial conversion results** by keeping successfully processed pages
4. **Logs the filtering action** for transparency

## Verification

The fix has been verified with comprehensive tests that:
1. ✅ Confirm timeout scenarios don't cause AssertionError
2. ✅ Validate that filtered pages are compatible with ReadingOrderModel
3. ✅ Ensure normal processing (without timeout) still works correctly

## Status

✅ **FIXED** - The issue has been resolved and the fix is working correctly.

The conversion will now complete with `ConversionStatus.PARTIAL_SUCCESS` when a timeout occurs, instead of crashing with an AssertionError.