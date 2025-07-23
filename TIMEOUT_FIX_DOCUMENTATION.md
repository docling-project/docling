# Fix for ReadingOrderModel AssertionError with document_timeout

## Problem Description

When `pipeline_options.document_timeout` was set in the latest version of docling (v2.24.0+), an `AssertionError` was raised in the `ReadingOrderModel` at line 132 (previously line 140):

```python
assert size is not None, "Page size is not initialized."
```

This error occurred in `ReadingOrderModel._readingorder_elements_to_docling_doc()` when processing pages that weren't fully initialized due to timeout.

Additionally, there was a secondary issue where the `ConversionStatus.PARTIAL_SUCCESS` status that was correctly set during timeout was being overwritten by the `_determine_status` method.

## Root Cause

The issue had two parts:

1. **Uninitialized Pages**: When a document processing timeout occurs:
   - The pipeline stops processing pages mid-way through the document
   - Some pages remain uninitialized with `page.size = None`
   - These uninitialized pages are passed to the `ReadingOrderModel`
   - The `ReadingOrderModel` expects all pages to have `size != None`, causing the assertion to fail

2. **Status Overwriting**: The `_determine_status` method would:
   - Always start with `ConversionStatus.SUCCESS`
   - Only change to `PARTIAL_SUCCESS` based on backend validation issues
   - Ignore that timeout might have already set the status to `PARTIAL_SUCCESS`

## Solution

The fix was implemented in two parts in `docling/pipeline/base_pipeline.py`:

### Part 1: Page Filtering (lines 196-206)

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

### Part 2: Status Preservation (lines 220-221)

```python
def _determine_status(self, conv_res: ConversionResult) -> ConversionStatus:
    # Preserve PARTIAL_SUCCESS status if already set (e.g., due to timeout)
    status = ConversionStatus.SUCCESS if conv_res.status != ConversionStatus.PARTIAL_SUCCESS else ConversionStatus.PARTIAL_SUCCESS
    
    for page in conv_res.pages:
        if page._backend is None or not page._backend.is_valid():
            conv_res.errors.append(
                ErrorItem(
                    component_type=DoclingComponentType.DOCUMENT_BACKEND,
                    module_name=type(page._backend).__name__,
                    error_message=f"Page {page.page_no} failed to parse.",
                )
            )
            status = ConversionStatus.PARTIAL_SUCCESS

    return status
```

This fix:
1. **Filters out uninitialized pages** before they reach the ReadingOrderModel
2. **Prevents the AssertionError** by ensuring all pages have `size != None`
3. **Preserves timeout-induced PARTIAL_SUCCESS status** through the status determination process
4. **Maintains partial conversion results** by keeping successfully processed pages
5. **Logs the filtering action** for transparency

## Verification

The fix has been verified with comprehensive tests that:
1. ✅ Confirm timeout scenarios don't cause AssertionError
2. ✅ Validate that filtered pages are compatible with ReadingOrderModel
3. ✅ Ensure timeout-induced PARTIAL_SUCCESS status is preserved
4. ✅ Ensure normal processing (without timeout) still works correctly

## Status

✅ **FIXED** - The issue has been resolved and the fix is working correctly.

The conversion will now complete with `ConversionStatus.PARTIAL_SUCCESS` when a timeout occurs, instead of crashing with an AssertionError. The status is properly preserved throughout the pipeline execution.