# Performance Statistics Feature

This document describes the new `--stats` performance statistics feature added to the Docling CLI.

## Overview

The `--stats` flag provides detailed performance metrics and timing information for document conversion operations. This feature is valuable for:

- Understanding processing bottlenecks
- Optimizing conversion workflows
- Benchmarking performance across different systems
- Debugging slow conversion processes

## Usage

Add the `--stats` flag to any `docling convert` command:

```bash
# Single document with stats
docling document.pdf --stats

# Multiple documents with stats
docling documents/ --stats --output ./converted

# With other options
docling document.pdf --stats --to json --to md --output ./output
```

## Output Format

The statistics output includes two main sections:

### 1. Performance Statistics Table

Shows high-level conversion metrics:

- **Total Documents**: Number of documents processed
- **Successful**: Number of successfully converted documents  
- **Failed**: Number of failed conversions
- **Total Pages**: Sum of all pages across documents
- **Avg Pages/Doc**: Average pages per document
- **Total Time**: Total processing time in seconds
- **Throughput (docs/s)**: Documents processed per second
- **Throughput (pages/s)**: Pages processed per second

### 2. Pipeline Timings Table

Provides detailed breakdown of processing time by pipeline operation:

- **Operation**: Name of the pipeline stage (e.g., layout, table_structure, ocr)
- **Total (s)**: Total time spent in this operation across all documents
- **Avg (s)**: Average time per operation instance
- **Min (s)**: Minimum time observed
- **Max (s)**: Maximum time observed  
- **Count**: Number of times this operation was executed

## Implementation Details

- Enabling `--stats` automatically enables pipeline profiling (`DOCLING_DEBUG_PROFILE_PIPELINE_TIMINGS=true`)
- Statistics are collected during processing and displayed after completion
- The feature works with single documents, multiple documents, and batch processing
- All timing measurements use high-precision monotonic time

## Example Output

```
     📊 Performance Statistics     
┏━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
┃ Metric                  ┃ Value ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━┩
│ 📄 Total Documents      │     1 │
│ ✅ Successful           │     1 │
│ ❌ Failed               │     0 │
│ 📃 Total Pages          │     1 │
│ 📊 Avg Pages/Doc        │   1.0 │
│ ⏱️  Total Time           │ 5.13s │
│ 🚀 Throughput (docs/s)  │  0.20 │
│ 📄 Throughput (pages/s) │  0.20 │
└─────────────────────────┴───────┘

                         ⚙️  Pipeline Timings                         
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━┓
┃ Operation       ┃ Total (s) ┃ Avg (s) ┃ Min (s) ┃ Max (s) ┃ Count ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━┩
│ pipeline_total  │     1.456 │   1.456 │   1.456 │   1.456 │     1 │
│ doc_build       │     1.410 │   1.410 │   1.410 │   1.410 │     1 │
│ table_structure │     0.673 │   0.673 │   0.673 │   0.673 │     1 │
│ layout          │     0.508 │   0.508 │   0.508 │   0.508 │     1 │
│ ocr             │     0.115 │   0.115 │   0.115 │   0.115 │     1 │
│ page_parse      │     0.061 │   0.061 │   0.061 │   0.061 │     1 │
│ doc_assemble    │     0.046 │   0.046 │   0.046 │   0.046 │     1 │
│ page_init       │     0.045 │   0.045 │   0.045 │   0.045 │     1 │
│ reading_order   │     0.005 │   0.005 │   0.005 │   0.005 │     1 │
│ page_assemble   │     0.001 │   0.001 │   0.001 │   0.001 │     1 │
│ doc_enrich      │     0.000 │   0.000 │   0.000 │   0.000 │     1 │
└─────────────────┴───────────┴─────────┴─────────┴─────────┴───────┘
```

## Performance Insights

From the example above, you can see that:

- **Table structure detection** (0.673s) and **layout analysis** (0.508s) consume most processing time
- **OCR processing** takes 0.115s for this document
- **Document parsing** and **assembly** are relatively fast operations

This information helps identify optimization opportunities and understand where processing time is spent.