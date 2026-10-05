# EBCDIC test data

Mainframe samples and COBOL layouts derived from
[`ebcdic-parser`](https://github.com/larandvit/ebcdic-parser) by larandvit, used
here under the MIT License (Copyright (c) 2018 larandvit).

## Sources

Each `sources/<name>.ebc` pairs with a `sources/<name>.layout.json` describing
how to decode it. The layouts were translated from the reference project's
`layout_repository/` rules into the `EbcdicLayout` schema; the record bytes are
unmodified.

| Sample | Origin | Exercises |
|---|---|---|
| `311_calls_for_service` | `311_calls_for_service_requests_sample.dat` | single schema, 17 character fields |
| `gas_disposition` | `gsf102.ebc` | single schema, 26 character fields, 200-byte records |
| `ola013k` | `olf001l.ebc` | four schemas behind a record-type prefix, 483 packed-decimal fields |

The upstream samples run from 20 MB to 67 MB, so each one is cut down here:

- Single-schema samples keep the leading 40 records.
- `ola013k` keeps the first 5 records of each of its four schemas. A plain
  prefix would not do — its first type-4 record only appears 1382 records in, so
  covering all four schemas that way would take megabytes.

Every retained record is byte-for-byte upstream data, and each sample's decoded
values were cross-checked field by field against the reference project's own
`validate/`, `validation/` output before being committed. Scaled numeric fields
are the one intentional difference: the reference divides by a power of ten and
prints a float, while this backend keeps the declared scale as a `Decimal`, so
its `0.0` is `0.0000` here.

The reference project's fourth sample, `service_segment_data`, is not included.
It relies on `OCCURS DEPENDING ON` repeating groups (`layoutvariable`), which
this backend does not implement.

## Groundtruth

`groundtruth/<name>.ebc.md` holds the Markdown export. These documents are
nothing but tables of decoded values, so the serialized `DoclingDocument` is not
kept as groundtruth: it would add megabytes of cell scaffolding without covering
anything the tables do not already show. Document structure is asserted directly
in `tests/test_backend_ebcdic.py`.

Regenerate with:

```bash
DOCLING_GEN_TEST_DATA=1 uv run pytest tests/test_backend_ebcdic.py
```
