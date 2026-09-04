# OCR in Docling

## Overview

Docling supports multiple OCR engines that can be installed as extra packages:

- [RapidOCR](https://github.com/RapidAI/RapidOCR)
- [Nemotron-OCR](https://huggingface.co/nvidia/nemotron-ocr-v2)
- [EasyOCR](https://github.com/jaidedai/easyocr)
- [ocrmac](https://github.com/straussmaximilian/ocrmac)
- [tesseract-CLI](https://github.com/tesseract-ocr/tesseract)
- [tesserocr](https://github.com/sirfz/tesserocr)

## Language selection

Every OCR engine takes its languages through the same field, `OcrOptions.lang`.

An entry of `lang` is written in one of exactly two forms:

- A code of the engine you selected, handed to that engine untouched: `deu`, `ch`,
  `script/Cyrillic`
- A **[BCP-47 (RFC 5646)](https://en.wikipedia.org/wiki/IETF_language_tag) language tag** behind the
  **`iso:` prefix**, canonicalized to a `(language, script)` pair and then mapped onto that engine's
  own notation: `iso:de`, `iso:en-US`, `iso:zh-Hant`

```python
from docling.datamodel.pipeline_options import TesseractCliOcrOptions

TesseractCliOcrOptions(lang=["deu", "eng"])          # -> tesseract -l deu+eng
TesseractCliOcrOptions(lang=["iso:de", "iso:en"])    # the same thing, said portably
```

By default a language is written in the spelling native to the selected OCR engine. An alternative
syntax is to provide the language in [BCP-47 format](https://en.wikipedia.org/wiki/IETF_language_tag)
prefixed with `iso:` (e.g. `iso:el`). There is no need to provide the script, unless it is a
non-default script. For example if you want serbian latin you *must* specific the script `sr-Latn`
because the default script for serbian is Cyrillic.

All OCR engines report which language codes they support via the `supported_ocr_languages()` API
call. This method returns a list of the `native` and the `BCP47` supported languages.


### The tags docling refuses

Three BCP-47 tags name something other than a language and are rejected behind `iso:`:

| Tag   | Means                 | Say this instead                                                                       |
| ----- | --------------------- | -------------------------------------------------------------------------------------- |
| `mul` | multiple languages    | The engine's own code for its multilingual model, e.g. `multilingual` for Nemotron-OCR |
| `und` | undetermined          | An empty list, or a language written in the script you want                            |
| `zxx` | no linguistic content | Turn OCR off: `--no-ocr`, or `do_ocr=False`                                            |

An **empty list** is how you say "let the engine decide".
On the CLI, omitting `--ocr-lang` applies the engine's default languages; an empty value,
`--ocr-lang ""`, is how you ask for the `lang=[]` column below:

| Engine            | `lang=[]`                          |
| ----------------- | ---------------------------------- |
| Tesseract (both)  | Per-page orientation and script    |
|                   | detection; needs the `osd` file    |
| EasyOCR           | English (`en`)                     |
| RapidOCR          | The Simplified Chinese default     |
| KServe            | Sends `en`                         |
| Nemotron-OCR      | The English model                  |
| ocrmac            | Vision's own automatic behaviour   |

The KServe client is the exception to everything in this page: it canonicalizes nothing at all.
Only the deployed model knows which languages it serves, so `lang` is neither validated nor mapped
-- the first entry is sent to the server exactly as written, and the rest are dropped with a
warning. Use the codes your deployment expects (`english`, `chinese`, `ch`, ...); an `iso:` tag is
only right if the server itself speaks that prefix, which no deployment does.

### Native engine codes

A language input without the `iso:` prefix is native to the selected OCR engine. The input is
validated against the vocabulary of that engine and propagated verbatim to it.


```python
from docling.datamodel.pipeline_options import RapidOcrOptions

RapidOcrOptions(lang=["ch"]).lang          # -> ["ch"]
RapidOcrOptions(lang=["iso:zh-Hans"]).lang # -> ["iso:zh-Hans"], the same PP-OCR recognizer
```

Two cases need the bare code:

- the model has no `(language, script)` name at all -- see
  [Models no tag can name](#models-no-tag-can-name)
- you want the engine's reading of a code that is also a tag for something else -- see
  [Codes that shadow a tag](#codes-that-shadow-a-tag)


| Engine            | What it accepts as a bare code                                        |
| ----------------- | --------------------------------------------------------------------- |
| RapidOCR          | Any PP-OCR token the resolved backbone serves: `ch`, `chinese_cht`,   |
|                   | `japan`, `korean`, `ka`, `eslav`, `rs_latin`, `french`, `german`,     |
|                   | and the script recognizers `latin`, `cyrillic`, `arabic`,             |
|                   | `devanagari`                                                          |
| Tesseract (both)  | Any installed traineddata name: `chi_sim`, `chi_tra`, `srp_latn`,     |
|                   | `aze_cyrl`, `uzb_cyrl`, `deu_latf`, `frk`, `script/<Name>`, and       |
|                   | files you trained yourself                                            |
| EasyOCR           | Any EasyOCR code: `ch_sim`, `ch_tra`, `rs_latin`, `rs_cyrillic`,      |
|                   | `tjk`, `ang`, `mah`, `tab`                                            |
| Nemotron-OCR      | `english`, `multilingual`                                             |
| ocrmac            | Any recognition language the running macOS reports, e.g. `en-US`,     |
|                   | `zh-Hans`                                                             |
| KServe            | Everything -- `lang` is sent verbatim either way                      |


### Codes that shadow a tag

A handful of engine codes are also BCP-47 subtags for an unrelated language. Bare, they are always
the engine's own reading:

| Code  | Bare, it reaches                                 | `iso:<code>` means         |
| ----- | ------------------------------------------------ | -------------------------- |
| `ch`  | PP-OCR's `ch`, Chinese Simplified                | `ch-Latn`, Chamorro        |
| `ang` | EasyOCR's `ang`, Angika                          | `ang-Latn`, Old English    |
| `frk` | Tesseract's `frk`, German Fraktur                | `frk-Latn`, Frankish       |
| `tab` | EasyOCR's `tab`, which is Cyrillic Tabasaran     | `tab-Latn`, Tabasaran      |
| `ka`  | PP-OCR's `ka`, Kannada; Tesseract has no such    | `ka-Geor`, Georgian, which |
|       | file, so it is an error there                    | PP-OCR cannot serve at all |
| `mah` | EasyOCR's Magahi                                 | `mh-Latn`, Marshallese     |

Write the tag when you mean the language, and the bare code when you mean the model:

```python
RapidOcrOptions(lang=["iso:zh-Hans"])        # Chinese Simplified, said portably
RapidOcrOptions(lang=["ka"])                 # PP-OCR's Kannada recognizer
TesseractCliOcrOptions(lang=["iso:de-Latf"]) # German Fraktur, said portably
EasyOcrOptions(lang=["ang"])                 # EasyOCR's Angika recognizer
```

### When an engine has no model

An exception is raised whenever an OCR engine cannot serve the input language. Docling never quietly
substitutes a different recognizer. The message reports what that engine *can* serve in a spelling
you can paste straight back into `lang`:

- `Engine codes:` -- the engine's own name for every model no tag can reach, written bare
- `Supported:` -- the canonical tags, each in its shortest spelling and carrying the `iso:` prefix

```console
TesseractOcrCli has no model for the OCR language 'iso:th-Thai'. No traineddata file 'tha' is installed. Engine codes: jpn_vert, script/Cyrillic. Supported: iso:de, iso:en, iso:ja, iso:zh.
```

Engines that run one language at a time (RapidOCR, Nemotron-OCR) take the **first** tag and warn
about the rest. The KServe client also sends only the first entry.


## RapidOCR

The engine's own vocabulary, in its own codes, is listed in
[Native OCR engines](OCR_native.md#rapidocr).

### RapidOCR language input

RapidOCR runs a **single** language per conversion. If `lang` holds more than one tag the first is
used and the rest are dropped with a warning.

An `iso:` tag resolves to a PP-OCR recognizer in this order: an explicit entry in the table below,
then the primary subtag if PP-OCR has it under that name, then the script family, then an error.
The following table shows how the language resolution works:

| You write                                            | PP-OCR token            | PP-OCR version    |
| ---------------------------------------------------- | ----------------------- | ----------------- |
| `iso:zh-Hans` / `iso:zh-Hant`                        | `ch` / `chinese_cht`    | v6                |
| `iso:ja`                                             | `japan`                 | v6                |
| `iso:ko`                                             | `korean`                | v5 / v4           |
| `iso:en`, `iso:de`, `iso:fr`, and the other v6 codes | the primary subtag      | v6                |
| `iso:sr-Latn`                                        | `rs_latin`              | v6                |
| `iso:ru`, `iso:uk`, `iso:be`                         | `eslav`                 | v5                |
| other Cyrillic-script languages                      | `cyrillic`              | v5 / v4           |
| Arabic and Devanagari (script languages)             | `arabic` / `devanagari` | v5 / v4           |
| `iso:el`, `iso:ta`, `iso:te`, `iso:th`               | `el`, `ta`, `te`, `th`  | v5                |
| `iso:kn`                                             | `ka` (PP-OCR's Kannada) | v4                |
| `iso:ka-Geor` (Georgian)                             | -- (`ka` is Kannada)    | **error**         |
| `ka`                                                 | `ka` (PP-OCR's Kannada) | v4                |
| `latin`, `cyrillic`, `arabic`, `devanagari`          | the token itself        | v5 / v4           |
| any other PP-OCR token                               | the token itself        | wherever it lives |
| an empty list                                        | `ch`                    | the default       |


## EasyOCR

The engine's own vocabulary, in its own codes, is listed in
[Native OCR engines](OCR_native.md#easyocr).

### EasyOCR language input

EasyOCR takes several languages at once. Its own codes -- `ch_tra`, `rs_cyrillic`, `tjk` -- are
written bare. A BCP-47 tag behind `iso:` is translated into one of them: `iso:zh-Hant` becomes
`ch_tra`, `iso:sr-Cyrl` becomes `rs_cyrillic`, `iso:tg` becomes `tjk`. EasyOCR then picks the one
checkpoint covering every requested code, by the script they share, so `iso:ru` reaches the
Cyrillic model unnamed. The grouping is in [Native OCR engines](OCR_native.md#easyocr).

## Nemotron-OCR

The engine's own vocabulary, in its own codes, is listed in
[Native OCR engines](OCR_native.md#nemotron-ocr).

### Nemotron-OCR language input

`english` selects the English recognizer, and so do `iso:en` and an empty list. `multilingual`
selects the multilingual one, as do the five languages it is trained on: `iso:zh`, `iso:zh-Hant`,
`iso:ja`, `iso:ko` and `iso:ru`. Any other language raises.


## Tesseract - TesserOCR

The engine's own vocabulary, in its own codes, is listed in
[Native OCR engines](OCR_native.md#tesseract-tesserocr).

### Tesseract language input

A traineddata file is named by its own stem: `deu`, `chi_tra`, `script/Latin`, `jpn_vert`,
`ita_old`, or a file you trained yourself. That is the only way to reach the ones no tag describes.

Tesseract's own vocabulary *is* ISO 639-2/T, so most `iso:` tags map straight through: `iso:de`
becomes `deu`, `iso:el` becomes `ell`, `iso:cs` becomes `ces`. Docling handles the deviations for
you -- `iso:zh-Hant` becomes `chi_tra`, `iso:sr-Latn` becomes `srp_latn`, `iso:az-Cyrl` becomes
`aze_cyrl`, `iso:ku` becomes `kmr`.

Languages are checked against the installed tessdata **at construction time**, so a missing
traineddata file fails immediately with the installed set in the message, instead of failing
per page during conversion. That set is reported back in the spelling you would write: an `iso:`
tag where one exists, and the bare file name for everything else.

An empty `lang` list runs Tesseract's per-page orientation and script detection. That requires
the `osd` traineddata; without it, `lang=[]` raises with an install hint.

## OcrMac

The engine's own vocabulary, in its own codes, is listed in
[Native OCR engines](OCR_native.md#ocrmac).

### OcrMac language input

Vision's vocabulary is BCP-47 with regions. Docling therefore matches an `iso:` tag against the
languages the running macOS reports, instead of mapping it through a table: `iso:de` finds `de-DE`,
`iso:pt` finds `pt-BR`, `iso:zh-CN` finds `zh-Hans`. A tag with no close match raises.

Some Vision codes carry a region that is not ISO valid like `vi-VT`. Such cases should be passed as
bare/native inputs. An empty `lang` list lets Vision choose.

