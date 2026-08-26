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

Every OCR engine takes its languages through the same field, `OcrOptions.lang`, and every engine
takes them in the same vocabulary: **BCP-47 (RFC 5646) language tags**, the notation behind `en`,
`de-DE` and `zh-Hant`. Docling canonicalizes each tag to a `(language, script)` pair and each engine
translates that pair into its own notation.

```python
from docling.datamodel.pipeline_options import TesseractCliOcrOptions

TesseractCliOcrOptions(lang=["de", "en"])  # -> tesseract -l deu+eng
```

Canonicalization drops the region once it has told us the script, because no OCR engine
distinguishes `de-DE` from `de-AT`:

| You write                     | Docling stores | Why                                       |
| ----------------------------- | -------------- | ----------------------------------------- |
| `de`, `de-DE`, `deu`, `ger`   | `de-Latn`      | ISO 639-1/2/3 fold together; region drops |
| `en`, `en-US`, `eng`          | `en-Latn`      |                                           |
| `zh`, `zh-CN`, `zho`          | `zh-Hans`      | Simplified is the likely script for `zh`  |
| `zh-TW`, `zh-HK`, `zh-Hant`   | `zh-Hant`      | Traditional                               |
| `sr`                          | `sr-Cyrl`      | Serbian defaults to Cyrillic              |
| `sr-Latn`                     | `sr-Latn`      | Same language, a different model          |
| `pa`, `pa-IN`                 | `pa-Guru`      | Gurmukhi                                  |
| `pa-PK`                       | `pa-Arab`      | Shahmukhi                                 |

The list order is preference order. Duplicates collapse, so `["de", "de-AT"]` is one language.

### The reserved tag

One tag carries engine-independent meaning, and it must be used **alone**.

| Tag   | Meaning            | Behaviour                                |
| ----- | ------------------ | ---------------------------------------- |
| `mul` | multiple languages | The engine's broadest multilingual model |

An **empty list** is how you say "let the engine decide", and a script is named by naming a
language written in it. To skip OCR altogether, turn the stage off — `--no-ocr` on the CLI,
`do_ocr=False` in the pipeline options.

What each engine does with an empty `lang` and with `mul`:

| Engine            | `lang=[]`                          | `mul`                       |
| ----------------- | ---------------------------------- | --------------------------- |
| Tesseract (both)  | Per-page orientation and script    | Error                       |
|                   | detection; needs the `osd` file    |                             |
| EasyOCR           | EasyOCR's own default              | Error -- list the languages |
| RapidOCR / KServe | The Simplified Chinese default     | Error -- list the languages |
| Nemotron-OCR      | The English model                  | The multilingual model      |
| ocrmac            | Vision's own automatic behaviour   | Error                       |

On the CLI, omitting `--ocr-lang` applies the engine's default languages; an empty value,
`--ocr-lang ""`, is how you ask for the `lang=[]` column above.

Engines that ship a recognizer named after a script rather than a language expose it under
that engine's own token: `latin`, `cyrillic`, `arabic` and `devanagari` for RapidOCR and
KServe, and `script/<Name>` files such as `script/Cyrillic` for Tesseract. They are engine
vocabulary, not portable tags, so they are only accepted by the engine that defines them.

### Engine-native language codes

You can also write the language codes of the engine you selected, and they mean what that engine
means by them. Docling canonicalizes them on the way in, so `lang` always ends up holding BCP-47:

```python
from docling.datamodel.pipeline_options import RapidOcrOptions

RapidOcrOptions(lang=["ch"]).lang  # -> ["zh-Hans"]
```

| Engine            | Codes accepted in addition to BCP-47                               |
| ----------------- | ------------------------------------------------------------------ |
| RapidOCR / KServe | PP-OCR tokens: `ch`, `chinese_cht`, `japan`, `korean`, `ka`,       |
|                   | `eslav`, `latin`, `cyrillic`, `arabic`, `devanagari`, `rs_latin`   |
| Tesseract (both)  | tessdata names: `chi_sim`, `chi_tra`, `srp_latn`, `aze_cyrl`,      |
|                   | `uzb_cyrl`, `deu_latf`, `frk`, and any `script/<Name>` file        |
| EasyOCR           | EasyOCR codes: `ch_sim`, `ch_tra`, `rs_latin`, `rs_cyrillic`,      |
|                   | `tjk`, `ang`, `mah`, `tab`                                          |
| Nemotron-OCR      | `english`, `multilingual`                                           |
| ocrmac            | none needed -- Vision's own vocabulary already is BCP-47            |

Only codes the engine spells differently from ISO 639 are listed. The rest of every engine's
vocabulary (`deu`, `fra`, `ru`, `ta`, `en-US`, ...) is valid BCP-47 already and has always worked.

A code belongs to **one** engine. Asking RapidOCR for Tesseract's `chi_sim` is an error naming the
tag to write instead, because reading another engine's vocabulary would make the same string mean
different things depending on a setting elsewhere in your config.

### Common surprises

Six codes are a legitimate BCP-47 tag for one language and an engine's own name for a different one.
The engine that owns the code wins, which is what makes existing configurations keep working:

| Code  | Engine means         | BCP-47 means | Owned by  |
| ----- | -------------------- | ------------ | --------- |
| `ch`  | Chinese Simplified   | Chamorro     | PP-OCR    |
| `ka`  | Kannada              | Georgian     | PP-OCR    |
| `ang` | Angika               | Old English  | EasyOCR   |
| `mah` | Magahi               | Marshallese  | EasyOCR   |
| `tab` | Tabasaran (Cyrillic) | Tabasaran    | EasyOCR   |
| `frk` | German Fraktur       | Frankish     | Tesseract |

This is safe rather than lucky: no engine ships a recognizer for any of the shadowed readings, so
nothing that was reachable before becomes unreachable. A test enforces that property, so an engine
that later gains one of those models turns the code into a reported ambiguity instead of a silent
wrong answer.

To ask for the BCP-47 meaning anyway, write the script out. The tables above are keyed on the bare
code, so a qualified tag bypasses them:

```python
RapidOcrOptions(lang=["ch-Latn"])   # Chamorro -- and so, an honest "no model" error
RapidOcrOptions(lang=["ka-Geor"])   # Georgian, not Kannada
EasyOcrOptions(lang=["ang-Latn"])   # Old English, not Angika
```

With no engine selected -- `docling convert-remote`, or `--ocr-engine auto` before it has picked one
-- there is nothing to prefer an engine's reading over the standard one, so only the codes every
engine agrees on carry their engine meaning. `ch` is refused there and names `zh-Hans` in the
message; the other five parse as ordinary BCP-47 and mean the *BCP-47* column above, so `ka` is
Georgian and `ang` is Old English. Name the engine, or write the qualified tag, to be explicit.

### Codes with no tag

A few native codes name a model that a `(language, script)` pair cannot describe. Docling refuses
them rather than quietly selecting a neighbouring recognizer:

| Code                                  | Names                        |
| ------------------------------------- | ---------------------------- |
| `jpn_vert`, `chi_sim_vert`, ...       | Vertical-text models         |
| `ita_old`, `spa_old`, `kat_old`       | Historical orthographies     |
| `equ`                                 | Mathematical notation        |

Custom traineddata files you trained yourself fall in the same category and are not reachable
through `lang`.

### When an engine has no model

A language the selected engine cannot serve is an **error**, uniformly, for every engine. Docling
never quietly substitutes a different recognizer; the message names the languages that engine does
support, as canonical tags. Engines that run one language at a time (RapidOCR, Nemotron-OCR, the
KServe client) take the **first** tag and warn about the rest.

## RapidOCR

This section describes RapidOCR for versions `v3.9.1`, `v3.9.2`.

### RapidOCR backends

RapidOCR supports multiple backends.
Docling currently (2026.07.28) supports: "onnxruntime" (default), "openvino", "paddle", "torch".

RapidOCR relies on the [PP-OCR](https://rapidai.github.io/RapidOCRDocs/main/model_list/#_2) models.
Docling currently (2026.07.28) supports: "PP-OCR v4", "PP-OCR v5", "PP-OCR v6".

**PP-OCR versions supported by each rapidocr backend:**

| Backend     | PP-OCR versions      |
| ----------- | -------------------- |
| onnxruntime | v4, v5, v6           |
| openvino    | v4, v5, v6           |
| paddle      | v4, v5, v6           |
| torch       | v4, v5 (ch only), v6 |

<u>Notice</u>: torch on PP-OCRv5 supports ONLY chinese.


### RapidOCR language support

**PP-OCRv4 supported languages/scripts:**

```
arabic, ch, chinese_cht, cyrillic, devanagari, en, japan, ka, korean, latin, ta, te
```

<u>Notice</u>: `cyrillic`, `devanagari`, `latin` are actually scripts and each one supports multiple
languages.


**PP-OCRv5 supported languages/scripts:**

```
arabic, ch, cyrillic, devanagari, el, en, eslav, korean, latin, ta, te, th
```


**PP-OCRv6 supported languages:**

```
ch, chinese_cht, en, japan, af, az, bs, ca, cs, cy, da, de, es, et, eu, fi, fr, ga, gl,
hr, hu, id, is, it, ku, la, lb, lt, lv, mi, ms, mt, nl, no, oc, pl, pt, qu, rm, ro,
rs_latin, sk, sl, sq, sv, sw, tl, tr, uz, vi, french, german
```

<u>Notices</u>:

- These are PP-OCR's own tokens, listed here to document what the checkpoints cover. You never
  write them: docling takes BCP-47 tags and maps them onto these tokens for you.
- German exists in 2 formats: `de`, `german`; French in `fr`, `french`. Docling always picks the
  two-letter one.
- Korean is actually not supported in PP-OCR v6 (only the alias exists).


### RapidOCR language input

RapidOCR runs a **single** language per conversion. If `lang` holds more than one tag the first is
used and the rest are dropped with a warning.

Tags resolve to a PP-OCR recognizer in this order: an explicit entry in the table below, then the
primary subtag if PP-OCR has it under that name, then the script family, then an error.

| You write                                  | PP-OCR token            | Backbone           |
| ------------------------------------------ | ----------------------- | ------------------ |
| `zh-Hans` / `zh-Hant`                      | `ch` / `chinese_cht`    | v6                 |
| `ja` / `ko`                                | `japan` / `korean`      | v6 / v5 / v4       |
| `en`, `de`, `fr`, and the other v6 codes   | the primary subtag      | v6                 |
| `sr-Latn`                                  | `rs_latin`              | v6                 |
| `ru`, `uk`, `be`                           | `eslav`                 | v5 -- narrower     |
| other Cyrillic-script languages            | `cyrillic`              | v5 / v4            |
| Arabic- and Devanagari-script languages    | `arabic` / `devanagari` | v5 / v4            |
| `el`, `ta`, `te`, `th`                     | `el`, `ta`, `te`, `th`  | v5                 |
| `kn`                                       | `ka` (PP-OCR's Kannada) | v4                 |
| `ka` (Georgian)                            | --                      | **error**          |
| `latin`, `cyrillic`, `arabic`,             | the token itself        | v5 / v4            |
| `devanagari` (PP-OCR's own tokens)         |                         |                    |
| an empty list                              | `ch`                    | the default        |
| `mul`                                      | --                      | **error**          |

A language written in a script PP-OCR does not serve under that language's own name falls back to
the script family: `uz` is PP-OCR's Latin Uzbek, so `uz-Cyrl` resolves to `cyrillic` rather than
silently using the Latin recognizer.

Prefetching follows the same vocabulary:

```console
docling-tools models download rapidocr --rapidocr-backend-lang onnxruntime:th-Thai
```

## EasyOCR

This section describes EasyOCR for versions `v1.7.2`, `v1.7.1`.
The model checkpoints are those of `gen2`.

### EasyOCR language support

EasyOCR accepts a list of languages and picks the recognition model that covers all of them.
Docling translates each BCP-47 tag into EasyOCR's own code -- `zh-Hant` becomes `ch_tra`,
`sr-Cyrl` becomes `rs_cyrillic`, `tg` becomes `tjk` -- and EasyOCR then selects the recognition
checkpoint for the script those codes share, so `ru` reaches the Cyrillic model without you
naming it. EasyOCR has no multilingual model, so `mul` raises; list the languages instead.

The following table shows which recognition model is enabled per language combination
(the detection checkpoint `craft_mlt_25k.pth` is required in all cases; the codes are EasyOCR's
own, shown here to explain the grouping):

| Recognition checkpoint | Supported languages                                                     |
| ---------------------- | ----------------------------------------------------------------------- |
| `english_g2.pth`       | `en`                                                                    |
| `latin_g2.pth`         | `af`, `az`, `bs`, `cs`, `cy`, `da`, `de`, `en`, `es`, `et`, `fr`, `ga`, |
|                        | `hr`, `hu`, `id`, `is`, `it`, `ku`, `la`, `lt`, `lv`, `mi`, `ms`, `mt`, |
|                        | `nl`, `no`, `oc`, `pi`, `pl`, `pt`, `ro`, `rs_latin`, `sk`, `sl`, `sq`, |
|                        | `sv`, `sw`, `tl`, `tr`, `uz`, `vi`                                      |
| `zh_sim_g2.pth`        | `ch_sim` + `en`                                                         |
| `japanese_g2.pth`      | `ja` + `en`                                                             |
| `korean_g2.pth`        | `ko` + `en`                                                             |
| `telugu.pth`           | `te` + `en`                                                             |
| `kannada.pth`          | `kn` + `en`                                                             |
| `cyrillic_g2.pth`      | `ru`, `rs_cyrillic`, `be`, `bg`, `uk`, `mn`, `abq`, `ady`, `kbd`,       |
|                        | `ava`, `dar`, `inh`, `che`, `lbe`, `lez`, `tab`, `tjk`, `en`            |

<u>Notice</u>: keep the requested language list as short and specific as possible. Because the
resolution picks a model that covers *all* requested languages, adding a language you do not need
downgrades the model for the ones you do. For example, `["en"]` selects the English-specific
`english_g2.pth`, while `["en", "de"]` falls back to the broader `latin_g2.pth`, which is generally
less accurate on English text.

Check the semantic of easyocr language inputs here: https://www.jaided.ai/easyocr/



## Nemotron-OCR

This section describes Nemotron-OCR for versions `v2.0.0`, `v2.0.2`.

Nemotron works only on Linux and requires CUDA (Docling enforces 13.x).

The following table shows the supported Python versions and languages


| Nemotron version | Python version   | Supported language inputs             |
| ---------------- | ---------------- | ------------------------------------- |
| v2.0.0           | 3.12 only        | `en`, `mul`                           |
| v2.0.2           | 3.11, 3.12, 3.13 | `en`, `mul`                           |

`en`, and an empty list, select the English recognizer. `mul` selects the multilingual one, as do
the languages it is trained on: English, Chinese (Simplified and Traditional), Japanese, Korean and
Russian. Any other language raises rather than silently loading the multilingual model -- ask for
`mul` explicitly if that is what you want.


## Tesseract - TesserOCR

Tesseract must be installed as a system package (see
[installation](../getting_started/installation.md)).
TesserOCR is a python library that wraps the Tesseract engine.

Tesseract's own vocabulary *is* ISO 639-2/T, so most tags map straight through: `de` becomes `deu`,
`el` becomes `ell`, `cs` becomes `ces`. Docling handles the deviations for you -- `zh-Hant` becomes
`chi_tra`, `sr-Latn` becomes `srp_latn`, `az-Cyrl` becomes `aze_cyrl`, `ku` becomes `kmr`. A
`script/` traineddata file can be named directly, e.g. `lang=["script/Latin"]`.

Languages are checked against the installed tessdata **at construction time**, so a missing
traineddata file now fails immediately with the installed set in the message, instead of failing
per page during conversion.

An empty `lang` list runs Tesseract's per-page orientation and script detection. That requires
the `osd` traineddata; without it, `lang=[]` raises with an install hint.

[Languages support](https://tesseract-ocr.github.io/tessdoc/Data-Files-in-different-versions.html)


## OcrMac

This section describes ocrmac for versions `v1.0.0`, `v1.0.1`.

ocrmac is a thin wrapper around Apple's Vision framework. It is macOS-only and ships no model
artifacts of its own — the recognizers are part of the operating system. The supported language set
is therefore a property of the macOS version, not of the ocrmac release.

Vision's own vocabulary is BCP-47 with regions, so docling matches your tag against the list the
running macOS reports rather than mapping it through a table: `de` finds `de-DE`, `pt` finds
`pt-BR`, `zh-CN` finds `zh-Hans`. A tag with no close enough match raises, and the message lists
what this particular macOS actually offers. An empty `lang` list hands the choice to Vision's own
automatic behaviour; `mul` is not supported.

