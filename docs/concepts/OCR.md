# OCR engines in Docling

## Overview

Docling supports multiple OCR engines that can be installed as extra packages:

- [RapidOCR](https://github.com/RapidAI/RapidOCR)
- [Nemotron-OCR](https://huggingface.co/nvidia/nemotron-ocr-v2)
- [EasyOCR](https://github.com/jaidedai/easyocr)
- [ocrmac](https://github.com/straussmaximilian/ocrmac)
- [tesseract-CLI](https://github.com/tesseract-ocr/tesseract)
- [tesserocr](https://github.com/sirfz/tesserocr)


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

Additionally the following aliases exist for PP-OCR v6:

```
zh      -> ch
zh_cn   -> ch
zh-cn   -> ch
zh_tw   -> chinese_cht
zh-tw   -> chinese_cht
ja      -> japan
jp      -> japan
ko      -> korean
```


<u>Notices</u>:

- German exists in 2 formats: `de`, `german`.
- French exists in 2 formats: `fr`, `french`.
- Korean is actually not supported in PP-OCR v6 (only the alias exists).


### RapidOCR language input semantic

The following table explains the semantic of each language input for RapidOCR

```
token        meaning
-----------  ---------------------------------------------------------------------------------------
af           Afrikaans
arabic       Arabic-script family (9): Arabic, Persian, Uyghur, Urdu, Pashto, Kurdish, Sindhi,
             Baluchi, English
az           Azerbaijani
bs           Bosnian
ca           Catalan
ch           Chinese (Simplified)
chinese_cht  Chinese (Traditional)
cs           Czech
cy           Welsh
cyrillic     Cyrillic-script family (34): Russian, Belarusian, Ukrainian, Serbian (Cyrillic),
             Bulgarian, Mongolian, Abkhaz, Adyghe, Kabardian, Avar, Dargwa, Ingush, Chechen, Lak,
             Lezgian, Tabasaran, Kazakh, Kyrgyz, Tajik, Macedonian, Tatar, Chuvash, Bashkir, Meadow
             Mari, Moldovan, Udmurt, Komi, Ossetian, Buriat, Kalmyk, Tuvan, Yakut, Karakalpak,
             English
da           Danish
de           German
devanagari   Devanagari-script family (14): Hindi, Marathi, Nepali, Bihari, Maithili, Angika,
             Bhojpuri, Magahi, Sadri, Newari, Konkani (Goan), Sanskrit, Haryanvi, English
el           Greek
en           English
es           Spanish
eslav        East Slavic family, Cyrillic script (4): Russian, Belarusian, Ukrainian, English
et           Estonian
eu           Basque
fi           Finnish
fr           French
ga           Irish
gl           Galician
hr           Croatian
hu           Hungarian
id           Indonesian
is           Icelandic
it           Italian
japan        Japanese
ka           Kannada
korean       Korean
ku           Kurdish
la           Latin
lb           Luxembourgish
lt           Lithuanian
lv           Latvian
mi           Maori
ms           Malay
mt           Maltese
nl           Dutch
no           Norwegian
oc           Occitan
pl           Polish
pt           Portuguese
qu           Quechua
rm           Romansh
ro           Romanian
rs_latin     Serbian (Latin)
sk           Slovak
sl           Slovenian
sq           Albanian
sv           Swedish
sw           Swahili
ta           Tamil
te           Telugu
th           Thai
tl           Tagalog
tr           Turkish
uz           Uzbek
vi           Vietnamese
```

## EasyOCR

This section describes EasyOCR for versions `v1.7.2`, `v1.7.1`.
The model checkpoints are those of `gen2`.

### EasyOCR language support

EasyOCR accepts as input a list of languages.
The language resolution that takes place inside EasyOCR enables those models that can support all input languages.

The following table shows which recognition model is enabled per language combination
(the detection checkpoint `craft_mlt_25k.pth` is required in all cases):

| Recognition checkpoint | Supported languages                                          |
| ---------------------- | ------------------------------------------------------------ |
| `english_g2.pth`       | en                                                           |
| `latin_g2.pth`         | af, az, bs, cs, cy, da, de, en, es, et, fr, ga, hr, hu, id,  |
|                        | is, it, ku, la, lt, lv, mi, ms, mt, nl, no, oc, pi, pl, pt,  |
|                        | ro, rs_latin, sk, sl, sq, sv, sw, tl, tr, uz, vi             |
| `zh_sim_g2.pth`        | ch_sim + en                                                  |
| `japanese_g2.pth`      | ja + en                                                      |
| `korean_g2.pth`        | ko + en                                                      |
| `telugu.pth`           | te + en                                                      |
| `kannada.pth`          | kn + en                                                      |
| `cyrillic_g2.pth`      | ru, rs_cyrillic, be, bg, uk, mn, abq, ady, kbd, ava, dar,    |
|                        | inh, che, lbe, lez, tab, tjk, en                             |

<u>Notice</u>: keep the requested language list as short and specific as possible. Because the resolution
picks a model that covers *all* requested languages, adding a language you do not need downgrades the
model for the ones you do. For example, `["en"]` selects the English-specific `english_g2.pth`, while
`["en", "de"]` falls back to the broader `latin_g2.pth`, which is generally less accurate on English
text.

Check the semantic of easyocr language inputs here: https://www.jaided.ai/easyocr/



## Nemotron-OCR

This section describes Nemotron-OCR for versions `v2.0.0`, `v2.0.2`.

Nemotron works only on Linux and requires CUDA (Docling enforces 13.x).

The following table shows the supported Python versions and languages


| Nemotron version | Python version   | Supported language inputs                              |
| ---------------- | ---------------- | ------------------------------------------------------ |
| v2.0.0           | 3.12 only        | "english" (alias "en"), "multilingual" (alias "multi") |
| v2.0.2           | 3.11, 3.12, 3.13 | "english" (alias "en"), "multilingual" (alias "multi") |


The "multi/multilingual" languages cover: English, Chinese (Simplified and Traditional), Japanese, Korean, and Russian


## Tesseract - TesserOCR

Tesseract must be installed as a system package (see [installation](../getting_started/installation.md)).
TesserOCR is a python library that wraps the Tesseract engine.


[Languages support](https://tesseract-ocr.github.io/tessdoc/Data-Files-in-different-versions.html)


## OcrMac

This section describes ocrmac for versions `v1.0.0`, `v1.0.1`.

ocrmac is a thin wrapper around Apple's Vision framework. It is macOS-only and ships no model
artifacts of its own — the recognizers are part of the operating system. The supported language set
is therefore a property of the macOS version, not of the ocrmac release.

