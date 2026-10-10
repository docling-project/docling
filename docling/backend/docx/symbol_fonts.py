# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Unicode mapping for text in the Symbol font.

Word stores characters of symbol fonts as font code points, either as the plain
byte (``a`` for alpha) or shifted into the Private Use Area (``U+F061``). The
`SYMBOL_FONT_TO_UNICODE` table maps the byte to Unicode.

The table is generated from the Adobe Symbol Encoding to Unicode mapping
(https://unicode.org/Public/MAPPINGS/VENDORS/ADOBE/symbol.txt), with these
changes:

- The source also maps the Greek letters Delta, Omega and mu (``0x44``,
  ``0x57``, ``0x6D``) to signs. The table uses the Greek letters.
- The serif and sans serif registered, copyright and trade mark signs
  (``0xD2``-``0xD4``, ``0xE2``-``0xE4``) map to the standard signs, not to the
  Corporate Use Subarea. Other Corporate Use Subarea characters are not included.

Notice of the source file:

     Name:             Adobe Symbol Encoding to Unicode
     Unicode version:  2.0
     Table version:    1.0
     Date:             2011 July 12

     Copyright (c) 1991-2011 Unicode, Inc. All Rights reserved.

     This file is provided as-is by Unicode, Inc. (The Unicode Consortium). No
     claims are made as to fitness for any particular purpose. No warranties of
     any kind are expressed or implied. The recipient agrees to determine
     applicability of information provided. If this file has been provided on
     magnetic media by Unicode, Inc., the sole remedy for any claim will be
     exchange of defective media within 90 days of receipt.

     Unicode, Inc. hereby grants the right to freely use the information
     supplied in this file in the creation of products supporting the
     Unicode Standard, and to make copies of this file in any form for
     internal or external distribution as long as this notice remains
     attached.
"""

SYMBOL_FONT_TO_UNICODE: dict[int, str] = {
    0x20: "\u0020",  # SPACE
    0x21: "\u0021",  # EXCLAMATION MARK
    0x22: "\u2200",  # FOR ALL
    0x23: "\u0023",  # NUMBER SIGN
    0x24: "\u2203",  # THERE EXISTS
    0x25: "\u0025",  # PERCENT SIGN
    0x26: "\u0026",  # AMPERSAND
    0x27: "\u220b",  # CONTAINS AS MEMBER
    0x28: "\u0028",  # LEFT PARENTHESIS
    0x29: "\u0029",  # RIGHT PARENTHESIS
    0x2A: "\u2217",  # ASTERISK OPERATOR
    0x2B: "\u002b",  # PLUS SIGN
    0x2C: "\u002c",  # COMMA
    0x2D: "\u2212",  # MINUS SIGN
    0x2E: "\u002e",  # FULL STOP
    0x2F: "\u002f",  # SOLIDUS
    0x30: "\u0030",  # DIGIT ZERO
    0x31: "\u0031",  # DIGIT ONE
    0x32: "\u0032",  # DIGIT TWO
    0x33: "\u0033",  # DIGIT THREE
    0x34: "\u0034",  # DIGIT FOUR
    0x35: "\u0035",  # DIGIT FIVE
    0x36: "\u0036",  # DIGIT SIX
    0x37: "\u0037",  # DIGIT SEVEN
    0x38: "\u0038",  # DIGIT EIGHT
    0x39: "\u0039",  # DIGIT NINE
    0x3A: "\u003a",  # COLON
    0x3B: "\u003b",  # SEMICOLON
    0x3C: "\u003c",  # LESS-THAN SIGN
    0x3D: "\u003d",  # EQUALS SIGN
    0x3E: "\u003e",  # GREATER-THAN SIGN
    0x3F: "\u003f",  # QUESTION MARK
    0x40: "\u2245",  # APPROXIMATELY EQUAL TO
    0x41: "\u0391",  # GREEK CAPITAL LETTER ALPHA
    0x42: "\u0392",  # GREEK CAPITAL LETTER BETA
    0x43: "\u03a7",  # GREEK CAPITAL LETTER CHI
    0x44: "\u0394",  # GREEK CAPITAL LETTER DELTA
    0x45: "\u0395",  # GREEK CAPITAL LETTER EPSILON
    0x46: "\u03a6",  # GREEK CAPITAL LETTER PHI
    0x47: "\u0393",  # GREEK CAPITAL LETTER GAMMA
    0x48: "\u0397",  # GREEK CAPITAL LETTER ETA
    0x49: "\u0399",  # GREEK CAPITAL LETTER IOTA
    0x4A: "\u03d1",  # GREEK THETA SYMBOL
    0x4B: "\u039a",  # GREEK CAPITAL LETTER KAPPA
    0x4C: "\u039b",  # GREEK CAPITAL LETTER LAMDA
    0x4D: "\u039c",  # GREEK CAPITAL LETTER MU
    0x4E: "\u039d",  # GREEK CAPITAL LETTER NU
    0x4F: "\u039f",  # GREEK CAPITAL LETTER OMICRON
    0x50: "\u03a0",  # GREEK CAPITAL LETTER PI
    0x51: "\u0398",  # GREEK CAPITAL LETTER THETA
    0x52: "\u03a1",  # GREEK CAPITAL LETTER RHO
    0x53: "\u03a3",  # GREEK CAPITAL LETTER SIGMA
    0x54: "\u03a4",  # GREEK CAPITAL LETTER TAU
    0x55: "\u03a5",  # GREEK CAPITAL LETTER UPSILON
    0x56: "\u03c2",  # GREEK SMALL LETTER FINAL SIGMA
    0x57: "\u03a9",  # GREEK CAPITAL LETTER OMEGA
    0x58: "\u039e",  # GREEK CAPITAL LETTER XI
    0x59: "\u03a8",  # GREEK CAPITAL LETTER PSI
    0x5A: "\u0396",  # GREEK CAPITAL LETTER ZETA
    0x5B: "\u005b",  # LEFT SQUARE BRACKET
    0x5C: "\u2234",  # THEREFORE
    0x5D: "\u005d",  # RIGHT SQUARE BRACKET
    0x5E: "\u22a5",  # UP TACK
    0x5F: "\u005f",  # LOW LINE
    0x61: "\u03b1",  # GREEK SMALL LETTER ALPHA
    0x62: "\u03b2",  # GREEK SMALL LETTER BETA
    0x63: "\u03c7",  # GREEK SMALL LETTER CHI
    0x64: "\u03b4",  # GREEK SMALL LETTER DELTA
    0x65: "\u03b5",  # GREEK SMALL LETTER EPSILON
    0x66: "\u03c6",  # GREEK SMALL LETTER PHI
    0x67: "\u03b3",  # GREEK SMALL LETTER GAMMA
    0x68: "\u03b7",  # GREEK SMALL LETTER ETA
    0x69: "\u03b9",  # GREEK SMALL LETTER IOTA
    0x6A: "\u03d5",  # GREEK PHI SYMBOL
    0x6B: "\u03ba",  # GREEK SMALL LETTER KAPPA
    0x6C: "\u03bb",  # GREEK SMALL LETTER LAMDA
    0x6D: "\u03bc",  # GREEK SMALL LETTER MU
    0x6E: "\u03bd",  # GREEK SMALL LETTER NU
    0x6F: "\u03bf",  # GREEK SMALL LETTER OMICRON
    0x70: "\u03c0",  # GREEK SMALL LETTER PI
    0x71: "\u03b8",  # GREEK SMALL LETTER THETA
    0x72: "\u03c1",  # GREEK SMALL LETTER RHO
    0x73: "\u03c3",  # GREEK SMALL LETTER SIGMA
    0x74: "\u03c4",  # GREEK SMALL LETTER TAU
    0x75: "\u03c5",  # GREEK SMALL LETTER UPSILON
    0x76: "\u03d6",  # GREEK PI SYMBOL
    0x77: "\u03c9",  # GREEK SMALL LETTER OMEGA
    0x78: "\u03be",  # GREEK SMALL LETTER XI
    0x79: "\u03c8",  # GREEK SMALL LETTER PSI
    0x7A: "\u03b6",  # GREEK SMALL LETTER ZETA
    0x7B: "\u007b",  # LEFT CURLY BRACKET
    0x7C: "\u007c",  # VERTICAL LINE
    0x7D: "\u007d",  # RIGHT CURLY BRACKET
    0x7E: "\u223c",  # TILDE OPERATOR
    0xA0: "\u20ac",  # EURO SIGN
    0xA1: "\u03d2",  # GREEK UPSILON WITH HOOK SYMBOL
    0xA2: "\u2032",  # PRIME
    0xA3: "\u2264",  # LESS-THAN OR EQUAL TO
    0xA4: "\u2044",  # FRACTION SLASH
    0xA5: "\u221e",  # INFINITY
    0xA6: "\u0192",  # LATIN SMALL LETTER F WITH HOOK
    0xA7: "\u2663",  # BLACK CLUB SUIT
    0xA8: "\u2666",  # BLACK DIAMOND SUIT
    0xA9: "\u2665",  # BLACK HEART SUIT
    0xAA: "\u2660",  # BLACK SPADE SUIT
    0xAB: "\u2194",  # LEFT RIGHT ARROW
    0xAC: "\u2190",  # LEFTWARDS ARROW
    0xAD: "\u2191",  # UPWARDS ARROW
    0xAE: "\u2192",  # RIGHTWARDS ARROW
    0xAF: "\u2193",  # DOWNWARDS ARROW
    0xB0: "\u00b0",  # DEGREE SIGN
    0xB1: "\u00b1",  # PLUS-MINUS SIGN
    0xB2: "\u2033",  # DOUBLE PRIME
    0xB3: "\u2265",  # GREATER-THAN OR EQUAL TO
    0xB4: "\u00d7",  # MULTIPLICATION SIGN
    0xB5: "\u221d",  # PROPORTIONAL TO
    0xB6: "\u2202",  # PARTIAL DIFFERENTIAL
    0xB7: "\u2022",  # BULLET
    0xB8: "\u00f7",  # DIVISION SIGN
    0xB9: "\u2260",  # NOT EQUAL TO
    0xBA: "\u2261",  # IDENTICAL TO
    0xBB: "\u2248",  # ALMOST EQUAL TO
    0xBC: "\u2026",  # HORIZONTAL ELLIPSIS
    0xBF: "\u21b5",  # DOWNWARDS ARROW WITH CORNER LEFTWARDS
    0xC0: "\u2135",  # ALEF SYMBOL
    0xC1: "\u2111",  # BLACK-LETTER CAPITAL I
    0xC2: "\u211c",  # BLACK-LETTER CAPITAL R
    0xC3: "\u2118",  # SCRIPT CAPITAL P
    0xC4: "\u2297",  # CIRCLED TIMES
    0xC5: "\u2295",  # CIRCLED PLUS
    0xC6: "\u2205",  # EMPTY SET
    0xC7: "\u2229",  # INTERSECTION
    0xC8: "\u222a",  # UNION
    0xC9: "\u2283",  # SUPERSET OF
    0xCA: "\u2287",  # SUPERSET OF OR EQUAL TO
    0xCB: "\u2284",  # NOT A SUBSET OF
    0xCC: "\u2282",  # SUBSET OF
    0xCD: "\u2286",  # SUBSET OF OR EQUAL TO
    0xCE: "\u2208",  # ELEMENT OF
    0xCF: "\u2209",  # NOT AN ELEMENT OF
    0xD0: "\u2220",  # ANGLE
    0xD1: "\u2207",  # NABLA
    0xD2: "\u00ae",  # REGISTERED SIGN
    0xD3: "\u00a9",  # COPYRIGHT SIGN
    0xD4: "\u2122",  # TRADE MARK SIGN
    0xD5: "\u220f",  # N-ARY PRODUCT
    0xD6: "\u221a",  # SQUARE ROOT
    0xD7: "\u22c5",  # DOT OPERATOR
    0xD8: "\u00ac",  # NOT SIGN
    0xD9: "\u2227",  # LOGICAL AND
    0xDA: "\u2228",  # LOGICAL OR
    0xDB: "\u21d4",  # LEFT RIGHT DOUBLE ARROW
    0xDC: "\u21d0",  # LEFTWARDS DOUBLE ARROW
    0xDD: "\u21d1",  # UPWARDS DOUBLE ARROW
    0xDE: "\u21d2",  # RIGHTWARDS DOUBLE ARROW
    0xDF: "\u21d3",  # DOWNWARDS DOUBLE ARROW
    0xE0: "\u25ca",  # LOZENGE
    0xE1: "\u2329",  # LEFT-POINTING ANGLE BRACKET
    0xE2: "\u00ae",  # REGISTERED SIGN
    0xE3: "\u00a9",  # COPYRIGHT SIGN
    0xE4: "\u2122",  # TRADE MARK SIGN
    0xE5: "\u2211",  # N-ARY SUMMATION
    0xF1: "\u232a",  # RIGHT-POINTING ANGLE BRACKET
    0xF2: "\u222b",  # INTEGRAL
    0xF3: "\u2320",  # TOP HALF INTEGRAL
    0xF5: "\u2321",  # BOTTOM HALF INTEGRAL
}
