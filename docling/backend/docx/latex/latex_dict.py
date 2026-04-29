"""LaTeX dictionary for OMML to LaTeX conversion.

This module contains constants and dictionaries used for converting Office Math
Markup Language (OMML) to LaTeX format. It includes mappings for special characters,
mathematical symbols, functions, and formatting templates.

Adapted from https://github.com/xiilei/dwml/blob/master/dwml/latex_dict.py on 23/01/2025
"""

from typing import Final

CHARS: Final[tuple[str, ...]] = ("{", "}", "_", "^", "#", "&", "$", "%", "~")

BLANK: Final[str] = ""
BACKSLASH: Final[str] = "\\"
ALN: Final[str] = "&"

# Characters that indicate mathematical expressions (not plain text)
# Used to detect when spaces should be escaped in limit labels
MATH_CHARS: Final[tuple[str, ...]] = (
    BACKSLASH,
    "<",
    ">",
    "=",
    "+",
    "*",
    "/",
    "^",
    "_",
    "{",
    "}",
)

CHR: Final[dict[str, str]] = {
    # Unicode : Latex Math Symbols
    # Top accents
    "\u0300": "\\grave{%s}",
    "\u0301": "\\acute{%s}",
    "\u0302": "\\hat{%s}",
    "\u0303": "\\tilde{%s}",
    "\u0304": "\\bar{%s}",
    "\u0305": "\\overbar{%s}",
    "\u0306": "\\breve{%s}",
    "\u0307": "\\dot{%s}",
    "\u0308": "\\ddot{%s}",
    "\u0309": "\\ovhook{%s}",
    "\u030a": "\\ocirc{%s}}",
    "\u030c": "\\check{%s}}",
    "\u0310": "\\candra{%s}",
    "\u0312": "\\oturnedcomma{%s}",
    "\u0315": "\\ocommatopright{%s}",
    "\u031a": "\\droang{%s}",
    "\u0338": "\\not{%s}",
    "\u20d0": "\\leftharpoonaccent{%s}",
    "\u20d1": "\\rightharpoonaccent{%s}",
    "\u20d2": "\\vertoverlay{%s}",
    "\u20d6": "\\overleftarrow{%s}",
    "\u20d7": "\\vec{%s}",
    "\u20db": "\\dddot{%s}",
    "\u20dc": "\\ddddot{%s}",
    "\u20e1": "\\overleftrightarrow{%s}",
    "\u20e7": "\\annuity{%s}",
    "\u20e9": "\\widebridgeabove{%s}",
    "\u20f0": "\\asteraccent{%s}",
    # Bottom accents
    "\u0330": "\\wideutilde{%s}",
    "\u0331": "\\underbar{%s}",
    "\u20e8": "\\threeunderdot{%s}",
    "\u20ec": "\\underrightharpoondown{%s}",
    "\u20ed": "\\underleftharpoondown{%s}",
    "\u20ee": "\\underledtarrow{%s}",
    "\u20ef": "\\underrightarrow{%s}",
    # Over | group
    "\u23b4": "\\overbracket{%s}",
    "\u23dc": "\\overparen{%s}",
    "\u23de": "\\overbrace{%s}",
    # Under| group
    "\u23b5": "\\underbracket{%s}",
    "\u23dd": "\\underparen{%s}",
    "\u23df": "\\underbrace{%s}",
}

CHR_BO: Final[dict[str, str]] = {
    # Big operators
    "\u2140": "\\Bbbsum",
    "\u220f": "\\prod",
    "\u2210": "\\coprod",
    "\u2211": "\\sum",
    "\u222b": "\\int",
    "\u222c": "\\iint",
    "\u222d": "\\iiint",
    "\u222e": "\\oint",
    "\u222f": "\\oiint",
    "\u2230": "\\oiiint",
    "\u22c0": "\\bigwedge",
    "\u22c1": "\\bigvee",
    "\u22c2": "\\bigcap",
    "\u22c3": "\\bigcup",
    "\u2a00": "\\bigodot",
    "\u2a01": "\\bigoplus",
    "\u2a02": "\\bigotimes",
}

T: Final[dict[str, str]] = {
    # Greek letters
    "\U0001d6fc": "\\alpha ",
    "\U0001d6fd": "\\beta ",
    "\U0001d6fe": "\\gamma ",
    "\U0001d6ff": "\\theta ",
    "\U0001d700": "\\epsilon ",
    "\U0001d701": "\\zeta ",
    "\U0001d702": "\\eta ",
    "\U0001d703": "\\theta ",
    "\U0001d704": "\\iota ",
    "\U0001d705": "\\kappa ",
    "\U0001d706": "\\lambda ",
    "\U0001d707": "\\m ",
    "\U0001d708": "\\n ",
    "\U0001d709": "\\xi ",
    "\U0001d70a": "\\omicron ",
    "\U0001d70b": "\\pi ",
    "\U0001d70c": "\\rho ",
    "\U0001d70d": "\\varsigma ",
    "\U0001d70e": "\\sigma ",
    "\U0001d70f": "\\ta ",
    "\U0001d710": "\\upsilon ",
    "\U0001d711": "\\phi ",
    "\U0001d712": "\\chi ",
    "\U0001d713": "\\psi ",
    "\U0001d714": "\\omega ",
    "\U0001d715": "\\partial ",
    "\U0001d716": "\\varepsilon ",
    "\U0001d717": "\\vartheta ",
    "\U0001d718": "\\varkappa ",
    "\U0001d719": "\\varphi ",
    "\U0001d71a": "\\varrho ",
    "\U0001d71b": "\\varpi ",
    # Relation symbols
    "\u2190": "\\leftarrow ",
    "\u2191": "\\uparrow ",
    "\u2192": "\\rightarrow ",
    "\u2193": "\\downright ",
    "\u2194": "\\leftrightarrow ",
    "\u2195": "\\updownarrow ",
    "\u2196": "\\nwarrow ",
    "\u2197": "\\nearrow ",
    "\u2198": "\\searrow ",
    "\u2199": "\\swarrow ",
    "\u22ee": "\\vdots ",
    "\u22ef": "\\cdots ",
    "\u22f0": "\\adots ",
    "\u22f1": "\\ddots ",
    "\u2260": "\\ne ",
    "\u2264": "\\leq ",
    "\u2265": "\\geq ",
    "\u2266": "\\leqq ",
    "\u2267": "\\geqq ",
    "\u2268": "\\lneqq ",
    "\u2269": "\\gneqq ",
    "\u226a": "\\ll ",
    "\u226b": "\\gg ",
    "\u2208": "\\in ",
    "\u2209": "\\notin ",
    "\u220b": "\\ni ",
    "\u220c": "\\nni ",
    # Ordinary symbols
    "\u221e": "\\infty ",
    # Binary relations
    "\u00b1": "\\pm ",
    "\u2213": "\\mp ",
    # Italic, Latin, uppercase
    "\U0001d434": "A",
    "\U0001d435": "B",
    "\U0001d436": "C",
    "\U0001d437": "D",
    "\U0001d438": "E",
    "\U0001d439": "F",
    "\U0001d43a": "G",
    "\U0001d43b": "H",
    "\U0001d43c": "I",
    "\U0001d43d": "J",
    "\U0001d43e": "K",
    "\U0001d43f": "L",
    "\U0001d440": "M",
    "\U0001d441": "N",
    "\U0001d442": "O",
    "\U0001d443": "P",
    "\U0001d444": "Q",
    "\U0001d445": "R",
    "\U0001d446": "S",
    "\U0001d447": "T",
    "\U0001d448": "U",
    "\U0001d449": "V",
    "\U0001d44a": "W",
    "\U0001d44b": "X",
    "\U0001d44c": "Y",
    "\U0001d44d": "Z",
    # Italic, Latin, lowercase
    "\U0001d44e": "a",
    "\U0001d44f": "b",
    "\U0001d450": "c",
    "\U0001d451": "d",
    "\U0001d452": "e",
    "\U0001d453": "f",
    "\U0001d454": "g",
    "\U0001d456": "i",
    "\U0001d457": "j",
    "\U0001d458": "k",
    "\U0001d459": "l",
    "\U0001d45a": "m",
    "\U0001d45b": "n",
    "\U0001d45c": "o",
    "\U0001d45d": "p",
    "\U0001d45e": "q",
    "\U0001d45f": "r",
    "\U0001d460": "s",
    "\U0001d461": "t",
    "\U0001d462": "u",
    "\U0001d463": "v",
    "\U0001d464": "w",
    "\U0001d465": "x",
    "\U0001d466": "y",
    "\U0001d467": "z",
}

FUNC: Final[dict[str, str]] = {
    "sin": "\\sin({fe})",
    "cos": "\\cos({fe})",
    "tan": "\\tan({fe})",
    "arcsin": "\\arcsin({fe})",
    "arccos": "\\arccos({fe})",
    "arctan": "\\arctan({fe})",
    "arccot": "\\arccot({fe})",
    "sinh": "\\sinh({fe})",
    "cosh": "\\cosh({fe})",
    "tanh": "\\tanh({fe})",
    "coth": "\\coth({fe})",
    "sec": "\\sec({fe})",
    "csc": "\\csc({fe})",
    "mod": "\\mod {fe}",
    "max": "\\max({fe})",
    "min": "\\min({fe})",
    "log": "\\log({fe})",
    "ln": "\\ln({fe})",
    "exp": "\\exp({fe})",
    "det": "\\det({fe})",
    "gcd": "\\gcd({fe})",
    "deg": "\\deg({fe})",
    "hom": "\\hom({fe})",
    "ker": "\\ker({fe})",
    "dim": "\\dim({fe})",
    "arg": "\\arg({fe})",
    "inf": "\\inf({fe})",
    "sup": "\\sup({fe})",
    "lim": "\\lim({fe})",
    "Pr": "\\Pr({fe})",
}

FUNC_PLACE: Final[str] = "{fe}"

BRK: Final[str] = "\\\\"

CHR_DEFAULT: Final[dict[str, str]] = {
    "ACC_VAL": "\\hat{%s}",
    "GROUPCHR_VAL": "\\underbrace{%s}",
}

# Grouping functions that can have subscripts/superscripts
# These are bracket/brace functions, not limit functions
GROUPING_FUNCS: Final[tuple[str, ...]] = (
    "\\underbrace",
    "\\overbrace",
    "\\underparen",
    "\\overparen",
    "\\underbracket",
    "\\overbracket",
)

POS: Final[dict[str, str]] = {
    "top": "\\overline{%s}",
    "bot": "\\underline{%s}",
}

POS_DEFAULT: Final[dict[str, str]] = {
    "BAR_VAL": "\\overline{%s}",
}

SUB: Final[str] = "_{%s}"

SUP: Final[str] = "^{%s}"

F: Final[dict[str, str]] = {
    "bar": "\\frac{%(num)s}{%(den)s}",
    "skw": r"^{%(num)s}/_{%(den)s}",
    "noBar": "\\genfrac{}{}{0pt}{}{%(num)s}{%(den)s}",
    "lin": "{%(num)s}/{%(den)s}",
}
F_DEFAULT: Final[str] = "\\frac{%(num)s}{%(den)s}"

D: Final[str] = "\\left%(left)s%(text)s\\right%(right)s"

D_DEFAULT: Final[dict[str, str]] = {
    "left": "(",
    "right": ")",
    "null": ".",
}

RAD: Final[str] = "\\sqrt[%(deg)s]{%(text)s}"
RAD_DEFAULT: Final[str] = "\\sqrt{%(text)s}"
ARR: Final[str] = "%(text)s"

LIM_FUNC: Final[dict[str, str]] = {
    "lim": "\\lim_{%(lim)s}",
    "max": "\\max_{%(lim)s}",
    "min": "\\min_{%(lim)s}",
    "argmax": "\\operatorname{argmax}_{%(lim)s}",
    "argmin": "\\operatorname{argmin}_{%(lim)s}",
}

LIM_TO: Final[tuple[str, str]] = ("\\rightarrow", "\\to")

LIM_UPP: Final[str] = "\\overset{%(lim)s}{%(text)s}"

M: Final[str] = "\\begin{matrix}%(text)s\\end{matrix}"
