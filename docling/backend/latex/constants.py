MACROS_NEWCOMMAND = frozenset(["newcommand", "renewcommand", "providecommand"])

MACROS_PREAMBLE_METADATA = frozenset(["title", "author", "date"])

MACROS_INLINE_VERBATIM = frozenset(["%", "$", "&", "#", "_", "{", "}", "~"])

MACROS_TEXT_FORMATTING = frozenset(["textbf", "textit", "emph", "texttt", "underline"])

MACROS_CITATION = frozenset(["cite", "citep", "citet", "ref", "eqref"])

MACROS_COLOR = frozenset(["color", "definecolor", "colorlet"])

MACROS_STRUCTURAL = frozenset(
    [
        "section",
        "subsection",
        "subsubsection",
        "chapter",
        "part",
        "paragraph",
        "subparagraph",
        "caption",
        "label",
        "includegraphics",
        "bibliography",
        "title",
        "author",
        "maketitle",
        "footnote",
        "marginpar",
        "textsc",
        "textsf",
        "textrm",
        "textnormal",
        "mbox",
        "href",
        "newline",
        "hfill",
        "break",
        "centering",
        "textcolor",
        "colorbox",
        "item",
        "input",
        "include",
    ]
)

MACROS_HEADING = frozenset(
    [
        "part",
        "chapter",
        "section",
        "subsection",
        "subsubsection",
        "paragraph",
        "subparagraph",
    ]
)

MACROS_TEXT_STYLE = frozenset(["textsc", "textsf", "textrm", "textnormal", "mbox"])

MACROS_IGNORED = frozenset(
    [
        "documentclass",
        "usepackage",
        "geometry",
        "hypersetup",
        "lstset",
        "bibliographystyle",
        "newcommand",
        "renewcommand",
        "def",
        "let",
        "edef",
        "gdef",
        "xdef",
        "newenvironment",
        "renewenvironment",
        "DeclareMathOperator",
        "DeclareMathSymbol",
        "setlength",
        "setcounter",
        "addtolength",
        "color",
        "definecolor",
        "colorlet",
        "AtBeginDocument",
        "AtEndDocument",
        "newlength",
        "newcounter",
        "newif",
        "providecommand",
        "DeclareOption",
        "RequirePackage",
        "ProvidesPackage",
        "LoadClass",
        "makeatletter",
        "makeatother",
        "NeedsTeXFormat",
        "ProvidesClass",
        "DeclareRobustCommand",
        "newtheorem",
        "theoremstyle",
        "newtheoremstyle",
        "documentstyle",
        "pagestyle",
        "thispagestyle",
        "pagenumbering",
        "tableofcontents",
        "listoffigures",
        "listoftables",
        "appendix",
        "cleardoublepage",
        "clearpage",
        "newpage",
        "markboth",
        "markright",
        "lhead",
        "rhead",
        "cfoot",
        "hyphenation",
        "overfullrule",
        "protect",
    ]
)

MACROS_ACCENTS = frozenset(
    ["'", '"', "^", "`", "~", "=", ".", "c", "d", "b", "H", "k", "r", "t", "u", "v"]
)

MACROS_SPACING = frozenset(
    [
        "newline",
        "hfill",
        "break",
        "centering",
        "noindent",
        "par",
        "smallskip",
        "medskip",
        "bigskip",
        "vfill",
        "vskip",
        "hskip",
    ]
)

MACROS_LEGACY_FORMATTING = frozenset(
    [
        "bf",
        "it",
        "rm",
        "sc",
        "sf",
        "sl",
        "tt",
        "cal",
        "em",
        "tiny",
        "scriptsize",
        "footnotesize",
        "small",
        "large",
        "Large",
        "LARGE",
        "huge",
        "Huge",
        "color",
    ]
)

MACROS_ESCAPED = frozenset(["&", "%", "$", "#", "_", "{", "}"])

ENV_MATH_DISPLAY_PREFIXES = (
    "$$",
    "\\[",
    "\\begin{equation}",
    "\\begin{align}",
    "\\begin{gather}",
    "\\begin{displaymath}",
)

ENV_MATH_CLEAN = frozenset(
    [
        "equation",
        "equation*",
        "displaymath",
        "math",
        "eqnarray",
        "eqnarray*",
        "dmath",
        "dmath*",
    ]
)

ENV_MATH = frozenset(
    [
        "equation",
        "align",
        "gather",
        "multline",
        "flalign",
        "alignat",
        "displaymath",
        "eqnarray",
        "dmath",
        "dgroup",
        "darray",
    ]
)

ENV_THEOREM = frozenset(
    [
        "theorem",
        "lemma",
        "corollary",
        "proposition",
        "definition",
        "remark",
        "example",
        "conjecture",
    ]
)

ENV_LIST = frozenset(["itemize", "enumerate", "description"])

ENV_QUOTE = frozenset(["quote", "quotation", "verse"])

TABLE_MACROS_RULE = frozenset(
    ["hline", "cline", "toprule", "midrule", "bottomrule", "cmidrule", "specialrule"]
)

TABLE_MACROS_IGNORE = frozenset(
    [
        "rule",
        "vspace",
        "hspace",
        "vskip",
        "hskip",
        "smallskip",
        "medskip",
        "bigskip",
        "strut",
        "phantom",
        "hphantom",
        "vphantom",
        "noalign",
    ]
)
