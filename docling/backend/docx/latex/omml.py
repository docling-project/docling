"""Office Math Markup Language (OMML) to LaTeX converter.

This module provides functionality to convert Office Math Markup Language (OMML)
elements from Word documents into LaTeX format. It handles various mathematical
constructs including fractions, subscripts, superscripts, matrices, limits, and
special characters.

Adapted from https://github.com/xiilei/dwml/blob/master/dwml/omml.py on 23/01/2025.
"""

import logging
from typing import Any, Iterator

import lxml.etree as ET
from lxml.etree import _Element
from pylatexenc.latexencode import UnicodeToLatexEncoder

from docling.backend.docx.latex.latex_dict import (
    ALN,
    ARR,
    BACKSLASH,
    BLANK,
    BRK,
    CHARS,
    CHR,
    CHR_BO,
    CHR_DEFAULT,
    D_DEFAULT,
    F_DEFAULT,
    FUNC,
    FUNC_PLACE,
    GROUPING_FUNCS,
    LIM_FUNC,
    LIM_TO,
    LIM_UPP,
    MATH_CHARS,
    POS,
    POS_DEFAULT,
    RAD,
    RAD_DEFAULT,
    SUB,
    SUP,
    D,
    F,
    M,
    T,
)

OMML_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"

_log = logging.getLogger(__name__)


def load(stream: Any) -> Iterator[str]:
    """Load and convert OMML elements from a stream.

    Args:
        stream: Input stream containing OMML XML data.

    Yields:
        LaTeX string representation of each oMath element found.
    """
    tree = ET.parse(stream)
    for omath in tree.findall(OMML_NS + "oMath"):
        yield str(oMath2Latex(omath))


def load_string(string: str) -> Iterator[str]:
    """Load and convert OMML elements from a string.

    Args:
        string: XML string containing OMML data.

    Yields:
        LaTeX string representation of each oMath element found.
    """
    root = ET.fromstring(string)
    for omath in root.findall(OMML_NS + "oMath"):
        yield str(oMath2Latex(omath))


def escape_latex(strs: str) -> str:
    """Escape special LaTeX characters in a string.

    Args:
        strs: Input string to escape.

    Returns:
        String with LaTeX special characters properly escaped.
    """
    last = None
    new_chr = []
    strs = strs.replace(r"\\", "\\")
    for c in strs:
        if (c in CHARS) and (last != BACKSLASH):
            new_chr.append(BACKSLASH + c)
        else:
            new_chr.append(c)
        last = c
    return BLANK.join(new_chr)


def get_val(key: str | None, default: str, store: dict | None = CHR) -> str:
    """Get a value from a dictionary store or return the key/default.

    When default is provided, this function always returns a non-None string.

    Args:
        key: Key to look up in the store.
        default: Default value if key is None.
        store: Dictionary to look up the key in. If None, returns key directly.

    Returns:
        Value from store, the key itself, or the default value.
    """
    if key is not None:
        return key if not store else store.get(key, key)
    else:
        return default


class Tag2Method:
    """Base class for processing XML elements by mapping tags to methods."""

    tag2meth: dict[str, Any] = {}

    def call_method(self, elm: _Element, stag: str | None = None) -> Any | None:
        """Call the appropriate method for an XML element based on its tag.

        Args:
            elm: XML element to process.
            stag: Optional simplified tag name (without namespace).

        Returns:
            Result of the method call, or None if no method is found.
        """
        getmethod = self.tag2meth.get
        if stag is None:
            stag = elm.tag.replace(OMML_NS, "")
        method = getmethod(stag)
        if method:
            return method(self, elm)
        else:
            return None

    def process_children_list(
        self, elm: _Element, include: tuple[str, ...] | None = None
    ) -> Iterator[tuple[str, Any, _Element]]:
        """Process children of an element and yield results as tuples.

        Args:
            elm: Parent XML element.
            include: Optional tuple of tag names to include. If None, includes all.

        Yields:
            Tuple of (tag_name, processed_result, element) for each child.
        """
        for _e in list(elm):
            if OMML_NS not in _e.tag:
                continue
            stag = _e.tag.replace(OMML_NS, "")
            if include and (stag not in include):
                continue
            t = self.call_method(_e, stag=stag)
            if t is None:
                t = self.process_unknow(_e, stag)
                if t is None:
                    continue
            yield (stag, t, _e)

    def process_children_dict(
        self, elm: _Element, include: tuple[str, ...] | None = None
    ) -> dict[str, Any]:
        """Process children of an element and return results as a dictionary.

        Args:
            elm: Parent XML element.
            include: Optional tuple of tag names to include. If None, includes all.

        Returns:
            Dictionary mapping tag names to their processed results.
        """
        latex_chars = dict()
        for stag, t, e in self.process_children_list(elm, include):
            latex_chars[stag] = t
        return latex_chars

    def process_children(
        self, elm: _Element, include: tuple[str, ...] | None = None
    ) -> str:
        """Process children of an element and return concatenated string result.

        Args:
            elm: Parent XML element.
            include: Optional tuple of tag names to include. If None, includes all.

        Returns:
            Concatenated string of all processed children.
        """
        return BLANK.join(
            (
                t if not isinstance(t, Tag2Method) else str(t)
                for stag, t, e in self.process_children_list(elm, include)
            )
        )

    def process_unknow(self, elm: _Element, stag: str) -> Any | None:
        """Handle unknown element types.

        Args:
            elm: XML element.
            stag: Simplified tag name.

        Returns:
            None by default. Subclasses can override to provide custom handling.
        """
        return None


class Pr(Tag2Method):
    """Properties element processor for OMML elements."""

    text: str
    __val_tags: tuple[str, ...]
    __innerdict: dict[str, str | None]  # can't use the __dict__

    def __init__(self, elm: _Element):
        """Initialize properties processor.

        Args:
            elm: XML element containing properties.
        """
        self.__val_tags = ("chr", "pos", "begChr", "endChr", "type")
        self.__innerdict = {}
        self.text = self.process_children(elm)

    def __str__(self) -> str:
        """Return string representation."""
        return self.text

    def __unicode__(self) -> str:
        """Return unicode representation."""
        return self.__str__()

    def __getattr__(self, name: str) -> Any | None:
        """Get attribute from internal dictionary.

        Args:
            name: Attribute name.

        Returns:
            Attribute value or None if not found.
        """
        return self.__innerdict.get(name, None)

    def do_brk(self, elm: _Element) -> str:
        """Process line break element.

        Args:
            elm: XML element.

        Returns:
            LaTeX line break string.
        """
        self.__innerdict["brk"] = BRK
        return BRK

    def do_common(self, elm: _Element) -> None:
        """Process common property elements.

        Args:
            elm: XML element.

        Returns:
            None (stores value in internal dictionary).
        """
        stag = elm.tag.replace(OMML_NS, "")
        if stag in self.__val_tags:
            t = elm.get(f"{OMML_NS}val")
            self.__innerdict[stag] = t
        return None

    tag2meth = {
        "brk": do_brk,
        "chr": do_common,
        "pos": do_common,
        "begChr": do_common,
        "endChr": do_common,
        "type": do_common,
    }


class oMath2Latex(Tag2Method):
    """Convert OMML oMath elements to LaTeX format."""

    _t_dict: dict[str, str] = T
    __direct_tags: tuple[str, ...] = ("box", "num", "den", "deg", "e")
    u: UnicodeToLatexEncoder = UnicodeToLatexEncoder(
        replacement_latex_protection="braces-all",
        unknown_char_policy="keep",
        unknown_char_warning=False,
    )

    def __init__(self, element: _Element):
        """Initialize OMML to LaTeX converter.

        Args:
            element: Root oMath XML element to convert.
        """
        self._latex = self.process_children(element)

    def __str__(self) -> str:
        """Return LaTeX string with normalized spacing."""
        return self.latex.replace("  ", " ")

    def __unicode__(self) -> str:
        """Return unicode representation."""
        return self.__str__()

    def process_unknow(self, elm: _Element, stag: str) -> Any | None:
        """Handle unknown element types.

        Args:
            elm: XML element.
            stag: Simplified tag name.

        Returns:
            Processed children for direct tags, Pr object for property tags,
            or None for other unknown tags.
        """
        if stag in self.__direct_tags:
            return self.process_children(elm)
        elif stag[-2:] == "Pr":
            return Pr(elm)
        else:
            return None

    @property
    def latex(self) -> str:
        """Get the LaTeX representation.

        Returns:
            LaTeX string.
        """
        return self._latex

    def do_acc(self, elm: _Element) -> str:
        """Process accent element.

        Args:
            elm: XML element containing accent.

        Returns:
            LaTeX string with accent applied.
        """
        c_dict = self.process_children_dict(elm)
        latex_s = get_val(
            c_dict["accPr"].chr, default=CHR_DEFAULT["ACC_VAL"], store=CHR
        )
        # If latex_s contains %s, format it; otherwise return as-is (unmapped character)
        if "%s" in latex_s:
            return latex_s % (c_dict["e"],)
        else:
            return latex_s

    def do_bar(self, elm: _Element) -> str:
        """Process bar element (overline/underline).

        Args:
            elm: XML element containing bar.

        Returns:
            LaTeX string with bar applied.
        """
        c_dict = self.process_children_dict(elm)
        pr = c_dict["barPr"]
        latex_s = get_val(pr.pos, default=POS_DEFAULT["BAR_VAL"], store=POS)
        # If latex_s contains %s, format it; otherwise return as-is (unmapped character)
        if "%s" in latex_s:
            return pr.text + (latex_s % (c_dict["e"],))
        else:
            return pr.text + latex_s

    def do_d(self, elm: _Element) -> str:
        """Process delimiter element.

        Args:
            elm: XML element containing delimiter.

        Returns:
            LaTeX string with delimiters applied.
        """
        c_dict = self.process_children_dict(elm)
        pr = c_dict["dPr"]
        null = D_DEFAULT["null"]

        s_val = get_val(pr.begChr, default=D_DEFAULT["left"], store=T)
        e_val = get_val(pr.endChr, default=D_DEFAULT["right"], store=T)
        delim = pr.text + (
            D
            % {
                "left": null if not s_val else escape_latex(s_val),
                "text": c_dict["e"],
                "right": null if not e_val else escape_latex(e_val),
            }
        )
        return delim

    def do_spre(self, elm: _Element) -> None:
        """Process pre-sub-superscript element (not supported yet).

        Args:
            elm: XML element.

        Returns:
            None.
        """
        _log.warning("pre-sub-superscript element not supported yet")
        return None

    @staticmethod
    def _needs_grouping(latex_str: str) -> bool:
        """Check if a LaTeX string needs wrapping in braces for sub/superscript.

        Args:
            latex_str: LaTeX string to check.

        Returns:
            True if the string contains constructs that need grouping.
        """
        return "\\frac" in latex_str or "\\sqrt" in latex_str

    @staticmethod
    def _unwrap_script(script: str, marker: str) -> str:
        """Remove outer script wrapper if present.

        Args:
            script: Script string (subscript or superscript).
            marker: Script marker ("_" or "^").

        Returns:
            Unwrapped script content.
        """
        prefix = f"{marker}{{"
        if script.startswith(prefix) and script.endswith("}"):
            return script[len(prefix) : -1]
        return script

    def do_ssub(self, elm: _Element) -> str:
        """Process subscript element.

        Args:
            elm: XML element containing subscript.

        Returns:
            LaTeX string with subscript applied.
        """
        c_dict = self.process_children_dict(elm, include=("e", "sub", "sSubPr"))
        base = c_dict.get("e", "")
        sub = self._unwrap_script(c_dict.get("sub", ""), "_")
        if self._needs_grouping(base):
            base = "{" + base + "}"
        return base + (SUB % sub)

    def do_ssup(self, elm: _Element) -> str:
        """Process superscript element.

        Args:
            elm: XML element containing superscript.

        Returns:
            LaTeX string with superscript applied.
        """
        c_dict = self.process_children_dict(elm, include=("e", "sup", "sSupPr"))
        base = c_dict.get("e", "")
        sup = self._unwrap_script(c_dict.get("sup", ""), "^")
        if self._needs_grouping(base):
            base = "{" + base + "}"
        return base + (SUP % sup)

    def do_ssubsup(self, elm: _Element) -> str:
        """Process combined sub-superscript element.

        Args:
            elm: XML element containing both subscript and superscript.

        Returns:
            LaTeX string with both scripts applied.
        """
        c_dict = self.process_children_dict(
            elm, include=("e", "sub", "sup", "sSubSupPr")
        )
        base = c_dict.get("e", "")
        sub = self._unwrap_script(c_dict.get("sub", ""), "_")
        sup = self._unwrap_script(c_dict.get("sup", ""), "^")
        if self._needs_grouping(base):
            base = "{" + base + "}"
        return base + (SUB % sub) + (SUP % sup)

    def do_sub(self, elm: _Element) -> str:
        """Process standalone subscript content.

        Args:
            elm: XML element containing subscript content.

        Returns:
            LaTeX subscript string.
        """
        text = self.process_children(elm)
        return SUB % text

    def do_sup(self, elm: _Element) -> str:
        """Process standalone superscript content.

        Args:
            elm: XML element containing superscript content.

        Returns:
            LaTeX superscript string.
        """
        text = self.process_children(elm)
        return SUP % text

    def do_f(self, elm: _Element) -> str:
        """Process fraction element.

        Args:
            elm: XML element containing fraction.

        Returns:
            LaTeX fraction string.
        """
        c_dict = self.process_children_dict(elm)
        pr = c_dict.get("fPr")
        if pr is None:
            _log.debug("Missing fPr element in fraction, using default formatting")
            return F_DEFAULT % {
                "num": c_dict.get("num"),
                "den": c_dict.get("den"),
            }
        latex_s = get_val(pr.type, default=F_DEFAULT, store=F)
        return pr.text + (
            latex_s % {"num": c_dict.get("num"), "den": c_dict.get("den")}
        )

    def do_func(self, elm: _Element) -> str:
        """Process function application element.

        Args:
            elm: XML element containing function application.

        Returns:
            LaTeX function string.
        """
        c_dict = self.process_children_dict(elm)
        func_name = c_dict.get("fName", "")
        return func_name.replace(FUNC_PLACE, c_dict.get("e", ""))

    def do_fname(self, elm: _Element) -> str:
        """Process function name element.

        Args:
            elm: XML element containing function name.

        Returns:
            LaTeX function name with placeholder.
        """
        latex_chars = []
        for stag, t, e in self.process_children_list(elm):
            if stag == "r":
                if FUNC.get(t):
                    latex_chars.append(FUNC[t])
                else:
                    _log.warning("Function not supported, will default to text: %s", t)
                    if isinstance(t, str):
                        latex_chars.append(t)
            elif isinstance(t, str):
                latex_chars.append(t)
        t = BLANK.join(latex_chars)
        return t if FUNC_PLACE in t else t + FUNC_PLACE

    def do_groupchr(self, elm: _Element) -> str:
        """Process group character element (e.g., underbrace, overbrace).

        According to OMML spec, when chr is not specified and pos is not specified,
        the default position is "bot" (bottom), which corresponds to underbrace.

        Args:
            elm: XML element containing group character.

        Returns:
            LaTeX string with group character applied.
        """
        c_dict = self.process_children_dict(elm)
        pr = c_dict["groupChrPr"]
        latex_s = get_val(pr.chr, default=CHR_DEFAULT["GROUPCHR_VAL"], store=CHR)
        # If latex_s contains %s, format it; otherwise return as-is (unmapped character)
        if "%s" in latex_s:
            return pr.text + (latex_s % (c_dict["e"],))
        else:
            return pr.text + latex_s

    def do_rad(self, elm: _Element) -> str:
        """Process radical (root) element.

        Args:
            elm: XML element containing radical.

        Returns:
            LaTeX radical string.
        """
        c_dict = self.process_children_dict(elm)
        text = c_dict.get("e")
        deg_text = c_dict.get("deg")
        if deg_text:
            return RAD % {"deg": deg_text, "text": text}
        else:
            return RAD_DEFAULT % {"text": text}

    def do_eqarr(self, elm: _Element) -> str:
        """Process equation array element.

        Args:
            elm: XML element containing equation array.

        Returns:
            LaTeX array string.
        """
        return ARR % {
            "text": BRK.join(
                [t for stag, t, e in self.process_children_list(elm, include=("e",))]
            )
        }

    def do_limlow(self, elm: _Element) -> str:
        """Process lower limit element.

        Args:
            elm: XML element containing lower limit.

        Returns:
            LaTeX string with lower limit applied.
        """
        t_dict = self.process_children_dict(elm, include=("e", "lim"))
        base = t_dict.get("e", "")
        lim = t_dict.get("lim", "")

        # Check if base is a known limit function
        latex_s = LIM_FUNC.get(base)
        if latex_s:
            return latex_s % {"lim": lim}

        # Check if base is a grouping function (underbrace, overbrace, etc.)
        # These are already formatted LaTeX commands that just need a subscript
        if any(base.startswith(f"{func}{{") for func in GROUPING_FUNCS):
            return f"{base}_{{{lim}}}"

        # For unknown functions, log warning and use fallback
        _log.warning(
            f"Limit function {base} not in LIM_FUNC dictionary, using fallback format"
        )
        return f"{base}_{{{lim}}}"

    def do_limupp(self, elm: _Element) -> str:
        """Process upper limit element.

        Args:
            elm: XML element containing upper limit.

        Returns:
            LaTeX string with upper limit applied.
        """
        t_dict = self.process_children_dict(elm, include=("e", "lim"))
        return LIM_UPP % {"lim": t_dict.get("lim"), "text": t_dict.get("e")}

    def do_lim(self, elm: _Element) -> str:
        """Process limit content element.

        This processes the lower limit of limLow and upper limit of limUpp.
        It handles special formatting for limit labels including:
        - Replacing `rightarrow` with `to`
        - Stripping trailing line breaks
        - Escaping spaces in plain text labels

        Args:
            elm: XML element containing limit content.

        Returns:
            Processed limit string.
        """
        result = self.process_children(elm).replace(LIM_TO[0], LIM_TO[1])

        # Strip trailing LaTeX line breaks (\\) and whitespace
        result = result.rstrip()
        if result.endswith(BACKSLASH + BACKSLASH):
            result = result[:-2].rstrip()

        # Escape spaces with backslash-space for plain text labels in math mode
        if result and not any(char in result for char in MATH_CHARS):
            result = result.replace(" ", BACKSLASH + " ")
        return result

    def do_m(self, elm: _Element) -> str:
        """Process matrix element.

        Args:
            elm: XML element containing matrix.

        Returns:
            LaTeX matrix string.
        """
        rows = []
        for stag, t, e in self.process_children_list(elm):
            if stag == "mPr":
                pass
            elif stag == "mr":
                rows.append(t)
        return M % {"text": BRK.join(rows)}

    def do_mr(self, elm: _Element) -> str:
        """Process matrix row element.

        Args:
            elm: XML element containing matrix row.

        Returns:
            LaTeX matrix row string.
        """
        return ALN.join(
            [t for stag, t, e in self.process_children_list(elm, include=("e",))]
        )

    def do_nary(self, elm: _Element) -> str:
        """Process n-ary operator element (e.g., sum, product, integral).

        Args:
            elm: XML element containing n-ary operator.

        Returns:
            LaTeX n-ary operator string.
        """
        res = []
        bo = ""
        for stag, t, e in self.process_children_list(elm):
            if stag == "naryPr":
                # if <m:naryPr> contains no <m:chr>, the n-ary represents an integral
                bo = get_val(t.chr, default="\\int", store=CHR_BO)
            else:
                res.append(t)
        return bo + BLANK.join(res)

    _MATH_CHAR_MAP: dict[str, str] = {
        "\u2013": "-",  # EN DASH → minus
        "\u2014": "-",  # EM DASH → minus
        "\u2212": "-",  # MINUS SIGN → minus
        "\u005e": "^",  # CIRCUMFLEX → superscript operator
    }

    def process_unicode(self, s: str) -> str:
        """Process Unicode character and convert to LaTeX.

        Args:
            s: Unicode character to process.

        Returns:
            LaTeX representation of the character.
        """
        # Map characters that are math operators before the text encoder
        # converts them to text-mode macros like \textendash.
        if s in self._MATH_CHAR_MAP:
            return self._MATH_CHAR_MAP[s]

        out_latex_str = self.u.unicode_to_latex(s)

        if (
            s.startswith("{") is False
            and out_latex_str.startswith("{")
            and s.endswith("}") is False
            and out_latex_str.endswith("}")
        ):
            out_latex_str = f" {out_latex_str[1:-1]} "

        if "ensuremath" in out_latex_str:
            out_latex_str = out_latex_str.replace("\\ensuremath{", " ")
            out_latex_str = out_latex_str.replace("}", " ")

        if out_latex_str.strip().startswith("\\text"):
            out_latex_str = f" \\text{{{out_latex_str}}} "

        return out_latex_str

    def do_r(self, elm: _Element) -> str:
        """Process run element containing text.

        Args:
            elm: XML element containing text run.

        Returns:
            LaTeX string with text properly escaped and formatted.
        """
        _str = []
        _base_str = []
        found_text = elm.findtext(f"./{OMML_NS}t")
        if found_text:
            for s in found_text:
                out_latex_str = self.process_unicode(s)
                _str.append(out_latex_str)
                _base_str.append(s)

        proc_str = escape_latex(BLANK.join(_str))
        base_proc_str = BLANK.join(_base_str)

        if "{" not in base_proc_str and "\\{" in proc_str:
            proc_str = proc_str.replace("\\{", "{")

        if "}" not in base_proc_str and "\\}" in proc_str:
            proc_str = proc_str.replace("\\}", "}")

        # Undo escaping of characters that process_unicode intentionally
        # mapped to math operators (e.g. U+005E caret → ^).  escape_latex
        # treats them as text-mode specials, but inside <m:r> they are math.
        for orig, mapped in self._MATH_CHAR_MAP.items():
            if (
                mapped in CHARS
                and orig in (found_text or "")
                and f"\\{mapped}" in proc_str
            ):
                proc_str = proc_str.replace(f"\\{mapped}", mapped)

        return proc_str

    tag2meth = {
        "acc": do_acc,
        "r": do_r,
        "bar": do_bar,
        "sSub": do_ssub,
        "sSup": do_ssup,
        "sSubSup": do_ssubsup,
        "sub": do_sub,
        "sup": do_sup,
        "f": do_f,
        "func": do_func,
        "fName": do_fname,
        "groupChr": do_groupchr,
        "d": do_d,
        "rad": do_rad,
        "eqArr": do_eqarr,
        "limLow": do_limlow,
        "limUpp": do_limupp,
        "lim": do_lim,
        "m": do_m,
        "mr": do_mr,
        "nary": do_nary,
    }
