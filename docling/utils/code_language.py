"""Best-effort detection of a code block's programming language.

The blocks docling extracts are usually only a few lines. An explicit hint (a
Markdown fence token, an HTML ``language-*`` class) is therefore trusted first,
and content detection commits to a language only on a strong marker: downstream
consumers treat ``code_language`` as authoritative, so a wrong guess is worse
than ``UNKNOWN``.
"""

import json
import re
from typing import Optional

from docling_core.types.doc import CodeLanguageLabel

_LABEL_BY_VALUE: dict[str, CodeLanguageLabel] = {
    label.value.lower(): label for label in CodeLanguageLabel
}

_ALIASES = {
    "py": CodeLanguageLabel.PYTHON,
    "python2": CodeLanguageLabel.PYTHON,
    "python3": CodeLanguageLabel.PYTHON,
    "golang": CodeLanguageLabel.GO,
    "js": CodeLanguageLabel.JAVASCRIPT,
    "jsx": CodeLanguageLabel.JAVASCRIPT,
    "node": CodeLanguageLabel.JAVASCRIPT,
    "nodejs": CodeLanguageLabel.JAVASCRIPT,
    "ts": CodeLanguageLabel.TYPESCRIPT,
    "tsx": CodeLanguageLabel.TYPESCRIPT,
    "cpp": CodeLanguageLabel.C_PLUS_PLUS,
    "cxx": CodeLanguageLabel.C_PLUS_PLUS,
    "cc": CodeLanguageLabel.C_PLUS_PLUS,
    "cs": CodeLanguageLabel.C_SHARP,
    "csharp": CodeLanguageLabel.C_SHARP,
    "yml": CodeLanguageLabel.YAML,
    "mysql": CodeLanguageLabel.SQL,
    "postgres": CodeLanguageLabel.SQL,
    "postgresql": CodeLanguageLabel.SQL,
    "psql": CodeLanguageLabel.SQL,
    "sqlite": CodeLanguageLabel.SQL,
    "plsql": CodeLanguageLabel.SQL,
    "tsql": CodeLanguageLabel.SQL,
    "sh": CodeLanguageLabel.BASH,
    "shell": CodeLanguageLabel.BASH,
    "zsh": CodeLanguageLabel.BASH,
    "rb": CodeLanguageLabel.RUBY,
    "rs": CodeLanguageLabel.RUST,
    "kt": CodeLanguageLabel.KOTLIN,
    "kts": CodeLanguageLabel.KOTLIN,
    "objc": CodeLanguageLabel.OBJECTIVEC,
    "objective-c": CodeLanguageLabel.OBJECTIVEC,
    "tex": CodeLanguageLabel.LATEX,
    "vb": CodeLanguageLabel.VISUALBASIC,
    "vbnet": CodeLanguageLabel.VISUALBASIC,
    "htm": CodeLanguageLabel.HTML,
    "html5": CodeLanguageLabel.HTML,
    "xhtml": CodeLanguageLabel.HTML,
}

_HINT_PREFIXES = ("language-", "lang-")

_SHEBANG_INTERPRETERS = {
    "bash": CodeLanguageLabel.BASH,
    "sh": CodeLanguageLabel.BASH,
    "zsh": CodeLanguageLabel.BASH,
    "node": CodeLanguageLabel.JAVASCRIPT,
    "perl": CodeLanguageLabel.PERL,
    "php": CodeLanguageLabel.PHP,
    "python": CodeLanguageLabel.PYTHON,
    "ruby": CodeLanguageLabel.RUBY,
}

_SHEBANG_RE = re.compile(
    r"^#![^\n]*?\b(bash|zsh|sh|node|perl|php|python|ruby)[0-9.]*\b"
)
_PHP_RE = re.compile(r"<\?php\b")
_HTML_RE = re.compile(
    r"<!doctype\s+html\b|</(html|head|body)>|<(head|body)[\s>]", re.IGNORECASE
)
_DOCKERFILE_FROM_RE = re.compile(r"^\s*FROM\s+\S+", re.IGNORECASE | re.MULTILINE)
_DOCKERFILE_DIRECTIVE_RE = re.compile(
    r"^\s*(RUN|CMD|COPY|ADD|ENTRYPOINT|WORKDIR|ENV|EXPOSE)\b",
    re.IGNORECASE | re.MULTILINE,
)
_CPP_RE = re.compile(r"\bstd::|\bcout\b|\btemplate\s*<|\bnamespace\b")
_C_RE = re.compile(r"\bint\s+main\s*\(|\bprintf\s*\(|\bscanf\s*\(")
_JSON_PREFIX = ("{", "[")

# The JavaScript/TypeScript/Java/C# family shares too many generic keywords
# (import, const, public, class, ...) to tell apart from a handful of lines, so
# each rule keys on a marker distinctive to one language and a snippet that could
# be several of them is left UNKNOWN rather than guessed.
_CONTENT_RULES = (
    (
        CodeLanguageLabel.GO,
        re.compile(
            r"^\s*package\s+main\b|\bfunc\s+\(\w+\s+\*?\w+\)|\bfmt\.(Print|Println|Printf)\b",
            re.MULTILINE,
        ),
    ),
    (
        CodeLanguageLabel.RUST,
        re.compile(
            r"\bfn\s+main\s*\(|\blet\s+mut\b|\bprintln!\s*\(|\bfn\s+\w+[^\n]*->"
        ),
    ),
    (
        CodeLanguageLabel.PYTHON,
        re.compile(
            r"^\s*def\s+\w+\s*\([^\n]*\)\s*(->[^\n:]+)?:"
            r"|^\s*elif\b|\b__name__\b|^\s*from\s+\S+\s+import\b",
            re.MULTILINE,
        ),
    ),
    (
        CodeLanguageLabel.JAVA,
        re.compile(
            r"\bimport\s+java\.|\bSystem\.out\.print|\bpublic\s+static\s+void\s+main"
        ),
    ),
    (
        CodeLanguageLabel.C_SHARP,
        re.compile(
            r"\busing\s+System\b|\bConsole\.(Write|WriteLine)\b"
            r"|\bnamespace\s+[\w.]+\s*[{;]"
        ),
    ),
    (
        CodeLanguageLabel.SQL,
        re.compile(
            r"^\s*select\b.*\bfrom\b.*(\bwhere\b|\bjoin\b|\bgroup\s+by\b"
            r"|\border\s+by\b|;)"
            r"|^\s*insert\s+into\s+\w+\s*(\(|values\b|select\b)"
            r"|^\s*update\s+\w+\s+set\b.*="
            r"|^\s*delete\s+from\s+\w+\s*(\bwhere\b|;)"
            r"|^\s*create\s+(table|view|index|database)\s+(if\s+not\s+exists\s+)?"
            r"\w+\s*(\(|as\b)"
            r"|^\s*alter\s+table\s+\w+\s+(add|drop|modify|alter|rename)\b"
            r"|^\s*drop\s+(table|view|index|database)\s+(if\s+exists\s+)?\w+\s*;",
            re.IGNORECASE | re.MULTILINE,
        ),
    ),
    (
        CodeLanguageLabel.TYPESCRIPT,
        re.compile(
            r"\b(readonly|public|private|protected)\s+\w+\s*:\s*\w+"
            r"|:\s*(string|number|boolean)(\[\])?\s*[;,)=]"
        ),
    ),
    (
        CodeLanguageLabel.JAVASCRIPT,
        re.compile(
            r"\bconsole\.log\s*\(|\brequire\s*\(|\bmodule\.exports\b"
            r"|\bdocument\.(getElementById|querySelector)\b"
        ),
    ),
)


def normalize_code_language(hint: Optional[str]) -> CodeLanguageLabel:
    if not hint:
        return CodeLanguageLabel.UNKNOWN

    token = hint.strip().lower()
    for prefix in _HINT_PREFIXES:
        if token.startswith(prefix):
            token = token[len(prefix) :]
            break

    label = _LABEL_BY_VALUE.get(token)
    if label is not None:
        return label
    return _ALIASES.get(token, CodeLanguageLabel.UNKNOWN)


def detect_code_language(text: str, hint: Optional[str] = None) -> CodeLanguageLabel:
    """An explicit ``hint`` wins over content analysis."""
    label = normalize_code_language(hint)
    if label is not CodeLanguageLabel.UNKNOWN:
        return label

    if not text or not text.strip():
        return CodeLanguageLabel.UNKNOWN

    return _detect_from_content(text)


def _detect_from_content(text: str) -> CodeLanguageLabel:
    head = text.lstrip()

    shebang = _SHEBANG_RE.match(head)
    if shebang:
        return _SHEBANG_INTERPRETERS[shebang.group(1)]

    if _PHP_RE.search(text):
        return CodeLanguageLabel.PHP
    if _HTML_RE.search(text):
        return CodeLanguageLabel.HTML
    if _DOCKERFILE_FROM_RE.search(text) and _DOCKERFILE_DIRECTIVE_RE.search(text):
        return CodeLanguageLabel.DOCKERFILE

    if "#include" in text:
        if _CPP_RE.search(text):
            return CodeLanguageLabel.C_PLUS_PLUS
        if _C_RE.search(text):
            return CodeLanguageLabel.C

    for label, pattern in _CONTENT_RULES:
        if pattern.search(text):
            return label

    if _looks_like_json(text):
        return CodeLanguageLabel.JSON

    return CodeLanguageLabel.UNKNOWN


def _looks_like_json(text: str) -> bool:
    stripped = text.strip()
    if not stripped or stripped[0] not in _JSON_PREFIX:
        return False
    try:
        json.loads(stripped)
    except ValueError:
        return False
    return True
