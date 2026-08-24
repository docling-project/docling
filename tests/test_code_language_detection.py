import threading
import time

import pytest
from docling_core.types.doc import CodeLanguageLabel

from docling.utils.code_language import (
    detect_code_language,
    normalize_code_language,
)

PYTHON_FUNC = "def greet(name):\n    print(f'hi {name}')\n"
GO_MAIN = 'package main\n\nfunc main() {\n    fmt.Println("hi")\n}\n'
RUST_MAIN = 'fn main() {\n    let mut total = 0;\n    println!("{}", total);\n}\n'
JAVA_CLASS = "import java.util.List;\n\npublic class App {\n    int x;\n}\n"
CSHARP_HELLO = 'using System;\n\nConsole.WriteLine("hi");\n'
C_HELLO = '#include <stdio.h>\n\nint main() {\n    printf("hi");\n}\n'
CPP_HELLO = '#include <iostream>\n\nint main() {\n    std::cout << "hi";\n}\n'
SQL_QUERY = "SELECT id, name FROM users WHERE active = 1;"
SQL_MULTILINE = "SELECT id, name\nFROM users\nWHERE active = 1"
TS_INTERFACE = "interface User {\n    name: string;\n    age: number;\n}\n"
JS_SNIPPET = "const add = (a, b) => a + b;\nconsole.log(add(1, 2));\n"
PHP_SNIPPET = '<?php echo "hi"; ?>'
HTML_DOC = "<!DOCTYPE html>\n<html><body><p>hi</p></body></html>"
DOCKERFILE = 'FROM python:3.11-slim\nRUN pip install docling\nCMD ["docling"]\n'
JSON_DOC = '{"name": "docling", "tags": ["pdf", "html"]}'
BASH_SCRIPT = "#!/bin/bash\nset -euo pipefail\necho hello\n"


@pytest.mark.parametrize(
    "text, expected",
    [
        (PYTHON_FUNC, CodeLanguageLabel.PYTHON),
        (GO_MAIN, CodeLanguageLabel.GO),
        (RUST_MAIN, CodeLanguageLabel.RUST),
        (JAVA_CLASS, CodeLanguageLabel.JAVA),
        (CSHARP_HELLO, CodeLanguageLabel.C_SHARP),
        (C_HELLO, CodeLanguageLabel.C),
        (CPP_HELLO, CodeLanguageLabel.C_PLUS_PLUS),
        (SQL_QUERY, CodeLanguageLabel.SQL),
        (SQL_MULTILINE, CodeLanguageLabel.SQL),
        (TS_INTERFACE, CodeLanguageLabel.TYPESCRIPT),
        (JS_SNIPPET, CodeLanguageLabel.JAVASCRIPT),
        (PHP_SNIPPET, CodeLanguageLabel.PHP),
        (HTML_DOC, CodeLanguageLabel.HTML),
        (DOCKERFILE, CodeLanguageLabel.DOCKERFILE),
        (JSON_DOC, CodeLanguageLabel.JSON),
        (BASH_SCRIPT, CodeLanguageLabel.BASH),
    ],
)
def test_detect_from_content(text, expected):
    assert detect_code_language(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   \n  ",
        "x = compute(y)",
        "foo(bar, baz)",
        "const value = 1;",
        "public void render() {}",
        "{'name': 'docling'}",
        # Prose that happens to contain keywords must not be misread as code.
        "Please select an option from the list below to proceed.",
        "We select features from the model based on importance scores.",
        "The namespace of the project should be unique.",
        "Create table layouts that adapt to screen size.",
        "Update the set of rules to apply next.",
        "Drop table support was requested by users.",
        # interface/enum exist in Java and C# too, so they are not TS-distinctive.
        "public interface Repository {\n    void save();\n}",
        "public enum Color { RED, GREEN, BLUE }",
        # SQL embedded in a string literal is not a SQL document.
        'cur.execute("select id from users where active = 1")',
        # Multi-line prose must not read as SQL just because a line opens with a
        # keyword and a later line happens to carry from/where/order by/=.
        "Select the items you want to keep.\nGet them out from the cupboard.\n"
        "This is where things get tricky.",
        "Select a plan that fits.\nChoose from our tiers.\nWe will join you shortly.",
        "Update your set of preferences soon.\nThe answer = yes.",
    ],
)
def test_ambiguous_content_stays_unknown(text):
    assert detect_code_language(text) == CodeLanguageLabel.UNKNOWN


@pytest.mark.parametrize(
    "text",
    [
        # A line of repeated keywords is the bait for polynomial backtracking:
        # untempered ".*" gaps around the `from` pivot would re-scan the tail
        # for every `from`. The tempered gaps keep this linear.
        pytest.param("select " + "from " * 40000, id="repeated-from-keywords"),
        # Many SELECT line-start anchors under re.MULTILINE.
        pytest.param("select x from y\n" * 40000, id="repeated-select-lines"),
        # A literal line followed by a long blank run: a leading `\s` in a
        # line-anchored rule would swallow the blanks and go quadratic.
        pytest.param("FROM ubuntu\n" + "\n" * 40000, id="dockerfile-long-blank-run"),
        pytest.param("def f():\n" + "\n" * 40000, id="python-long-blank-run"),
    ],
)
def test_detection_stays_linear_on_large_input(text):
    result: list[CodeLanguageLabel] = []

    def _run() -> None:
        result.append(detect_code_language(text))

    worker = threading.Thread(target=_run, daemon=True)
    t0 = time.monotonic()
    worker.start()
    worker.join(timeout=10.0)
    elapsed = time.monotonic() - t0

    assert not worker.is_alive(), (
        f"detect_code_language() still running after {elapsed:.1f}s on a large block."
    )
    assert isinstance(result[0], CodeLanguageLabel)


@pytest.mark.parametrize(
    "hint, expected",
    [
        ("python", CodeLanguageLabel.PYTHON),
        ("Python", CodeLanguageLabel.PYTHON),
        ("py", CodeLanguageLabel.PYTHON),
        ("language-go", CodeLanguageLabel.GO),
        ("lang-rust", CodeLanguageLabel.RUST),
        ("c++", CodeLanguageLabel.C_PLUS_PLUS),
        ("cpp", CodeLanguageLabel.C_PLUS_PLUS),
        ("cs", CodeLanguageLabel.C_SHARP),
        ("ts", CodeLanguageLabel.TYPESCRIPT),
        ("yml", CodeLanguageLabel.YAML),
        ("sh", CodeLanguageLabel.BASH),
        (None, CodeLanguageLabel.UNKNOWN),
        ("", CodeLanguageLabel.UNKNOWN),
        ("not-a-language", CodeLanguageLabel.UNKNOWN),
    ],
)
def test_normalize_hint(hint, expected):
    assert normalize_code_language(hint) == expected


@pytest.mark.parametrize(
    "text, hint, expected",
    [
        # A valid hint overrides content-based detection.
        (PYTHON_FUNC, "javascript", CodeLanguageLabel.JAVASCRIPT),
        # An unrecognized hint falls back to content detection.
        (PYTHON_FUNC, "not-a-language", CodeLanguageLabel.PYTHON),
    ],
)
def test_detect_with_hint(text, hint, expected):
    assert detect_code_language(text, hint=hint) == expected
