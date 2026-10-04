# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import os


def pytest_addoption(parser) -> None:
    parser.addoption(
        "--update-groundtruth",
        action="store_true",
        default=False,
        help="Rewrite checked-in ground-truth fixtures.",
    )


def pytest_configure(config) -> None:
    if config.getoption("--update-groundtruth"):
        os.environ["DOCLING_GEN_TEST_DATA"] = "1"
