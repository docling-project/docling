import sys
from typing import NoReturn


def missing_cli_dependency_error(
    error: ImportError,
    *,
    cli_name: str,
    fallback_package: str = "typer or rich",
) -> NoReturn:
    missing_package = (
        str(error).split("'")[1] if "'" in str(error) else fallback_package
    )
    print(
        f"Error: Missing required CLI dependency '{missing_package}'", file=sys.stderr
    )
    print(f"\nThe {cli_name} CLI requires additional dependencies.", file=sys.stderr)
    print("Please install them using one of the following options:\n", file=sys.stderr)
    print("  1. Install the full docling package (recommended):", file=sys.stderr)
    print("     pip install docling\n", file=sys.stderr)
    print("  2. Install docling-slim with CLI support:", file=sys.stderr)
    print("     pip install docling-slim[cli]\n", file=sys.stderr)
    print("  3. Install just the missing dependencies:", file=sys.stderr)
    print("     pip install typer rich\n", file=sys.stderr)
    sys.exit(1)
