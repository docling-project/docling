from importlib import import_module
from typing import Any

from docling.cli._dependencies import missing_cli_dependency_error
from docling.cli.models import app as models_app

# Check for CLI dependencies
typer: Any = None
try:
    typer = import_module("typer")
except ImportError as e:
    missing_cli_dependency_error(e, cli_name="docling-tools", fallback_package="typer")

app = typer.Typer(
    name="Docling helpers",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)

app.add_typer(models_app, name="models")

click_app = typer.main.get_command(app)

if __name__ == "__main__":
    app()
