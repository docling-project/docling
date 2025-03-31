import typer

from docling.cli.models import app as models_app

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
