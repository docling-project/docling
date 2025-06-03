## Contributing In General
Our project welcomes external contributions. If you have an itch, please feel
free to scratch it.

For more details on the contributing guidelines head to the Docling Project [community repository](https://github.com/docling-project/community).

## Developing

### Usage of uv

We use [uv](https://docs.astral.sh/uv/) as package and project manager.

#### Installation

To install `uv`, check the documentation on [Installing uv](https://docs.astral.sh/uv/getting-started/installation/).

#### Create an environment and sync it

You can use the `uv sync` to create a project virtual environment (if it does not already exist) and sync
the project's dependencies with the environment.

```bash
uv sync
```

#### Use a specific Python version (optional)

If you need to work with a specific version of Python, you can create a new virtual environment for that version
and run the sync command:

```bash
uv venv --python 3.12
uv sync
```

More detailed options are described on the [Using Python environments](https://docs.astral.sh/uv/pip/environments/) documentation.

#### Add a new dependency

Simply use the `uv add` command. The `pyproject.toml` and `uv.lock` files will be updated.

```bash
uv add [OPTIONS] <PACKAGES|--requirements <REQUIREMENTS>>
```

## Coding Style Guidelines

We use the following tools to enforce code style:

- [Ruff](https://docs.astral.sh/ruff/), as linter and code formatter
- [MyPy](https://mypy.readthedocs.io), as static type checker

A set of styling checks, as well as regression tests, are defined and managed through the [pre-commit](https://pre-commit.com/) framework.
To ensure that those scripts run automatically before a commit is finalized, install `pre-commit` on your local repository:

```bash
pre-commit install
```

To run the checks on-demand, run:

```bash
pre-commit run --all-files
```

Note: Checks like `Ruff` will "fail" if they modify files. This is because `pre-commit` doesn't like to see files modified by its hooks. In these cases, `git add` the modified files and `git commit` again.

## Tests

When submitting a new feature or fix, please consider adding a short test for it.

### Reference test documents

When a change improves the conversion results, multiple reference documents must be regenerated and reviewed.

The reference data can be regenerated with

```sh
DOCLING_GEN_TEST_DATA=1 uv run pytest
```

All PRs modifying the reference test data require a double review to guarantee we don't miss edge cases.


## Documentation

We use [MkDocs](https://www.mkdocs.org/) to write documentation.

To run the documentation server, run:

```bash
mkdocs serve
```

The server will be available at [http://localhost:8000](http://localhost:8000).

### Pushing Documentation to GitHub Pages

Run the following:

```bash
mkdocs gh-deploy
```