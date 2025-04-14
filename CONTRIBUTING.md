## Contributing In General
Our project welcomes external contributions. If you have an itch, please feel
free to scratch it.

For more details on the contributing guidelines head to the Docling Project [community repository](https://github.com/docling-project/community).

## Developing

### Usage of Poetry

We use Poetry to manage dependencies.

#### Installation

To install Poetry, follow the documentation here: https://python-poetry.org/docs/master/#installing-with-the-official-installer

1. Install Poetry globally on your machine:
    ```bash
    curl -sSL https://install.python-poetry.org | python3 -
    ```
    The installation script will print the installation bin folder `POETRY_BIN` which you need in the next steps.

2. Make sure Poetry is in your `$PATH`:
    - for `zsh`:
        ```sh
        echo 'export PATH="POETRY_BIN:$PATH"' >> ~/.zshrc
        ```
    - for `bash`:
        ```sh
        echo 'export PATH="POETRY_BIN:$PATH"' >> ~/.bashrc
        ```

3. The official guidelines linked above include useful details on configuring autocomplete for most shell environments, e.g., Bash and Zsh.

#### Create a Virtual Environment and Install Dependencies

To activate the Virtual Environment, run:

```bash
poetry shell
```

This will spawn a shell with the Virtual Environment activated. If the Virtual Environment doesn't exist, Poetry will create one for you. Then, to install dependencies, run:

```bash
poetry install
```

**(Advanced) Use a Specific Python Version**

If you need to work with a specific (older) version of Python, run:

```bash
poetry env use $(which python3.8)
```

This creates a Virtual Environment with Python 3.8. For other versions, replace `$(which python3.8)` with the path to the interpreter (e.g., `/usr/bin/python3.8`) or use `$(which pythonX.Y)`.

#### Add a New Dependency

```bash
poetry add NAME
```

## Coding Style Guidelines

We use the following tools to enforce code style:

- iSort, to sort imports
- Black, to format code

We run a series of checks on the codebase on every commit using `pre-commit`. To install the hooks, run:

```bash
pre-commit install
```

To run the checks on-demand, run:

```bash
pre-commit run --all-files
```

Note: Checks like `Black` and `isort` will "fail" if they modify files. This is because `pre-commit` doesn't like to see files modified by its hooks. In these cases, `git add` the modified files and `git commit` again.

## Tests

When submitting a new feature or fix, please consider adding a short test for it.

### Reference test documents

When a change improves the conversion results, multiple reference documents must be regenerated and reviewed.

The reference data can be regenerated with

```sh
DOCLING_GEN_TEST_DATA=1 poetry run pytest
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