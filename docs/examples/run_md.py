import json
import logging
import os
from pathlib import Path

import yaml

from docling.backend.md_backend import MarkdownDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument

_log = logging.getLogger(__name__)


def main():
    input_paths = [Path("README.md")]

    for path in input_paths:
        in_doc = InputDocument(
            path_or_stream=path,
            format=InputFormat.PDF,
            backend=MarkdownDocumentBackend,
        )
        mdb = MarkdownDocumentBackend(in_doc=in_doc, path_or_stream=path)
        document = mdb.convert()

        out_path = Path("scratch")
        print(f"Document {path} converted.\nSaved markdown output to: {out_path!s}")

        # Export Docling document format to markdowndoc:
        fn = os.path.basename(path)

        with (out_path / f"{fn}.md").open("w") as fp:
            fp.write(document.export_to_markdown())

        with (out_path / f"{fn}.json").open("w") as fp:
            fp.write(json.dumps(document.export_to_dict()))

        with (out_path / f"{fn}.yaml").open("w") as fp:
            fp.write(yaml.safe_dump(document.export_to_dict()))


if __name__ == "__main__":
    main()
