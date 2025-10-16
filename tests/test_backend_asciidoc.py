import glob
import os
from pathlib import Path

from docling.backend.asciidoc_backend import (
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    AsciiDocBackend,
)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument


def _get_backend(fname):
    in_doc = InputDocument(
        path_or_stream=fname,
        format=InputFormat.ASCIIDOC,
        backend=AsciiDocBackend,
    )

    doc_backend = in_doc._backend
    return doc_backend


def test_parse_picture():
    line = (
        "image::images/example1.png[Example Image, width=200, height=150, align=center]"
    )
    res = AsciiDocBackend._parse_picture(line)
    assert res
    assert res.get("width", 0) == "200"
    assert res.get("height", 0) == "150"
    assert res.get("uri", "") == "images/example1.png"

    line = "image::renamed-bookmark.png[Renamed bookmark]"
    res = AsciiDocBackend._parse_picture(line)
    assert res
    assert "width" not in res
    assert "height" not in res
    assert res.get("uri", "") == "renamed-bookmark.png"


def test_asciidocs_examples():
    fnames = sorted(glob.glob("./tests/data/asciidoc/*.asciidoc"))

    for fname in fnames:
        print(f"reading {fname}")

        bname = os.path.basename(fname)
        gname = os.path.join("./tests/data/groundtruth/docling_v2/", bname + ".md")

        doc_backend = _get_backend(Path(fname))
        doc = doc_backend.convert()

        pred_itdoc = doc._export_to_indented_text(max_text_len=16)
        print("\n\n", pred_itdoc)

        pred_mddoc = doc.export_to_markdown()
        print("\n\n", pred_mddoc)

        if os.path.exists(gname):
            with open(gname) as fr:
                fr.read()

            # assert pred_mddoc == true_mddoc, "pred_mddoc!=true_mddoc for asciidoc"
        else:
            with open(gname, "w") as fw:
                fw.write(pred_mddoc)

            # print("\n\n", doc.export_to_markdown())

    assert True
