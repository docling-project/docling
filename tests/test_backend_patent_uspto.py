"""Test methods in module docling.backend.patent_uspto_backend.py."""

import logging
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest
from docling_core.types import DoclingDocument
from docling_core.types.doc import DocItemLabel, TableData, TextItem

from docling.backend.xml.uspto_backend import PatentUsptoDocumentBackend, XmlTable
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument

from .test_data_gen_flag import GEN_TEST_DATA
from .verify_utils import verify_document

GENERATE: bool = GEN_TEST_DATA
DATA_PATH: Path = Path("./tests/data/uspto/")
GT_PATH: Path = Path("./tests/data/groundtruth/docling_v2/")


def _generate_groundtruth(doc: DoclingDocument, file_stem: str) -> None:
    with open(GT_PATH / f"{file_stem}.itxt", "w", encoding="utf-8") as file_obj:
        file_obj.write(doc._export_to_indented_text())
    doc.save_as_json(GT_PATH / f"{file_stem}.json")
    doc.save_as_markdown(GT_PATH / f"{file_stem}.md")


@pytest.fixture(scope="module")
def patents() -> list[tuple[Path, DoclingDocument]]:
    patent_paths = (
        sorted(DATA_PATH.glob("ip*.xml"))
        + sorted(DATA_PATH.glob("pg*.xml"))
        + sorted(DATA_PATH.glob("pa*.xml"))
        + sorted(DATA_PATH.glob("pftaps*.txt"))
    )
    patents: list[dict[Path, DoclingDocument]] = []
    for in_path in patent_paths:
        in_doc = InputDocument(
            path_or_stream=in_path,
            format=InputFormat.XML_USPTO,
            backend=PatentUsptoDocumentBackend,
        )
        backend = PatentUsptoDocumentBackend(in_doc=in_doc, path_or_stream=in_path)
        logging.info(f"Converting patent from file {in_path}")
        doc = backend.convert()
        assert doc, f"Failed to parse document {in_path}"
        patents.append((in_path, doc))

    return patents


@pytest.fixture(scope="module")
def groundtruth() -> list[tuple[Path, str]]:
    patent_paths = (
        sorted(GT_PATH.glob("ip*"))
        + sorted(GT_PATH.glob("pg*"))
        + sorted(GT_PATH.glob("pa*"))
        + sorted(GT_PATH.glob("pftaps*"))
    )
    groundtruth: list[tuple[Path, str]] = []
    for in_path in patent_paths:
        with open(in_path, encoding="utf-8") as file_obj:
            content = file_obj.read()
            groundtruth.append((in_path, content))

    return groundtruth


@pytest.fixture(scope="module")
def tables() -> list[tuple[Path, TableData]]:
    table_paths = sorted(DATA_PATH.glob("tables*.xml"))
    tables: list[tuple[Path, TableData]] = []
    for in_path in table_paths:
        with open(in_path, encoding="utf-8") as file_obj:
            content = file_obj.read()
            parser = XmlTable(content)
            parsed_table = parser.parse()
            assert parsed_table
            tables.append((in_path, parsed_table))

    return tables


@pytest.mark.skip("Slow test")
def test_patent_export(patents):
    for _, doc in patents:
        with NamedTemporaryFile(suffix=".yaml", delete=False) as tmp_file:
            doc.save_as_yaml(Path(tmp_file.name))
            assert os.path.getsize(tmp_file.name) > 0

        with NamedTemporaryFile(suffix=".html", delete=False) as tmp_file:
            doc.save_as_html(Path(tmp_file.name))
            assert os.path.getsize(tmp_file.name) > 0

        with NamedTemporaryFile(suffix=".md", delete=False) as tmp_file:
            doc.save_as_markdown(Path(tmp_file.name))
            assert os.path.getsize(tmp_file.name) > 0


def test_patent_groundtruth(patents, groundtruth):
    gt_stems: list[str] = [item[0].stem for item in groundtruth]
    gt_names: dict[str, str] = {item[0].name: item[1] for item in groundtruth}
    for path, doc in patents:
        if path.stem not in gt_stems:
            continue
        md_name = path.stem + ".md"
        if md_name in gt_names:
            pred_md = doc.export_to_markdown()
            assert pred_md == gt_names[md_name], (
                f"Markdown file mismatch against groundtruth {md_name}"
            )
        json_path = path.with_suffix(".json")
        if json_path.stem in gt_names:
            assert verify_document(doc, str(json_path), GENERATE), (
                f"JSON file mismatch against groundtruth {json_path}"
            )
        itxt_name = path.stem + ".itxt"
        if itxt_name in gt_names:
            pred_itxt = doc._export_to_indented_text()
            assert pred_itxt == gt_names[itxt_name], (
                f"Indented text file mismatch against groundtruth {itxt_name}"
            )


def test_tables(tables):
    """Test the table parser."""
    # CHECK table in file tables_20180000016.xml
    file_name = "tables_ipa20180000016.xml"
    file_table = next(item[1] for item in tables if item[0].name == file_name)
    assert file_table.num_rows == 13
    assert file_table.num_cols == 10
    assert len(file_table.table_cells) == 130


def test_patent_uspto_ice(patents):
    """Test applications and grants Full Text Data/XML Version 4.x ICE."""

    # CHECK application doc number 20200022300
    file_name = "ipa20200022300.xml"
    doc = next(item[1] for item in patents if item[0].name == file_name)
    if GENERATE:
        _generate_groundtruth(doc, Path(file_name).stem)

    assert doc.name == file_name
    texts = doc.texts
    assert len(texts) == 78
    assert isinstance(texts[0], TextItem)
    assert (
        texts[0].text
        == "SYSTEM FOR CONTROLLING THE OPERATION OF AN ACTUATOR MOUNTED ON A SEED PLANTING IMPLEMENT"
    )
    assert texts[0].label == DocItemLabel.TITLE
    assert texts[0].parent.cref == "#/body"
    assert isinstance(texts[1], TextItem)
    assert texts[1].text == "ABSTRACT"
    assert texts[1].label == DocItemLabel.SECTION_HEADER
    assert texts[1].parent.cref == "#/texts/0"
    assert isinstance(texts[2], TextItem)
    assert texts[2].text == (
        "In one aspect, a system for controlling an operation of an actuator mounted "
        "on a seed planting implement may include an actuator configured to adjust a "
        "position of a row unit of the seed planting implement relative to a toolbar "
        "of the seed planting implement. The system may also include a flow restrictor"
        " fluidly coupled to a fluid chamber of the actuator, with the flow restrictor"
        " being configured to reduce a rate at which fluid is permitted to exit the "
        "fluid chamber in a manner that provides damping to the row unit. Furthermore,"
        " the system may include a valve fluidly coupled to the flow restrictor in a "
        "parallel relationship such that the valve is configured to permit the fluid "
        "exiting the fluid chamber to flow through the flow restrictor and the fluid "
        "entering the fluid chamber to bypass the flow restrictor."
    )
    assert texts[2].label == DocItemLabel.PARAGRAPH
    assert texts[2].parent.cref == "#/texts/1"
    assert isinstance(texts[3], TextItem)
    assert texts[3].text == "FIELD"
    assert texts[3].label == DocItemLabel.SECTION_HEADER
    assert texts[3].parent.cref == "#/texts/0"
    assert isinstance(texts[4], TextItem)
    assert texts[4].text == (
        "The present disclosure generally relates to seed planting implements and, "
        "more particularly, to systems for controlling the operation of an actuator "
        "mounted on a seed planting implement in a manner that provides damping to "
        "one or more components of the seed planting implement."
    )
    assert texts[4].label == DocItemLabel.PARAGRAPH
    assert texts[4].parent.cref == "#/texts/3"
    assert isinstance(texts[5], TextItem)
    assert texts[5].text == "BACKGROUND"
    assert texts[5].label == DocItemLabel.SECTION_HEADER
    assert texts[5].parent.cref == "#/texts/0"
    assert isinstance(texts[6], TextItem)
    assert texts[6].text == (
        "Modern farming practices strive to increase yields of agricultural fields. In"
        " this respect, seed planting implements are towed behind a tractor or other "
        "work vehicle to deposit seeds in a field. For example, seed planting "
        "implements typically include one or more ground engaging tools or openers "
        "that form a furrow or trench in the soil. One or more dispensing devices of "
        "the seed planting implement may, in turn, deposit seeds into the furrow(s). "
        "After deposition of the seeds, a packer wheel may pack the soil on top of the"
        " deposited seeds."
    )
    assert texts[6].label == DocItemLabel.PARAGRAPH
    assert texts[6].parent.cref == "#/texts/5"
    assert isinstance(texts[7], TextItem)
    assert texts[7].text == (
        "In certain instances, the packer wheel may also control the penetration depth"
        " of the furrow. In this regard, the position of the packer wheel may be moved"
        " vertically relative to the associated opener(s) to adjust the depth of the "
        "furrow. Additionally, the seed planting implement includes an actuator "
        "configured to exert a downward force on the opener(s) to ensure that the "
        "opener(s) is able to penetrate the soil to the depth set by the packer wheel."
        " However, the seed planting implement may bounce or chatter when traveling at"
        " high speeds and/or when the opener(s) encounters hard or compacted soil. As "
        "such, operators generally operate the seed planting implement with the "
        "actuator exerting more downward force on the opener(s) than is necessary in "
        "order to prevent such bouncing or chatter. Operation of the seed planting "
        "implement with excessive down pressure applied to the opener(s), however, "
        "reduces the overall stability of the seed planting implement."
    )
    assert texts[7].label == DocItemLabel.PARAGRAPH
    assert texts[7].parent.cref == "#/texts/5"
    assert isinstance(texts[8], TextItem)
    assert texts[8].text == (
        "Accordingly, an improved system for controlling the operation of an actuator "
        "mounted on s seed planting implement to enhance the overall operation of the "
        "implement would be welcomed in the technology."
    )
    assert texts[8].label == DocItemLabel.PARAGRAPH
    assert texts[8].parent.cref == "#/texts/5"
    assert isinstance(texts[9], TextItem)
    assert texts[9].text == "BRIEF DESCRIPTION"
    assert texts[9].label == DocItemLabel.SECTION_HEADER
    assert texts[9].parent.cref == "#/texts/0"
    assert isinstance(texts[15], TextItem)
    assert texts[15].text == "BRIEF DESCRIPTION OF THE DRAWINGS"
    assert texts[15].label == DocItemLabel.SECTION_HEADER
    assert texts[15].parent.cref == "#/texts/0"
    assert isinstance(texts[17], TextItem)
    assert texts[17].text == (
        "FIG. 1 illustrates a perspective view of one embodiment of a seed planting "
        "implement in accordance with aspects of the present subject matter;"
    )
    assert texts[17].label == DocItemLabel.PARAGRAPH
    assert texts[17].parent.cref == "#/texts/15"
    assert isinstance(texts[27], TextItem)
    assert texts[27].text == "DETAILED DESCRIPTION"
    assert texts[27].label == DocItemLabel.SECTION_HEADER
    assert texts[27].parent.cref == "#/texts/0"
    assert isinstance(texts[57], TextItem)
    assert texts[57].text == (
        "This written description uses examples to disclose the technology, including "
        "the best mode, and also to enable any person skilled in the art to practice "
        "the technology, including making and using any devices or systems and "
        "performing any incorporated methods. The patentable scope of the technology "
        "is defined by the claims, and may include other examples that occur to those "
        "skilled in the art. Such other examples are intended to be within the scope "
        "of the claims if they include structural elements that do not differ from the"
        " literal language of the claims, or if they include equivalent structural "
        "elements with insubstantial differences from the literal language of the "
        "claims."
    )
    assert texts[57].label == DocItemLabel.PARAGRAPH
    assert texts[57].parent.cref == "#/texts/27"
    assert isinstance(texts[58], TextItem)
    assert texts[58].text == "CLAIMS"
    assert texts[58].label == DocItemLabel.SECTION_HEADER
    assert texts[58].parent.cref == "#/texts/0"
    assert isinstance(texts[77], TextItem)
    assert texts[77].text == (
        "19. The system of claim 18, wherein the flow restrictor and the valve are "
        "fluidly coupled in a parallel relationship."
    )
    assert texts[77].label == DocItemLabel.PARAGRAPH
    assert texts[77].parent.cref == "#/texts/58"

    # CHECK application doc number 20180000016 for HTML entities, level 2 headings, tables
    file_name = "ipa20180000016.xml"
    doc = next(item[1] for item in patents if item[0].name == file_name)
    if GENERATE:
        _generate_groundtruth(doc, Path(file_name).stem)

    assert doc.name == file_name
    texts = doc.texts
    assert len(texts) == 183
    assert isinstance(texts[0], TextItem)
    assert texts[0].text == "LIGHT EMITTING DEVICE AND PLANT CULTIVATION METHOD"
    assert texts[0].label == DocItemLabel.TITLE
    assert texts[0].parent.cref == "#/body"
    assert isinstance(texts[1], TextItem)
    assert texts[1].text == "ABSTRACT"
    assert texts[1].label == DocItemLabel.SECTION_HEADER
    assert texts[1].parent.cref == "#/texts/0"
    assert isinstance(texts[2], TextItem)
    assert texts[2].text == (
        "Provided is a light emitting device that includes a light emitting element "
        "having a light emission peak wavelength ranging from 380 nm to 490 nm, and a "
        "fluorescent material excited by light from the light emitting element and "
        "emitting light having at a light emission peak wavelength ranging from 580 nm"
        " or more to less than 680 nm. The light emitting device emits light having a "
        "ratio R/B of a photon flux density R to a photon flux density B ranging from "
        "2.0 to 4.0 and a ratio R/FR of the photon flux density R to a photon flux "
        "density FR ranging from 0.7 to 13.0, the photon flux density R being in a "
        "wavelength range of 620 nm or more and less than 700 nm, the photon flux "
        "density B being in a wavelength range of 380 nm or more and 490 nm or less, "
        "and the photon flux density FR being in a wavelength range of 700 nm or more "
        "and 780 nm or less."
    )
    assert isinstance(texts[3], TextItem)
    assert texts[3].text == "CROSS-REFERENCE TO RELATED APPLICATION"
    assert texts[3].label == DocItemLabel.SECTION_HEADER
    assert texts[3].parent.cref == "#/texts/0"
    assert isinstance(texts[4], TextItem)
    assert texts[5].text == "BACKGROUND"
    assert texts[5].label == DocItemLabel.SECTION_HEADER
    assert texts[5].parent.cref == "#/texts/0"
    assert isinstance(texts[6], TextItem)
    assert texts[6].text == "Technical Field"
    assert texts[6].label == DocItemLabel.SECTION_HEADER
    assert texts[6].parent.cref == "#/texts/0"
    assert isinstance(texts[7], TextItem)
    assert texts[7].text == (
        "The present disclosure relates to a light emitting device and a plant "
        "cultivation method."
    )
    assert texts[7].label == DocItemLabel.PARAGRAPH
    assert texts[7].parent.cref == "#/texts/6"
    assert isinstance(texts[8], TextItem)
    assert texts[8].text == "Description of Related Art"
    assert texts[8].label == DocItemLabel.SECTION_HEADER
    assert texts[8].parent.cref == "#/texts/0"
    assert isinstance(texts[63], TextItem)
    assert texts[63].text == (
        "wherein r, s, and t are numbers satisfying 0≦r≦1.0, 0≦s≦1.0, 0<t<1.0, and "
        "r+s+t≦1.0."
    )
    assert texts[63].label == DocItemLabel.PARAGRAPH
    assert texts[63].parent.cref == "#/texts/51"
    assert isinstance(texts[89], TextItem)
    assert texts[89].text == (
        "Examples of the compound containing Al, Ga, or In specifically include Al₂O₃, "
        "Ga₂O₃, and In₂O₃."
    )
    assert texts[89].label == DocItemLabel.PARAGRAPH
    assert texts[89].parent.cref == "#/texts/87"

    # CHECK application doc number 20110039701 for complex long tables
    file_name = "ipa20110039701.xml"
    doc = next(item[1] for item in patents if item[0].name == file_name)
    assert doc.name == file_name
    assert len(doc.tables) == 17


def test_patent_uspto_grant_v2(patents):
    """Test applications and grants Full Text Data/APS."""

    # CHECK application doc number 06442728
    file_name = "pg06442728.xml"
    doc = next(item[1] for item in patents if item[0].name == file_name)
    if GENERATE:
        _generate_groundtruth(doc, Path(file_name).stem)

    assert doc.name == file_name
    texts = doc.texts
    assert len(texts) == 108
    assert isinstance(texts[0], TextItem)
    assert texts[0].text == "Methods and apparatus for turbo code"
    assert texts[0].label == DocItemLabel.TITLE
    assert texts[0].parent.cref == "#/body"
    assert isinstance(texts[1], TextItem)
    assert texts[1].text == "ABSTRACT"
    assert texts[1].label == DocItemLabel.SECTION_HEADER
    assert texts[1].parent.cref == "#/texts/0"
    assert isinstance(texts[2], TextItem)
    assert texts[2].text == (
        "An interleaver receives incoming data frames of size N. The interleaver "
        "indexes the elements of the frame with an N₁×N₂ index array. The interleaver "  # noqa: RUF001
        "then effectively rearranges (permutes) the data by permuting the rows of the "
        "index array. The interleaver employs the equation I(j,k)=I(j,αjk+βj)modP) to "  # noqa: RUF001
        "permute the columns (indexed by k) of each row (indexed by j). P is at least "
        "equal to N₂, βj is a constant which may be different for each row, and each "
        "αj is a relative prime number relative to P. After permuting, the "  # noqa: RUF001
        "interleaver outputs the data in a different order than received (e.g., "
        "receives sequentially row by row, outputs sequentially each column by column)."
    )
    # check that the formula has been skipped
    assert texts[43].text == (
        "Calculating the specified equation with the specified values for permuting "
        "row 0 of array D 350 into row 0 of array D₁ 360 proceeds as:"
    )
    assert texts[44].text == (
        "and the permuted data frame is contained in array D₁ 360 shown in FIG. 3. "
        "Outputting the array column by column outputs the frame elements in the "
        "order:"
    )


def test_patent_uspto_app_v1(patents):
    """Test applications Full Text Data/XML Version 1.x."""

    # CHECK application doc number 20010031492
    file_name = "pa20010031492.xml"
    doc = next(item[1] for item in patents if item[0].name == file_name)
    if GENERATE:
        _generate_groundtruth(doc, Path(file_name).stem)

    assert doc.name == file_name
    texts = doc.texts
    assert len(texts) == 103
    assert isinstance(texts[0], TextItem)
    assert texts[0].text == "Assay reagent"
    assert texts[0].label == DocItemLabel.TITLE
    assert texts[0].parent.cref == "#/body"
    assert isinstance(texts[1], TextItem)
    assert texts[1].text == "ABSTRACT"
    assert texts[1].label == DocItemLabel.SECTION_HEADER
    assert texts[1].parent.cref == "#/texts/0"
    # check that the formula has been skipped
    assert texts[62].text == (
        "5. The % toxic effect for each sample was calculated as follows:"
    )
    assert texts[63].text == "where: Cₒ=light in control at time zero"
    assert len(doc.tables) == 1
    assert doc.tables[0].data.num_rows == 6
    assert doc.tables[0].data.num_cols == 3


def test_patent_uspto_grant_aps(patents):
    """Test applications Full Text Data/APS."""

    # CHECK application doc number 057006474
    file_name = "pftaps057006474.txt"
    doc = next(item[1] for item in patents if item[0].name == file_name)
    if GENERATE:
        _generate_groundtruth(doc, Path(file_name).stem)

    assert doc.name == file_name
    texts = doc.texts
    assert len(texts) == 75
    assert isinstance(texts[0], TextItem)
    assert texts[0].text == "Carbocation containing cyanine-type dye"
    assert texts[0].label == DocItemLabel.TITLE
    assert texts[0].parent.cref == "#/body"
    assert isinstance(texts[1], TextItem)
    assert texts[1].text == "ABSTRACT"
    assert texts[1].label == DocItemLabel.SECTION_HEADER
    assert texts[1].parent.cref == "#/texts/0"
    assert isinstance(texts[2], TextItem)
    assert texts[2].text == (
        "To provide a reagent with excellent stability under storage, which can detect"
        " a subject compound to be measured with higher specificity and sensitibity. "
        "Complexes of a compound represented by the general formula (IV):"
    )
    assert len(doc.tables) == 0
    for item in texts:
        assert "##STR1##" not in item.text
