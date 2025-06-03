from pathlib import Path

from pytest import warns

from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult, DoclingDocument
from docling.document_converter import DocumentConverter

from .test_data_gen_flag import GEN_TEST_DATA
from .verify_utils import verify_document, verify_export

GENERATE = GEN_TEST_DATA


def get_csv_paths():
    # Define the directory you want to search
    directory = Path("./tests/data/csv/")

    # List all CSV files in the directory and its subdirectories
    return sorted(directory.rglob("*.csv"))


def get_csv_path(name: str):
    # Return the matching CSV file path
    return Path(f"./tests/data/csv/{name}.csv")


def get_converter():
    converter = DocumentConverter(allowed_formats=[InputFormat.CSV])

    return converter


def test_e2e_valid_csv_conversions():
    valid_csv_paths = get_csv_paths()
    converter = get_converter()

    for csv_path in valid_csv_paths:
        print(f"converting {csv_path}")

        gt_path = csv_path.parent.parent / "groundtruth" / "docling_v2" / csv_path.name
        if csv_path.stem in (
            "csv-too-few-columns",
            "csv-too-many-columns",
            "csv-inconsistent-header",
        ):
            with warns(UserWarning, match="Inconsistent column lengths"):
                conv_result: ConversionResult = converter.convert(csv_path)
        else:
            conv_result: ConversionResult = converter.convert(csv_path)

        doc: DoclingDocument = conv_result.document

        pred_md: str = doc.export_to_markdown()
        assert verify_export(pred_md, str(gt_path) + ".md"), "export to md"

        pred_itxt: str = doc._export_to_indented_text(
            max_text_len=70, explicit_tables=False
        )
        assert verify_export(pred_itxt, str(gt_path) + ".itxt"), (
            "export to indented-text"
        )

        assert verify_document(
            pred_doc=doc,
            gtfile=str(gt_path) + ".json",
            generate=GENERATE,
        ), "export to json"


def test_e2e_invalid_csv_conversions():
    csv_too_few_columns = get_csv_path("csv-too-few-columns")
    csv_too_many_columns = get_csv_path("csv-too-many-columns")
    csv_inconsistent_header = get_csv_path("csv-inconsistent-header")
    converter = get_converter()

    print(f"converting {csv_too_few_columns}")
    with warns(UserWarning, match="Inconsistent column lengths"):
        converter.convert(csv_too_few_columns)

    print(f"converting {csv_too_many_columns}")
    with warns(UserWarning, match="Inconsistent column lengths"):
        converter.convert(csv_too_many_columns)

    print(f"converting {csv_inconsistent_header}")
    with warns(UserWarning, match="Inconsistent column lengths"):
        converter.convert(csv_inconsistent_header)
