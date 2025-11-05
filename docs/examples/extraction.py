from __future__ import annotations

import argparse
from typing import Optional, Sequence

from pydantic import BaseModel, Field

from docling.datamodel.base_models import InputFormat
from docling.document_extractor import DocumentExtractor

DEFAULT_SOURCE = (
    "https://upload.wikimedia.org/wikipedia/commons/9/9f/Swiss_QR-Bill_example.jpg"
)


def build_extractor() -> DocumentExtractor:
    return DocumentExtractor(allowed_formats=[InputFormat.IMAGE, InputFormat.PDF])


def example_with_string_template(source: str = DEFAULT_SOURCE) -> None:
    extractor = build_extractor()
    result = extractor.extract(
        source=source,
        template='{"bill_no": "string", "total": "float"}',
    )
    print(result.pages)


def example_with_dict_template(source: str = DEFAULT_SOURCE) -> None:
    extractor = build_extractor()
    result = extractor.extract(
        source=source,
        template={
            "bill_no": "string",
            "total": "float",
        },
    )
    print(result.pages)


class Invoice(BaseModel):
    bill_no: str = Field(examples=["A123", "5414"])  # examples only
    total: float = Field(default=10, examples=[20])
    tax_id: Optional[str] = Field(default=None, examples=["1234567890"])


def example_with_pydantic_template(source: str = DEFAULT_SOURCE) -> None:
    extractor = build_extractor()

    # Using the model class directly
    result = extractor.extract(
        source=source,
        template=Invoice,
    )
    print(result.pages)

    # Using a model instance with defaults/overrides
    result = extractor.extract(
        source=source,
        template=Invoice(
            bill_no="41",
            total=100,
            tax_id="42",
        ),
    )
    print(result.pages)


class Contact(BaseModel):
    name: Optional[str] = Field(default=None, examples=["Smith"])
    address: str = Field(default="123 Main St", examples=["456 Elm St"])
    postal_code: str = Field(default="12345", examples=["67890"])
    city: str = Field(default="Anytown", examples=["Othertown"])
    country: Optional[str] = Field(default=None, examples=["Canada"])


class ExtendedInvoice(BaseModel):
    bill_no: str = Field(examples=["A123", "5414"])  # examples only
    total: float = Field(default=10, examples=[20])
    garden_work_hours: int = Field(default=1, examples=[2])
    sender: Contact = Field(default=Contact(), examples=[Contact()])
    receiver: Contact = Field(default=Contact(), examples=[Contact()])


def example_with_advanced_pydantic_template(source: str = DEFAULT_SOURCE) -> None:
    extractor = build_extractor()
    result = extractor.extract(
        source=source,
        template=ExtendedInvoice,
    )
    print(result.pages)

    if result.pages:
        invoice = ExtendedInvoice.model_validate(result.pages[0].extracted_data)
        print(invoice)
        print(
            f"Invoice #{invoice.bill_no} was sent by {invoice.sender.name} "
            f"to {invoice.receiver.name} at {invoice.sender.address}."
        )


def run_all_examples(source: str = DEFAULT_SOURCE) -> None:
    print("\n-- Example: string template --")
    example_with_string_template(source)

    print("\n-- Example: dict template --")
    example_with_dict_template(source)

    print("\n-- Example: Pydantic model --")
    example_with_pydantic_template(source)

    print("\n-- Example: Advanced Pydantic model --")
    example_with_advanced_pydantic_template(source)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Docling extraction examples")
    parser.add_argument(
        "--source",
        type=str,
        default=DEFAULT_SOURCE,
        help="Path or URL to the input document/image",
    )
    parser.add_argument(
        "--example",
        choices=["string", "dict", "pydantic", "advanced", "all"],
        default="all",
        help="Which example to run",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    match args.example:
        case "string":
            example_with_string_template(args.source)
        case "dict":
            example_with_dict_template(args.source)
        case "pydantic":
            example_with_pydantic_template(args.source)
        case "advanced":
            example_with_advanced_pydantic_template(args.source)
        case _:
            run_all_examples(args.source)


if __name__ == "__main__":
    main()
