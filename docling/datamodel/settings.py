import sys
from pathlib import Path
from typing import Annotated, Optional, Tuple

from pydantic import BaseModel, PlainValidator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _validate_page_range(v: Tuple[int, int]) -> Tuple[int, int]:
    if v[0] < 1 or v[1] < v[0]:
        raise ValueError(
            "Invalid page range: start must be ≥ 1 and end must be ≥ start."
        )
    return v


PageRange = Annotated[Tuple[int, int], PlainValidator(_validate_page_range)]

DEFAULT_PAGE_RANGE: PageRange = (1, sys.maxsize)


class DocumentLimits(BaseModel):
    max_num_pages: int = sys.maxsize
    max_file_size: int = sys.maxsize
    page_range: PageRange = DEFAULT_PAGE_RANGE


class BatchConcurrencySettings(BaseModel):
    doc_batch_size: int = 2
    doc_batch_concurrency: int = 2
    page_batch_size: int = 4
    page_batch_concurrency: int = 2
    elements_batch_size: int = 16

    # doc_batch_size: int = 1
    # doc_batch_concurrency: int = 1
    # page_batch_size: int = 1
    # page_batch_concurrency: int = 1

    # model_concurrency: int = 2

    # To force models into single core: export OMP_NUM_THREADS=1


class DebugSettings(BaseModel):
    visualize_cells: bool = False
    visualize_ocr: bool = False
    visualize_layout: bool = False
    visualize_raw_layout: bool = False
    visualize_tables: bool = False

    profile_pipeline_timings: bool = False

    # Path used to output debug information.
    debug_output_path: str = str(Path.cwd() / "debug")


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DOCLING_", env_nested_delimiter="_", env_nested_max_split=1
    )

    perf: BatchConcurrencySettings = BatchConcurrencySettings()
    debug: DebugSettings = DebugSettings()

    cache_dir: Path = Path.home() / ".cache" / "docling"
    artifacts_path: Optional[Path] = None


settings = AppSettings()
