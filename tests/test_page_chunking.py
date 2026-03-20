import math
import pytest
from pypdf import PdfWriter
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import ThreadedPdfPipelineOptions
from docling.datamodel.settings import settings

from io import BytesIO
from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import (
    ConversionStatus,
)

def create_dummy_pdf(path: str, num_pages: int):
    """Helper to generate a blank PDF with N pages."""
    writer = PdfWriter()
    for _ in range(num_pages):
        writer.add_blank_page(width=72, height=72)
    with open(path, "wb") as f:
        writer.write(f)

def generate_dummy_pdf_bytes(num_pages: int) -> bytes:
    """Helper to generate a blank PDF entirely in memory."""
    writer = PdfWriter()
    for _ in range(num_pages):
        writer.add_blank_page(width=72, height=72)
    buf = BytesIO()
    writer.write(buf)
    return buf.getvalue()

@pytest.fixture(scope="module")
def document_matrix(tmp_path_factory):
    """Generates a matrix of local, memory stream, and mixed documents just once."""
    tmp_path = tmp_path_factory.mktemp("pdf_chunks")
    page_counts = [50, 75, 100, 120, 150] 
    
    # 1. Local Documents
    local_docs = []
    for i, pc in enumerate(page_counts):
        path = tmp_path / f"local_doc_{i}.pdf"
        create_dummy_pdf(path, pc)
        local_docs.append(path)
        
    # 2. Memory Stream Documents
    stream_docs = []
    for i, pc in enumerate(page_counts):
        pdf_bytes = generate_dummy_pdf_bytes(pc)
        stream_docs.append((f"stream_doc_{i}.pdf", pdf_bytes))
        
    # 3. Mixed Documents (3 local, 2 stream)
    mixed_docs = local_docs[:3] + stream_docs[3:]
    mixed_counts = page_counts[:3] + page_counts[3:]

    return {
        "local": (local_docs, page_counts),
        "stream": (stream_docs, page_counts),
        "mixed": (mixed_docs, mixed_counts)
    }


@pytest.mark.parametrize("doc_type", ["local", "stream", "mixed"])
@pytest.mark.parametrize("concurrency, batch_size, chunk_size, mode_name", [
    (15, 15, 50, "concurrent_chunked"),
    (15, 15, None, "concurrent_unchunked"),
    (1, 1, 50, "sequential_chunked"),
    (1, 1, None, "sequential_unchunked")
])
def test_core_chunking_scenarios(document_matrix, doc_type, concurrency, batch_size, chunk_size, mode_name):
    """
    Tests the 3x4 matrix for page chunking:
    - 3 input types: Local, Memory Stream, Mixed
    - 4 execution modes: Concurrent/Sequential x Chunked/Unchunked
    """
    input_docs_raw, page_counts = document_matrix[doc_type]
    
    # Reconstruct input documents for each test to avoid closed file issues
    input_docs = []
    for item in input_docs_raw:
        if isinstance(item, tuple):
            input_docs.append(DocumentStream(name=item[0], stream=BytesIO(item[1])))
        else:
            input_docs.append(item)
            
    # 1. Configure Concurrency & Batching
    settings.perf.doc_batch_concurrency = concurrency
    settings.perf.doc_batch_size = batch_size
    
    # 2. Configure Pipeline
    pipeline_options = ThreadedPdfPipelineOptions()
    pipeline_options.page_chunk_size = chunk_size
    pipeline_options.do_ocr = False  # Speed up tests
    
    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)},
    )
    
    # 3. Execute Conversion
    results = list(converter.convert_all(input_docs, raises_on_error=True))
    
    # 4. Calculate Expected Number of Chunks
    if chunk_size:
        expected_chunks = sum(math.ceil(pc / chunk_size) for pc in page_counts)
    else:
        expected_chunks = len(input_docs)
        
    # 5. Assertions
    assert len(results) == expected_chunks, f"Expected {expected_chunks} chunks for {mode_name}, got {len(results)}"
    
    for res in results:
        assert res.status in [ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS]
        start, end = res.input.limits.page_range
        
        if chunk_size:
            assert (end - start + 1) <= chunk_size, f"Chunk size exceeded limit of {chunk_size}"


@pytest.mark.parametrize(
    "total_pages, chunk_size, batch_size, page_range, expected_chunk_ranges",
    [
        (10, 5, 20, (1, 10), [(1, 5), (6, 10)]),           # Batch > Chunk
        (12, 5, 2, (1, 12), [(1, 5), (6, 10), (11, 12)]),  # Batch < Chunk
        (3, 1, 4, (1, 3), [(1, 1), (2, 2), (3, 3)]),       # Chunk size = 1
        (6, 4, 1, (1, 6), [(1, 4), (5, 6)]),               # Batch size = 1
        (10, 5, 5, (1, 10), [(1, 5), (6, 10)]),            # Chunk == Batch
        (7, 3, 2, (1, 7), [(1, 3), (4, 6), (7, 7)]),       # Unaligned chunk size
        (3, 10, 4, (1, 3), [(1, 3)]),                      # Total < Chunk size
        (10, 2, 4, (3, 8), [(3, 4), (5, 6), (7, 8)]),      # Custom page ranges
    ],
)
def test_page_chunking_edge_cases(
    tmp_path, total_pages, chunk_size, batch_size, page_range, expected_chunk_ranges
):
    pdf_path = tmp_path / f"dummy_{total_pages}.pdf"
    create_dummy_pdf(pdf_path, total_pages)

    pipeline_options = ThreadedPdfPipelineOptions()
    pipeline_options.page_chunk_size = chunk_size
    pipeline_options.layout_batch_size = batch_size
    pipeline_options.ocr_batch_size = batch_size
    pipeline_options.table_batch_size = batch_size
    pipeline_options.do_ocr = False

    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)},
    )

    results = list(converter.convert_all([pdf_path], page_range=page_range))

    assert len(results) == len(expected_chunk_ranges)
    for result, expected_range in zip(results, expected_chunk_ranges):
        assert result.input.limits.page_range == expected_range
        assert result.status in [ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS]


def test_invalid_file_skips(tmp_path):
    valid_pdf = tmp_path / "valid.pdf"
    create_dummy_pdf(valid_pdf, 4)

    invalid_pdf = tmp_path / "invalid.pdf" # doesn't exist (n/a)
    invalid_pdf.write_text("This is corrupted gibberish, not a real PDF.")

    pipeline_options = ThreadedPdfPipelineOptions()
    pipeline_options.page_chunk_size = 2
    pipeline_options.do_ocr = False
    
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

    results = list(converter.convert_all([invalid_pdf, valid_pdf], raises_on_error=False))

    assert len(results) == 3  # 1 failure result + 2 valid chunks
    assert results[0].status in [ConversionStatus.FAILURE, ConversionStatus.SKIPPED]
    assert results[1].input.limits.page_range == (1, 2)
    assert results[2].input.limits.page_range == (3, 4)