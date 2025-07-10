import os
import tempfile
from pathlib import Path
from io import BytesIO

import pypdfium2 as pdfium
from PIL import Image, ImageDraw, ImageFont

from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument


def create_test_multipage_image(format_name, num_pages=3):
    """Create a test multi-page image file."""
    images = []
    
    for i in range(num_pages):
        # Create a unique image for each page
        img = Image.new('RGB', (200, 150), color='white')
        draw = ImageDraw.Draw(img)
        
        # Draw page identifier
        text = f"Page {i+1}"
        draw.text((10, 10), text, fill='black')
        
        # Draw some unique content for each page
        colors = ['red', 'green', 'blue', 'orange', 'purple']
        color = colors[i % len(colors)]
        draw.rectangle([10, 30, 190, 140], outline=color, width=2)
        
        images.append(img)
    
    # Save as multi-page image
    suffix = '.tif' if format_name == 'TIFF' else f'.{format_name.lower()}'
    temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    temp_file.close()
    
    try:
        if images:
            images[0].save(temp_file.name, format_name, save_all=True, append_images=images[1:])
        return temp_file.name
    except Exception as e:
        # Clean up on error
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
        raise e


def test_multipage_image_support():
    """Test that multi-page image files are properly handled."""
    
    # Test with different image formats and numbers of pages
    test_cases = [
        ('TIFF', 1, "single page TIFF"),
        ('TIFF', 2, "two-page TIFF"),
        ('TIFF', 3, "three-page TIFF"),
        ('GIF', 2, "two-page GIF"),
        ('WEBP', 2, "two-page WEBP"),
    ]
    
    for format_name, num_pages, description in test_cases:
        print(f"Testing {description}...")
        
        # Create test file
        test_file = create_test_multipage_image(format_name, num_pages)
        
        try:
            # Verify the image has the expected number of pages
            with Image.open(test_file) as img:
                expected_pages = getattr(img, 'n_frames', 1)
                assert expected_pages == num_pages, f"Test {format_name} should have {num_pages} pages, got {expected_pages}"
            
            # Test with docling backend
            input_doc = InputDocument(
                path_or_stream=Path(test_file),
                format=InputFormat.IMAGE,
                backend=DoclingParseV4DocumentBackend
            )
            
            backend = DoclingParseV4DocumentBackend(input_doc, Path(test_file))
            
            # Check the page count in the backend
            actual_pages = backend.page_count()
            assert actual_pages == expected_pages, f"Backend should report {expected_pages} pages, got {actual_pages}"
            
            # Also verify the PDF was created correctly
            if isinstance(backend.path_or_stream, BytesIO):
                backend.path_or_stream.seek(0)
                pdf_doc = pdfium.PdfDocument(backend.path_or_stream)
                pdf_pages = len(pdf_doc)
                pdf_doc.close()
                assert pdf_pages == expected_pages, f"PDF should have {expected_pages} pages, got {pdf_pages}"
            
            print(f"✅ {description} passed")
            
        finally:
            # Clean up
            if os.path.exists(test_file):
                os.unlink(test_file)


def test_single_page_image_unchanged():
    """Test that single-page image files still work as before."""
    
    # Test different single-page formats
    formats_to_test = ['TIFF', 'GIF', 'WEBP']
    
    for format_name in formats_to_test:
        print(f"Testing single-page {format_name}...")
        
        # Create a single-page image
        test_file = create_test_multipage_image(format_name, 1)
        
        try:
            # Test with docling backend
            input_doc = InputDocument(
                path_or_stream=Path(test_file),
                format=InputFormat.IMAGE,
                backend=DoclingParseV4DocumentBackend
            )
            
            backend = DoclingParseV4DocumentBackend(input_doc, Path(test_file))
            
            # Should have exactly 1 page
            assert backend.page_count() == 1, f"Single-page {format_name} should have 1 page, got {backend.page_count()}"
            
            print(f"✅ Single-page {format_name} test passed")
            
        finally:
            # Clean up
            if os.path.exists(test_file):
                os.unlink(test_file)


if __name__ == "__main__":
    test_multipage_image_support()
    test_single_page_image_unchanged()
    print("All tests passed!")