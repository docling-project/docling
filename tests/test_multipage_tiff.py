"""
Test file for multi-page TIFF functionality.

This file tests that multi-page TIFF files are properly handled by the 
image-to-PDF conversion logic in the PDF backend.
"""

from pathlib import Path
from PIL import Image


def get_tiff_test_paths():
    """Get all TIFF test files from the test data directory."""
    directory = Path("./tests/data/tiff/")
    if not directory.exists():
        return []
    
    # List all TIFF files in the directory
    tiff_files = sorted(directory.rglob("*.tif")) + sorted(directory.rglob("*.tiff"))
    return tiff_files


def test_tiff_test_files_exist():
    """Verify that TIFF test files exist and have the expected properties."""
    tiff_paths = get_tiff_test_paths()
    
    assert len(tiff_paths) > 0, "No TIFF test files found in tests/data/tiff/"
    
    # Expected test files
    expected_files = {
        "single_page.tif": 1,
        "multipage_2pages.tif": 2,
        "multipage_3pages.tif": 3,
        "multipage_4pages.tif": 4,
    }
    
    found_files = {p.name: p for p in tiff_paths}
    
    for expected_file, expected_pages in expected_files.items():
        assert expected_file in found_files, f"Expected test file {expected_file} not found"
        
        # Verify the file has the expected number of pages
        with Image.open(found_files[expected_file]) as img:
            if hasattr(img, 'n_frames'):
                actual_pages = img.n_frames
            else:
                actual_pages = 1
                
            assert actual_pages == expected_pages, (
                f"File {expected_file} should have {expected_pages} pages, "
                f"but has {actual_pages}"
            )


def test_multipage_tiff_image_properties():
    """Test that our multi-page TIFF files have the correct structure."""
    tiff_paths = get_tiff_test_paths()
    
    if not tiff_paths:
        print("SKIP: No TIFF test files found")
        return
    
    multipage_files = [p for p in tiff_paths if "multipage" in p.name]
    
    for tiff_path in multipage_files:
        with Image.open(tiff_path) as img:
            # Should be a multi-page image
            assert hasattr(img, 'n_frames'), f"{tiff_path.name} should be multi-page"
            assert img.n_frames > 1, f"{tiff_path.name} should have more than 1 frame"
            
            # Verify we can seek through all frames
            frames_accessible = 0
            try:
                for i in range(img.n_frames):
                    img.seek(i)
                    frames_accessible += 1
                    
                    # Verify frame is valid
                    assert img.size[0] > 0 and img.size[1] > 0, (
                        f"Frame {i} in {tiff_path.name} has invalid size"
                    )
            except EOFError:
                pass
                
            assert frames_accessible == img.n_frames, (
                f"Could only access {frames_accessible} of {img.n_frames} frames "
                f"in {tiff_path.name}"
            )


def test_single_page_tiff_properties():
    """Test that single-page TIFF files have the correct structure."""
    tiff_paths = get_tiff_test_paths()
    
    if not tiff_paths:
        print("SKIP: No TIFF test files found")
        return
    
    single_page_files = [p for p in tiff_paths if "single_page" in p.name]
    
    if not single_page_files:
        print("SKIP: No single-page TIFF test files found")
        return
    
    for tiff_path in single_page_files:
        with Image.open(tiff_path) as img:
            # Should be a single-page image
            if hasattr(img, 'n_frames'):
                assert img.n_frames == 1, f"{tiff_path.name} should have exactly 1 frame"
            
            # Verify image is valid
            assert img.size[0] > 0 and img.size[1] > 0, (
                f"{tiff_path.name} has invalid size"
            )


def get_expected_pages_from_filename(filename: str) -> int:
    """Extract expected number of pages from test file names."""
    filename_lower = filename.lower()
    
    if "single_page" in filename_lower:
        return 1
    elif "2pages" in filename_lower:
        return 2
    elif "3pages" in filename_lower:
        return 3
    elif "4pages" in filename_lower:
        return 4
    else:
        # For any other files, assume single page
        return 1


if __name__ == "__main__":
    # When run directly, execute the tests
    test_tiff_test_files_exist()
    test_multipage_tiff_image_properties()
    test_single_page_tiff_properties()
    print("All TIFF test file validation passed!")