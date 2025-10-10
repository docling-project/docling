import easyocr
import os
import cv2 # OpenCV is often required for image processing

# --- CONFIGURATION ---
# IMPORTANT: 
# 1. Create an image file named 'handwriting_sample.jpg'.
# 2. Put it in the same directory as this script (C:\docling-hackathon).
TEST_IMAGE_NAME = "handwriting_sample.jpg"
TEST_IMAGE_PATH = os.path.join(os.getcwd(), TEST_IMAGE_NAME)

# Language codes used by the OCR engine
LANGUAGES = ['en'] 

def run_ocr_test():
    """Tests the core EasyOCR engine's ability to read a sample image."""
    
    if not os.path.exists(TEST_IMAGE_PATH):
        print("🔴 ERROR: Test image not found!")
        print(f"Please create and place '{TEST_IMAGE_NAME}' at: {os.getcwd()}")
        return

    print(f"--- Running Isolated OCR Test on: {TEST_IMAGE_PATH} ---")
    
    # 1. Initialize the OCR Reader (loads the models)
    try:
        reader = easyocr.Reader(LANGUAGES, gpu=False)
    except Exception as e:
        print("🔴 ERROR: Failed to initialize EasyOCR Reader. Check installation.")
        print(f"Details: {e}")
        return

    # 2. Run the recognition on the image
    # detail=0 returns only the text recognized
    results = reader.recognize(TEST_IMAGE_PATH, detail=0)

    # 3. Analyze Results
    if results:
        print("✅ SUCCESS: Recognized Text:")
        for text in results:
            print(f"  - {text}")
        print("\nConclusion: The core OCR engine works. The bug is likely in Docling's pipeline.")
    else:
        print("❌ FAILURE: No text was recognized from the sample image.")
        print("\nConclusion: The core OCR engine itself is likely misconfigured or lacks robust handwriting support.")


if __name__ == "__main__":
    run_ocr_test()