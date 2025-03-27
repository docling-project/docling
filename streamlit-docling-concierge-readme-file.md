# Docling Concierge - Streamlit Web Interface

A user-friendly web application for Docling document processing built with Streamlit. This app allows both technical and non-technical users to process documents using natural language instructions or guided selections through an intuitive interface.

![Docling Concierge Screenshot](https://via.placeholder.com/800x450?text=Docling+Concierge+Web+App)

## Features

- **Simple User Interface**: Designed specifically for non-technical users
- **Visual Controls**: Checkboxes, toggles, and dropdowns for all options
- **Flexible Input**: Upload files or provide URLs
- **Rich Output Formats**: Markdown, HTML, JSON, Text, Doctags
- **Interactive Previews**: See processed documents before downloading
- **Batch Processing**: Handle multiple documents at once
- **Smart Suggestions**: Get guidance on best options for your documents
- **Visual Progress Tracking**: Monitor document processing with progress bars

## Compatibility

This version of Docling Concierge is optimized for Docling 2.25.1 and similar versions, with the following adjustments:
- No dependencies on deprecated `document_type` attributes
- No references to Docling backend classes
- Proper language code mappings for different OCR engines
- Enhanced error handling and progress visualization

## Installation

1. Make sure you have Python 3.8+ installed on your system
2. Install Docling (if not already installed):
   ```bash
   pip install -U docling==2.25.1
   ```
3. Install Streamlit and other dependencies:
   ```bash
   pip install streamlit pandas
   ```
4. Download the `docling_concierge_streamlit.py` file from this repository
5. Run the Streamlit app:
   ```bash
   streamlit run docling_concierge_streamlit.py
   ```

## Usage

1. **Upload Documents**:
   - Use the "Upload Files" tab to select documents from your computer
   - Or use the "Enter URLs" tab to process documents from the web

2. **Describe Your Task**:
   - Type what you want to do in natural language
   - Example: "Convert this PDF to markdown with tables"
   - The app will automatically detect intent and pre-select appropriate options

3. **Adjust Processing Options** (Optional):
   - **Basic Options**: Output format, image handling
   - **OCR Settings**: Enable text recognition, select language
   - **Advanced Features**: Table extraction, code detection, formula processing

4. **Process Documents**:
   - Click the "Process Documents" button
   - Monitor progress through the status indicator and progress bar

5. **Work with Results**:
   - Preview processed documents directly in the app
   - Download individual results or all files as a ZIP

## Language Support for OCR

The app includes smart language mapping that automatically translates user-friendly language names to the correct codes for each OCR engine:

- **EasyOCR**: Uses codes like "en", "es", "fr", etc.
- **Tesseract**: Uses codes like "eng", "spa", "fra", etc.
- **RapidOCR**: Uses appropriate codes for this engine

Users can simply check the languages they need without knowing the specific codes required by each engine.

## Common Processing Tasks

- **Document Conversion**: PDF to Markdown, DOCX to HTML, etc.
- **Table Extraction**: Pull tables from documents into structured formats (with fast or accurate options)
- **OCR Processing**: Extract text from scanned documents and images
- **Content Analysis**: Detect and enhance code blocks, formulas, images
- **Image Handling**: Choose between embedded, referenced, or placeholder modes

## Tips for Non-Technical Users

- Use the examples in the sidebar for quick starts
- Start with default options for most documents
- Enable OCR for scanned documents or images
- Use "Markdown" format for most readable output
- Choose "HTML" for web-ready content with formatting
- Use the "Reset All Settings" button in the sidebar to start fresh

## Requirements

- Python 3.8+
- Docling 2.25.1 (or compatible version)
- Streamlit
- Pandas

## License

This project is licensed under the same terms as Docling.

## Acknowledgements

Based on the original Docling Concierge command-line tool.
