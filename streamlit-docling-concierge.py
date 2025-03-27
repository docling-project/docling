#!/usr/bin/env python3
"""
Docling Concierge - Streamlit Web Interface (Docling 2.25.1 or similar)
Updated to remove references to `document_type`, which no longer exists.

Features:
- No docling.backends usage
- Spinner + progress bar for processing
- OCR language code mapping for Tesseract/EasyOCR
- Displays `res.input.format` instead of `document_type` in results
"""

import os
import re
import tempfile
from pathlib import Path
from typing import Optional, List, Dict

import streamlit as st
import pandas as pd

# --------------------------------
# Attempt to import Docling modules
# --------------------------------
missing_modules = []
try:
    import docling
except ImportError as e:
    missing_modules.append(f"docling: {str(e)}")

try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
except ImportError as e:
    missing_modules.append(f"docling.document_converter: {str(e)}")

try:
    from docling.datamodel.base_models import InputFormat, OutputFormat
except ImportError as e:
    missing_modules.append(f"docling.datamodel.base_models: {str(e)}")

try:
    from docling.datamodel.document import ConversionResult
except ImportError as e:
    missing_modules.append(f"docling.datamodel.document: {str(e)}")

try:
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions, OcrEngine, TableFormerMode,
        EasyOcrOptions, TesseractCliOcrOptions, TesseractOcrOptions,
        OcrMacOptions, RapidOcrOptions, OcrOptions,
    )
except ImportError as e:
    missing_modules.append(f"docling.datamodel.pipeline_options: {str(e)}")

try:
    from docling_core.types.doc import ImageRefMode
except ImportError as e:
    missing_modules.append(f"docling_core.types.doc: {str(e)}")


class DoclingConciergeStreamlit:
    """
    Streamlit interface to Docling functionality, tailored to Docling 2.25.1.
    No references to `document_type` (use input.format instead).
    """
    
    def __init__(self):
        self.converter = DocumentConverter()
        self.temp_dir = Path(tempfile.mkdtemp(prefix="docling_"))
        
        # Default pipeline options
        self.pipeline_options = PdfPipelineOptions(
            do_ocr=True,
            do_table_structure=True,
            do_code_enrichment=False,
            do_formula_enrichment=False,
            do_picture_classification=False,
            do_picture_description=False,
            enable_remote_services=False,
        )
        
        # Simple mappings for input detection
        self.format_aliases = {
            "pdf": InputFormat.PDF,
            "image": InputFormat.IMAGE,
            "docx": InputFormat.DOCX,
            "word": InputFormat.DOCX,
            "excel": InputFormat.XLSX,
            "xlsx": InputFormat.XLSX,
            "pptx": InputFormat.PPTX,
            "powerpoint": InputFormat.PPTX,
            "html": InputFormat.HTML,
            "markdown": InputFormat.MD,
            "md": InputFormat.MD,
            "asciidoc": InputFormat.ASCIIDOC,
            "csv": InputFormat.CSV,
            "xml": InputFormat.XML_JATS,
            "json": InputFormat.JSON_DOCLING,
        }
        
        # Mapping output format strings to docling's OutputFormat
        self.output_format_aliases = {
            "markdown": OutputFormat.MARKDOWN,
            "md": OutputFormat.MARKDOWN,
            "html": OutputFormat.HTML,
            "json": OutputFormat.JSON,
            "text": OutputFormat.TEXT,
            "txt": OutputFormat.TEXT,
            "doctags": OutputFormat.DOCTAGS,
        }
        
        # OCR engine choices for the UI
        self.ocr_engines = {
            "EasyOCR (recommended)": OcrEngine.EASYOCR,
            "Tesseract": OcrEngine.TESSERACT,
            "Tesseract CLI": OcrEngine.TESSERACT_CLI,
            "macOS OCR (Apple devices only)": OcrEngine.OCRMAC,
            "RapidOCR": OcrEngine.RAPIDOCR,
        }
        
        # ImageRefMode choices
        self.image_modes = {
            "Embedded (include images in file)": ImageRefMode.EMBEDDED,
            "Referenced (link to separate image files)": ImageRefMode.REFERENCED,
            "Placeholder (text only, no images)": ImageRefMode.PLACEHOLDER,
        }
        
        # Common language list for user-friendly display
        self.common_languages = [
            "English", "Spanish", "French", "German", "Chinese",
            "Japanese", "Korean", "Russian", "Arabic", "Hindi",
            "Portuguese", "Italian"
        ]
        
        # -----------
        # Define OCR language code mappings
        # -----------
        self.lang_code_map_tesseract = {
            "English": "eng",
            "Spanish": "spa",
            "French": "fra",
            "German": "deu",
            "Chinese": "chi_sim",  # Tesseract can use chi_tra for traditional
            "Japanese": "jpn",
            "Korean": "kor",
            "Russian": "rus",
            "Arabic": "ara",
            "Hindi": "hin",
            "Portuguese": "por",
            "Italian": "ita",
        }
        
        # EasyOCR language codes
        self.lang_code_map_easyocr = {
            "English": "en",
            "Spanish": "es",
            "French": "fr",
            "German": "de",
            "Chinese": "ch_tra",   # or ch_sim
            "Japanese": "ja",
            "Korean": "ko",
            "Russian": "ru",
            "Arabic": "ar",
            "Hindi": "hi",
            "Portuguese": "pt",
            "Italian": "it",
        }
        
        # RapidOCR might follow a similar convention to EasyOCR
        self.lang_code_map_rapidocr = {
            "English": "en",
            "Spanish": "es",
            "French": "fr",
            "German": "de",
            "Chinese": "ch",
            "Japanese": "ja",
            "Korean": "ko",
            "Russian": "ru",
            "Arabic": "ar",
            "Hindi": "hi",
            "Portuguese": "pt",
            "Italian": "it",
        }
    
    def configure_streamlit_page(self):
        """Set up the Streamlit page layout and style."""
        st.set_page_config(
            page_title="Docling Concierge",
            page_icon="📄",
            layout="wide",
            initial_sidebar_state="expanded",
        )
        
        # Custom CSS for styling
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1E88E5;
        }
        .sub-header {
            font-size: 1.5rem;
            color: #424242;
        }
        .card {
            border-radius: 5px;
            padding: 20px;
            margin-bottom: 20px;
            background-color: #f8f9fa;
            border-left: 4px solid #1E88E5;
        }
        .feature-title {
            font-weight: bold;
            color: #1E88E5;
        }
        .hint {
            font-size: 0.9rem;
            color: #757575;
            font-style: italic;
        }
        .success {
            padding: 10px;
            border-radius: 5px;
            background-color: #E8F5E9;
            border-left: 4px solid #4CAF50;
        }
        .error {
            padding: 10px;
            border-radius: 5px;
            background-color: #FFEBEE;
            border-left: 4px solid #F44336;
        }
        .stButton>button {
            width: 100%;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def show_welcome(self):
        """Welcome header and intro text."""
        st.markdown('<p class="main-header">📄 Docling Concierge</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Transform and extract content from your documents with ease</p>',
                    unsafe_allow_html=True)
        
        with st.expander("ℹ️ About this app", expanded=False):
            st.markdown("""
            **Docling Concierge** helps you process documents in various formats. You can:
            
            - **Convert between formats**: PDF to Markdown, Word to HTML, and more
            - **Extract content**: Tables, images, code snippets, and formulas
            - **Process text**: Apply OCR to scanned documents
            - **Analyze content**: Enhance with image descriptions and classifications
            
            Simply upload your documents or provide URLs, select the options you need,
            and let Docling handle the rest.
            """)
    
    def show_sidebar(self):
        """Sidebar with quick examples and info."""
        with st.sidebar:
            st.markdown("### 📚 Quick Examples")
            st.markdown("Try these examples (click to fill):")
            
            example_queries = [
                "Convert PDF to Markdown with tables",
                "Extract tables from document and save as HTML",
                "Process scanned PDF with OCR",
                "Convert document to text only (no images)",
                "Extract code blocks from programming PDF"
            ]
            
            for query in example_queries:
                if st.button(query, key=f"example_{query}"):
                    st.session_state.query = query
            
            st.markdown("---")
            st.markdown("### 🔍 Document Types")
            st.markdown("""
            - 📄 **Documents**: PDF, DOCX, PPTX
            - 📊 **Data**: XLSX, CSV
            - 🌐 **Web**: HTML
            - 📝 **Text**: Markdown, Text
            - 🖼️ **Images**: JPG, PNG
            """)
            
            st.markdown("---")
            st.markdown("### 🎯 Output Formats")
            st.markdown("""
            - **Markdown**: Perfect for documentation
            - **HTML**: For web publishing
            - **JSON**: For data processing
            - **Text**: Plain text extraction
            - **Doctags**: Docling token format
            """)
            
            st.markdown("---")
            st.markdown("### 💡 Tips")
            st.markdown("""
            - Use OCR for scanned documents
            - Enable table extraction for spreadsheet-like content
            - Choose image mode based on your needs
            - Batch process multiple files at once
            """)
            
            st.markdown("---")
            st.markdown("Made with ❤️ using Docling")
            
            # Reset button if you want to clear all state
            if st.button("🔄 Reset All Settings"):
                for key in list(st.session_state.keys()):
                    if key not in ["authenticated", "username"]:
                        del st.session_state[key]
                st.rerun()
    
    def show_input_section(self):
        """Section to upload files or enter URLs."""
        st.markdown('<p class="feature-title">📥 Input Documents</p>', unsafe_allow_html=True)
        
        input_tab1, input_tab2 = st.tabs(["Upload Files", "Enter URLs"])
        
        with input_tab1:
            st.markdown("""<p class="hint">
            Upload one or more documents to process. 
            Supports PDF, Word, Excel, PowerPoint, Images, and more.
            </p>""", unsafe_allow_html=True)
            
            uploaded_files = st.file_uploader(
                "Upload documents",
                accept_multiple_files=True,
                type=[
                    "pdf", "docx", "xlsx", "pptx", "jpg", "jpeg", "png",
                    "html", "md", "csv", "xml", "json", "txt"
                ],
                help="Select one or more files from your computer"
            )
            
            if uploaded_files:
                st.session_state.file_paths = []
                for uploaded_file in uploaded_files:
                    temp_file_path = self.temp_dir / uploaded_file.name
                    with open(temp_file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.session_state.file_paths.append(temp_file_path)
                
                # Show info about uploaded files
                files_df = pd.DataFrame({
                    "File Name": [file.name for file in uploaded_files],
                    "Size": [f"{file.size / 1024:.1f} KB" for file in uploaded_files],
                    "Type": [file.type for file in uploaded_files]
                })
                st.dataframe(files_df, use_container_width=True)
        
        with input_tab2:
            st.markdown("""<p class="hint">
            Enter one or more URLs to process (one per line).
            </p>""", unsafe_allow_html=True)
            
            url_input = st.text_area(
                "Enter document URLs",
                placeholder="https://example.com/document.pdf\nhttps://another-site.org/paper.pdf",
            )
            
            if url_input:
                urls = [url.strip() for url in url_input.split("\n") if url.strip().startswith("http")]
                if urls:
                    st.session_state.urls = urls
                    # Show data frame of URLs
                    urls_df = pd.DataFrame({
                        "URL": urls,
                        "Type": ["Web URL" for _ in urls]
                    })
                    st.dataframe(urls_df, use_container_width=True)
                else:
                    st.warning("No valid URLs found. URLs must start with http:// or https://")
    
    def show_natural_language_query(self):
        """User can type a plain-English request for what to do."""
        st.markdown('<p class="feature-title">🔍 What do you want to do?</p>', unsafe_allow_html=True)
        
        query = st.text_input(
            "Describe what you want to do with your documents",
            value=st.session_state.get("query", ""),
            placeholder="Example: Convert this PDF to markdown with tables"
        )
        
        if query:
            st.session_state.query = query
            # Try to guess an output format from the user's text
            output_format = self._extract_output_format_from_query(query)
            if output_format:
                for alias, format_value in self.output_format_aliases.items():
                    if format_value == output_format:
                        st.session_state.output_format = alias.capitalize()
                        break
    
    def show_visual_options(self):
        """Checkboxes and advanced features for the user."""
        st.markdown('<p class="feature-title">⚙️ Processing Options</p>', unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["Basic Options", "OCR Settings", "Advanced Features"])
        
        # ----------
        # Basic Options
        # ----------
        with tab1:
            output_formats = ["Markdown", "HTML", "JSON", "Text", "Doctags"]
            selected_format = st.selectbox(
                "Output Format",
                options=output_formats,
                index=output_formats.index(st.session_state.get("output_format", "Markdown")),
                help="Select the format you want to convert your documents to"
            )
            st.session_state.output_format = selected_format
            
            image_mode_options = list(self.image_modes.keys())
            selected_image_mode = st.selectbox(
                "Image Handling",
                options=image_mode_options,
                index=0,
                help="Choose how images should be handled in the output"
            )
            st.session_state.image_mode = self.image_modes[selected_image_mode]
        
        # ----------
        # OCR Settings
        # ----------
        with tab2:
            ocr_enabled = st.checkbox(
                "Enable OCR (Text Recognition)",
                value=True,
                help="Apply OCR for scanned documents or images"
            )
            st.session_state.ocr_enabled = ocr_enabled
            
            if ocr_enabled:
                ocr_engine_options = list(self.ocr_engines.keys())
                selected_ocr_engine = st.selectbox(
                    "OCR Engine",
                    options=ocr_engine_options,
                    index=0,
                    help="Select the OCR engine to use"
                )
                st.session_state.ocr_engine = self.ocr_engines[selected_ocr_engine]
                
                force_ocr = st.checkbox(
                    "Force Full-Page OCR",
                    value=False,
                    help="OCR entire pages even if they appear to have machine-readable text"
                )
                st.session_state.force_ocr = force_ocr
                
                # Expandable area for choosing languages
                with st.expander("OCR Languages", expanded=False):
                    selected_languages = []
                    
                    # Determine which language map we should use
                    if st.session_state.ocr_engine == OcrEngine.EASYOCR:
                        lang_map = self.lang_code_map_easyocr
                    elif st.session_state.ocr_engine == OcrEngine.RAPIDOCR:
                        lang_map = self.lang_code_map_rapidocr
                    else:
                        # Tesseract, Tesseract CLI, OCRMAC fallback
                        lang_map = self.lang_code_map_tesseract
                    
                    col1, col2 = st.columns(2)
                    half = len(self.common_languages) // 2
                    
                    with col1:
                        for lang in self.common_languages[:half]:
                            # If user checks the box, we map e.g. "English" to "eng" or "en"
                            if st.checkbox(lang, value=(lang == "English")):
                                code = lang_map.get(lang, "eng")  # fallback to "eng"
                                selected_languages.append(code)
                    
                    with col2:
                        for lang in self.common_languages[half:]:
                            if st.checkbox(lang):
                                code = lang_map.get(lang, "eng")
                                selected_languages.append(code)
                    
                    custom_lang = st.text_input(
                        "Add other language codes (comma-separated)",
                        placeholder="e.g., eng, spa, fra"
                    )
                    if custom_lang:
                        custom_langs = [x.strip().lower() for x in custom_lang.split(",")]
                        selected_languages.extend(custom_langs)
                    
                    if not selected_languages:
                        # fallback if user didn't check anything
                        selected_languages = ["eng"]
                    
                    st.session_state.ocr_langs = selected_languages
        
        # ----------
        # Advanced Features
        # ----------
        with tab3:
            st.markdown("#### Content Extraction")
            col1, col2 = st.columns(2)
            
            with col1:
                table_extraction = st.checkbox(
                    "Extract Tables",
                    value=True,
                    help="Recognize and extract tables from documents"
                )
                st.session_state.do_table_structure = table_extraction
                if table_extraction:
                    table_mode = st.radio(
                        "Table Extraction Mode",
                        options=["Fast", "Accurate"],
                        index=0,
                        horizontal=True,
                        help="Fast is quick but less precise, Accurate is slower but more reliable"
                    )
                    st.session_state.table_mode = (
                        TableFormerMode.FAST if table_mode == "Fast" else TableFormerMode.ACCURATE
                    )
                
                code_extraction = st.checkbox(
                    "Extract Code Blocks",
                    value=False,
                    help="Identify and extract programming code blocks"
                )
                st.session_state.do_code_enrichment = code_extraction
            
            with col2:
                formula_extraction = st.checkbox(
                    "Extract Mathematical Formulas",
                    value=False,
                    help="Detect mathematical equations and formulas"
                )
                st.session_state.do_formula_enrichment = formula_extraction
                
                image_classification = st.checkbox(
                    "Classify Images",
                    value=False,
                    help="Add classifications to images in the document"
                )
                st.session_state.do_picture_classification = image_classification
                
                image_description = st.checkbox(
                    "Describe Images",
                    value=False,
                    help="Generate descriptive text for images (may require remote services)"
                )
                st.session_state.do_picture_description = image_description
                
                if image_description:
                    st.session_state.enable_remote_services = True
    
    def process_documents(self):
        """Process documents based on user settings, with progress bar."""
        has_files = hasattr(st.session_state, 'file_paths') and st.session_state.file_paths
        has_urls = hasattr(st.session_state, 'urls') and st.session_state.urls
        
        if not (has_files or has_urls):
            st.warning("Please upload at least one document or provide a URL.")
            return
        
        # Gather sources
        sources = []
        if has_files:
            sources.extend(st.session_state.file_paths)
        if has_urls:
            sources.extend(st.session_state.urls)
        
        # Determine output format
        output_format_str = st.session_state.get("output_format", "Markdown").lower()
        output_format = self.output_format_aliases.get(output_format_str, OutputFormat.MARKDOWN)
        
        # OCR settings
        ocr_enabled = st.session_state.get("ocr_enabled", True)
        ocr_engine = st.session_state.get("ocr_engine", OcrEngine.EASYOCR)
        force_ocr = st.session_state.get("force_ocr", False)
        ocr_langs = st.session_state.get("ocr_langs", ["eng"])
        
        # Image handling
        image_mode = st.session_state.get("image_mode", ImageRefMode.EMBEDDED)
        
        # Configure pipeline options
        self.pipeline_options.do_ocr = ocr_enabled
        self.pipeline_options.do_table_structure = st.session_state.get("do_table_structure", True)
        self.pipeline_options.do_code_enrichment = st.session_state.get("do_code_enrichment", False)
        self.pipeline_options.do_formula_enrichment = st.session_state.get("do_formula_enrichment", False)
        self.pipeline_options.do_picture_classification = st.session_state.get("do_picture_classification", False)
        self.pipeline_options.do_picture_description = st.session_state.get("do_picture_description", False)
        self.pipeline_options.enable_remote_services = st.session_state.get("enable_remote_services", False)
        
        # If table extraction is on, set the mode
        if st.session_state.get("do_table_structure", True):
            mode = st.session_state.get("table_mode", TableFormerMode.FAST)
            self.pipeline_options.table_structure_options.mode = mode
        
        # Create the OCR options object
        if ocr_enabled:
            if ocr_engine == OcrEngine.EASYOCR:
                ocr_options = EasyOcrOptions(force_full_page_ocr=force_ocr)
            elif ocr_engine == OcrEngine.TESSERACT_CLI:
                ocr_options = TesseractCliOcrOptions(force_full_page_ocr=force_ocr)
            elif ocr_engine == OcrEngine.TESSERACT:
                ocr_options = TesseractOcrOptions(force_full_page_ocr=force_ocr)
            elif ocr_engine == OcrEngine.OCRMAC:
                ocr_options = OcrMacOptions(force_full_page_ocr=force_ocr)
            elif ocr_engine == OcrEngine.RAPIDOCR:
                ocr_options = RapidOcrOptions(force_full_page_ocr=force_ocr)
            else:
                ocr_options = EasyOcrOptions(force_full_page_ocr=force_ocr)
            
            # Assign the languages (e.g. ["eng", "spa"])
            ocr_options.lang = ocr_langs
            self.pipeline_options.ocr_options = ocr_options
        
        # Create PdfFormatOption with no custom backend
        pdf_format_option = PdfFormatOption(
            pipeline_options=self.pipeline_options
        )
        
        format_options = {
            InputFormat.PDF: pdf_format_option,
            InputFormat.IMAGE: pdf_format_option,
        }
        
        converter = DocumentConverter(format_options=format_options)
        
        results = []
        output_files = []
        
        # Show a spinner + progress bar for each file
        with st.spinner("Processing documents..."):
            progress_bar = st.progress(0)
            total_sources = len(sources)
            
            for i, source in enumerate(sources):
                short_name = os.path.basename(str(source))
                st.info(f"Processing {i+1}/{total_sources}: {short_name}")
                
                try:
                    result = converter.convert(source)
                    results.append(result)
                    
                    if result.status.value == "success":
                        base_filename = os.path.basename(str(result.input.file))
                        base_name = os.path.splitext(base_filename)[0]
                        
                        # Save based on the chosen output format
                        if output_format == OutputFormat.MARKDOWN:
                            out_file = self.temp_dir / f"{base_name}.md"
                            result.document.save_as_markdown(out_file, image_mode=image_mode)
                        
                        elif output_format == OutputFormat.HTML:
                            out_file = self.temp_dir / f"{base_name}.html"
                            result.document.save_as_html(out_file, image_mode=image_mode)
                        
                        elif output_format == OutputFormat.JSON:
                            out_file = self.temp_dir / f"{base_name}.json"
                            result.document.save_as_json(out_file, image_mode=image_mode)
                        
                        elif output_format == OutputFormat.TEXT:
                            out_file = self.temp_dir / f"{base_name}.txt"
                            result.document.save_as_markdown(
                                out_file, strict_text=True, image_mode=ImageRefMode.PLACEHOLDER
                            )
                        
                        elif output_format == OutputFormat.DOCTAGS:
                            out_file = self.temp_dir / f"{base_name}.doctags"
                            result.document.save_as_document_tokens(out_file)
                        
                        output_files.append(out_file)
                    
                except Exception as e:
                    st.error(f"Error processing {short_name}: {str(e)}")
                
                # Update the progress bar
                progress_percent = int((i + 1) / total_sources * 100)
                progress_bar.progress(progress_percent)
            
            st.success("Processing complete!")
        
        st.session_state.results = results
        st.session_state.output_files = output_files
        return results, output_files
    
    def show_results(self, results, output_files):
        """Display the final results. Replaced `document_type` with `res.input.format`."""
        if not results:
            return
        
        st.markdown('<p class="feature-title">📝 Results</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            success_count = sum(r.status.value == "success" for r in results)
            fail_count = len(results) - success_count
            
            if success_count > 0:
                st.markdown(f"<div class='success'>✅ Successfully processed {success_count} document(s)</div>",
                            unsafe_allow_html=True)
            if fail_count > 0:
                st.markdown(f"<div class='error'>❌ Failed to process {fail_count} document(s)</div>",
                            unsafe_allow_html=True)
        
        with col2:
            if output_files:
                import zipfile
                import io
                
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                    for fp in output_files:
                        zf.write(fp, arcname=fp.name)
                
                zip_buffer.seek(0)
                
                st.download_button(
                    "📦 Download All Results",
                    data=zip_buffer,
                    file_name="docling_results.zip",
                    mime="application/zip",
                    use_container_width=True
                )
        
        if output_files:
            st.markdown("### Processed Documents")
            for i, (res, out_fp) in enumerate(zip(results, output_files), start=1):
                if res.status.value != "success":
                    continue
                
                with st.expander(f"Document {i}: {out_fp.name}"):
                    st.markdown(f"**Source**: {res.input.file}")

                    # Replaced: st.markdown(f"**Type**: {res.document.document_type}")
                    # With: show the format from res.input
                    st.markdown(f"**Format**: {res.input.format}")

                    # If pages are present
                    st.markdown(f"**Pages**: {len(res.document.pages)}")
                    
                    preview_tab, download_tab = st.tabs(["Preview", "Download"])
                    
                    with preview_tab:
                        try:
                            with open(out_fp, "r", encoding="utf-8") as f:
                                content = f.read()
                        except UnicodeDecodeError:
                            content = f"<Cannot preview file {out_fp.suffix}>"
                        
                        ext = out_fp.suffix.lower()
                        if ext == ".md":
                            st.markdown(content)
                        elif ext == ".html":
                            st.components.v1.html(content, height=400)
                        elif ext == ".json":
                            st.json(content)
                        else:
                            st.text(content)
                    
                    with download_tab:
                        with open(out_fp, "rb") as f:
                            file_data = f.read()
                        st.download_button(
                            f"Download {out_fp.name}",
                            data=file_data,
                            file_name=out_fp.name,
                            mime="application/octet-stream",
                            use_container_width=True
                        )
    
    def _extract_output_format_from_query(self, query: str) -> Optional[OutputFormat]:
        """Try to guess an OutputFormat from user text like 'Convert to Markdown'."""
        to_format_match = re.search(r'to\s+(\w+)', query, re.IGNORECASE)
        if to_format_match:
            fmt_str = to_format_match.group(1).lower()
            if fmt_str in self.output_format_aliases:
                return self.output_format_aliases[fmt_str]
        
        # Check any direct mention (like 'markdown', 'html', etc.)
        for format_name, format_value in self.output_format_aliases.items():
            if format_name in query.lower():
                return format_value
        
        return None
    
    def run(self):
        """Run the entire Streamlit app."""
        if missing_modules:
            st.error("Error: Some Docling modules could not be imported:")
            for module in missing_modules:
                st.write(f"- {module}")
            
            st.info("""
            Possible solutions:
            1. Make sure you have the correct version of Docling installed:
               ```
               pip install -U docling
               ```
            2. If you installed from source or development version, make sure all dependencies are installed:
               ```
               pip install -e .[dev]
               ```
            3. Check your PYTHONPATH if using a development environment
            """)
            return
        
        self.configure_streamlit_page()
        self.show_welcome()
        self.show_sidebar()
        self.show_input_section()
        self.show_natural_language_query()
        self.show_visual_options()
        
        if st.button("🔄 Process Documents", type="primary", use_container_width=True):
            with st.spinner("Processing documents..."):
                results, output_files = self.process_documents()
                if results:
                    self.show_results(results, output_files)
        
        # If results already exist in session (e.g. user pressed 'Process' before),
        # show them again
        if hasattr(st.session_state, 'results') and hasattr(st.session_state, 'output_files'):
            self.show_results(st.session_state.results, st.session_state.output_files)


def main():
    """Entry point for Streamlit."""
    app = DoclingConciergeStreamlit()
    app.run()


if __name__ == "__main__":
    main()
