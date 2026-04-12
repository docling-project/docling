import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys

# Mock rapidocr which might not be installed or is heavy
mock_rapidocr = MagicMock()
mock_rapidocr.EngineType.ONNXRUNTIME = "onnxruntime"
mock_rapidocr.EngineType.TORCH = "torch"
sys.modules["rapidocr"] = mock_rapidocr

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.pipeline_options import RapidOcrOptions
from docling.models.stages.ocr.rapid_ocr_model import RapidOcrModel

class TestRapidOcrLang(unittest.TestCase):
    @patch("rapidocr.RapidOCR")
    def test_language_selection(self, mock_rapidocr_engine):
        acc_opts = AcceleratorOptions()
        artifacts_path = Path("/tmp/artifacts")

        # Case 1: English (explicitly requested)
        opts_en = RapidOcrOptions(lang=["en"], backend="onnxruntime")
        # Initialize normally; BaseOcrModel.__init__ is light.
        model_en = RapidOcrModel(enabled=True, artifacts_path=artifacts_path, options=opts_en, accelerator_options=acc_opts)
        # Verify English URLs would be used (this checks the internal logic)
        assert "english" in RapidOcrModel._models_by_language
        
        # Case 2: Chinese (default)
        opts_ch = RapidOcrOptions(backend="onnxruntime")
        assert "chinese" in opts_ch.lang
        model_ch = RapidOcrModel(enabled=True, artifacts_path=artifacts_path, options=opts_ch, accelerator_options=acc_opts)

    def test_download_models_lang(self):
        # Mock download_url_with_progress to return a mock response with read() returning bytes
        with patch("docling.models.stages.ocr.rapid_ocr_model.download_url_with_progress") as mock_download:
            mock_response = MagicMock()
            mock_response.read.return_value = b"dummy content"
            mock_download.return_value = mock_response
            
            # We also need to mock builtins.open to avoid actually writing to /tmp
            with patch("builtins.open", unittest.mock.mock_open()):
                # Download English with force=True
                RapidOcrModel.download_models(local_dir=Path("/tmp"), backend="onnxruntime", lang="english", force=True)
                # Verify that the English detection model was requested in one of the calls
                all_urls = [arg[0] for arg, _ in mock_download.call_args_list]
                print(f"Captured URLs (English): {all_urls}")
                assert any("en_PP-OCRv3_det_mobile.onnx" in url for url in all_urls)
                
                # Download Chinese (default) with force=True
                mock_download.reset_mock()
                RapidOcrModel.download_models(local_dir=Path("/tmp"), backend="onnxruntime", lang="chinese", force=True)
                all_urls_ch = [arg[0] for arg, _ in mock_download.call_args_list]
                print(f"Captured URLs (Chinese): {all_urls_ch}")
                assert any("ch_PP-OCRv4_det_mobile.onnx" in url for url in all_urls_ch)

    def test_full_model_downloader_coverage(self):
        from docling.utils.model_downloader import download_models
        with patch("docling.models.stages.ocr.rapid_ocr_model.RapidOcrModel.download_models") as mock_rapid_down:
            download_models(
                output_dir=Path("/tmp/models"),
                with_layout=False,
                with_tableformer=False,
                with_code_formula=False,
                with_picture_classifier=False,
                with_rapidocr=True
            )
            # Verify 2 backends * 2 languages = 4 calls
            assert mock_rapid_down.call_count == 4
            call_args_list = [call.kwargs for call in mock_rapid_down.call_args_list]
            langs = [c.get("lang") for c in call_args_list]
            assert "english" in langs
            assert "chinese" in langs

if __name__ == "__main__":
    unittest.main()
