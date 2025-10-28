from paddleocr import PaddleOCR

pipeline = PaddleOCR(
    lang="korean",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    text_detection_model_dir="/models/paddleocr_model/paddleocr_model/PP-OCRv5_server_det",
    text_recognition_model_dir="/models/paddleocr_model/paddleocr_model/korean_PP-OCRv5_mobile_rec",
    text_detection_model_name="PP-OCRv5_server_det",
    text_recognition_model_name="korean_PP-OCRv5_mobile_rec",
)
pipeline.export_paddlex_config_to_yaml("ocr_config.yaml")
