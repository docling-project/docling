def map_tesseract_script(script: str) -> str:
    r""" """
    if script == "Katakana" or script == "Hiragana":
        script = "Japanese"
    elif script == "Han":
        script = "HanS"
    elif script == "Korean":
        script = "Hangul"
    return script
