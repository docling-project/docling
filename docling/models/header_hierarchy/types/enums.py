from enum import Enum


class NumberingLevel(Enum):
    level_latin = "level_latin"
    level_alpha = "level_alpha"
    level_numerical = "level_numerical"


class StyleAttributes(Enum):
    font_size = "font_size"
    bold = "bold"
    italic = "italic"
