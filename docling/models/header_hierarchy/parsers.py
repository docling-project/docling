import re


class InvalidLetterException(Exception):
    def __init__(self, letter: str):
        super().__init__(f"Invalid letter: {letter}")


def infer_header_level_numerical(header_text: str) -> list[int]:
    # Match dot-, space-, or minus-separated numbers at the start
    match = re.match(r"^((?:\d+[.\s-])+)\d+", header_text.strip())
    if match:
        # Count the number of numeric groups (split by dot or space)
        numbering = match.group(0)
        # Split by dot or space, filter out empty strings
        try:
            groups = [int(g) for g in re.split(r"[.\s]", numbering) if g]
        except ValueError:
            return []
        return groups
    # Handle single number at the start (e.g., "2 Heading")
    match_single = re.match(r"^\d+", header_text.strip())
    if match_single:
        return [int(match_single.group(0))]
    # No numbering found
    return []


def letter_to_number(letter: str) -> int:
    """Convert a single letter (A-Z or a-z) to its corresponding number (A/a=1, B/b=2, ...)."""
    letter = letter.strip()
    if len(letter) != 1 or not letter.isalpha():
        raise InvalidLetterException(letter)
    return ord(letter.lower()) - ord("a") + 1


def infer_header_level_letter(header_text: str) -> list[int]:
    """
    Detects whether a header starts with a letter-numbered marker (A, B, C, ... or a, b, c, ...)
    and returns the numeric equivalent along with the raw match.
    """
    header_text = header_text.strip()
    # Match patterns like "A. ", "b) ", "C - Heading"
    match = re.match(r"^([A-Za-z])(?:[.)\s-]+)", header_text)
    if match:
        letter = match.group(1)
        try:
            return [letter_to_number(letter)]
        except InvalidLetterException:
            return []

    return []


# Roman numeral conversion helper
def roman_to_int(roman: str) -> int:
    roman = roman.upper()
    roman_map = {
        "M": 1000,
        "CM": 900,
        "D": 500,
        "CD": 400,
        "C": 100,
        "XC": 90,
        "L": 50,
        "XL": 40,
        "X": 10,
        "IX": 9,
        "V": 5,
        "IV": 4,
        "I": 1,
    }
    i, result = 0, 0
    while i < len(roman):
        # Check 2-letter symbols first (like 'CM', 'IX', etc.)
        if i + 1 < len(roman) and roman[i : i + 2] in roman_map:
            result += roman_map[roman[i : i + 2]]
            i += 2
        else:
            result += roman_map[roman[i]]
            i += 1
    return result


def infer_header_level_roman(header_text: str) -> list[int]:
    """
    Detects Roman numeral headers (at beginning of the string)
    and returns list of integer numbering levels.

    Examples:
        "II. Methods" -> [2]
        "IV-2 Results" -> [4, 2]
        "XIII Introduction" -> [13]
        "XI.2.3 Subsection" -> [11, 2, 3]
    """
    text = header_text.strip()

    # Match Roman numerals at start, optionally combined with dots/numbers
    match = re.match(r"^((?:[IVXLCDM]+[.\s-])+|[IVXLCDM]+$)", text, re.IGNORECASE)

    if match:
        numbering = match.group(0)
        # Split into tokens by dot, dash, space
        tokens = [t for t in re.split(r"[.\s-]", numbering) if t]

        groups = []
        try:
            for tok in tokens:
                if re.fullmatch(r"[IVXLCDM]+", tok, flags=re.IGNORECASE):
                    groups.append(roman_to_int(tok))
                elif tok.isdigit():
                    groups.append(int(tok))
        except KeyError:
            # KeyError from converting roman numbers to int.
            pass
        return groups

    return []
