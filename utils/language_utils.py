from langdetect import detect

def detect_language(text: str) -> str:
    """
    Detects language of the given text.
    Returns 'Kannada' or 'English' (default English).
    """
    try:
        lang_code = detect(text)
        if lang_code == "kn":
            return "Kannada"
        return "English"
    except Exception:
        return "English"
