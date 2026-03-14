"""
Translation layer for Hindi ↔ English name/text translation.
Uses deep-translator (Google Translate free tier) with a robust
fallback that returns the original text if the network is unavailable.

pip install deep-translator
"""
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy import so the server doesn't crash if library is missing
try:
    from deep_translator import GoogleTranslator
    _TRANSLATOR_AVAILABLE = True
except ImportError:
    _TRANSLATOR_AVAILABLE = False
    logger.warning("deep-translator not installed. pip install deep-translator to enable translation.")


# ── Common Hindi-to-transliteration mappings (Devanagari proper nouns) ────────
# Covers the most frequent names on Indian government documents.
# Used as a fast offline fallback before hitting Google Translate.
_HINDI_NAME_MAP: dict[str, str] = {
    "राम": "Ram",      "श्याम": "Shyam",    "रमेश": "Ramesh",
    "सुरेश": "Suresh", "महेश": "Mahesh",     "दिनेश": "Dinesh",
    "अर्जुन": "Arjun", "विजय": "Vijay",      "राजेश": "Rajesh",
    "अशोक": "Ashok",   "कमल": "Kamal",       "सुनील": "Sunil",
    "अनिल": "Anil",    "मनोज": "Manoj",      "संजय": "Sanjay",
    "राकेश": "Rakesh", "प्रकाश": "Prakash",  "विनोद": "Vinod",
    "सीता": "Sita",    "गीता": "Geeta",      "रीता": "Rita",
    "प्रिया": "Priya", "पूजा": "Pooja",      "अनीता": "Anita",
    "नीता": "Neeta",   "ममता": "Mamta",      "कमला": "Kamla",
    "देवी": "Devi",    "लक्ष्मी": "Lakshmi", "सरस्वती": "Saraswati",
    "कुमार": "Kumar",  "सिंह": "Singh",       "शर्मा": "Sharma",
    "वर्मा": "Verma",  "गुप्ता": "Gupta",    "जोशी": "Joshi",
    "पाण्डेय": "Pandey","तिवारी": "Tiwari",  "मिश्र": "Mishra",
    "यादव": "Yadav",   "पटेल": "Patel",      "चौधरी": "Chaudhary",
    "ठाकुर": "Thakur", "रावत": "Rawat",      "नेगी": "Negi",
    "बिष्ट": "Bisht",  "नायक": "Nayak",      "मेहता": "Mehta",
}

# Reverse map: English → Hindi
_ENGLISH_NAME_MAP: dict[str, str] = {v.lower(): k for k, v in _HINDI_NAME_MAP.items()}


def _contains_devanagari(text: str) -> bool:
    """Returns True if the text contains Hindi/Devanagari Unicode characters."""
    return bool(re.search(r'[\u0900-\u097F]', text))


def _fast_transliterate_hindi_to_english(text: str) -> str:
    """
    Offline word-by-word transliteration using the built-in name map.
    Returns the (partially) transliterated text; words not in the map are kept as-is.
    """
    words = text.strip().split()
    translated = []
    for w in words:
        translated.append(_HINDI_NAME_MAP.get(w, w))
    return " ".join(translated)


def _fast_transliterate_english_to_hindi(text: str) -> str:
    """
    Offline word-by-word transliteration using the reverse name map.
    """
    words = text.strip().split()
    translated = []
    for w in words:
        translated.append(_ENGLISH_NAME_MAP.get(w.lower(), w))
    return " ".join(translated)


def translate_to_english(text: str) -> dict:
    """
    Translates Hindi text to English.
    Cascade: fast offline map → Google Translate → original (fallback).
    Returns {
        "original":   <input text>,
        "translated": <English text>,
        "method":     "offline_map" | "google_translate" | "passthrough"
    }
    """
    if not text or not text.strip():
        return {"original": text, "translated": text, "method": "passthrough"}

    if not _contains_devanagari(text):
        return {"original": text, "translated": text, "method": "passthrough"}

    # Step 1: Try fast offline map
    fast_result = _fast_transliterate_hindi_to_english(text)
    still_has_devanagari = _contains_devanagari(fast_result)

    if not still_has_devanagari:
        return {"original": text, "translated": fast_result.title(), "method": "offline_map"}

    # Step 2: Google Translate (online)
    if _TRANSLATOR_AVAILABLE:
        try:
            result = GoogleTranslator(source='hi', target='en').translate(text)
            return {"original": text, "translated": result, "method": "google_translate"}
        except Exception as e:
            logger.warning(f"Google Translate failed: {e}. Falling back to partial map.")

    # Step 3: Return partial offline result
    return {"original": text, "translated": fast_result, "method": "offline_map_partial"}


def translate_to_hindi(text: str) -> dict:
    """
    Translates English text to Hindi (Devanagari).
    Cascade: fast offline map → Google Translate → original (fallback).
    """
    if not text or not text.strip():
        return {"original": text, "translated": text, "method": "passthrough"}

    if _contains_devanagari(text):
        return {"original": text, "translated": text, "method": "passthrough"}

    # Step 1: Fast offline map
    fast_result = _fast_transliterate_english_to_hindi(text)
    changed = fast_result != text
    if changed and _contains_devanagari(fast_result):
        return {"original": text, "translated": fast_result, "method": "offline_map"}

    # Step 2: Google Translate
    if _TRANSLATOR_AVAILABLE:
        try:
            result = GoogleTranslator(source='en', target='hi').translate(text)
            return {"original": text, "translated": result, "method": "google_translate"}
        except Exception as e:
            logger.warning(f"Google Translate failed: {e}.")

    return {"original": text, "translated": text, "method": "passthrough"}


def enrich_name_with_translations(name: str) -> dict:
    """
    Given a name in either English or Hindi, returns:
    {
        "original": <name>,
        "english":  <English version>,
        "hindi":    <Hindi version>,
        "script":   "latin" | "devanagari"
    }
    """
    if not name:
        return {"original": name, "english": name, "hindi": name, "script": "unknown"}

    is_hindi = _contains_devanagari(name)

    if is_hindi:
        en_result = translate_to_english(name)
        return {
            "original": name,
            "english":  en_result["translated"],
            "hindi":    name,
            "script":   "devanagari",
            "method":   en_result["method"]
        }
    else:
        hi_result = translate_to_hindi(name)
        return {
            "original": name,
            "english":  name,
            "hindi":    hi_result["translated"],
            "script":   "latin",
            "method":   hi_result["method"]
        }
