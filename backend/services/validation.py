import re
from rapidfuzz import fuzz


# ── Date normalisation helpers ────────────────────────────────────────────────
_DATE_FORMATS = [
    r"(\d{2})[/\-\.](\d{2})[/\-\.](\d{4})",   # DD/MM/YYYY  or  DD-MM-YYYY
    r"(\d{4})[/\-\.](\d{2})[/\-\.](\d{2})",   # YYYY/MM/DD  or  YYYY-MM-DD
    r"(\d{2})\s+(\w{3,9})\s+(\d{4})",          # 01 January 2000
]
_MONTHS = {
    "jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
    "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12,
    "january":1,"february":2,"march":3,"april":4,"june":6,
    "july":7,"august":8,"september":9,"october":10,"november":11,"december":12,
}


def _parse_date_to_tuple(s: str):
    """
    Return a (day, month, year) int-tuple from common date formats,
    or None if parsing fails.
    """
    s = s.strip()
    # DD/MM/YYYY or DD-MM-YYYY
    m = re.fullmatch(r"(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{4})", s)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    # YYYY/MM/DD
    m = re.fullmatch(r"(\d{4})[/\-\.](\d{1,2})[/\-\.](\d{1,2})", s)
    if m:
        return int(m.group(3)), int(m.group(2)), int(m.group(1))
    # "01 January 2000" or "01 Jan 2000"
    m = re.fullmatch(r"(\d{1,2})\s+(\w+)\s+(\d{4})", s, re.IGNORECASE)
    if m:
        mon = _MONTHS.get(m.group(2).lower()[:3])
        if mon:
            return int(m.group(1)), mon, int(m.group(3))
    return None


def _dates_match(expected: str, extracted: str) -> float:
    """
    Returns 0.0 – 100.0 based on how well two date strings match.
    Handles format differences (DD/MM/YYYY vs YYYY-MM-DD etc.)
    """
    t1 = _parse_date_to_tuple(expected)
    t2 = _parse_date_to_tuple(extracted)
    if t1 and t2:
        if t1 == t2:
            return 100.0
        # Same day+month, different year or vice versa → partial
        if t1[0] == t2[0] and t1[1] == t2[1]:
            return 60.0
        if t1[2] == t2[2]:   # same year
            return 40.0
        return 0.0
    # Fallback: normalise separators and exact
    e = re.sub(r"[/\-\.]", "", expected.strip())
    x = re.sub(r"[/\-\.]", "", extracted.strip())
    return 100.0 if e == x else 0.0


def validate_data(extracted_data: dict, application_metadata: dict) -> dict:
    """
    Validates extracted OCR fields against application metadata.
    Returns a structured validation_results dict with per-field status
    and an overall_status summary.
    """
    validation_results: dict = {
        "fields": {},
        "overall_status": "failed"
    }

    total = 0
    matched = 0

    # ── Aadhaar (exact 12-digit match) ───────────────────────────────────────
    if application_metadata.get("aadhaar_number"):
        total += 1
        expected = re.sub(r"\s", "", str(application_metadata["aadhaar_number"]))
        extracted = re.sub(r"\s", "", str(extracted_data.get("aadhaar_number", {}).get("value", "") or ""))
        if extracted and expected == extracted:
            validation_results["fields"]["aadhaar_number"] = {"status": "matched",    "score": 100.0}
            matched += 1
        elif extracted and expected[-4:] == extracted[-4:]:   # last 4 digits match
            validation_results["fields"]["aadhaar_number"] = {"status": "partial_match", "score": 50.0}
        else:
            validation_results["fields"]["aadhaar_number"] = {"status": "mismatched",   "score":   0.0}

    # ── PAN (exact match, case-insensitive) ──────────────────────────────────
    if application_metadata.get("pan_number"):
        total += 1
        expected = str(application_metadata["pan_number"]).strip().upper()
        extracted = str(extracted_data.get("pan_number", {}).get("value", "") or "").strip().upper()
        if extracted and expected == extracted:
            validation_results["fields"]["pan_number"] = {"status": "matched",    "score": 100.0}
            matched += 1
        elif extracted and len(extracted) == 10 and extracted[:5] == expected[:5]:
            # Same first 5 chars → likely same person, OCR error in digits
            validation_results["fields"]["pan_number"] = {"status": "partial_match", "score": 60.0}
        else:
            validation_results["fields"]["pan_number"] = {"status": "mismatched",   "score":   0.0}

    # ── Name (fuzzy, with Hindi→English translation) ─────────────────────────
    if application_metadata.get("name"):
        total += 1
        expected_name = application_metadata["name"].lower().strip()
        raw_text = extracted_data.get("raw_text", "")

        try:
            from services.translation import translate_to_english, _contains_devanagari
            if _contains_devanagari(raw_text):
                res = translate_to_english(raw_text)
                raw_for_match = res["translated"].lower()
            else:
                raw_for_match = raw_text.lower()
        except Exception:
            raw_for_match = raw_text.lower()

        score = fuzz.partial_ratio(expected_name, raw_for_match)
        extracted_data["name"] = {
            "value":      application_metadata["name"],
            "confidence": round(score / 100.0, 2)
        }
        if score > 85:
            validation_results["fields"]["name"] = {"status": "matched",       "score": score}
            matched += 1
        elif score > 65:
            validation_results["fields"]["name"] = {"status": "partial_match", "score": score}
        else:
            validation_results["fields"]["name"] = {"status": "mismatched",    "score": score}

    # ── DOB (fuzzy-format-aware) ──────────────────────────────────────────────
    if application_metadata.get("dob"):
        total += 1
        expected_dob   = application_metadata["dob"]
        extracted_dob  = extracted_data.get("dob", {}).get("value", "") or ""
        dob_score = _dates_match(expected_dob, extracted_dob) if extracted_dob else 0.0

        if dob_score == 100.0:
            validation_results["fields"]["dob"] = {"status": "matched",       "score": 100.0}
            matched += 1
        elif dob_score >= 40.0:
            validation_results["fields"]["dob"] = {"status": "partial_match", "score": dob_score}
        else:
            validation_results["fields"]["dob"] = {"status": "mismatched",    "score":   0.0}

    # ── District (fuzzy, 2-token) ─────────────────────────────────────────────
    if application_metadata.get("district"):
        total += 1
        expected_dist = application_metadata["district"].lower().strip()
        raw_lower = extracted_data.get("raw_text", "").lower()
        score = fuzz.partial_ratio(expected_dist, raw_lower)
        if score > 85:
            validation_results["fields"]["district"] = {"status": "matched",       "score": score}
            matched += 1
        elif score > 65:
            validation_results["fields"]["district"] = {"status": "partial_match", "score": score}
        else:
            validation_results["fields"]["district"] = {"status": "mismatched",    "score": score}

    # ── Overall status ────────────────────────────────────────────────────────
    if total == 0:
        validation_results["overall_status"] = "manual_review_needed"
    elif matched == total:
        validation_results["overall_status"] = "verified"
    elif matched > 0:
        validation_results["overall_status"] = "partial_verification"
    else:
        validation_results["overall_status"] = "failed"

    # Aggregate confidence: weighted mean of per-field scores
    field_scores = [v["score"] for v in validation_results["fields"].values() if "score" in v]
    validation_results["aggregate_confidence"] = round(
        (sum(field_scores) / len(field_scores)) / 100.0, 3
    ) if field_scores else 0.0

    return validation_results
