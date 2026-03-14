import re

# Noise words to exclude from name candidates
_NOISE_WORDS = {
    "government", "india", "republic", "department", "aadhaar", "aadhar",
    "unique", "identification", "authority", "income", "tax", "pan", "card",
    "certificate", "domicile", "caste", "address", "dob", "birth", "date",
    "male", "female", "other", "year", "years", "valid", "signature",
    "father", "mother", "husband", "wife", "son", "daughter", "guardian",
    "permanent", "account", "number", "enrollment", "enrolment", "uidai",
    "district", "state", "pincode", "pin", "tehsil", "village", "house",
    "door", "near", "post", "office", "phone", "mobile", "email", "age"
}

# Keywords (English + Hindi) that precede a name on the next token(s)
_NAME_LABELS = [
    r"name\s*[:\-]?\s*", r"नाम\s*[:\-]?\s*",
    r"applicant\s*name\s*[:\-]?\s*", r"holder\s*name\s*[:\-]?\s*",
    r"full\s*name\s*[:\-]?\s*", r"name\s+of\s+the\s+holder\s*[:\-]?\s*",
    r"nominated\s+name\s*[:\-]?\s*"
]

# Address label patterns
_ADDR_LABELS = [
    r"address\s*[:\-]?\s*",
    r"पता\s*[:\-]?\s*",
    r"addr\s*[:\-]?\s*",
    r"house\s*no\s*[:\-]?\s*",
    r"residential\s*address\s*[:\-]?\s*",
]


def _looks_like_name(text: str) -> bool:
    """
    Returns True if a string looks like a proper person name.
    Heuristic: 1-4 words, each 2+ chars, mostly alpha, no digits, not a noise word.
    """
    text = text.strip()
    if not text or len(text) < 3:
        return False
    words = text.split()
    if not (1 <= len(words) <= 4):
        return False
    for w in words:
        w_lower = w.lower()
        if len(w) < 2:
            return False
        if not re.match(r"^[A-Za-z\.'/-]+$", w):
            return False
        if w_lower in _NOISE_WORDS:
            return False
    return True


def _extract_name_from_ocr(ocr_data: list, full_text: str):
    """
    Two-pass approach to find an applicant name:
    Pass 1 – Look for a Name: label then grab the following text.
    Pass 2 – Scan each OCR box for a text segment that looks like a proper name.
    Returns (name_value, confidence) or (None, 0.0).
    """
    ordered = [(item["text"].strip(), item["confidence"]) for item in ocr_data]

    # ── PASS 1: label-based ─────────────────────────────────────────────────
    for label_pattern in _NAME_LABELS:
        for text, prob in ordered:
            m = re.search(label_pattern + r"(.+)", text, re.IGNORECASE)
            if m:
                candidate = m.group(1).strip()
                if _looks_like_name(candidate):
                    return candidate.title(), round(prob, 2)

        for i, (text, _) in enumerate(ordered[:-1]):
            if re.search(label_pattern, text, re.IGNORECASE):
                next_text, next_prob = ordered[i + 1]
                if _looks_like_name(next_text):
                    return next_text.title(), round(next_prob, 2)

    for label_pattern in _NAME_LABELS:
        m = re.search(label_pattern + r"([A-Za-z\s\.'/-]{3,40})", full_text, re.IGNORECASE)
        if m:
            candidate = m.group(1).strip()
            if _looks_like_name(candidate):
                return candidate.title(), 0.6

    # ── PASS 2: highest-confidence proper noun ─────────────────────────────
    best_candidate = None
    best_prob = 0.0
    for text, prob in ordered:
        text_clean = re.split(r"[,|;]", text)[0].strip()
        if _looks_like_name(text_clean) and prob > best_prob:
            best_candidate = text_clean.title()
            best_prob = prob

    if best_candidate:
        return best_candidate, round(best_prob * 0.75, 2)

    return None, 0.0


def _extract_address_from_ocr(ocr_data: list, full_text: str):
    """
    Three-pass address extractor:
    Pass 1 – Label-based (Address:, पता:) with continuation boxes.
    Pass 2 – Global regex on full_text.
    Pass 3 – Pincode fallback: grab text surrounding first 6-digit number.
    Returns (address_value, confidence) or (None, 0.0).
    """
    ordered = [(item["text"].strip(), item["confidence"]) for item in ocr_data]

    # Pass 1
    for lbl in _ADDR_LABELS:
        for i, (text, prob) in enumerate(ordered):
            m = re.search(lbl + r"(.+)", text, re.IGNORECASE)
            if m:
                candidate = m.group(1).strip()
                if len(candidate) < 6:
                    candidate = ""
                extra = []
                for j in range(i + 1, min(i + 5, len(ordered))):
                    nxt = ordered[j][0]
                    if any(kw in nxt.lower() for kw in ["name", "dob", "pan", "aadhaar", "date of birth"]):
                        break
                    extra.append(nxt)
                    if re.search(r"\d{6}", nxt):
                        break
                full_addr = ((candidate + " ") if candidate else "") + " ".join(extra)
                full_addr = full_addr.strip()
                if len(full_addr) >= 8:
                    return full_addr[:140], round(prob, 2)

    # Pass 2: combined label regex on full_text
    addr_pattern = r'(?:address|addr|पता)\s*[:\-]?\s*(.{10,120}?)(?=\s+(?:dob|date|aadhaar|pan|$)|\Z)'
    m = re.search(addr_pattern, full_text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()[:140], 0.5

    # Pass 3: pincode surroundings
    m = re.search(r"(.{15,80}?)\s*\b(\d{6})\b", full_text)
    if m:
        return (m.group(1) + " " + m.group(2)).strip(), 0.3

    return None, 0.0


def extract_fields(ocr_data: list) -> dict:
    """
    Extracts structured fields from unstructured OCR segments:
    - aadhaar_number
    - pan_number
    - dob
    - name  (two-pass heuristic)
    - address (three-pass heuristic)
    - raw_text
    """
    extracted_fields = {
        "aadhaar_number": {"value": None, "confidence": 0.0},
        "pan_number":     {"value": None, "confidence": 0.0},
        "dob":            {"value": None, "confidence": 0.0},
        "name":           {"value": None, "confidence": 0.0},
        "address":        {"value": None, "confidence": 0.0},
        "raw_text":       ""
    }

    if not ocr_data:
        return extracted_fields

    full_text = " ".join([item["text"] for item in ocr_data])
    extracted_fields["raw_text"] = full_text

    # ── Regex patterns ──────────────────────────────────────────────────────
    aadhaar_pattern = r'\b(\d{4}\s?\d{4}\s?\d{4})\b'
    pan_pattern     = r'\b([A-Z]{5}[0-9]{4}[A-Z]{1})\b'
    dob_pattern     = r'\b(\d{2}[/\-]\d{2}[/\-]\d{4})\b'

    def find_highest_confidence_match(pattern, data_list):
        best_match, best_prob = None, 0.0
        for item in data_list:
            text, prob = item["text"], item["confidence"]
            matches = re.findall(pattern, text)
            if matches and prob > best_prob:
                best_match = matches[0] if isinstance(matches[0], str) else "".join(matches[0])
                best_prob = prob
        if not best_match:
            global_matches = re.findall(pattern, full_text)
            if global_matches:
                best_match = global_matches[0]
                best_prob = 0.5
        return best_match, best_prob

    aadhaar_match, aadhaar_conf = find_highest_confidence_match(aadhaar_pattern, ocr_data)
    if aadhaar_match:
        extracted_fields["aadhaar_number"] = {
            "value": aadhaar_match.replace(" ", ""), "confidence": aadhaar_conf
        }

    pan_match, pan_conf = find_highest_confidence_match(pan_pattern, ocr_data)
    if pan_match:
        extracted_fields["pan_number"] = {"value": pan_match, "confidence": pan_conf}

    dob_match, dob_conf = find_highest_confidence_match(dob_pattern, ocr_data)
    if dob_match:
        extracted_fields["dob"] = {"value": dob_match, "confidence": dob_conf}

    name_value, name_conf = _extract_name_from_ocr(ocr_data, full_text)
    if name_value:
        extracted_fields["name"] = {"value": name_value, "confidence": name_conf}

    addr_value, addr_conf = _extract_address_from_ocr(ocr_data, full_text)
    if addr_value:
        extracted_fields["address"] = {"value": addr_value, "confidence": addr_conf}

    return extracted_fields
