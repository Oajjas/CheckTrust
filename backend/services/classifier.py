"""
Lightweight document classifier based on keyword scoring.
Runs on the raw OCR text BEFORE field extraction to determine document type.
No ML models required — pure heuristic keyword/pattern matching.

Supported document types (18 total):
  Identity:    aadhaar, pan, voter_id, passport, driving_license
  Certificates:income_certificate, domicile_certificate, caste_certificate,
               birth_certificate, death_certificate, marriage_certificate,
               character_certificate, medical_certificate, disability_certificate
  Property:    land_record
  Financial:   ration_card, bank_passbook
  Education:   mark_sheet
"""
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ClassificationResult:
    doc_type:   str         # e.g. "aadhaar", "pan", "voter_id"
    doc_label:  str         # Human-readable label
    confidence: float       # 0.0 – 1.0
    signals:    list[str]   # Which patterns triggered


# ── Signature table ────────────────────────────────────────────────────────────
# Format: (doc_type, human_label, [(regex_pattern, weight), ...])
# Patterns are case-insensitive. Higher weight = stronger signal.
_SIGNATURES: list[tuple[str, str, list[tuple[str, float]]]] = [

    # ── IDENTITY ────────────────────────────────────────────────────────────
    (
        "aadhaar", "Aadhaar Card",
        [
            (r"\baadhaar\b",                    4.0),
            (r"\baadhar\b",                     3.0),
            (r"\buidai\b",                      4.0),
            (r"unique\s+identification",         3.0),
            (r"enrollment\s*no",                2.0),
            (r"enrolment\s*no",                 2.0),
            (r"\d{4}\s\d{4}\s\d{4}",            2.5),  # 12-digit Aadhaar pattern
            (r"\bvid\b",                         1.5),  # Virtual ID
            (r"आधार",                            4.0),  # Hindi Aadhaar
            (r"भारतीय\s*विशिष्ट\s*पहचान",        3.0),  # Hindi UIDAI
        ]
    ),
    (
        "pan", "PAN Card",
        [
            (r"\bpan\b",                         3.0),
            (r"permanent\s+account\s+number",    5.0),
            (r"income\s+tax\s+department",        4.0),
            (r"[A-Z]{5}[0-9]{4}[A-Z]",           3.0),  # PAN regex
            (r"assessment\s+year",               2.0),
            (r"\bpancard\b",                     4.0),
            (r"स्थायी\s*खाता\s*संख्या",           4.0),  # Hindi PAN
        ]
    ),
    (
        "voter_id", "Voter ID Card (EPIC)",
        [
            (r"\bepic\b",                        4.0),
            (r"election\s+commission",           4.5),
            (r"voter\s*(id|card|identity)",      4.5),
            (r"electoral\s+roll",                3.5),
            (r"electors?\s+photo",               3.5),
            (r"\bdelimitation\b",                2.0),
            (r"मतदाता\s*पहचान",                  4.0),  # Hindi Voter ID
            (r"निर्वाचन\s*आयोग",                 4.0),  # Hindi Election Commission
        ]
    ),
    (
        "passport", "Passport",
        [
            (r"\bpassport\b",                    5.0),
            (r"republic\s+of\s+india",           2.0),
            (r"ministry\s+of\s+external\s+affairs", 4.0),
            (r"\btype\b.*\bpassport\b",           3.0),
            (r"\bmrz\b",                         3.0),  # Machine Readable Zone
            (r"passport\s+no",                   4.0),
            (r"पासपोर्ट",                         5.0),  # Hindi Passport
            (r"nationality.*indian",             2.0),
        ]
    ),
    (
        "driving_license", "Driving Licence",
        [
            (r"driving\s+licen[sc]e",            5.0),
            (r"\bdl\s*no\b",                     4.0),
            (r"\bdl\b",                          2.0),
            (r"transport\s+department",          3.0),
            (r"motor\s+vehicle",                 3.0),
            (r"validity",                        1.0),
            (r"ड्राइविंग\s*लाइसेंस",             5.0),  # Hindi DL
            (r"वाहन\s*चालन",                     3.5),
            (r"class\s+of\s+vehicle",            3.0),
        ]
    ),

    # ── CERTIFICATES ────────────────────────────────────────────────────────
    (
        "income_certificate", "Income Certificate",
        [
            (r"income\s+certificate",            5.0),
            (r"annual\s+income",                 3.5),
            (r"family\s+income",                 3.0),
            (r"tahsildar|tehsildar",             3.0),
            (r"patwari",                         2.5),
            (r"block\s+development\s+officer",   2.5),
            (r"\bbdo\b",                         1.5),
            (r"rupees?\s+per\s+annum",           3.0),
            (r"आय\s*प्रमाण\s*पत्र",              5.0),  # Hindi income cert
            (r"वार्षिक\s*आय",                    3.5),
        ]
    ),
    (
        "domicile_certificate", "Domicile Certificate",
        [
            (r"domicile",                        5.0),
            (r"permanent\s+resident",            3.5),
            (r"resident\s+certificate",          4.0),
            (r"\binhabitant\b",                  2.5),
            (r"bond\s+of\s+domicile",            3.0),
            (r"state\s+of\s+uttarakhand",        2.0),
            (r"state\s+of\s+himachal",           2.0),
            (r"state\s+of\s+uttar\s+pradesh",    2.0),
            (r"मूल\s*निवास\s*प्रमाण",             5.0),  # Hindi domicile
            (r"स्थायी\s*निवासी",                  3.5),
        ]
    ),
    (
        "caste_certificate", "Caste / OBC Certificate",
        [
            (r"caste\s+certificate",             5.0),
            (r"scheduled\s+caste",               4.0),
            (r"scheduled\s+tribe",               4.0),
            (r"\bsc\b",                          1.5),
            (r"\bst\b",                          1.5),
            (r"\bobc\b",                         4.0),
            (r"other\s+backward\s+class",        4.0),
            (r"backwardness",                    2.5),
            (r"sub\s+caste",                     2.5),
            (r"जाति\s*प्रमाण\s*पत्र",             5.0),  # Hindi caste cert
            (r"अनुसूचित\s*जाति",                  4.0),
        ]
    ),
    (
        "birth_certificate", "Birth Certificate",
        [
            (r"birth\s+certificate",             5.0),
            (r"place\s+of\s+birth",              3.0),
            (r"municipal\s+corporation",          2.5),
            (r"registrar\s+of\s+births",         4.5),
            (r"born\s+(on|at)",                  2.5),
            (r"time\s+of\s+birth",               3.0),
            (r"name\s+of\s+hospital",            2.0),
            (r"जन्म\s*प्रमाण\s*पत्र",             5.0),  # Hindi birth cert
            (r"जन्म\s*स्थान",                     3.0),
        ]
    ),
    (
        "death_certificate", "Death Certificate",
        [
            (r"death\s+certificate",             5.0),
            (r"date\s+of\s+death",               4.0),
            (r"cause\s+of\s+death",              4.5),
            (r"deceased",                        3.0),
            (r"registrar\s+of\s+deaths",         4.5),
            (r"मृत्यु\s*प्रमाण\s*पत्र",           5.0),  # Hindi death cert
            (r"मृत्यु\s*की\s*तारीख",              4.0),
        ]
    ),
    (
        "marriage_certificate", "Marriage Certificate",
        [
            (r"marriage\s+certificate",          5.0),
            (r"registration\s+of\s+marriage",    4.5),
            (r"bridegroom|bride\b",              3.5),
            (r"husband|wife",                    2.0),
            (r"matrimonial",                     3.5),
            (r"solemnized",                      3.0),
            (r"विवाह\s*प्रमाण\s*पत्र",            5.0),  # Hindi marriage cert
            (r"वर\s*वधु",                         3.5),
        ]
    ),
    (
        "character_certificate", "Character / Conduct Certificate",
        [
            (r"character\s+certificate",         5.0),
            (r"conduct\s+certificate",           5.0),
            (r"good\s+conduct",                  3.5),
            (r"no\s+criminal\s+record",          4.0),
            (r"police\s+verification",           3.5),
            (r"चरित्र\s*प्रमाण\s*पत्र",           5.0),  # Hindi character cert
        ]
    ),
    (
        "medical_certificate", "Medical Certificate",
        [
            (r"medical\s+certificate",           5.0),
            (r"fit\s+(for|to)\s+work",           3.5),
            (r"hospital|clinic",                 1.5),
            (r"registered\s+medical\s*practitioner", 4.0),
            (r"\brmp\b",                         2.0),
            (r"chief\s+medical\s+officer",       3.5),
            (r"\bcmo\b",                         2.0),
            (r"चिकित्सा\s*प्रमाण\s*पत्र",         5.0),  # Hindi medical cert
        ]
    ),
    (
        "disability_certificate", "Disability Certificate",
        [
            (r"disability\s+certificate",        5.0),
            (r"person\s+with\s+disability",      4.0),
            (r"\bpwd\b",                         3.5),
            (r"physically\s+handicapped",        4.0),
            (r"percentage\s+of\s+disability",    4.5),
            (r"दिव्यांग\s*प्रमाण\s*पत्र",         5.0),  # Hindi disability cert
            (r"विकलांगता\s*प्रतिशत",              4.5),
        ]
    ),

    # ── PROPERTY & LAND ─────────────────────────────────────────────────────
    (
        "land_record", "Land Record (Khasra / Khatauni)",
        [
            (r"khasra",                          4.5),
            (r"khatauni",                        4.5),
            (r"khata\s*no",                      3.5),
            (r"jamabandi",                       4.0),
            (r"revenue\s+record",                3.5),
            (r"patwari|lekhpal",                 3.0),
            (r"tehsil",                          2.0),
            (r"agricultural\s+land",             2.5),
            (r"खसरा",                             5.0),  # Hindi Khasra
            (r"खतौनी",                            5.0),  # Hindi Khatauni
        ]
    ),

    # ── FINANCIAL ──────────────────────────────────────────────────────────
    (
        "ration_card", "Ration Card",
        [
            (r"ration\s+card",                   5.0),
            (r"public\s+distribution\s+system",  4.0),
            (r"\bpds\b",                         3.5),
            (r"food\s+supply\s+department",      4.0),
            (r"below\s+poverty\s*line",          3.0),
            (r"\bbpl\b",                         3.0),
            (r"\bapl\b",                         2.5),
            (r"fair\s+price\s+shop",             3.5),
            (r"राशन\s*कार्ड",                     5.0),  # Hindi ration card
            (r"सार्वजनिक\s*वितरण",                4.0),
        ]
    ),
    (
        "bank_passbook", "Bank Passbook / Statement",
        [
            (r"bank\s+passbook",                 5.0),
            (r"account\s+(no|number)",           2.5),
            (r"ifsc\s*code",                     4.0),
            (r"branch\s+code",                   2.5),
            (r"savings\s+account",               3.0),
            (r"current\s+account",               3.0),
            (r"transaction\s+details?",          3.0),
            (r"balance",                         1.5),
            (r"\bpassbook\b",                    5.0),
            (r"बैंक\s*पासबुक",                    5.0),  # Hindi bank passbook
        ]
    ),

    # ── EDUCATION ──────────────────────────────────────────────────────────
    (
        "mark_sheet", "Mark Sheet / Certificate",
        [
            (r"mark\s*sheet",                    5.0),
            (r"statement\s+of\s+marks",          4.5),
            (r"certificate\s+of\s+(passing|merit)", 4.0),
            (r"board\s+of\s+education",          3.5),
            (r"\bcbse\b",                         4.0),
            (r"\bicse\b",                         4.0),
            (r"\bubse\b",                         3.5),
            (r"roll\s*(no|number)",              2.0),
            (r"aggregate\s+(marks|percentage)",  3.5),
            (r"अंक\s*पत्र",                       5.0),  # Hindi mark sheet
        ]
    ),
]


def classify_document(ocr_data: list, raw_text: Optional[str] = None) -> ClassificationResult:
    """
    Accepts OCR segment list (or pre-joined raw_text) and
    returns the best-matching document type with confidence and matched signals.
    """
    text = (raw_text or " ".join(seg["text"] for seg in ocr_data))

    scores: dict[str, float] = {}
    signals_map: dict[str, list[str]] = {}

    for doc_type, _doc_label, patterns in _SIGNATURES:
        total_weight  = sum(w for _, w in patterns)
        matched_score = 0.0
        matched_sigs: list[str] = []

        for pattern, weight in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                matched_score += weight
                # Clean the regex pattern for display
                clean = re.sub(r'[\\()?+*^${\[].*?[}\]]', '', pattern).strip()
                matched_sigs.append(clean or pattern[:30])

        if matched_score > 0:
            scores[doc_type] = min(matched_score / total_weight, 1.0)
            signals_map[doc_type] = matched_sigs

    if not scores:
        return ClassificationResult(
            doc_type="unknown",
            doc_label="Unknown Document",
            confidence=0.0,
            signals=[]
        )

    best_type  = max(scores, key=lambda k: scores[k])
    best_score = scores[best_type]
    label_map  = {dt: lbl for dt, lbl, _ in _SIGNATURES}

    return ClassificationResult(
        doc_type=best_type,
        doc_label=label_map.get(best_type, best_type.replace("_", " ").title()),
        confidence=round(best_score, 3),
        signals=signals_map[best_type]
    )
