import os
import time
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional

from services.compression import process_file_content
from services.preprocessing import preprocess_image
from services.ocr_engine import extract_text
from services.extraction import extract_fields
from services.validation import validate_data
from services.report_engine import generate_verification_report
from services.classifier import classify_document

app = FastAPI(title="GovDoc AI Pipeline", version="2.0.0")

templates = Jinja2Templates(directory="templates")

# ── In-memory store (hackathon prototype) ─────────────────────────────────────
PROCESSING_CACHE: dict = {}

# Global session impact counter
_SESSION_STATS = {"docs_processed": 0, "total_size_saved_mb": 0.0}


# ── Request Models ────────────────────────────────────────────────────────────
class VerifyRequest(BaseModel):
    file_id:        str
    name:           Optional[str] = None
    aadhaar_number: Optional[str] = None
    pan_number:     Optional[str] = None
    dob:            Optional[str] = None
    district:       Optional[str] = None   # used by income/domicile/birth certs


# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/stats")
async def get_session_stats():
    return _SESSION_STATS


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Receive PDF/Image, compress it, and cache it ready for OCR."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    start_time = time.time()
    file_bytes = await file.read()
    original_size_mb = len(file_bytes) / (1024 * 1024)

    try:
        compressed_images = process_file_content(file_bytes, file.filename)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    file_id      = f"file_{int(time.time() * 1000)}"
    new_size_mb  = sum(len(img) for img in compressed_images) / (1024 * 1024)
    saved_mb     = max(0.0, original_size_mb - new_size_mb)
    compression  = round((saved_mb / original_size_mb) * 100, 1) if original_size_mb > 0 else 0.0

    PROCESSING_CACHE[file_id] = {
        "images":         compressed_images,
        "original_size":  original_size_mb,
        "optimized_size": new_size_mb,
        "compression_pct": compression,
        "filename":       file.filename,
    }

    # Update global stats
    _SESSION_STATS["total_size_saved_mb"] = round(
        _SESSION_STATS["total_size_saved_mb"] + saved_mb, 2
    )

    return {
        "status":  "success",
        "file_id": file_id,
        "file_info": {
            "filename":           file.filename,
            "original_size_mb":   round(original_size_mb, 2),
            "optimized_size_mb":  round(new_size_mb, 2),
            "compression_pct":    compression,
            "pages":              len(compressed_images),
        },
        "time_taken": round(time.time() - start_time, 2),
    }


@app.post("/process-document")
async def process_document(file_id: str = Form(...)):
    """Run Preprocessing → OCR → Field Extraction → Classification on cached file."""
    if file_id not in PROCESSING_CACHE:
        raise HTTPException(status_code=404, detail="File ID not found. Upload first.")

    start_time = time.time()
    data       = PROCESSING_CACHE[file_id]

    all_text: list = []
    for img_bytes in data["images"]:
        preprocessed = preprocess_image(img_bytes)
        text_data    = extract_text(preprocessed)
        all_text.extend(text_data)

    # ── Classify BEFORE extraction so type context is available ──────────
    classification = classify_document(all_text)

    # ── Extract structured fields ──────────────────────────────────────
    extracted_fields = extract_fields(all_text)
    raw_text         = extracted_fields.get("raw_text", "")

    # Cache for verification step
    PROCESSING_CACHE[file_id]["extracted_fields"] = extracted_fields
    PROCESSING_CACHE[file_id]["classification"]   = classification

    return {
        "status":  "success",
        "file_id": file_id,
        "document_type": {
            "type":       classification.doc_type,
            "label":      classification.doc_label,
            "confidence": classification.confidence,
            "signals":    classification.signals,
        },
        "extracted_fields": {k: v for k, v in extracted_fields.items() if k != "raw_text"},
        "raw_text":  raw_text,
        "time_taken": round(time.time() - start_time, 2),
    }


@app.post("/verify-application")
async def verify_application(request: VerifyRequest):
    """Validate OCR-extracted fields against submitted application metadata."""
    start_time = time.time()

    if request.file_id not in PROCESSING_CACHE:
        raise HTTPException(status_code=404, detail="File ID not found.")

    data = PROCESSING_CACHE[request.file_id]

    if "extracted_fields" not in data:
        raise HTTPException(status_code=400, detail="Run /process-document first.")

    extracted_fields = data["extracted_fields"]
    classification   = data.get("classification")
    name_translations = data.get("name_translations")

    metadata = {
        "name":           request.name,
        "aadhaar_number": request.aadhaar_number,
        "pan_number":     request.pan_number,
        "dob":            request.dob,
        "district":       request.district,
    }

    validation_results = validate_data(extracted_fields, metadata)

    orig_mb  = data["original_size"]
    opt_mb   = data["optimized_size"]
    comp_pct = data.get("compression_pct", 0.0)

    # Build richer file info
    file_info = {
        "filename":          data["filename"],
        "original_size_mb":  round(orig_mb, 2),
        "optimized_size_mb": round(opt_mb, 2),
        "compression_pct":   comp_pct,
    }

    report = generate_verification_report(
        file_info=file_info,
        extracted_fields=extracted_fields,
        validation_results=validation_results,
        processing_time=(time.time() - start_time),
    )

    # \u2500\u2500 Enrich report with classification \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    report_data = report.get("data", report)
    report_data["document_type"]   = {
        "type":       classification.doc_type  if classification else "unknown",
        "label":      classification.doc_label if classification else "Unknown",
        "confidence": classification.confidence if classification else 0.0,
    }
    report_data["compression_pct"] = comp_pct

    # \u2500\u2500 Update session impact counter \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    _SESSION_STATS["docs_processed"] += 1

    # Free cache
    del PROCESSING_CACHE[request.file_id]

    return report


# \u2550\u2550 SINGLE-CALL ENDPOINT \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
@app.post("/verify-documents")
async def verify_documents_single_call(
    file:           UploadFile      = File(...),
    name:           Optional[str]   = Form(None),
    aadhaar_number: Optional[str]   = Form(None),
    pan_number:     Optional[str]   = Form(None),
    dob:            Optional[str]   = Form(None),
    district:       Optional[str]   = Form(None),
):
    """
    All-in-one endpoint: Upload + Compress + OCR + Classify + Extract + Validate.
    Returns the canonical API spec response:
    {
      "documents": [{...}],
      "processing_time_seconds": 3.2
    }
    Suitable for external integrations that don\u2019t want the 3-step flow.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    pipeline_start = time.time()

    # \u2500\u2500 1. Compress \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    file_bytes = await file.read()
    orig_mb = len(file_bytes) / (1024 * 1024)

    try:
        compressed_images = process_file_content(file_bytes, file.filename)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    opt_mb   = sum(len(img) for img in compressed_images) / (1024 * 1024)
    comp_pct = round((max(0, orig_mb - opt_mb) / orig_mb) * 100, 1) if orig_mb > 0 else 0.0

    # \u2500\u2500 2. Preprocess + OCR \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    all_text: list = []
    for img_bytes in compressed_images:
        preprocessed = preprocess_image(img_bytes)
        all_text.extend(extract_text(preprocessed))

    # \u2500\u2500 3. Classify + Extract \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    classification   = classify_document(all_text)
    extracted_fields = extract_fields(all_text)

    # \u2500\u2500 4. Validate \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    metadata = {
        "name":           name,
        "aadhaar_number": aadhaar_number,
        "pan_number":     pan_number,
        "dob":            dob,
        "district":       district,
    }
    validation_results = validate_data(extracted_fields, metadata)

    file_info = {
        "filename":          file.filename,
        "original_size_mb":  round(orig_mb, 2),
        "optimized_size_mb": round(opt_mb, 2),
        "compression_pct":   comp_pct,
    }

    # \u2500\u2500 5. Report \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    report = generate_verification_report(
        file_info=file_info,
        extracted_fields=extracted_fields,
        validation_results=validation_results,
        processing_time=time.time() - pipeline_start,
    )
    report.get("data", {})["document_type"] = {
        "type":       classification.doc_type,
        "label":      classification.doc_label,
        "confidence": classification.confidence,
    }
    report.get("data", {})["compression_pct"] = comp_pct

    _SESSION_STATS["docs_processed"] += 1
    _SESSION_STATS["total_size_saved_mb"] = round(
        _SESSION_STATS["total_size_saved_mb"] + max(0, orig_mb - opt_mb), 2
    )

    return report

