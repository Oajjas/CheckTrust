# CheckTrust — AI Document Verification Pipeline

> Intelligent document verification for Indian e-governance portals.  
> Hackathon demo · Python · FastAPI · EasyOCR

---

## Quick Start

```bash
# 1. Create and activate virtual environment
cd backend
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env              # edit USE_GPU / DEBUG as needed

# 4. Start the server
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000** for the dashboard.  
Swagger docs at **http://localhost:8000/docs**.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Dashboard UI |
| GET | `/health` | System health check |
| POST | `/warmup` | Pre-load OCR model (run before demo) |
| GET | `/stats` | Session impact counters |
| POST | `/upload` | Compress & cache document |
| POST | `/process-document` | OCR + classify + extract fields |
| POST | `/verify-application` | Validate extracted fields vs metadata |
| **POST** | **`/verify-documents`** | **Single-call full pipeline** |

### Single-call example

```bash
curl -X POST http://localhost:8000/verify-documents \
  -F 'file=@/path/to/aadhaar.jpg' \
  -F 'name=Ramesh Kumar' \
  -F 'aadhaar_number=1234 5678 9012' \
  -F 'dob=15/06/1985'
```

---

## Pipeline

```
Upload → Compress (Pillow) → Deskew/Preprocess (OpenCV) → OCR (EasyOCR)
       → Classify (18 doc types) → Extract fields → Validate (RapidFuzz)
       → Report (JSON)
```

### Key features
- **Compression** — up to 94% file size reduction (JPEG/PNG/PDF)
- **Auto-deskew** — corrects camera/scan tilt before OCR
- **Bilingual OCR** — English + Hindi Devanagari
- **18 document types** — Aadhaar, PAN, Voter ID, Passport, Driving Licence, Income/Domicile/Caste/Birth/Death/Marriage Certificates, Land Record, Ration Card, and more
- **Hindi↔English name matching** — user types English; Hindi in document is translated silently before validation
- **Fuzzy DOB matching** — `15/06/1985`, `1985-06-15`, `15 June 1985` all match

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_GPU` | `false` | Enable CUDA GPU for EasyOCR |
| `DEBUG` | `false` | Verbose logging |
| `MAX_UPLOAD_SIZE_MB` | `10` | File size limit |

---

## Generate Synthetic Test Documents

```bash
pip install faker
cd ..           # Blockathon/ root
python aadhar.py
# → synthetic_documents/   (50 test images, Aadhaar/PAN/certs)
# → synthetic_documents/metadata.json   (ground-truth field values)
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.10+ / FastAPI |
| OCR | EasyOCR (EN + HI) |
| Image processing | OpenCV, Pillow |
| Validation | RapidFuzz |
| Translation | deep-translator (Google Translate, no API key) |
| Server | Uvicorn |
