# CheckTrust — AI Document Verification Pipeline

> Intelligent document verification for Indian e-governance portals.  
> Hackathon demo · Python · FastAPI · EasyOCR

---

## Quick Start

```bash
# 1. Create and activate virtual environment
cd backend
python -m venv venv
source venv/bin/activate          # Windows: venv\\Scripts\\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env              # edit USE_GPU / DEBUG as needed

# 4. Start the server
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000** for the dashboard.  
Swagger docs at **http://localhost:8000/docs**.
