import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
PROCESSED_DIR = BASE_DIR / "processed"

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Application Config
MAX_UPLOAD_SIZE_MB = 10
TARGET_IMAGE_SIZE_KB = 500

# Supported Formats
SUPPORTED_IMAGE_FORMATS = {"image/jpeg", "image/png", "image/jpg"}
SUPPORTED_DOCUMENT_FORMATS = SUPPORTED_IMAGE_FORMATS.union({"application/pdf"})

# OCR Config
OCR_LANGUAGES = ['en', 'hi']
USE_GPU = True # Set to True if testing on a machine with a compatible GPU

# Debug Mode
DEBUG = True
