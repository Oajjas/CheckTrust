import easyocr
import logging
from config import OCR_LANGUAGES, USE_GPU

logger = logging.getLogger(__name__)

# Initialize single reader instance to save time on subsequent runs
logger.info("Initializing EasyOCR reader. This may take a while on first run...")
reader = easyocr.Reader(OCR_LANGUAGES, gpu=USE_GPU)
logger.info("EasyOCR reader initialized.")

def extract_text(image_path_or_array) -> list:
    """
    Takes an image array or path and returns the extracted raw text segments.
    Each segment is a tuple: (bounding_box, text, confidence)
    We return merely a list of text strings with their overall confidence for later parsing.
    """
    try:
        # detail=1 means we get (bbox, text, prob)
        results = reader.readtext(image_path_or_array, detail=1)
        
        extracted_data = []
        for (bbox, text, prob) in results:
            if prob > 0.2: # Filter very low-confidence gibberish
                extracted_data.append({"text": text.strip(), "confidence": float(prob)})
                
        return extracted_data
    except Exception as e:
        logger.error(f"Error extracting text with EasyOCR: {str(e)}")
        raise e
