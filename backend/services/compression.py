import os
from io import BytesIO
from PIL import Image
from pdf2image import convert_from_bytes
import logging

logger = logging.getLogger(__name__)

def compress_image(image: Image.Image, max_size_kb: int = 500) -> bytes:
    """
    Compresses a PIL Image iteratively until its size is below max_size_kb or quality gets too low.
    """
    quality = 90
    output = BytesIO()
    
    # Ensure image is in RGB for JPEG saving
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
        
    image.save(output, format="JPEG", quality=quality)
    
    while output.tell() > max_size_kb * 1024 and quality > 30:
        quality -= 10
        output = BytesIO()
        image.save(output, format="JPEG", quality=quality)
        
    return output.getvalue()

def process_file_content(file_bytes: bytes, filename: str) -> list[bytes]:
    """
    Processes the uploaded file bytes.
    If PDF, converts to a list of image bytes.
    If Image, directly compresses it.
    Returns a list of JPEG image bytes (one for each page/image).
    """
    ext = os.path.splitext(filename)[1].lower()
    processed_images = []
    
    try:
        if ext == ".pdf":
            # Convert PDF to list of PIL images
            images = convert_from_bytes(file_bytes)
            for img in images:
                compressed_bytes = compress_image(img)
                processed_images.append(compressed_bytes)
        elif ext in [".jpg", ".jpeg", ".png"]:
            img = Image.open(BytesIO(file_bytes))
            compressed_bytes = compress_image(img)
            processed_images.append(compressed_bytes)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
            
    except Exception as e:
        logger.error(f"Error processing file {filename}: {str(e)}")
        raise e
        
    return processed_images
