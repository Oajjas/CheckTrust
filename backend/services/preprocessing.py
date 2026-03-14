import cv2
import numpy as np

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Applies standard preprocessing techniques to an image for better OCR accuracy.
    - Grayscale conversion
    - Denoising
    - Thresholding / Binarization
    """
    # Load image from bytes
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("Could not decode image bytes into numpy array")

    # 1. Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. Denoising
    denoised = cv2.fastNlMeansDenoising(gray, h=30)
    
    # 3. Increase Contrast (CLAHE setup)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast_img = clahe.apply(denoised)
    
    # 4. Binarization (Otsu's Thresholding)
    # _, binary_img = cv2.threshold(contrast_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Often, just passing the high-contrast grayscale image to EasyOCR works better
    # than aggressive binarization which might cut off thin fonts.
    return contrast_img

def deskew_image(image: np.ndarray) -> np.ndarray:
    """
    Detects document skew and rotates the image to correct it.
    """
    # Simple skew detection using edge detection and Hough lines
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    
    if lines is not None:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            angles.append(angle)
            
        median_angle = np.median(angles)
        
        # Only correct if the skew is significant but not 90 degrees layout
        if abs(median_angle) > 0.5 and abs(median_angle) < 45:
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return rotated
            
    return image
