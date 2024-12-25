import os
import fitz  # PyMuPDF
from typing import Tuple
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('document_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def is_scanned_pdf(pdf_path: str) -> Tuple[bool, float]:

    try:
        doc = fitz.open(pdf_path)
        total_pages = doc.page_count
        text_chars = 0
        image_count = 0
        
        for page_num in range(total_pages):
            page = doc[page_num]
            
            text = page.get_text()
            text_chars += len(text)
            
            image_list = page.get_images()
            image_count += len(image_list)
        
        doc.close()
        
        chars_per_page = text_chars / total_pages if total_pages > 0 else 0
        images_per_page = image_count / total_pages if total_pages > 0 else 0
        
        is_scanned = chars_per_page < 100 and images_per_page > 0
        
        if chars_per_page < 50 and images_per_page > 0:
            confidence = 0.9  
        elif chars_per_page > 500:
            confidence = 0.9  
        else:
            confidence = 0.6 
            
        logger.info(f"PDF Analysis - Chars per page: {chars_per_page:.2f}, Images per page: {images_per_page:.2f}")
        logger.info(f"Scan detection result: {'Scanned' if is_scanned else 'Digital'} (confidence: {confidence:.2f})")
        
        return is_scanned
        
    except Exception as e:
        logger.error(f"Error analyzing PDF: {str(e)}")
        return False

if __name__ == "__main__":
    pdf_path = r"C:\Users\Srinjoy\OneDrive\Desktop\Appian\AppianChallenge\kutta_pdf.pdf"
    is_scanned = is_scanned_pdf(pdf_path)
    print(f"Is scanned: {is_scanned}")
