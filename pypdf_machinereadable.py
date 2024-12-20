#works only for machine readble pdfs


import PyPDF2
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from tabula import read_pdf
import cv2
import numpy as np

def is_scanned_pdf(pdf_path):
    """
    Determines if a PDF is scanned or machine-readable.
    
    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        bool: True if the PDF is likely scanned, False otherwise.
    """
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            
            # Check if text can be extracted
            extracted_text = ''
            for page in reader.pages:
                extracted_text += page.extract_text() or ''
            
            if extracted_text.strip():
                return False  # Text present, likely machine-readable
            
            # Check for images using pdfplumber
            with pdfplumber.open(pdf_path) as plumber_pdf:
                for page in plumber_pdf.pages:
                    if page.images:
                        return True  # Images present, likely scanned
            
            # If no text and no images, it's ambiguous
            return True  # Default to scanned for safety
    except Exception as e:
        print(f"Error analyzing PDF: {e}")
        return False  # Assume unscanned on error for robustness

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF. Handles both machine-readable and scanned PDFs.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Extracted text from the PDF.
    """
    try:
        # Try to extract text directly (for machine-readable PDFs)
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + '\n'
        
        if text.strip():
            return text.strip()  # Return text if successfully extracted

    except Exception as e:
        return f"Error: {e}"

def extract_tables_from_pdf(pdf_path):
    # Open the PDF file
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # Extract tables from the page
            tables = page.extract_tables()
            if(tables):
                for table in tables:
                    print(table)
            else:
                print("No tables found in the PDF.")


def preprocess_image(image):
    """Preprocess image for better OCR accuracy."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)  # Binarize
    return binary

def extract_text_from_scanned_pdf(pdf_path):
    """Extract text from a scanned PDF."""
    pages = convert_from_path(pdf_path)
    full_text = ""
    for page in pages:
        # Convert PIL image to NumPy array for preprocessing
        np_image = np.array(page)
        processed_image = preprocess_image(np_image)
        text = pytesseract.image_to_string(processed_image)
        full_text += text + "\n"
    return full_text

def extract_tables_from_scanned_pdf(pdf_path):
    """Extract tables from a scanned PDF using Tesseract."""
    pages = convert_from_path(pdf_path)
    tables = []
    for page in pages:
        np_image = np.array(page)
        processed_image = preprocess_image(np_image)
        data = pytesseract.image_to_string(processed_image, config='--psm 6')  # Table mode
        tables.append(data)
    return tables



# Example usage
if __name__ == "__main__":
    pdf_file_path = "text_extraction/output_maskrcnn_original.pdf"  # Replace with your PDF path
    
    if(is_scanned_pdf(pdf_file_path)):
        extracted_text = extract_text_from_scanned_pdf(pdf_file_path)
        extracted_tables = extract_tables_from_scanned_pdf(pdf_file_path)
        print("Extracted Text:", extracted_text)
        print("Extracted Tables:", extracted_tables)
    else:
        extracted_text = extract_text_from_pdf(pdf_file_path)
        print("Extracted Text:", extracted_text)
        print("Extracted Tables:")
        extract_tables_from_pdf(pdf_file_path)
        
