import PyPDF2
import pdfplumber

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
                print(extracted_text)
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

# Example usage
if __name__ == "__main__":
    pdf_file_path = "file_name.pdf"  # Replace with your PDF path
    is_scanned = is_scanned_pdf(pdf_file_path)
    print(f"The PDF is {'scanned' if is_scanned else 'machine-readable'}.")
