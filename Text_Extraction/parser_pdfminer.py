from pdfminer.high_level import extract_text

def extract_text_from_pdf(pdf_path):

    try:
        text = extract_text(pdf_path)
        
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""  



if __name__ == "__main__":
    pdf_path = r"C:\Users\Srinjoy\OneDrive\Desktop\Appian\AppianChallenge\kutta_pdf.pdf" 
    extracted_text = extract_text_from_pdf(pdf_path)
    print(extracted_text)