import os
from pdfminer.high_level import extract_text
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar, LTTextLineHorizontal
import json
import time
import urllib.request
import logging
import ocrmypdf
import tempfile
import pdfplumber
import PyPDF2
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, List
from dotenv import load_dotenv
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from groundx import GroundX, Document
from PyPDF2 import PdfReader, PdfWriter
import math
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
if os.name == 'nt':  # Windows
   os.environ['PATH'] = r'C:\\Program Files\\gs\\gs10.04.0\\bin\\gswin64.exe' + os.pathsep + os.environ['PATH']
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('document_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class APIKeyError(Exception):
    """Custom exception for API key related errors."""
    pass

class CombinedDocumentParser:

    def is_scanned_pdf(self,pdf_path):
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


    def __init__(self, groundx_api_keys: List[str], llama_api_keys: List[str], max_pages_per_chunk: int = 20):
        """Initialize with multiple API keys for both services."""
        if not groundx_api_keys or not llama_api_keys:
            raise APIKeyError("API keys must be provided for both services")
        self.groundx_api_keys = groundx_api_keys
        self.llama_api_keys = llama_api_keys
        self.current_groundx_key_index = 0
        self.current_llama_key_index = 0
        self.max_pages_per_chunk = max_pages_per_chunk
        
        # Create temp directory for OCR files
        self.temp_dir = Path("temp_ocr")
        self.temp_dir.mkdir(exist_ok=True)
        
        # Initialize clients
        self.groundx_client = self._initialize_groundx_client()
        self.llama_parser = self._initialize_llama_parser()


    def extract_text_from_pdf(self, pdf_path):
        """Extracts all text content from the given PDF file."""
        text = extract_text(pdf_path)
        return text

    def extract_table_like_data_from_pdf(self, pdf_path):
        """Extracts text arranged in tabular format"""
        table_data = []
        for page_layout in extract_pages(pdf_path):
            page_rows = []
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    row = []
                    for text_line in element:
                        if isinstance(text_line, LTTextLineHorizontal):
                            row.append(text_line.get_text().strip())
                    if row:
                        page_rows.append(row)
            table_data.append(page_rows)
        return table_data

    def process_pdf(self, pdf_path):
        """Processes a PDF to extract its text and table-like structures."""
        extracted_text = self.extract_text_from_pdf(pdf_path)
        table_data = self.extract_table_like_data_from_pdf(pdf_path)
        return {
            "text": extracted_text,
            "table_data": table_data,
        }


    def _initialize_groundx_client(self) -> GroundX:
        """Initialize GroundX client with current API key."""
        return GroundX(api_key=self.groundx_api_keys[self.current_groundx_key_index])

    def _initialize_llama_parser(self) -> LlamaParse:
        """Initialize LlamaParse with current API key."""
        os.environ['LLAMA_CLOUD_API_KEY'] = self.llama_api_keys[self.current_llama_key_index]
        return LlamaParse(result_type='markdown')

    def _switch_groundx_key(self) -> None:
        """Switch to next available GroundX API key."""
        self.current_groundx_key_index += 1
        if self.current_groundx_key_index >= len(self.groundx_api_keys):
            raise APIKeyError("All GroundX API keys have been exhausted")
        logger.info(f"Switching to GroundX API key at index: {self.current_groundx_key_index}")
        self.groundx_client = self._initialize_groundx_client()

    def _switch_llama_key(self) -> None:
        """Switch to next available LlamaParse API key."""
        self.current_llama_key_index += 1
        if self.current_llama_key_index >= len(self.llama_api_keys):
            raise APIKeyError("All LlamaParse API keys have been exhausted")
        logger.info(f"Switching to LlamaParse API key at index: {self.current_llama_key_index}")
        self.llama_parser = self._initialize_llama_parser()

    def _ocr_pdf(self, input_path: str) -> str:
        """Apply OCR to PDF and return path to OCRed file."""
        try:
            input_path = Path(input_path)
            output_path = self.temp_dir / f"ocr_{input_path.name}"
            
            logger.info(f"Starting OCR process for {input_path}")
            ocrmypdf.ocr(
                input_file=str(input_path),
                output_file=str(output_path),
                skip_text=True,     # Skip pages that already contain text
                force_ocr=False,    # Don't force OCR if text exists
                language='eng',     # Specify language(s)
                output_type='pdf',
                optimize=0,         # No optimization to maintain quality
                deskew=True         # Straighten skewed pages
            )
            logger.info(f"OCR completed: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"OCR failed: {str(e)}")
            return str(input_path)  # Return original file if OCR fails

    def _split_pdf(self, input_path: str) -> List[str]:
        """Split PDF into smaller chunks and return list of temporary file paths."""
        temp_files = []
        pdf = PdfReader(input_path)
        total_pages = len(pdf.pages)
        chunks = math.ceil(total_pages / self.max_pages_per_chunk)
        
        logger.info(f"Splitting PDF with {total_pages} pages into {chunks} chunks")
        
        for chunk in range(chunks):
            start_page = chunk * self.max_pages_per_chunk
            end_page = min((chunk + 1) * self.max_pages_per_chunk, total_pages)
            
            pdf_writer = PdfWriter()
            for page_num in range(start_page, end_page):
                pdf_writer.add_page(pdf.pages[page_num])
            
            temp_output = f"temp_chunk_{chunk}.pdf"
            with open(temp_output, "wb") as output_file:
                pdf_writer.write(output_file)
            temp_files.append(temp_output)
            
        return temp_files

    def parse_with_groundx(self, bucket_name: str, file_name: str, file_path: str, file_type: str) -> Dict[str, Any]:
        """Parse document using GroundX with retry mechanism."""
        while True:
            try:
                # Create bucket
                bucket_response = self.groundx_client.buckets.create(name=bucket_name)
                bucket_id = bucket_response.bucket.bucket_id
                logger.info(f"Created bucket: {bucket_id}")

                # Ingest document
                ingest_response = self.groundx_client.ingest(
                    documents=[
                        Document(
                            bucket_id=bucket_id,
                            file_name=file_name,
                            file_path=file_path,
                            file_type=file_type
                        )
                    ]
                )

                # Wait for processing
                while True:
                    status_response = self.groundx_client.documents.get_processing_status_by_id(
                        process_id=ingest_response.ingest.process_id
                    )
                    if status_response.ingest.status in ["complete", "cancelled"]:
                        break
                    if status_response.ingest.status == "error":
                        raise ValueError("Error processing document with GroundX")
                    time.sleep(3)

                # Get results and clean them
                document_response = self.groundx_client.documents.lookup(id=bucket_id)
                if not document_response.documents:
                    raise ValueError("No documents found in bucket")

                # Fetch XRay results
                xray_url = document_response.documents[0].xray_url
                with urllib.request.urlopen(xray_url) as url:
                    data = json.loads(url.read().decode())
                
                # Clean and format the data
                cleaned_data = self._clean_groundx_output(data)
                return cleaned_data

            except Exception as e:
                logger.error(f"GroundX error: {str(e)}")
                try:
                    self._switch_groundx_key()
                    logger.info("Retrying with new GroundX API key...")
                except APIKeyError as ake:
                    logger.error("All GroundX API keys exhausted")
                    raise ake

    def _clean_groundx_output(self, data: Dict) -> Dict:
        """Clean and format GroundX output."""
        def clean_text(text: str) -> str:
            if not isinstance(text, str):
                return text
            
            # Replace common Unicode characters with ASCII equivalents
            replacements = {
                '\u092d': 'bh',  # भ
                '\u093e': 'aa',  # ा
                '\u0930': 'r',   # र
                '\u0924': 't',   # त
                # Add more replacements as needed
            }
            
            for unicode_char, replacement in replacements.items():
                text = text.replace(unicode_char, replacement)
            
            # Remove any remaining Unicode characters
            text = text.encode('ascii', 'ignore').decode()
            
            # Clean up whitespace
            text = ' '.join(text.split())
            return text

        def clean_dict(d: Dict) -> Dict:
            cleaned = {}
            for key, value in d.items():
                if isinstance(value, dict):
                    cleaned[key] = clean_dict(value)
                elif isinstance(value, list):
                    cleaned[key] = [clean_dict(item) if isinstance(item, dict) 
                                  else clean_text(item) if isinstance(item, str)
                                  else item for item in value]
                elif isinstance(value, str):
                    cleaned[key] = clean_text(value)
                else:
                    cleaned[key] = value
            return cleaned

        # Clean the data
        cleaned_data = clean_dict(data)
        
        # Extract only relevant fields
        formatted_data = {
            'document_info': {
                'file_name': cleaned_data.get('fileName', ''),
                'file_type': cleaned_data.get('fileType', ''),
                'language': cleaned_data.get('language', '')
            },
            'content': []
        }

        # Format chunks data
        if 'chunks' in cleaned_data:
            for chunk in cleaned_data['chunks']:
                chunk_data = {
                    'text': chunk.get('text', ''),
                    'suggested_text': chunk.get('suggestedText', ''),
                    'page_numbers': chunk.get('pageNumbers', []),
                    'content_type': chunk.get('contentType', [])
                }
                formatted_data['content'].append(chunk_data)

        return formatted_data

    def parse_with_llama(self, file_path: str) -> str:
        """Parse document using LlamaParse with retry mechanism."""
        while True:
            try:
                file_extractor = {".pdf": self.llama_parser}
                documents = SimpleDirectoryReader(
                    input_files=[file_path], 
                    file_extractor=file_extractor,
                    num_files_limit=None,  # No limit on number of files
                    filename_as_id=True
                ).load_data()
                
                combined_text = []
                for doc in documents:
                    # Properly format the text and remove any unwanted characters
                    formatted_text = doc.text.replace('\\n', '\n').strip()
                    if formatted_text:  # Only add non-empty text
                        combined_text.append(formatted_text)
                
                # Join all text with double newlines for better readability
                return '\n\n'.join(combined_text)

            except Exception as e:
                logger.error(f"LlamaParse error: {str(e)}")
                try:
                    self._switch_llama_key()
                    logger.info("Retrying with new LlamaParse API key...")
                except APIKeyError as ake:
                    logger.error("All LlamaParse API keys exhausted")
                    raise ake

    def parse_document_groundx(self, file_path: str, output_dir: str = "parsed_output") -> str:
        """Parse document using GroundX and save results."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = Path(file_path).stem
        
        try:
            # OCR the PDF first if it's a PDF file
            if file_path.lower().endswith('.pdf'):
                ocr_path = self._ocr_pdf(file_path)
                file_path = ocr_path  # Use OCRed version for further processing
                
                # Split PDF into chunks
                chunk_files = self._split_pdf(file_path)
                combined_groundx_data = []
                
                try:
                    for idx, chunk_path in enumerate(chunk_files):
                        logger.info(f"Processing chunk {idx + 1}/{len(chunk_files)}")
                        
                        # Process with GroundX
                        groundx_data = self.parse_with_groundx(
                            bucket_name=f"bucket_{file_name}_chunk_{idx}_{timestamp}",
                            file_name=f"{file_name}_chunk_{idx}",
                            file_path=chunk_path,
                            file_type="pdf"
                        )
                        combined_groundx_data.append(groundx_data)
                        
                finally:
                    # Cleanup temporary files
                    for chunk_file in chunk_files:
                        try:
                            os.remove(chunk_file)
                        except Exception as e:
                            logger.warning(f"Failed to remove temporary file {chunk_file}: {e}")
                    
                    # Cleanup OCR file
                    try:
                        if file_path != str(Path(file_path)):  # If we created an OCR file
                            Path(file_path).unlink()
                    except Exception as e:
                        logger.warning(f"Failed to remove OCR file: {e}")
                
                # Save combined results
                groundx_output = output_path / f"{file_name}_groundx_{timestamp}.json"
                with open(groundx_output, 'w', encoding='utf-8') as f:
                    json.dump(combined_groundx_data, f, indent=2, ensure_ascii=True, sort_keys=True)
                
            else:
                # Process single file normally
                groundx_data = self.parse_with_groundx(
                    bucket_name=f"bucket_{file_name}_{timestamp}",
                    file_name=file_name,
                    file_path=file_path,
                    file_type="pdf"
                )
                
                groundx_output = output_path / f"{file_name}_groundx_{timestamp}.json"
                with open(groundx_output, 'w', encoding='utf-8') as f:
                    json.dump(groundx_data, f, indent=2, ensure_ascii=True, sort_keys=True)
            
            logger.info(f"GroundX results saved to: {groundx_output}")
            return str(groundx_output)

        except Exception as e:
            logger.error(f"Error in parse_document_groundx: {str(e)}")
            raise

    def parse_document_llama(self, file_path: str, output_dir: str = "parsed_output") -> str:
        """Parse document using LlamaParse and save results."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = Path(file_path).stem
        
        try:
            # OCR the PDF first if it's a PDF file
            if file_path.lower().endswith('.pdf'):
                ocr_path = self._ocr_pdf(file_path)
                file_path = ocr_path  # Use OCRed version for further processing
                
                # Split PDF into chunks
                chunk_files = self._split_pdf(file_path)
                combined_llama_text = []
                
                try:
                    for idx, chunk_path in enumerate(chunk_files):
                        logger.info(f"Processing chunk {idx + 1}/{len(chunk_files)}")
                        
                        # Process with LlamaParse
                        llama_text = self.parse_with_llama(chunk_path)
                        if llama_text.strip():  # Only add non-empty text
                            combined_llama_text.append(llama_text)
                        
                finally:
                    # Cleanup temporary files
                    for chunk_file in chunk_files:
                        try:
                            os.remove(chunk_file)
                        except Exception as e:
                            logger.warning(f"Failed to remove temporary file {chunk_file}: {e}")
                    
                    # Cleanup OCR file
                    try:
                        if file_path != str(Path(file_path)):  # If we created an OCR file
                            Path(file_path).unlink()
                    except Exception as e:
                        logger.warning(f"Failed to remove OCR file: {e}")
                
                # Join all chunks with clear separators
                final_llama_text = "\n\n=== Page Break ===\n\n".join(combined_llama_text)
                
            else:
                # Process single file normally
                final_llama_text = self.parse_with_llama(file_path)
            
            # Save LlamaParse results
            llama_output = output_path / f"{file_name}_llama_{timestamp}.txt"
            with open(llama_output, 'w', encoding='utf-8') as f:
                f.write(final_llama_text.strip())
            
            logger.info(f"LlamaParse results saved to: {llama_output}")
            return str(llama_output)

        except Exception as e:
            logger.error(f"Error in parse_document_llama: {str(e)}")
            raise

    def mode_selection(self)->str:
        print("Enter \"groundx\" for GroundX mode")
        print("Enter \"llama\" for LlamaParse mode")
        mode = input()
        return mode
    

    def execution(self, pdf_path: str, output_dir: str = "parsed_output"):
        if(self.is_scanned_pdf(pdf_path)):
            print("scanned pdf")
            mode = self.mode_selection()
            if(mode == "groundx"):
                groundx_output = self.parse_document_groundx(
                    file_path=pdf_path,
                    output_dir="parsed_output"
                )
                return groundx_output
            else:
                llama_output = self.parse_document_llama(
                    file_path=pdf_path,
                    output_dir="parsed_output"
                )
                return llama_output
        else:
            print("machine readable pdf")
            extracted_content = self.process_pdf(pdf_path)
            
            # Create output directory if it doesn't exist
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Generate timestamp and filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = Path(pdf_path).stem
            
            # Save extracted text
            text_output = output_path / f"{file_name}_extracted_{timestamp}.txt"
            with open(text_output, 'w', encoding='utf-8') as f:
                f.write("=== Extracted Text ===\n\n")
                f.write(extracted_content["text"])
                f.write("\n\n=== Table Data ===\n")
                for page_idx, tables in enumerate(extracted_content["table_data"]):
                    f.write(f"\nPage {page_idx + 1} Tables:\n")
                    for row in tables:
                        f.write("\t".join(row) + "\n")
            
            logger.info(f"Machine-readable PDF results saved to: {text_output}")
            return str(text_output)

def main():
    # GroundX API keys
    groundx_api_keys = [
        "bfa1f574-8d3e-46dd-ac94-d7e6cf4b29e2",
        "9c758966-e5a5-4135-86b4-71e71c1063bb",
        "a666a712-d99f-43cb-be2c-31c77addc456",
        "3b3c586f-476e-494b-8b2f-c91f1b07af2f",
        "bb60249a-e71b-438f-a701-ccc63b88d71c",
        "239cdd42-e258-46fd-a89f-5210ba52bc34"
    ]

    # LlamaParse API keys
    llama_api_keys = [
        "llx-TONwLNMZee82X68phw1R6lcUgS0sXvTBRwgYn8IfoSPDj1IW",
        "llx-U9jmQbrvmzCasd6cu5c6843RPXrxAhNZVqujFQTlazpgotrF",
        "llx-NBW6S9wkcKwcxaxN7t6TclHOVntON27WKc0kDGBiCRWfuM3D",
        "llx-K1BQ2MhtL8C7XtTjM2bjcUhmoI2oWht435bqLTPYjx6TQpI6",
        "llx-8hKTBd9bI5Csjk7FH4XvA4ShPFCuCI4M6aspONEg4tIZsNCv",
        "llx-QNSS2RZRqE7Z6ToFGCOgiSdDoX7Q4ZvrA5vyC53c6gZsxbMc"
    ]

    try:
        # Initialize parser with multiple API keys and chunk size
        parser = CombinedDocumentParser(
            groundx_api_keys=groundx_api_keys,
            llama_api_keys=llama_api_keys,
            max_pages_per_chunk=15  # Adjust this value as needed
        )
        
        
        # Process document
        # #
        # # #
        # # # #
        input_file = "text_extraction/Dz18s4RU0AAxcWU.pdf"  # Replace with your PDF file
        # # # #
        # # #
        # #
        #
        
        parser.execution(input_file, "parsed_output")
        
    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()