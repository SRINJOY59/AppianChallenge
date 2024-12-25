import os
import json
import time
import logging
import urllib.request
from pathlib import Path
from typing import Dict, Any
from groundx import GroundX, Document
from dotenv import load_dotenv
load_dotenv()

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

def initialize_groundx_client(api_key: str) -> GroundX:
    """Initialize GroundX client with API key."""
    return GroundX(api_key=api_key)

def parse_with_groundx(groundx_client: GroundX, bucket_name: str, file_name: str, file_path: str, file_type: str) -> Dict[str, Any]:
    """Parse document using GroundX with retry mechanism."""
    try:
        bucket_response = groundx_client.buckets.create(name=bucket_name)
        bucket_id = bucket_response.bucket.bucket_id
        logger.info(f"Created bucket: {bucket_id}")

        ingest_response = groundx_client.ingest(
            documents=[
                Document(
                    bucket_id=bucket_id,
                    file_name=file_name,
                    file_path=file_path,
                    file_type=file_type
                )
            ]
        )

        while True:
            status_response = groundx_client.documents.get_processing_status_by_id(
                process_id=ingest_response.ingest.process_id
            )
            if status_response.ingest.status in ["complete", "cancelled"]:
                break
            if status_response.ingest.status == "error":
                raise ValueError("Error processing document with GroundX")
            time.sleep(3)

        # Get results and clean them
        document_response = groundx_client.documents.lookup(id=bucket_id)
        if not document_response.documents:
            raise ValueError("No documents found in bucket")

        # Fetch XRay results
        xray_url = document_response.documents[0].xray_url
        with urllib.request.urlopen(xray_url) as url:
            data = json.loads(url.read().decode())

        cleaned_data = clean_groundx_output(data)
        return cleaned_data['content'][0]['text']

    except Exception as e:
        logger.error(f"GroundX error: {str(e)}")
        raise

def clean_groundx_output(data: Dict) -> Dict:
    """Clean and format GroundX output."""
    def clean_text(text: str) -> str:
        if not isinstance(text, str):
            return text

        replacements = {
            '\u092d': 'bh',  # भ
            '\u093e': 'aa',  # ा
            '\u0930': 'r',   # र
            '\u0924': 't',   # त
            # Add more replacements as needed
        }

        for unicode_char, replacement in replacements.items():
            text = text.replace(unicode_char, replacement)

        text = text.encode('ascii', 'ignore').decode()

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

    cleaned_data = clean_dict(data)

    formatted_data = {
        'document_info': {
            'file_name': cleaned_data.get('fileName', ''),
            'file_type': cleaned_data.get('fileType', ''),
            'language': cleaned_data.get('language', '')
        },
        'content': []
    }

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

def main():
    """
    Main function to parse a sample PDF document using GroundX and print the cleaned results.
    """
    api_key = os.getenv("GROUNDX_API_KEY")  
    bucket_name = "sample_bucket"
    file_name = "kutta_pdf"  
    file_path = r"C:\Users\Srinjoy\OneDrive\Desktop\Appian\AppianChallenge\kutta_pdf.pdf"  
    file_type = "pdf"

    try:
        groundx_client = initialize_groundx_client(api_key)
        logger.info("GroundX client initialized successfully.")

        logger.info("Starting document parsing...")
        cleaned_data = parse_with_groundx(
            groundx_client,
            bucket_name=bucket_name,
            file_name=file_name,
            file_path=file_path,
            file_type=file_type
        )
        logger.info("Document parsing completed successfully.")

        print("Cleaned GroundX Data:")
        print(cleaned_data)

    except APIKeyError:
        logger.error("API key is missing or invalid. Please check your configuration.")

if __name__ == "__main__":
    main()