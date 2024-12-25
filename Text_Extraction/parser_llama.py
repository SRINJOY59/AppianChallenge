import os
from dotenv import load_dotenv
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
load_dotenv()

def extract_text_from_llama_parse(pdf_path):
    try:
        parser = LlamaParse(
            result_type="markdown"
        )

        file_extractor = {".pdf": parser}
        
        documents = SimpleDirectoryReader(
            input_files=[pdf_path],
            file_extractor=file_extractor
        ).load_data()
        
        if not documents:
            return "No text could be extracted"
            
        extracted_text = documents[0].text_resource.text
        
        return extracted_text

    except Exception as e:
        return f"Error extracting text: {str(e)}"

