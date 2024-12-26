import os
import warnings
from KGAgent.KnowledgeGraphAgent import KnowledgeGraphAgent, load_as_json
from Specific_Agents.bank_specification_agent import DocumentSummaryAgentBank
from Specific_Agents.financial_specification_agent import FinancialSummaryAgent
from Specific_Agents.identity_specification_agent import IdentityClassificationAgent
from Specific_Agents.receipt_specification_agent import ReceiptClassificationAgent
from Main_Clf_Agents.base_ml_model import load_models_and_predict
from Main_Clf_Agents.mistral_base_agent import MistralBaseAgent
from Main_Clf_Agents.gemini_base_agent import DocumentCategoryAgentGemini
from Text_Extraction.parser_groundX import initialize_groundx_client, parse_with_groundx
from Text_Extraction.parser_llama import extract_text_from_llama_parse
from Text_Extraction.parser_pdfminer import extract_text_from_pdf
from Text_Extraction.scan_checker import is_scanned_pdf
from dotenv import load_dotenv
from collections import Counter

load_dotenv()
warnings.filterwarnings("ignore") 

def main():

    input_file = "TEST_PDFs/Aaadhar.pdf" 
    bucket_name = "sample_bucket"
    api_key = os.getenv("GROUNDX_API_KEY")
    groundx_client = initialize_groundx_client(api_key)
    file_name = input_file.split("\\")[-1]
    file_path = input_file
    file_type = "pdf"
    is_scanned = is_scanned_pdf(input_file)
    if is_scanned:
        mode = input("Enter the mode: ")
        if mode == "groundx":
            sample_text = parse_with_groundx(groundx_client, bucket_name, file_name, file_path, file_type)
        else:
            sample_text = extract_text_from_llama_parse(input_file)
    else:
        sample_text = extract_text_from_pdf(input_file)

    prediction = load_models_and_predict(sample_text)
    categorizer_mistral = MistralBaseAgent()
    category_mistral = categorizer_mistral.classify_document(sample_text)

    categorizer_gemini = DocumentCategoryAgentGemini()
    category_gemini = categorizer_gemini.categorize_document(sample_text)


    predictions = [prediction, category_mistral, category_gemini]

    most_common_prediction = Counter(predictions).most_common(1)
    base_prediction = most_common_prediction[0][0] if most_common_prediction else 'Uncategorized'

    print("Final Prediction based on Majority Voting:", base_prediction)
    
    
    knowledge_graph_agent = KnowledgeGraphAgent()
    knowledge_graph = knowledge_graph_agent.generate_knowledge_graph(sample_text, graph_type=base_prediction)
    knowledge_graph_json = load_as_json(knowledge_graph)

    if base_prediction == "bank":
        bank_agent = DocumentSummaryAgentBank()
        bank_summary = bank_agent.generate_summary(knowledge_graph_json)
        print("Bank Summary:", bank_summary)
    elif base_prediction == "finance":
        financial_agent = FinancialSummaryAgent()
        financial_summary = financial_agent.generate_summary(knowledge_graph_json)
        print("Financial Summary:", financial_summary)
    elif base_prediction == "identity":
        identity_agent = IdentityClassificationAgent()
        identity_summary = identity_agent.classify_identity(knowledge_graph_json)
        print("Identity Summary:", identity_summary)
    elif base_prediction == "receipt":
        receipt_agent = ReceiptClassificationAgent()
        receipt_summary = receipt_agent.classify_receipt(knowledge_graph_json)
        print("Receipt Summary:", receipt_summary)
    else:
        print("This is a uncategorized document.")



if __name__ == "__main__":
    main()