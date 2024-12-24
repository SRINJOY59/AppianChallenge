from KGAgent.KnowledgeGraphAgent import KnowledgeGraphAgent, load_as_json
from Specific_Agents.bank_specification_agent import DocumentSummaryAgentBank
from Specific_Agents.financial_specification_agent import FinancialSummaryAgent
from Specific_Agents.identity_specification_agent import IdentityClassificationAgent
from Specific_Agents.receipt_specification_agent import ReceiptClassificationAgent
from Main_Clf_Agents.base_ml_model import load_models_and_predict
from Main_Clf_Agents.mistral_base_agent import MistralBaseAgent
from Main_Clf_Agents.gemini_base_agent import DocumentCategoryAgentGemini
def main():
    
    sample_text = """
    ----------------------------------------
                 Loan Application
                 Name: John Doe
                 DOB: 1985-05-12
                 Gender: Male
                 Nationality: Canadian
                 Passport Number: AB1234567
                 Issue Date: 2020-01-15
                 Expiry Date: 2030-01-15
                 Issuing Authority: Canadian Government
                 Loan Amount: $100,000
                 Loan Tenure: 10 years
                 Loan Type: Personal Loan
                 Loan Purpose: Home Renovation
                 Loan Start Date: 2024-01-01
                 Loan End Date: 2034-01-01
                 Loan Status: Approved
    ----------------------------------------
    """
    
    prediction = load_models_and_predict(sample_text)
    categorizer_mistral = MistralBaseAgent()
    category_mistral = categorizer_mistral.classify_document(sample_text)

    categorizer_gemini = DocumentCategoryAgentGemini()
    category_gemini = categorizer_gemini.categorize_document(sample_text)

    from collections import Counter

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
