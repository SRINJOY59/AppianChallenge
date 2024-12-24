import json
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache

set_llm_cache(InMemoryCache())

load_dotenv()

class ReceiptClassificationAgent:
    def __init__(self, model="gemini-1.5-pro", temperature=0, max_tokens=None, timeout=None, max_retries=2):
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
        )

    def classify_receipt(self, receipt_knowledge_graph_json: dict) -> dict:
        receipt_knowledge_graph_str = json.dumps(receipt_knowledge_graph_json, indent=2)

        system_prompt = f"""
        You are a highly skilled assistant that specializes in identifying and summarizing receipts and invoices based on structured knowledge graphs.
        Your task involves two main parts:

        1. **Identifying the type of receipt/invoice**:
            - Based on the provided entities and relationships, determine whether the document is a Retail Receipt, Restaurant Receipt, Invoice, Utility Bill, Hotel Invoice, Online Purchase Receipt, or any other type of receipt/invoice.
            - Use **Chain of Thought (COT) reasoning** to analyze the entities, their attributes, and relationships in the knowledge graph to infer the receipt type. 
            - **Do not rely on any explicit type information** in the knowledge graph (e.g., "graph_type" attribute). Instead, reason step-by-step about the receipt type based on its structure.

        2. **Extracting key points**:
            - After identifying the receipt/invoice type, extract important details and present them in a **bulleted list**.
            - This should include key entities, their attributes (e.g., items purchased, transaction date, total amount, tax), and relationships that are critical to understanding the receipt or invoice. 
            - Use the entities and relationships in the knowledge graph to build a clear and concise list of essential points.

        Your response should be structured as follows:
        - **Receipt/Invoice Type**: The type of receipt or invoice.
        - **Key Points (Bulleted List)**: A list of the most important information extracted from the knowledge graph.
        - **Summary**: A concise summary of the receipt or invoice's contents.

        Please process the following knowledge graph and provide your response:
        """

        messages = [
            ("system", system_prompt),
            ("human", f"Receipt/Invoice Knowledge Graph:\n\n{receipt_knowledge_graph_str}"),
        ]

        ai_msg = self.llm.invoke(messages)
        return ai_msg.content


if __name__ == "__main__":
    receipt_classification_agent = ReceiptClassificationAgent()

    retail_receipt_knowledge_graph = {
        "entities": [
            {"name": "John Doe", "type": "Customer", "attributes": {"Transaction Date": "2024-12-24", "Total Amount": "$120.50"}},
            {"name": "Store XYZ", "type": "Retail Store", "attributes": {"Store Name": "Store XYZ", "Location": "123 Main Street"}},
            {"name": "Item 1", "type": "Product", "attributes": {"Name": "Shirt", "Price": "$50.00", "Quantity": 2}},
            {"name": "Item 2", "type": "Product", "attributes": {"Name": "Pants", "Price": "$20.50", "Quantity": 1}},
        ],
        "relationships": [
            {"from": "John Doe", "to": "Retail Store", "relationship": "Purchased from"},
            {"from": "Retail Store", "to": "Item 1", "relationship": "Sells"},
            {"from": "Retail Store", "to": "Item 2", "relationship": "Sells"},
        ]
    }

    retail_receipt_summary = receipt_classification_agent.classify_receipt(retail_receipt_knowledge_graph)
    print("Retail Receipt Summary:\n", retail_receipt_summary)

    invoice_knowledge_graph = {
        "entities": [
            {"name": "ABC Corp", "type": "Seller", "attributes": {"Invoice Number": "INV-123456", "Issue Date": "2024-12-20", "Due Date": "2025-01-20", "Total Amount": "$1,200.00"}},
            {"name": "XYZ Ltd", "type": "Buyer", "attributes": {"Name": "XYZ Ltd", "Address": "456 Corporate Blvd"}},
            {"name": "Item 1", "type": "Product", "attributes": {"Name": "Office Furniture", "Price": "$500.00", "Quantity": 2}},
            {"name": "Item 2", "type": "Product", "attributes": {"Name": "Chairs", "Price": "$200.00", "Quantity": 2}},
        ],
        "relationships": [
            {"from": "ABC Corp", "to": "XYZ Ltd", "relationship": "Issued to"},
            {"from": "ABC Corp", "to": "Item 1", "relationship": "Sells"},
            {"from": "ABC Corp", "to": "Item 2", "relationship": "Sells"},
        ]
    }

    invoice_summary = receipt_classification_agent.classify_receipt(invoice_knowledge_graph)
    print("Invoice Summary:\n", invoice_summary)

    utility_bill_knowledge_graph = {
        "entities": [
            {"name": "John Doe", "type": "Customer", "attributes": {"Bill Number": "BILL-2024-12", "Due Date": "2025-01-15", "Amount Due": "$75.30"}},
            {"name": "XYZ Utility Co", "type": "Utility Provider", "attributes": {"Service": "Electricity", "Issue Date": "2024-12-15"}},
        ],
        "relationships": [
            {"from": "John Doe", "to": "XYZ Utility Co", "relationship": "Received bill from"},
        ]
    }

    utility_bill_summary = receipt_classification_agent.classify_receipt(utility_bill_knowledge_graph)
    print("Utility Bill Summary:\n", utility_bill_summary)
