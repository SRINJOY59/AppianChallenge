import re
import json
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import json
from dataclasses import dataclass
from typing import Dict, Any
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache


set_llm_cache(InMemoryCache())

def clean_json_string(input_string):
    cleaned_string = re.sub(r"```json|```", '', input_string)
    return cleaned_string.strip()

def load_as_json(input_string):
    cleaned_string = clean_json_string(input_string)
    return json.loads(cleaned_string.strip())

load_dotenv()

class KnowledgeGraphAgent:
    def __init__(self, model="mixtral-8x7b-32768", temperature=0, max_tokens=None, timeout=None, max_retries=2):
        self.llm = ChatGroq(
            model=model,
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
        )

    def generate_knowledge_graph(self, text: str, graph_type: str) -> dict:
        system_prompt = f"""
        You are a highly skilled assistant specializing in extracting structured information from unstructured text and organizing it into a JSON-based Knowledge Graph (KG). Your task is to analyze the provided unstructured text and identify key entities, attributes, and relationships within the content. You will then generate a JSON representation of the information, ensuring it includes the most relevant data.

        The input will also specify the type of Knowledge Graph to generate, which is provided as `graph_type`. The graph type can be one of the following:

        1. **Bank Application KG**: Personal details, identity proof, contact information, employment details, and card selection.
        2. **Identity Document KG**: Personal details and document details.
        3. **Financial Statement KG**: Financial summary, transactions, and key metrics.
        4. **Receipt or Invoice KG**: Store details, purchase details, and payment information.

        Based on the provided graph type, you should extract and organize the relevant information accordingly.

        Your output should be a structured JSON object that represents the information extracted from the unstructured text, with the following attributes:
        - `graph_type`: The type of knowledge graph (e.g., "Bank Application KG").
        - `attributes`: Relevant attributes and relationships between entities, categorized based on the `graph_type`.
        - `relevant_information` : The relevant key-value pairs extracted from the text in dictionary format.
        Do not include any extra commentary or explanations in the output. Ensure that the structure is clean and only includes relevant data points.
        """

        messages = [
            ("system", system_prompt),
            ("human", f"Graph Type: {graph_type}\n\nText: {text}"),
        ]
        

        ai_msg = self.llm.invoke(messages)
        

        return ai_msg.content

if __name__ == "__main__":
    kg_agent = KnowledgeGraphAgent()

    # Example inputs
    bank_application_text = """
    Karina Richards has applied for a Union Bank of India credit card, as indicated by the provided application details. While the application includes personal information like her name, birthdate (September 9, 1970), and PAN number (I1570), several fields are incomplete. Her gender, nationality, and category (General/SC/ST/OBC/Minority) are missing. For identification, she provided her Aadhaar number (746666835556), but fields for other ID types and numbers are left blank. Her contact information includes a mobile number (905.581.5443) and email address (kingheather@example.org), but the alternate number and residential address are incomplete, lacking the street address and only providing a PIN code (51141). Crucially, the employment and income section is entirely blank, omitting details about her employment type, employer name, office address, monthly income, and other income sources. While she indicates an existing relationship with Union Bank, providing an account number (ICMY58011763128333), the home branch and IFSC code are missing. Finally, she has selected a Signature card type but hasn't specified whether an add-on card is required. The application reference number is also missing from the provided details. Overall, the application is significantly incomplete, lacking crucial information required for processing, particularly regarding employment and income details.
    """

    kg_bank_application = kg_agent.generate_knowledge_graph(bank_application_text, graph_type="Bank Application KG")
    print("Bank Application Knowledge Graph:", kg_bank_application)
    kg_bank_json = load_as_json(kg_bank_application)
    print("Bank Application JSON: \n")
    print(kg_bank_json)
    print(kg_bank_json['relevant_information'])

    passport_text = """
    Name: John Doe
    DOB: 1985-05-12
    Gender: Male
    Nationality: Canadian
    Passport Number: AB1234567
    Issue Date: 2020-01-15
    Expiry Date: 2030-01-15
    Issuing Authority: Canadian Government
    Address: 123 Maple Street, Toronto, Ontario, M5H 2N2, Canada
    """

    kg_passport = kg_agent.generate_knowledge_graph(passport_text, graph_type="Identity Document KG")
    print("Passport Knowledge Graph:", kg_passport)
    kg_passport_json = load_as_json(kg_passport)
    print("Passport KG JSON: \n", kg_passport_json)
    print(kg_passport_json['relevant_information'])

    financial_statement_text = """
    Total Income: $120,000
    Total Expense: $80,000
    Net Savings: $40,000

    Transactions:
    - Date: 2024-01-10, Description: Grocery Store, Debit: $200, Credit: $0, Balance: $39,800
    - Date: 2024-01-12, Description: Salary, Debit: $0, Credit: $10,000, Balance: $49,800
    - Date: 2024-01-15, Description: Utility Bill, Debit: $150, Credit: $0, Balance: $49,650

    Highest Transaction: $10,000
    Lowest Transaction: $150
    Average Monthly Spending: $6,667
    """

    kg_financial_statement = kg_agent.generate_knowledge_graph(financial_statement_text, graph_type="Financial Statement KG")
    print("Financial Statement Knowledge Graph:", kg_financial_statement)
    kg_financial_json = load_as_json(kg_financial_statement)
    print("Financial KG JSON: \n", kg_financial_json)
    print(kg_financial_json['relevant_information'])
    

    receipt_text = """
    Store Name: Best Buy
    Address: 456 Tech Avenue, San Jose, CA 95110
    Contact: (408) 555-0199

    Items Purchased:
    - Laptop: 1 x $1,200
    - Mouse: 2 x $25

    Subtotal: $1,250
    Taxes: $100
    Total: $1,350

    Payment Method: Visa
    Card Number: **** **** **** 1234
    Transaction ID: TXN456789123
    """

    kg_receipt = kg_agent.generate_knowledge_graph(receipt_text, graph_type="Receipt or Invoice KG")
    print("Receipt Knowledge Graph:", kg_receipt)
    kg_receipt_json = load_as_json(kg_receipt)
    print("Receipt KG JSON: \n", kg_receipt_json)
    print(kg_receipt_json['relevant_information'])
