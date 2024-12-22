import re
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

def clean_json_string(input_string):
    cleaned_string = re.sub(r"```json|```", '', input_string)
    return cleaned_string.strip()

def load_as_json(input_string):
    cleaned_string = clean_json_string(input_string)
    return json.loads(cleaned_string)

load_dotenv()

class KnowledgeGraphAgent:
    def __init__(self, model="gemini-1.5-pro", temperature=0, max_tokens=None, timeout=None, max_retries=2):
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
        )

    def generate_knowledge_graph(self, text: str, graph_type: str) -> dict:
        system_prompt = f"""
        You are a highly skilled assistant that specializes in extracting information from text and structuring it into a JSON-based Knowledge Graph (KG). Your task is to analyze the provided text and generate a JSON KG based on the specified document type.

        The following are the supported types of knowledge graphs you can generate:

        1. **Bank Application KG:**
            - Extract personal details (Name, DOB, Gender, Nationality, Category).
            - Extract identity proof details (PAN, Aadhaar, Other IDs).
            - Extract contact information (Mobile, Email, Address, PIN Code).
            - Extract employment and income details.
            - Extract existing relationship details (Account Number, IFSC Code, etc.).
            - Extract card selection details (Card Type, Add-on Card Requirement).

        2. **Identity Document KG (e.g., Passport):**
            - Extract personal information (Name, DOB, Gender, Nationality).
            - Extract document details (Passport Number, Issue Date, Expiry Date, Issuing Authority).
            - Extract address details (Residential Address, City, PIN Code, Country).

        3. **Financial Statement KG:**
            - Extract financial summary (Total Income, Total Expense, Net Savings).
            - Extract transaction details (Date, Description, Debit, Credit, Balance).
            - Extract key metrics (Highest Transaction, Lowest Transaction, Average Monthly Spending).

        4. **Receipt or Invoice KG:**
            - Extract store details (Store Name, Address, Contact Info).
            - Extract purchase details (Item Names, Quantities, Prices, Subtotal, Taxes, Total).
            - Extract payment information (Payment Method, Card Number, Transaction ID).

        You must output the Knowledge Graph as a JSON object strictly following the specified structure. Do not include any extra text, explanation, or commentary. If any required fields are missing in the provided text, don't include those fields in the JSON.
        """

        messages = [
            ("system", system_prompt),
            ("human", f"Document Type: {graph_type}\n\n{text}"),
        ]

        ai_msg = self.llm.invoke(messages)
        return ai_msg.content

if __name__ == "__main__":
    kg_agent = KnowledgeGraphAgent()

    bank_application_text = """
    Karina Richards has applied for a Union Bank of India credit card, as indicated by the provided application details.  While the application includes personal information like her name, birthdate (September 9, 1970), and PAN number (I1570), several fields are incomplete.  Her gender, nationality, and category (General/SC/ST/OBC/Minority) are missing.  For identification, she provided her Aadhaar number (746666835556), but fields for other ID types and numbers are left blank.  Her contact information includes a mobile number (905.581.5443) and email address (kingheather@example.org), but the alternate number and residential address are incomplete, lacking the street address and only providing a PIN code (51141).  Crucially, the employment and income section is entirely blank, omitting details about her employment type, employer name, office address, monthly income, and other income sources.  While she indicates an existing relationship with Union Bank, providing an account number (ICMY58011763128333), the home branch and IFSC code are missing.  Finally, she has selected a Signature card type but hasn't specified whether an add-on card is required.  The application reference number is also missing from the provided details.  Overall, the application is significantly incomplete, lacking crucial information required for processing, particularly regarding employment and income details.   
    """

    kg_bank_application = kg_agent.generate_knowledge_graph(bank_application_text, graph_type="Bank Application KG")
    print("Bank Application Knowledge Graph:", kg_bank_application)
    kg_bank_json = load_as_json(kg_bank_application)
    print("Bank Application json: ", kg_bank_json)

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
    print("Passport KG json: ", kg_passport_json)

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
    print("Financial KG as JSON: ", kg_financial_json)

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
    print("Receipt KG JSON: ", kg_receipt_json)