import json
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache

set_llm_cache(InMemoryCache())

load_dotenv()

class FinancialSummaryAgent:
    def __init__(self, model="llama3-70b-8192", temperature=0, max_tokens=None, timeout=None, max_retries=2):
        self.llm = ChatGroq(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
        )

    def generate_summary(self, financial_knowledge_graph_json: dict) -> dict:
        financial_knowledge_graph_str = json.dumps(financial_knowledge_graph_json, indent=2)

        system_prompt = f"""
        You are a highly skilled assistant that specializes in understanding and summarizing financial documents based on structured knowledge graphs. 
        Your task involves two main parts:

        1. **Identifying the type of financial document**: 
            - Based on the provided entities and relationships, determine whether the document is a Loan Agreement, Investment Agreement, Non-Disclosure Agreement (NDA), 
              Purchase Agreement, Employment Contract, Partnership Agreement, Lease Agreement, Credit Agreement, Divorce Settlement Agreement, M&A Agreement, 
              Warranty Agreement, Supply Agreement, Guarantee Agreement, Stock Option Agreement, Debt Settlement Agreement, or another type of financial document. 
            - Use **Chain of Thought (COT) reasoning** to analyze the entities, their attributes, and relationships in the knowledge graph to infer the document type. 
            - **Do not rely on any explicit type information** in the knowledge graph (e.g., "graph_type" attribute). Instead, reason step-by-step about the document type based on its structure.

        2. **Extracting key points**:
            - After identifying the document type, extract important details and present them in a **bulleted list**.
            - This should include key entities, their attributes, and relationships that are critical to understanding the financial document. 
            - Use the entities and relationships in the knowledge graph to build a clear and concise list of essential points.

        Your response should be structured as follows:
        - **Document Type**: The type of financial document.
        - **Key Points (Bulleted List)**: A list of the most important information extracted from the knowledge graph.
        - **Summary**: A concise summary of the document's contents, based on the extracted key points.
        
        ** Don't put your COT thoughts in the response. Just give the key points and summary. **

        Please process the following knowledge graph and provide your response:
        """

        messages = [
            ("system", system_prompt),
            ("human", f"Financial Knowledge Graph:\n\n{financial_knowledge_graph_str}"),
        ]

        ai_msg = self.llm.invoke(messages)
        end_time = time.time()
        return ai_msg.content


if __name__ == "__main__":
    financial_summary_agent = FinancialSummaryAgent()

    loan_agreement_knowledge_graph = {
        "entities": [
            {"name": "John Doe", "type": "Borrower", "attributes": {"Loan Amount": "$500,000", "Interest Rate": "5%", "Loan Term": "10 years"}},
            {"name": "XYZ Bank", "type": "Lender", "attributes": {"Loan Amount": "$500,000", "Interest Rate": "5%", "Loan Term": "10 years", "Repayment Schedule": "Monthly"}},
            {"name": "Collateral", "type": "Loan Clause", "attributes": {"Collateral": "Property in Downtown"}},
        ],
        "relationships": [
            {"from": "John Doe", "to": "XYZ Bank", "relationship": "Borrows from"},
            {"from": "XYZ Bank", "to": "Loan Clause", "relationship": "Secured by"},
        ]
    }

    loan_agreement_summary = financial_summary_agent.generate_summary(loan_agreement_knowledge_graph)
    print("Loan Agreement Summary:\n", loan_agreement_summary)

    purchase_agreement_knowledge_graph = {
        "entities": [
            {"name": "John Doe", "type": "Buyer", "attributes": {"Product": "Office Furniture", "Price": "$50,000", "Payment Terms": "Full upfront"}},
            {"name": "ABC Corp", "type": "Seller", "attributes": {"Product": "Office Furniture", "Price": "$50,000", "Delivery Date": "2024-02-01"}},
        ],
        "relationships": [
            {"from": "John Doe", "to": "ABC Corp", "relationship": "Purchases from"},
            {"from": "ABC Corp", "to": "John Doe", "relationship": "Delivers to"},
        ]
    }

    purchase_agreement_summary = financial_summary_agent.generate_summary(purchase_agreement_knowledge_graph)
    print("Purchase Agreement Summary:\n", purchase_agreement_summary)
