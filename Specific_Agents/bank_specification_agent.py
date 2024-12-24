import json
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache

set_llm_cache(InMemoryCache())

load_dotenv()

class DocumentSummaryAgentBank:
    def __init__(self, model="gemini-1.5-pro", temperature=0, max_tokens=None, timeout=None, max_retries=2):
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
        )

    def generate_summary(self, knowledge_graph_json: dict) -> dict:
        knowledge_graph_str = json.dumps(knowledge_graph_json, indent=2)

        system_prompt = f"""
        You are a highly skilled assistant specializing in extracting structured information from a JSON-based Knowledge Graph (KG) and summarizing it.

        The input will be a JSON knowledge graph that represents a document. You need to:
        1. Identify the **type** of document (e.g., KYC, Loan, Passbook, etc.).
        2. Provide a **bulleted list** of the key points extracted from the JSON knowledge graph.
        3. Write a **summary** of the document.

        Your output should contain the following:
        - **Document Type**: The type of document.
        - **Key Points (Bulleted List)**: A list of important information extracted from the knowledge graph.
        - **Summary**: A concise summary of the document.
        
        Do not include any extra commentary or explanations in the output. Keep the structure clean and relevant to the document.
        """

        messages = [
            ("system", system_prompt),
            ("human", f"Knowledge Graph:\n\n{knowledge_graph_str}"),
        ]

        ai_msg = self.llm.invoke(messages)
        return ai_msg.content


if __name__ == "__main__":
    doc_summary_agent = DocumentSummaryAgentBank()

    knowledge_graph_example = {
        "graph_type": "KYC",
        "entities": [
            {"name": "John Doe", "type": "Person", "attributes": {"DOB": "1985-05-12", "Gender": "Male", "Nationality": "Canadian"}},
            {"name": "John's Passport", "type": "Document", "attributes": {"Passport Number": "AB1234567", "Issue Date": "2020-01-15", "Expiry Date": "2030-01-15"}},
            {"name": "Address", "type": "Location", "attributes": {"Street": "123 Maple Street", "City": "Toronto", "Country": "Canada"}},
        ],
        "relationships": [
            {"from": "John Doe", "to": "John's Passport", "relationship": "Holds"},
            {"from": "John Doe", "to": "Address", "relationship": "Lives At"}
        ]
    }

    doc_summary = doc_summary_agent.generate_summary(knowledge_graph_example)
    print("Document Summary:\n", doc_summary)
