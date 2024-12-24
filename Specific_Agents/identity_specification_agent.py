import json
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache

set_llm_cache(InMemoryCache())

load_dotenv()

class IdentityClassificationAgent:
    def __init__(self, model="gemini-1.5-pro", temperature=0, max_tokens=None, timeout=None, max_retries=2):
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
        )

    def classify_identity(self, identity_knowledge_graph_json: dict) -> dict:
        identity_knowledge_graph_str = json.dumps(identity_knowledge_graph_json, indent=2)

        system_prompt = f"""
        You are a highly skilled assistant that specializes in identifying and summarizing identity documents based on structured knowledge graphs. 
        Your task involves two main parts:

        1. **Identifying the type of identity document**: 
            - Based on the provided entities and relationships, determine whether the document is an Aadhar Card, Passport, Driving License, National ID, Voter ID, or any other type of identity document. 
            - Use **Chain of Thought (COT) reasoning** to analyze the entities, their attributes, and relationships in the knowledge graph to infer the identity document type. 
            - **Do not rely on any explicit type information** in the knowledge graph (e.g., "graph_type" attribute). Instead, reason step-by-step about the document type based on its structure.

        2. **Extracting key points**:
            - After identifying the identity document type, extract important details and present them in a **bulleted list**.
            - This should include key entities, their attributes, and relationships that are critical to understanding the identity document. 
            - Use the entities and relationships in the knowledge graph to build a clear and concise list of essential points.

        Your response should be structured as follows:
        - **Document Type**: The type of identity document.
        - **Key Points (Bulleted List)**: A list of the most important information extracted from the knowledge graph.
        - **Summary**: A concise summary of the document's contents, based on the extracted key points.
        
        ** Don't put your COT thoughts in the response. Just give the key points and summary. **

        Please process the following knowledge graph and provide your response:
        """

        messages = [
            ("system", system_prompt),
            ("human", f"Identity Knowledge Graph:\n\n{identity_knowledge_graph_str}"),
        ]

        ai_msg = self.llm.invoke(messages)
        return ai_msg.content


if __name__ == "__main__":
    identity_classification_agent = IdentityClassificationAgent()

    aadhar_knowledge_graph = {
        "entities": [
            {"name": "John Doe", "type": "Individual", "attributes": {"Aadhar Number": "1234-5678-9101", "Full Name": "John Doe", "Gender": "Male", "Date of Birth": "1985-10-25"}},
            {"name": "Aadhar Card", "type": "Identity Document", "attributes": {"Aadhar Number": "1234-5678-9101", "Issuing Authority": "UIDAI", "Status": "Active"}},
        ],
        "relationships": [
            {"from": "John Doe", "to": "Aadhar Card", "relationship": "Possesses"},
        ]
    }

    aadhar_summary = identity_classification_agent.classify_identity(aadhar_knowledge_graph)
    print("Aadhar Card Summary:\n", aadhar_summary)

    passport_knowledge_graph = {
        "entities": [
            {"name": "Jane Doe", "type": "Individual", "attributes": {"Passport Number": "XYZ123456", "Full Name": "Jane Doe", "Gender": "Female", "Date of Birth": "1990-06-15", "Nationality": "Indian"}},
            {"name": "Passport", "type": "Identity Document", "attributes": {"Passport Number": "XYZ123456", "Issuing Authority": "Government of India", "Status": "Active", "Issue Date": "2020-05-10", "Expiry Date": "2030-05-10"}},
        ],
        "relationships": [
            {"from": "Jane Doe", "to": "Passport", "relationship": "Possesses"},
        ]
    }

    passport_summary = identity_classification_agent.classify_identity(passport_knowledge_graph)
    print("Passport Summary:\n", passport_summary)

    driving_license_knowledge_graph = {
        "entities": [
            {"name": "Mark Smith", "type": "Individual", "attributes": {"License Number": "DL-123456789", "Full Name": "Mark Smith", "Date of Birth": "1982-02-20", "Gender": "Male", "License Type": "Car"}},
            {"name": "Driving License", "type": "Identity Document", "attributes": {"License Number": "DL-123456789", "Issuing Authority": "Delhi Transport Department", "Status": "Active", "Issue Date": "2015-06-12", "Expiry Date": "2025-06-12"}},
        ],
        "relationships": [
            {"from": "Mark Smith", "to": "Driving License", "relationship": "Possesses"},
        ]
    }

    driving_license_summary = identity_classification_agent.classify_identity(driving_license_knowledge_graph)
    print("Driving License Summary:\n", driving_license_summary)
