from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()


class SummarizationAgent:
    def __init__(self, model="gemini-1.5-pro", temperature=0, max_tokens=None, timeout=None, max_retries=2):
        
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
        )

    def get_summary(self, text: str, summary_type: str = "brief") -> str:
        system_prompt = f"""
        You are a highly capable assistant that specializes in text summarization. Your task is to summarize the given text in a manner that suits the specified summary type.

        The following are the different types of summaries you can generate:

        1. **Brief Summary:**
            - Summarize the text in a concise manner while retaining the key points and core message.
            - Do not exceed more than 3-4 sentences. Focus on the main idea, avoiding unnecessary details.
            - The summary should be direct and to the point.

        2. **Detailed Summary:**
            - Provide a thorough and comprehensive summary of the text.
            - Include important details, examples, and supporting information. Aim to cover all major points and nuances of the original text.
            - The summary should be longer, but avoid overwhelming detail; it should be informative yet succinct.

        3. **Key Points Summary:**
            - Provide a bullet-point list of the key points or takeaways from the text.
            - Each bullet should represent a separate, important idea or fact, with no unnecessary elaboration.
            - This should be shorter than the detailed summary but provide more granularity than the brief summary.

        4. **Thematic Summary:**
            - Summarize the text by identifying the major themes or concepts discussed, rather than detailing individual facts.
            - Focus on the overarching message or core concepts that the text conveys.
            - This summary type may be slightly abstract and more conceptual than the others.

        Please summarize the following text based on the chosen summary type: {summary_type}.
        """

        messages = [
            ("system", system_prompt),
            ("human", text),
        ]
        
        ai_msg = self.llm.invoke(messages)
        return ai_msg.content

if __name__ == "__main__":
    summarizer = SummarizationAgent()

    text_to_summarize = """
**UNION BANK OF INDIA - CREDIT CARD APPLICATION**
Application Reference: 

PERSONAL DETAILS
--------------
Name: Karina Richards
Date of Birth: 1970-09-09
Gender: 
Nationality: 
Category:   # General/SC/ST/OBC/Minority

IDENTITY PROOF
------------
PAN Number: I1570
Aadhaar Number: 746666835556
Other ID Type: 
Other ID Number: 

CONTACT INFORMATION
-----------------
Mobile Number: 905.581.5443
Alternate Number: 
Email Address: kingheather@example.org
Residential Address: 
PIN Code: 51141

EMPLOYMENT & INCOME
----------------
Employment Type: 
Employer Name: 
Office Address: 
Monthly Income: 
Other Income Sources: 

EXISTING RELATIONSHIPS
-------------------
Union Bank Account: 
Account Number: ICMY58011763128333
Home Branch: 
IFSC Code: 

CARD SELECTION
------------
Card Type: Signature  # Classic/Gold/Platinum/Signature
Add-on Card Required:     """

    key_points_summary = summarizer.get_summary(text_to_summarize, summary_type="key points")
    print("Key Points Summary:", key_points_summary)

    detailed_summary = summarizer.get_summary(text_to_summarize, summary_type="detailed")
    print("Detailed Summary:", detailed_summary)
