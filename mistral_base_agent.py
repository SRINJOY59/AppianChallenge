from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

class DocumentCategoryAgent:
    def __init__(self, model="mixtral-8x7b-32768", temperature=0, max_tokens=None, timeout=None, max_retries=2):
        """
        Initialize the Document Category Agent with the Mistral AI model and configuration parameters.
        """
        self.llm = ChatGroq(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
        )

    def categorize_document(self, input_text: str) -> str:
        """
        Classify the given input text into one of the predefined categories:
        - Bank
        - Finance
        - Receipt
        - Identity
        If none apply, return 'Uncategorized'.
        """
        system_prompt = """
        You are an expert document classifier assistant. Your task is to analyze the given text and determine its category based on its content.

        The possible categories are:
        - **Bank**: Texts that are related to banking operations, such as applications for accounts, credit card details, or account numbers.
        - **Finance**: Texts related to financial documents like income statements, pay stubs, tax returns, or other monetary transactions.
        - **Receipt**: Texts that resemble transaction records, purchase receipts, or payment acknowledgments.
        - **Identity**: Texts containing personal identification information, such as driver's licenses, passports, or government-issued ID numbers.

        ### Guidelines:
        1. Analyze the overall context, structure, and keywords in the text.
        2. Return the **most relevant category**. If the text does not fit any category, respond with **'Uncategorized'**.

        ### Few-Shot Examples:

        **Example 1:**
        Input: 
        Application for a new checking account with Union Bank of India. 
        Account Number: 1234567890
        Home Branch: Mumbai
        IFSC Code: UBIN123456
        Output: bank

        **Example 2:**
        Input: 
        Gross Monthly Income: $4,500
        Deductions: $1,200
        Net Pay: $3,300
        Income Statement for January 2024
        Output: finance

        **Example 3:**
        Input: 
        Receipt No: 1456
        Date: 2024-08-01
        Amount Paid: $200
        Payment Method: Credit Card
        Description: Purchase of electronics from Tech Store
        Output: receipt

        **Example 4:**
        Input: 
        Driver's License
        Name: John Doe
        License Number: D12345678
        Date of Birth: 1980-01-01
        Issued: California, USA
        Output: identity

        **Example 5:**
        Input: 
        This is a random paragraph discussing the benefits of exercise for mental health. It contains no financial or personal identification information.
        Output: Uncategorized

        Output will be just a single word : bank or finance or identity or receipt
        no other text will be there in output.
        Now, classify the following text:
        {input_text}
        """
        
        messages = [
            ("system", system_prompt),
            ("human", input_text),
        ]
        
        ai_msg = self.llm.invoke(messages)
        return ai_msg.content.strip()

if __name__ == "__main__":
    categorizer = DocumentCategoryAgent()


    sample_text = """
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
    """

    category = categorizer.categorize_document(sample_text)
    print("Document Category:", category)
