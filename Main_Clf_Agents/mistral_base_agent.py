from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache

set_llm_cache(InMemoryCache())

load_dotenv()

class MistralBaseAgent:
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

    def classify_document(self, input_text: str) -> str:
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

        ### Chain-of-Thought Reasoning:
        Please carefully consider the text and think through the category it most likely falls into. 
        - Identify keywords that could be related to financial operations, banking, identification, or receipts.
        - Focus on the structure and content. 
        Only one word should be returned in the output: bank, finance, receipt, identity, or Uncategorized.
        Don't put any other details in the output

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
    categorizer = MistralBaseAgent()

    sample_text = """
    ----------------------------------------
                  SHOP NAME
               Address: 123 Main St
               Phone: (123) 456-7890
    ----------------------------------------
    Date: 2023-10-01
    Time: 14:30
    ----------------------------------------
    Item                Qty     Price
    ----------------------------------------
    Item 1              2       $10.00
    Item 2              1       $5.50
    Item 3              3       $7.25
    ----------------------------------------
    Subtotal:                     $32.75
    Tax (5%):                    $1.64
    ----------------------------------------
    Total:                      $34.39
    ----------------------------------------
    Thank you for shopping with us!
    ----------------------------------------
    """

    category = categorizer.classify_document(sample_text)
    print("Document Category:", category)
